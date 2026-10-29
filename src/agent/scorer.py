"""R3 模型推理封装：多窗口主观评分（免参考兜底客观分同出）

推理分层策略（与模型卡一致）：
  - pitch/rhythm 有参考时用算法真值呈现，网络头仅作免参考兜底/交叉校验
  - 主观四头（stability/timbre/expression/overall）由网络直出
  - dm/fame 组对用户唱段天然缺失 -> null token（训练时同款 mask 机制）
  - 长音频均匀取多窗 10s 推理取均值，降低单窗偶然性
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.model.heads import (GROUPS, HEADS, PROSODY_KEYS, TIMBRE_KEYS, PREF_KEYS,
                             LOG1P_KEYS, SR, WIN)
from src.model.singing_net import SingingAssessor

ROOT = Path(__file__).resolve().parents[2]
KEYS = {"prosody": PROSODY_KEYS, "timbre": TIMBRE_KEYS, "pref": PREF_KEYS}


class Scorer:
    def __init__(self, ckpt_path=None, mert_path=None, stats_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if ckpt_path is None:
            for cand in (ROOT / "models" / "singing_scorer_v3.pt",
                         ROOT / "models" / "singing_scorer_v2.pt",
                         ROOT / "models" / "singing_scorer_v1.pt",
                         ROOT / "runs" / "r3" / "best.pt"):
                if cand.exists():
                    ckpt_path = cand
                    break
        if ckpt_path is None:
            raise RuntimeError(
                "未找到评分模型权重：请按 README「模型资产」下载 "
                "singing_scorer_v3.pt 到 models/ 目录后重试")
        if mert_path is None:
            local = ROOT / "models" / "MERT-v1-95M"
            mert_path = local if local.exists() else "m-a-p/MERT-v1-95M"
        stats_path = Path(stats_path or ROOT / "models" / "feat_stats.json")
        if not stats_path.exists():
            raise RuntimeError(
                f"未找到特征归一化统计 {stats_path}：请按 README「模型资产」"
                "下载 feat_stats.json 到 models/ 目录")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.mean = {(g, k): v[0] for g, d in stats.items() for k, v in d.items()}
        self.std = {(g, k): v[1] for g, d in stats.items() for k, v in d.items()}
        group_dims = {"prosody": len(PROSODY_KEYS), "timbre": len(TIMBRE_KEYS),
                      "pref": len(PREF_KEYS), "dm": 68, "fame": 2}
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:  # 旧版 torch 或 ckpt 含非张量对象时回退
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # B3 起的 ckpt 在 args 里带 n_faults；旧 ckpt 无此键 -> 0（无故障头）
        n_faults = int((ckpt.get("args") or {}).get("n_faults", 0) or 0)
        self.n_faults = n_faults
        self.model = SingingAssessor(str(mert_path), group_dims, n_faults=n_faults)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)

    def _norm_group(self, g, vals: dict):
        dim = {"prosody": len(PROSODY_KEYS), "timbre": len(TIMBRE_KEYS),
               "pref": len(PREF_KEYS)}[g]
        x = np.zeros(dim, dtype=np.float32)
        if not vals:
            return x, 0.0
        for i, k in enumerate(KEYS[g]):
            v = vals.get(k)
            if v is None:
                continue
            v = math.log1p(max(float(v), 0.0)) if k in LOG1P_KEYS else float(v)
            z = (v - self.mean.get((g, k), 0.0)) / self.std.get((g, k), 1.0)
            x[i] = float(np.clip(z, -5, 5))
        return x, 1.0

    @torch.no_grad()
    def score(self, wav: np.ndarray, feats: dict, n_windows: int = 5) -> dict:
        """wav: float32 一维 @24kHz（长度不限）；feats: {prosody|timbre|pref: dict}。
        返回 {head: 0-5 分}（多窗均值）。"""
        n = wav.shape[0]
        if n <= WIN:
            starts = [0]
            pad = WIN - n
        else:
            n_windows = max(1, min(n_windows, (n - 1) // WIN + 1))
            starts = np.linspace(0, n - WIN, n_windows).astype(int).tolist()
            pad = 0
        f_dict, m_dict = {}, {}
        for g in GROUPS:
            if g in KEYS:
                x, m = self._norm_group(g, feats.get(g))
            else:  # dm / fame：用户唱段无此信息，固定缺失
                x, m = np.zeros(68 if g == "dm" else 2, dtype=np.float32), 0.0
            f_dict[g] = torch.from_numpy(x).unsqueeze(0).to(self.device)
            m_dict[g] = torch.tensor([m], device=self.device)
        acc = {h: [] for h in HEADS}
        facc = []
        for s in starts:
            w = torch.from_numpy(wav[s:s + WIN]).float()
            if pad:
                w = torch.nn.functional.pad(w, (0, pad))
            wm = torch.zeros(WIN, dtype=torch.bool)
            wm[:min(n, WIN)] = True
            out, _, flog = self.model(w.unsqueeze(0).to(self.device),
                                      wm.unsqueeze(0).to(self.device),
                                      f_dict, m_dict)
            for h in HEADS:
                acc[h].append(float(out[h].item()))
            if flog is not None:
                facc.append(torch.sigmoid(flog[0]).float().cpu().numpy())
        scores = {h: round(float(np.mean(v)), 2) for h, v in acc.items()}
        if facc:
            from src.model.heads import FAULT_TYPES
            prob = np.mean(facc, axis=0)
            scores["faults"] = {t: round(float(prob[i]), 3)
                                for i, t in enumerate(FAULT_TYPES)}
        return scores
