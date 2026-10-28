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
        ckpt_path = ckpt_path or ROOT / "runs" / "r3" / "best.pt"
        mert_path = mert_path or ROOT / "models" / "MERT-v1-95M"
        stats_path = stats_path or ROOT / "models" / "feat_stats.json"
        stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        self.mean = {(g, k): v[0] for g, d in stats.items() for k, v in d.items()}
        self.std = {(g, k): v[1] for g, d in stats.items() for k, v in d.items()}
        group_dims = {"prosody": len(PROSODY_KEYS), "timbre": len(TIMBRE_KEYS),
                      "pref": len(PREF_KEYS), "dm": 68, "fame": 2}
        self.model = SingingAssessor(str(mert_path), group_dims)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
        for s in starts:
            w = torch.from_numpy(wav[s:s + WIN]).float()
            if pad:
                w = torch.nn.functional.pad(w, (0, pad))
            wm = torch.zeros(WIN, dtype=torch.bool)
            wm[:min(n, WIN)] = True
            out, _ = self.model(w.unsqueeze(0).to(self.device),
                                wm.unsqueeze(0).to(self.device), f_dict, m_dict)
            for h in HEADS:
                acc[h].append(float(out[h].item()))
        return {h: round(float(np.mean(v)), 2) for h, v in acc.items()}
