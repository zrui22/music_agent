"""声乐评估网络：MERT 主干 + 多路专家特征 cross-attention 融合 + 多任务评分头

结构（对应设计白皮书"混合专家特征双路输入"）：
  主路: MERT-95M(24kHz) -> 末层隐状态 masked mean+std 池化 -> audio 摘要 z_a
  辅路: 5 组专家特征(prosody/timbre/pref/dm/fame) 各自 MLP -> d_exp token
        缺失组用可学习 null token 替代
  融合: z_a 作 query 对专家 token 做 multi-head cross-attention（互相整合）
  评分: [z_a, fused] -> MLP -> 6 个独立头(pitch/rhythm/stability/timbre/expression/overall)
        输出 sigmoid*5 到 0-5 分
"""
import torch
import torch.nn as nn

from src.model.heads import GROUPS, HEADS


class SingingAssessor(nn.Module):
    def __init__(self, mert_path, group_dims, d_audio=768, d_fuse=512,
                 d_exp=128, freeze_layers=6, dropout=0.1, n_faults=0):
        super().__init__()
        from transformers import AutoModel
        # ModelScope 版自带 modeling_MERT.py（继承 HubertModel），需 trust_remote_code
        self.mert = AutoModel.from_pretrained(mert_path, trust_remote_code=True)
        # 冻结卷积特征提取器 + 底部 N 层（Stage A：稳定优先，只调高层）
        for p in self.mert.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.mert.feature_projection.parameters():
            p.requires_grad = False
        layers = self.mert.encoder.layers
        for layer in layers[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False

        self.group_dims = group_dims
        self.group_proj = nn.ModuleDict({
            g: nn.Sequential(nn.Linear(d, d_exp), nn.GELU(), nn.Linear(d_exp, d_exp))
            for g, d in group_dims.items()})
        self.null_tokens = nn.ParameterDict({
            g: nn.Parameter(torch.randn(d_exp) * 0.02) for g in group_dims})
        self.exp_out = nn.Linear(d_exp, d_fuse)

        self.audio_pool = nn.Sequential(
            nn.Linear(d_audio * 2, d_fuse), nn.GELU(), nn.Linear(d_fuse, d_fuse))
        self.cross = nn.MultiheadAttention(d_fuse, num_heads=8, batch_first=True)
        self.fuse_norm = nn.LayerNorm(d_fuse)

        self.head_mlp = nn.Sequential(
            nn.Linear(d_fuse * 2, d_fuse), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({h: nn.Linear(d_fuse, 1) for h in HEADS})
        # B3 归因头：n_faults=0 时缺省不建（兼容 Stage A 旧 ckpt 严格加载）
        self.n_faults = n_faults
        if n_faults:
            self.fault_head = nn.Sequential(
                nn.Linear(d_fuse, d_fuse // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_fuse // 2, n_faults))

    def forward(self, wav, wav_mask, feats, masks):
        """wav: [B,T] 24kHz; wav_mask: [B,T] bool; feats/masks: dict g->[B,d]/[B]"""
        hidden = self.mert(input_values=wav).last_hidden_state  # [B,T',768]
        # MERT 内部卷积下采样 ~320x，池化掩码按同比例折算
        t = hidden.shape[1]
        m = wav_mask.float()
        m = torch.nn.functional.avg_pool1d(
            m.unsqueeze(1), kernel_size=max(wav.shape[1] // t, 1),
            stride=max(wav.shape[1] // t, 1))[:, 0, :t]
        m = (m > 0.5).float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (hidden * m).sum(dim=1) / denom
        var = ((hidden - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
        z_a = self.audio_pool(torch.cat([mean, var.sqrt()], dim=-1))  # [B,d_fuse]

        tokens = []
        for g in GROUPS:
            x = self.group_proj[g](feats[g])              # [B,d_exp]
            null = self.null_tokens[g].expand_as(x)
            x = torch.where(masks[g].unsqueeze(-1) > 0, x, null)
            tokens.append(self.exp_out(x))
        kv = torch.stack(tokens, dim=1)                    # [B,5,d_fuse]
        fused, attn_w = self.cross(z_a.unsqueeze(1), kv, kv)   # [B,1,d_fuse]
        fused = self.fuse_norm(fused.squeeze(1) + z_a)

        h = self.head_mlp(torch.cat([z_a, fused], dim=-1))
        scores = {name: torch.sigmoid(head(h)).squeeze(-1) * 5.0
                  for name, head in self.heads.items()}
        flog = self.fault_head(h) if self.n_faults else None
        return scores, attn_w.squeeze(1), flog
