import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class RetainedGuidedTemporalAttention(nn.Module):
    """
    RGTA (Retained-Guided Temporal Attention)

    - Excluded span(t_exc): 제거해야 할 프레임을 찾는 attention
    - Retained span(t_ret): 유지해야 할 프레임을 찾는 attention
    - 두 신호를 샘플별로 동적으로 합쳐 erasure_mask를 만든 뒤 temporal pooling

    원본 구현 출처:
    - `models/llm_cvr_combiner_ultra_v3.py` 의 `RetainedGuidedTemporalAttention`
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, excluded_balance_init: float = 0.5):
        super().__init__()
        self.excluded_balance_init = excluded_balance_init

        self.attn_excluded = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_retained = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )

        # outputs: [w_exc, w_ret]
        self.mask_balance = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

        # optional bias init to prefer excluded at start
        if excluded_balance_init != 0.5:
            logit_diff = math.log(excluded_balance_init / (1.0 - excluded_balance_init))
            with torch.no_grad():
                self.mask_balance[2].bias[0] = logit_diff / 2
                self.mask_balance[2].bias[1] = -logit_diff / 2

    def forward(
        self,
        ref_frames: torch.Tensor,  # (B, T, D)
        t_ret: torch.Tensor,  # (B, D)
        t_exc: torch.Tensor,  # (B, D)
        *,
        return_balance_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = ref_frames.shape

        # excluded attention map
        _, attn_exc = self.attn_excluded(
            query=t_exc.unsqueeze(1),
            key=ref_frames,
            value=ref_frames,
        )
        attn_exc = attn_exc.squeeze(1)  # (B, T)

        # retained attention map
        _, attn_ret = self.attn_retained(
            query=t_ret.unsqueeze(1),
            key=ref_frames,
            value=ref_frames,
        )
        attn_ret = attn_ret.squeeze(1)  # (B, T)

        # dynamic balance
        balance = self.mask_balance(torch.cat([t_exc, t_ret], dim=-1))  # (B, 2)
        w_exc = balance[:, 0:1]  # (B,1)
        w_ret = balance[:, 1:2]  # (B,1)

        # erasure mask (clamped)
        erasure_mask = (1.0 - attn_exc) * w_exc + attn_ret * w_ret
        erasure_mask = torch.clamp(erasure_mask, min=0.0, max=1.0)  # (B,T)

        # weighted temporal pooling
        denom = erasure_mask.sum(dim=1, keepdim=True) + 1e-6
        ref_cleaned = (ref_frames * erasure_mask.unsqueeze(-1)).sum(dim=1) / denom
        ref_cleaned = F.normalize(ref_cleaned, dim=-1)

        if return_balance_weights:
            return ref_cleaned, erasure_mask, balance
        return ref_cleaned, erasure_mask

