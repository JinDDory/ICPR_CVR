from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class InjectedPrioritizedFusion(nn.Module):
    """
    IPAF (Injected-Prioritized Adaptive Fusion)

    Stage 1) ref_original vs ref_cleaned를 샘플별로 선택/보간 (ref selector)
    Stage 2) [ref_final, t_exc, t_inj]를 샘플별로 가중합 하되 injected에 bias + clamp로 우선권 부여
             -> transformer fusion

    원본 구현 출처:
    - `models/llm_cvr_combiner_ultra_v3.py` 의 `InjectedPrioritizedFusion`
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()

        self.ref_selector = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
        )

        # [ref, exc, inj] bias (learnable)
        self.injected_bias = nn.Parameter(torch.tensor([0.35, 0.10, 0.55], dtype=torch.float32))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.trans_fusion = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        ref_original: torch.Tensor,  # (B,D)
        ref_cleaned: torch.Tensor,  # (B,D)
        t_exc: torch.Tensor,  # (B,D)
        t_inj: torch.Tensor,  # (B,D)
        *,
        inj_min: float = 0.45,
        inj_max: float = 0.70,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stage 1: reference selection
        stage1_weights = self.ref_selector(torch.cat([ref_original, ref_cleaned], dim=-1))  # (B,2)
        ref_stack = torch.stack([ref_original, ref_cleaned], dim=1)  # (B,2,D)
        ref_final = (ref_stack * stage1_weights.unsqueeze(-1)).sum(dim=1)  # (B,D)
        ref_final = F.normalize(ref_final, dim=-1)

        # Stage 2: injected-prioritized fusion
        logits = self.fusion_mlp(torch.cat([ref_final, t_exc, t_inj], dim=-1))  # (B,3)
        logits = logits + self.injected_bias.unsqueeze(0)
        stage2_weights = F.softmax(logits, dim=-1)  # (B,3)

        # clamp injected weight only, then renormalize
        stage2_weights_clamped = stage2_weights.clone()
        stage2_weights_clamped[:, 2] = torch.clamp(stage2_weights[:, 2], min=inj_min, max=inj_max)
        stage2_weights = stage2_weights_clamped / stage2_weights_clamped.sum(dim=-1, keepdim=True)

        features = torch.stack([ref_final, t_exc, t_inj], dim=1)  # (B,3,D)
        weighted = features * stage2_weights.unsqueeze(-1)  # (B,3,D)

        fused = self.trans_fusion(weighted)  # (B,3,D)
        out = self.fc(fused.mean(dim=1))  # (B,D)
        predicted = F.normalize(out, dim=-1)
        return predicted, stage1_weights, stage2_weights

