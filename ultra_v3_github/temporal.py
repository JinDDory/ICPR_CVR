from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MultiScaleTemporalEncoder(nn.Module):
    """
    MSTE (Multi-Scale Temporal Encoder)

    - frame-level transformer (세밀한 동작)
    - segment-level transformer (긴 맥락)
    - concat 후 projection으로 하나의 ref representation을 만듦

    원본 구현 출처:
    - `models/llm_cvr_combiner_ultra_v3.py` 의 `MultiScaleTemporalEncoder`
    """

    def __init__(self, feature_dim: int, segment_size: int = 4):
        super().__init__()
        self.segment_size = segment_size

        self.frame_encoder = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.segment_encoder = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

        self.scale_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def forward(self, ref_frames: torch.Tensor) -> torch.Tensor:
        # ref_frames: (B, T, D)
        B, T, D = ref_frames.shape

        frame_features = self.frame_encoder(ref_frames)  # (B,T,D)
        frame_pooled = frame_features.mean(dim=1)  # (B,D)

        num_segments = T // self.segment_size
        if num_segments > 0:
            clipped = ref_frames[:, : num_segments * self.segment_size, :]  # (B, S*seg, D)
            clipped = clipped.view(B, num_segments, self.segment_size, D)
            segment_tokens = clipped.mean(dim=2)  # (B, S, D)
            segment_features = self.segment_encoder(segment_tokens)  # (B,S,D)
            segment_pooled = segment_features.mean(dim=1)  # (B,D)
        else:
            segment_pooled = frame_pooled

        fused = self.scale_fusion(torch.cat([frame_pooled, segment_pooled], dim=-1))
        return F.normalize(fused, dim=-1)


class FrameTemporalEncoder(nn.Module):
    """
    Ablation 용 single-scale (frame-only) encoder.

    원본 구현 출처:
    - `models/llm_cvr_combiner_ultra_v3.py` 의 `FrameTemporalEncoder`
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.frame_encoder = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, ref_frames: torch.Tensor) -> torch.Tensor:
        frame_features = self.frame_encoder(ref_frames)
        frame_pooled = frame_features.mean(dim=1)
        return F.normalize(frame_pooled, dim=-1)

