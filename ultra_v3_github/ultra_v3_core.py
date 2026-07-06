from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .rgta import RetainedGuidedTemporalAttention
from .temporal import FrameTemporalEncoder, MultiScaleTemporalEncoder
from .ipaf import InjectedPrioritizedFusion


@dataclass(frozen=True)
class UltraV3Output:
    predicted: torch.Tensor  # (B,D)
    ref_original: torch.Tensor  # (B,D)
    ref_cleaned: torch.Tensor  # (B,D)
    erasure_mask: Optional[torch.Tensor]  # (B,T) if RGTA enabled else None
    stage1_weights: Optional[torch.Tensor]  # (B,2) if IPAF enabled else None
    stage2_weights: Optional[torch.Tensor]  # (B,3) if IPAF enabled else None


class UltraV3Core(nn.Module):
    """
    Ultra V3의 "핵심 forward 로직"만 분리한 GitHub용 코어 모듈.

    포함:
    - Multi-Scale Temporal Encoder (또는 frame-only ablation)
    - Retained-Guided Temporal Attention (RGTA, optional)
    - Injected-Prioritized Adaptive Fusion (IPAF, optional)

    제외(원본 레포에 있음):
    - SpanDecomposer (문장에서 retained/excluded/injected 텍스트 추출)
    - BLIP/CLIP 인코더 (텍스트/비디오 feature extraction)
    - 학습 루프, loss, 데이터로더

    사용자는 이미 만들어진 span embedding(t_ret/t_exc/t_inj)과 ref_frames를 넣어 forward 하면 됩니다.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 256,
        hidden_dim: int = 2048,
        segment_size: int = 4,
        excluded_balance_init: float = 0.5,
        use_rgta: bool = True,
        use_multi_scale: bool = True,
        use_ipaf: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size
        self.excluded_balance_init = excluded_balance_init
        self.use_rgta = use_rgta
        self.use_multi_scale = use_multi_scale
        self.use_ipaf = use_ipaf

        self.multi_scale_encoder: Optional[nn.Module]
        if use_multi_scale:
            self.multi_scale_encoder = MultiScaleTemporalEncoder(feature_dim, segment_size=segment_size)
            self.frame_encoder = None
        else:
            self.multi_scale_encoder = None
            self.frame_encoder = FrameTemporalEncoder(feature_dim)

        self.rgta = (
            RetainedGuidedTemporalAttention(
                feature_dim, num_heads=4, excluded_balance_init=excluded_balance_init
            )
            if use_rgta
            else None
        )

        self.ipaf = InjectedPrioritizedFusion(feature_dim, hidden_dim) if use_ipaf else None
        self.simple_fusion = (
            nn.Sequential(
                nn.Linear(feature_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, feature_dim),
            )
            if not use_ipaf
            else None
        )

    def forward(
        self,
        *,
        ref_frames: torch.Tensor,  # (B,T,D)
        t_ret: torch.Tensor,  # (B,D)
        t_exc: torch.Tensor,  # (B,D)
        t_inj: torch.Tensor,  # (B,D)
        normalize_inputs: bool = True,
    ) -> UltraV3Output:
        if normalize_inputs:
            t_ret = F.normalize(t_ret, dim=-1)
            t_exc = F.normalize(t_exc, dim=-1)
            t_inj = F.normalize(t_inj, dim=-1)

        # Step 1: (Multi-)scale encode
        if self.use_multi_scale:
            ref_original = self.multi_scale_encoder(ref_frames)  # type: ignore[operator]
        else:
            ref_original = self.frame_encoder(ref_frames)  # type: ignore[union-attr]

        # Step 2: RGTA (optional)
        if self.use_rgta:
            ref_cleaned, erasure_mask = self.rgta(ref_frames, t_ret, t_exc)  # type: ignore[misc]
        else:
            ref_cleaned, erasure_mask = ref_original, None

        # Step 3: Fusion (IPAF or simple)
        if self.use_ipaf:
            predicted, stage1_w, stage2_w = self.ipaf(ref_original, ref_cleaned, t_exc, t_inj)  # type: ignore[misc]
        else:
            ref_final = ref_cleaned if self.use_rgta else ref_original
            fusion_in = torch.cat([ref_final, t_exc, t_inj], dim=-1)
            predicted = F.normalize(self.simple_fusion(fusion_in), dim=-1)  # type: ignore[union-attr]
            stage1_w, stage2_w = None, None

        return UltraV3Output(
            predicted=predicted,
            ref_original=ref_original,
            ref_cleaned=ref_cleaned,
            erasure_mask=erasure_mask,
            stage1_weights=stage1_w,
            stage2_weights=stage2_w,
        )

