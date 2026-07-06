"""
Ultra V3 (GitHub-friendly core modules)

이 폴더는 논문/설명/코드 공유 목적의 "핵심 모듈"만 포함합니다.
- 학습 파이프라인, 데이터셋, SpanDecomposer, BLIP/CLIP 로딩은 포함하지 않습니다.
"""

from .rgta import RetainedGuidedTemporalAttention
from .temporal import MultiScaleTemporalEncoder, FrameTemporalEncoder
from .ipaf import InjectedPrioritizedFusion
from .ultra_v3_core import UltraV3Core

__all__ = [
    "RetainedGuidedTemporalAttention",
    "MultiScaleTemporalEncoder",
    "FrameTemporalEncoder",
    "InjectedPrioritizedFusion",
    "UltraV3Core",
]

