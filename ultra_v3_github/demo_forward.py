"""
Ultra V3 core forward demo (학습/데이터 없이 "모듈 동작 형태"만 확인)

실행:
  python3 ultra_v3_github/demo_forward.py
"""

import os
import sys

try:
    import torch
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "이 데모는 PyTorch가 필요합니다.\n"
        "- 설치 예: pip install torch\n"
        "- (GPU/OS에 따라 설치 방법이 다를 수 있어요)\n"
    ) from e

# 어디서 실행해도 import가 되도록 LLM-CVR 루트를 sys.path에 추가
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ultra_v3_github import UltraV3Core  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    B, T, D = 2, 8, 256
    ref_frames = torch.randn(B, T, D)

    # span embeddings는 원래 SpanDecomposer + BLIP/CLIP 인코더에서 만들어집니다.
    # 여기서는 형태만 맞춰서 dummy로 넣습니다.
    t_ret = torch.randn(B, D)
    t_exc = torch.randn(B, D)
    t_inj = torch.randn(B, D)

    model = UltraV3Core(
        feature_dim=D,
        hidden_dim=2048,
        segment_size=4,
        use_rgta=True,
        use_multi_scale=True,
        use_ipaf=True,
    )

    out = model(ref_frames=ref_frames, t_ret=t_ret, t_exc=t_exc, t_inj=t_inj)
    print("predicted:", tuple(out.predicted.shape))
    print("ref_original:", tuple(out.ref_original.shape))
    print("ref_cleaned:", tuple(out.ref_cleaned.shape))
    print("erasure_mask:", None if out.erasure_mask is None else tuple(out.erasure_mask.shape))
    print("stage1_weights:", None if out.stage1_weights is None else tuple(out.stage1_weights.shape))
    print("stage2_weights:", None if out.stage2_weights is None else tuple(out.stage2_weights.shape))


if __name__ == "__main__":
    main()

