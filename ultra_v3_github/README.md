# Ultra V3 (GitHub용 핵심 코드 정리)

이 폴더는 `LLM-CVR` 레포의 Ultra V3 구현 중에서, **논문/설명/공유에 필요한 핵심 모듈만** 추려서 정리한 버전입니다.  
**재현(학습/데이터/체크포인트 로딩)이 목적이 아니라**, *“v3가 무엇을 했는지 코드로 보여주기”*가 목적입니다.

---

## 포함된 코드 (핵심 3가지 개선)

1) **RGTA: Retained-Guided Temporal Attention**  
- 파일: `rgta.py`  
- 역할: `t_exc`로 “지울 프레임”을 찾고, `t_ret`로 “남길 프레임”을 찾은 뒤  
  \((1-A_{exc})\cdot w_{exc} + A_{ret}\cdot w_{ret}\) 형태로 **erasure mask**를 만들어 temporal pooling 합니다.

2) **MSTE: Multi-Scale Temporal Encoder**  
- 파일: `temporal.py`  
- 역할: frame-level + segment-level transformer로 **짧은 동작/긴 맥락**을 같이 반영합니다.

3) **IPAF: Injected-Prioritized Adaptive Fusion**  
- 파일: `ipaf.py`  
- 역할:
  - Stage1: `ref_original` vs `ref_cleaned`를 샘플별로 선택/보간  
  - Stage2: `[ref_final, t_exc, t_inj]` 가중치를 예측하되 **injected bias + injected clamp**로 injected를 우선합니다.

마지막으로 위 3개를 묶는 최소 forward 래퍼가 있습니다.
- 파일: `ultra_v3_core.py` (`UltraV3Core`)

---

## 포함하지 않은 것 (원본 레포에 존재)

- **SpanDecomposer**: 문장에서 retained/excluded/injected 텍스트 span을 뽑는 부분  
- **텍스트/비디오 인코딩(BLIP/CLIP)**: `t_ret/t_exc/t_inj`, `ref_frames`를 만드는 부분  
- **학습 루프 / 데이터셋 / loss / 평가**

즉, 이 폴더는 아래 입력이 이미 준비됐다고 가정합니다.
- `ref_frames`: (B, T, D) reference video frame features
- `t_ret/t_exc/t_inj`: (B, D) role별 span embeddings

---

## 빠른 실행 (모양 확인용)

```bash
cd CoVR_jk/LLM-CVR
python3 ultra_v3_github/demo_forward.py
```

출력은 tensor shape 위주로만 나옵니다(학습/정확도 목적 아님).

---

## 원본 코드와의 대응 관계

- 원본 메인 구현: `models/llm_cvr_combiner_ultra_v3.py`
  - `RetainedGuidedTemporalAttention` → `ultra_v3_github/rgta.py`
  - `MultiScaleTemporalEncoder`, `FrameTemporalEncoder` → `ultra_v3_github/temporal.py`
  - `InjectedPrioritizedFusion` → `ultra_v3_github/ipaf.py`
  - `LLMCVRCombinerUltraV3Revised`의 핵심 흐름 → `ultra_v3_github/ultra_v3_core.py`

원본 실험 엔트리포인트(학습):
- `run_ultra_v3.sh` → `train_experiments.py --config configs/ultra_v3.yaml`

