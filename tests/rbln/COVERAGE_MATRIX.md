# RBLN 벤치마크 커버리지 매트릭스

이 문서는 RBLN 백엔드 테스트가 **어떤 wrapper 패밀리 × 모달리티 × tier**를
커버하는지 명시한다. ~180개 데이터셋 × ~208개 모델을 전수 실행하지 않고,
**모든 wrapper 패밀리와 모든 모달리티를 최소 1회씩** 대표 셀로 커버한다.

## Tier 개요

| Tier | 목적 | 하드웨어 | 게이팅 |
|---|---|---|---|
| T0   | 오프라인 단위 테스트 (프롬프트 패리티/디스패치/config merge/레지스트리) | 불필요 | CI |
| T0.5 | `evaluate` 룰베이스 채점 (judge 없음) | 불필요 | CI |
| T1   | MockRBLNModel 로 `vlmeval.inference` 파이프라인 스모크 | 불필요 | CI |
| T2   | 실제 RBLN 추론 per 패밀리 (컴파일-or-로드) | **NPU 필요** | NPU 호스트 |
| T3   | 점수 회귀 (real judge, ±epsilon) | NPU + judge | 주기적(선택) |

T0/T0.5/T1 은 `/workspace/.eval` venv 의 `pytest tests/rbln/` 로 NPU 없이 통과한다.
T2 는 `scripts/rbln_smoke.sh` 로 NPU 호스트에서 수행한다 (CI 필수 체크 아님, 아티팩트로 기록).

## 패밀리 × 모달리티 (11개 concrete 패밀리)

`tensor_parallel_size` 는 **컴파일 시 결정**되는 값이며 고정 디바이스 카운트 게이트가
아니다. 캐시 아티팩트가 있으면 그대로 로드하고, 없으면 아래 default tp(=`_ARCH_TABLE`)
또는 `--rbln-kwargs` override 로 컴파일한다.

| Wrapper 클래스 | 모달리티 | 대표 모델 (basename) | 대표 데이터셋 | default tp (`_ARCH_TABLE`) | tp override 키 (`--rbln-kwargs rbln_config.*`) | Tier |
|---|---|---|---|---|---|---|
| RBLNQwen2VL      | image MCQ + video | Qwen2.5-VL-7B-Instruct | MMBench_DEV_EN / Video-MME_8frame | 8 | `tensor_parallel_size` (+ `visual.max_seq_lens`) | T0,T2 |
| RBLNQwen3VL¹     | image MCQ (video) | Qwen3-VL-8B-Instruct | MMStar | 8 | `tensor_parallel_size` (+ `visual.max_seq_lens`) | T0,T2 |
| RBLNCosmosReason1| video (reasoning) | Cosmos-Reason1-7B | Video-MME_8frame | 8 | `tensor_parallel_size` (+ `visual.max_seq_lens`) | T0,T2 |
| RBLNLlava        | image MCQ/VQA | llava-1.5-7b-hf | MMStar | 4 | `language_model.tensor_parallel_size` | T0,T2 |
| RBLNLlavaNext    | image MCQ (multi) | llava-v1.6-mistral-7b-hf | MMBench_DEV_EN | 4 | `language_model.tensor_parallel_size` | T0,T2 |
| RBLNIdefics3²    | interleaved multi-image | Idefics3-8B-Llama3 | MMStar | 8 | `text_model.tensor_parallel_size` | T0,T2 |
| RBLNGemma3       | image MCQ (4b/12b/27b) | gemma-3-4b-it | AI2D_TEST | 4 (12b=8, 27b=16) | `language_model.tensor_parallel_size` | T0,T2 |
| RBLNPixtral      | multi-image | pixtral-12b | MMBench_DEV_EN | 8 | `language_model.tensor_parallel_size` | T0,T2 |
| RBLNPaliGemma    | single-image (`INTERLEAVE=False`) | paligemma-3b-mix-448 | AI2D_TEST | 4 | `language_model.tensor_parallel_size` | T0,T2 |
| RBLNPaliGemma2   | single-image (`INTERLEAVE=False`) | paligemma2-3b-mix-224 | AI2D_TEST | 4 | `language_model.tensor_parallel_size` | T0,T2 |
| RBLNBlip2³       | single-image VQA/caption | blip2-opt-2.7b | OCRBench / ChartQA_TEST | 1 | `language_model.tensor_parallel_size` | T0,T2 |

각주:
- ¹ **알려진 quirk**: `Qwen3-VL-*-RBLN` 레지스트리 엔트리는 현재 `RBLNQwen2VL` 에
  바인딩되어 `Qwen2VLPromptMixin` 을 사용한다 (Qwen3 mixin 미적용). `RBLNQwen3VL`
  클래스 자체는 존재하며 `Qwen3VLPromptMixin` 을 쓴다. `test_registry_integrity.py::test_qwen3_binding_uses_qwen2_prompt_mixin`
  가 이 현재 동작을 잠금/표면화한다. 의도 여부는 미해결 질문.
- ² Idefics3 는 모델레벨 `build_prompt` 가 없고 `_format_for_dataset` 에 프롬프트
  로직이 내장되어 있다 (`test_prompt_parity.py` EMBEDDED 버킷).
- ³ BLIP-2: optimum-rbln 0.10.3 의 `generate` 출력은 프롬프트 토큰을 **포함**하므로 trim 을 켠다
  (`_DECODE_TRIM=True` 상속, `_DECODE_STRIP=True`). 또한 BLIP-2-OPT 는 `"Question: {q} Answer:"`
  컨벤션이라야 답을 생성하므로 `generate_inner` 가 프롬프트를 그 형식으로 감싼다 (실 NPU 검증됨).

## 모달리티 커버리지 (스칼라 채점기)

| 모달리티 | 채점 방식 | 대표 데이터셋 | judge | Tier |
|---|---|---|---|---|
| image MCQ | exact / choice match | MMBench, MMStar, AI2D | (대개) judge 또는 룰 | T1 |
| image VQA (OCR/chart) | relaxed_accuracy / anls | ChartQA_TEST | 룰베이스 (judge 불필요) | T0.5 |
| video | frame 추출 후 동일 채점 | Video-MME_8frame | 데이터셋별 | T2 (`requires_video_deps`) |
| text-only | 데이터셋 채점기 | TEXT 데이터셋(`--limit`) | 데이터셋별 | T2 |

## 캐시 격리 (basename 충돌 방지)

`RBLNVLMBase._save_dir()` 는 **CWD 기준 `basename(model_path)`** 로 아티팩트를
캐시한다 (`--work-dir` 와 독립). 서로 다른 모델이 같은 basename 을 가지면
캐시가 충돌할 수 있으므로, `scripts/rbln_smoke.sh` 는 **패밀리별 디렉토리로 cd**
하여 캐시를 격리한다.

## 검증 명령

```sh
# 오프라인 (CI): T0 + T0.5 + T1
source /workspace/.eval/bin/activate
export LMUData=/workspace/quantization/LMUData
pytest tests/rbln/ -q

# lazy-import 불변식 단독
python -c "import sys, vlmeval.vlm.rbln; assert 'optimum.rbln' not in sys.modules"

# T2 (NPU 호스트): 컴파일-or-로드 스모크
bash scripts/rbln_smoke.sh            # 전 패밀리
bash scripts/rbln_smoke.sh qwen2_vl   # 단일 패밀리
```
