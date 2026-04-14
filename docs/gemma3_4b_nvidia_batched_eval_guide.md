# Gemma3 4B on NVIDIA host OpenCL — batched eval-ll guide

NVIDIA 호스트 OpenCL에서 Gemma3 4B `--eval-ll`이 10문항 이상 배치에서 `CL_OUT_OF_RESOURCES`/`CL_INVALID_COMMAND_QUEUE`로 파열하는 문제(`issues/gemma3_4b_nvidia_batch_accumulation_20260414.md`)의 실용 워크어라운드.

요지: **배치를 쪼개서 `generate` 프로세스를 재시작하면** 드라이버의 deferred-release 누적이 프로세스 경계에서 완전히 정리된다. 실패 시 자동 재시도까지 포함한 래퍼 스크립트를 제공한다.

## 언제 사용하는가

- 호스트 개발 환경(x86_64 + NVIDIA OpenCL 1.2)에서 Gemma3 4B eval-ll을 **GPU로** 돌리고 싶을 때.
- 프로덕션 Android/Adreno에는 영향 없음 — 그쪽은 이 현상 자체가 없다.
- CPU(`--backend cpu`) / PoCL(`OCL_PLATFORM=portable`)이 이미 안정이지만 느리다고 판단할 때.

## 성능 기준 (40Q RACE-H smoke)

| 경로 | wall time | 비고 |
|------|-----------|------|
| `--backend cpu` | ~450 s | 완벽 안정, 단일 실행 |
| PoCL (`OCL_PLATFORM=portable`) | ~650 s | 완벽 안정, OpenCL stack 유지 |
| **NVIDIA GPU + 배치 분할** | **~260 s** | 본 가이드 — chunk 4, retry 3회 |

## 사전 준비

1. 릴리스 빌드: `cargo build --release --bin generate`
2. 평가 배치 JSON (항목별 `prompt` + `continuation` 또는 `choices` 필드).
3. Python 3.10+ (stdlib만 사용).

## 사용법

```bash
OCL_PLATFORM=NVIDIA python3 scripts/eval_ll_batched.py \
  --model-path /path/to/gemma3-4b \
  --eval-batch /path/to/questions.json \
  --output /tmp/eval_result.json \
  --chunk-size 4 \
  --max-retries 3 \
  -- --backend opencl --kv-type f32 --max-seq-len 4096 \
     --qcf-mode both --greedy
```

- `--` 이후의 모든 플래그는 `generate` 바이너리에 그대로 전달.
- stderr의 per-question 진행 로그는 실시간 스트리밍 (배치별로 묶여 나옴).
- 각 chunk의 stdout(JSON)만 파싱해 `--output`에 통합.

## 파라미터 가이드

| 옵션 | 기본값 | 권장 |
|------|--------|------|
| `--chunk-size` | 4 | NVIDIA Q5-Q10 실패 패턴을 회피하는 안전선. 8은 가끔 실패, 2~3은 startup overhead 증가. |
| `--max-retries` | 3 | NVIDIA 드라이버 상태는 **확률적** — 같은 입력이 한 번 실패 후 통과하는 경우가 많다. 2~3회면 충분. |
| `--work-dir` | (tempdir) | 실패 분석 시 chunk별 stdout/JSON을 남기려면 경로 지정. |

## 실패 시 동작

- chunk 실행 실패 → stderr 기록 → 최대 `max_retries` 회까지 재실행 → 모두 실패 시 스크립트가 exit 1.
- 재시도에서 JSON 파싱 실패하면 해당 attempt의 stdout을 `work_dir/chunk_NNN_stdout_attemptM.txt`에 저장.

## 결과 JSON 포맷

단일 `generate --eval-ll` 출력과 동일한 스키마. `results`는 전체 질문 concat, `wall_time_s`는 GPU 작업 시간 합(프로세스 startup 제외). 기존 분석 스크립트와 호환.

## 알려진 제약

- **재현성 비결정적**: 동일 입력에서도 드라이버 상태에 따라 실패 지점이 달라진다. `max_retries`가 있어도 극히 드물게 모두 실패할 수 있음. 이 경우 `OCL_PLATFORM=portable`로 fallback.
- **chunk_size=1~2**는 startup overhead로 CPU보다 느려질 수 있음 (40Q × 15s startup = 10분).
- `--qcf-mode` / `--eviction-policy` 등 질문 간 상태를 유지하는 기능은 **chunk 경계에서 초기화**된다. 이번 이슈의 재현 시나리오는 eval-ll full-prefill(kv_budget=0)이라 문제 없음. Ratio-mode/per-question budget이 chunk-local이 되는 걸 주의.

## 참고

- 이슈: `issues/gemma3_4b_nvidia_batch_accumulation_20260414.md`
- 상위 이슈(부분 해결): `issues/host_opencl_nvidia_fallback_20260414.md`
- 이 wrapper: `scripts/eval_ll_batched.py`
- 구현 배경: DK=256 flash attention 커널 추가로 CPU-fallback read_buffer를 제거한 뒤에도 드라이버 누적이 잔존 → 외부 프로세스 경계로 경감.
