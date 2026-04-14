# Long-context Decode 최적화 Sprint — Handoff (2026-04-14)

본 세션에서 long-context decode 갭(llm_rs2 vs llama.cpp)을 76% 해소했고, 잔존 작업을 다음 세션으로 인계.

## 1. 본 sprint 결과 요약

### 최종 수치 (Galaxy S25, Adreno 830, Qwen 2.5-1.5B Q4_0)

| 지표 | 이전 (B-4 ON) | 현재 (B-4 revert) | 개선 |
|---|---:|---:|---:|
| Decode wall slope | 12.45 μs/n_kv | **7.32 μs/n_kv** | **-41%** |
| 갭 vs llama (5.70) | 6.75 | **1.62** | **-76%** |
| n_kv=2048 TBT | 58.88 ms | **40.08 ms** | -32% |
| n_kv=4500 TBT | 83.30 ms | **53.28 ms** | -36% |

### 한 줄 요약

`engine/kernels/flash_attn_f32_f16.cl` Q1 kernel만 SLM tree-reduce 패턴으로 교체 (B-4 sub_group_reduce → tree-reduce). 정확도 회귀 0, 재현성 ±0.5%, prefill 영향 없음.

## 2. 핵심 발견 (다음 세션이 알아야 할 것)

### 2.1 측정 방법론 — apples-to-apples 주의

- `--profile-events` 켜면 wall slope이 +2.29 μs/n_kv 부풀려짐. 같은 세션 내 비교는 OK, 다른 세션 비교는 NG.
- llama.cpp는 `CL_QUEUE_PROFILING_ENABLE` 자체에 더 큰 패널티 (3-4× 부풀림). **engine 간 profile-events 직접 비교 절대 금지**. wall-clock만 신뢰.
- thermal AP `mStatus=0` 강제 + 240s cooldown 필수. 본 세션 ctx 2048/6k 한 번 thermal 오염 사례 있음.

### 2.2 microbench_flash_attn — 디바이스 격리 측정 도구

`engine/src/bin/microbench_flash_attn.rs` 신규. 핵심 능력:
- production `flash_attn_f32_f16.cl` 무수정 컴파일
- vendored `llamacpp_q1_flash_attn.cl` 동시 컴파일 (cross-run)
- 4 variants × 2 layouts × 4 n_kv × engines 매트릭스 측정
- CL event μs로 thermal-격리, 단일 dispatch 격리
- ROI 가장 명확한 디바이스 측정 도구. **다른 kernel 변경 검증 시 적극 활용 권장**.

### 2.3 가설 verdict 누적

| # | 가설 | 결론 | 증거 |
|---|---|---|---|
| 1 | Zero-copy KV가 long-ctx에서 BW 저하 | REJECT | 코드 확인 — 이미 device-local |
| 2 | KV layout HeadMajor vs SeqMajor | REJECT | 동일 패턴, slope diff 0.00 |
| 3 | Flash attn kernel 정적 동등 | PARTIAL | prefill만 동등, Q1은 다름 (B-4 차이) |
| 4 | Launch count context 비례 | REJECT | 506 dispatches/token context-invariant |
| 5 | 우리 attention이 더 빠름 | REJECT | cross-run에서 우리가 33-55% 느림 (post-revert는 TIE) |
| 6 | KV stride 차이 (256B vs 512B) | REJECT | microbench layout swap diff < 2% |
| 7 | Production 환경 effects 지배 | PARTIAL | 22%만 (FFN cache thrashing); 나머지는 kernel |
| 8 | 우리 Q1이 더 고도화 (A1 추정) | REJECT | cross-run direct measurement로 정반대 |
| 9 | 갭 자체가 phantom | REJECT | direct 측정으로 6.75 → 1.62 검증 |
| **10** | **B-4 sub_group_reduce가 Adreno에서 손해** | **CONFIRMED** | cross-run 33-55% diff, revert로 76% 회수 |

### 2.4 무엇이 작동했나

- **Vendor-and-compare 패턴**: llama.cpp 커널을 그대로 vendor 폴더에 넣고 우리 ocl harness에서 직접 컴파일·dispatch. 같은 디바이스 같은 입력에서 직접 비교 → **유일한 결정적 방법**. 다른 kernel 의심 시 같은 패턴 사용 권장.
- **변형 매트릭스 (Single/Repeat28/+Mask/+QVar)**: production 조건을 점진적으로 모방. 어느 단계에서 overhead가 들어오는지 isolate. 새 kernel 도입 시 동일 매트릭스로 검증 권장.

### 2.5 무엇이 잘못됐나 (반복하지 말 것)

- **Phase A는 prefill만 비교**했는데 "byte-identical, 우리가 빠를 것" 결론을 decode에까지 적용. **decode와 prefill은 완전히 다른 kernel** — 따로 비교해야 함.
- **간접 추정 (4.5 μs/n_kv = wall × 0.8)** 으로 phantom target 만들어 5시간 낭비. 직접 측정 가능한 상황에서 추정값 사용 금지.
- **이론 (barrier 0 vs 236) > 실측** 가정. Adreno 같은 closed-source 드라이버에서는 항상 실측이 우선.

## 3. 잔존 작업 (다음 세션 우선순위)

### A. D path (eviction) — **우선 추천**
- ROI: 명확. 정확도 trade-off 수용 시 즉시 추가 win.
- 작업: long-context decode에서 Sliding/H2O/D2O 적용 측정.
- 측정: `--eviction-policy sliding --eviction-window 1024` 등으로 effective n_kv 줄여 wall TBT 감소 확인.
- 잔존 1.6-2.6 μs/n_kv를 우회 가능 (절대값 감소).

### B. Op-level isolation — 잔존 갭의 진짜 원인
- 작업: 비-attention ops (matmul_qkv/wo/ffn/lm_head, rms_norm, kv_update)을 microbench로 격리 측정. llama.cpp 대응 op도 vendor-and-compare.
- 가설: 우리 op들 중 일부도 llama.cpp보다 느릴 가능성. matmul_ffn (Q4_0 GEMV) 가능성 높음.
- 도구: 본 sprint의 microbench_flash_attn 패턴 그대로 적용.

### C. Snapdragon Profiler (B path) — 보류 권장
- 잔존 갭이 1.6-2.6 μs/n_kv로 작음. SP 셋업 비용 대비 ROI 약함.
- D 또는 B로 추가 개선이 한계에 부딪치면 고려.

### D. Prefill 최적화 (별도 sprint)
- Prefill: ~2× 느림 vs llama.cpp (decode와 별개 sprint)
- 본 sprint의 발견: prefill의 `REQD_SUBGROUP_SIZE_64` attribute 자체는 무해 (격리 검증)
- A-3 B-1 subgroup split (DK==128) 효과 격리 필요
- llama.cpp prefill kernel cross-run으로 결정적 검증 권장 (decode와 같은 패턴)
- 참고: §1~10 (이전 sprint의 prefill 분석)

## 4. 본 sprint 산출물

### 코드 변경 (production)
- `engine/kernels/flash_attn_f32_f16.cl` Q1 kernel: B-4 → SLM tree-reduce
- `engine/src/bin/microbench_flash_attn.rs`: 신규 측정 도구

### 벤더 카피 (참조)
- `.agent/research/microbench_flash_attn/llamacpp_q1_flash_attn.cl`: llama.cpp Q1 vendored

### 분석 문서 (시간순)
1. `2026-04-14_a1_phase_a_reaudit.md` — A1: Phase A 재감사 (researcher)
2. `microbench_flash_attn/option_c_op_slope_audit.md` — Option C: per-op slope vs TOTAL
3. `microbench_flash_attn/option_a_production_decomp.md` — Option A: production 조건 분해
4. `microbench_flash_attn/cross_run_verdict.md` — Cross-run verdict (B-4 LOSS 확정)
5. `microbench_flash_attn/post_revert_verdict.md` — Revert 후 검증 결과
6. `microbench_flash_attn/HANDOFF.md` — **본 문서**

### Raw 측정 데이터
- `microbench_flash_attn/run1_*.txt` — Option C/A KV stride 매트릭스
- `microbench_flash_attn/optionA_run{1,2}_*.txt` — production 조건 분해
- `microbench_flash_attn/crossrun_run{1,2}_*.txt` — llm_rs2 vs llama.cpp 직접 비교
- `microbench_flash_attn/postrevert_run1_*.txt` — revert 후 microbench
- `microbench_flash_attn/postrevert_decode_*.txt` — revert 후 production decode (run1, run2)
- `microbench_flash_attn/prefill_noattr_*.txt` — prefill attribute 격리 테스트

### TODO 갱신
- `.agent/todos/long_context_attention_optimization.md` §12에 6 단계 (Option C → A → A1 → cross-run → revert → 잔존 검증) 전체 기록

## 5. 빠른 재현 가이드 (다음 세션이 본 결과를 reproduce 하려면)

### 5.1 microbench cross-run

```bash
# 호스트 빌드 + 디바이스 푸시
source android.source  # (linux NDK 경로로 override 필요시: NDK_HOME=/opt/android-ndk HOST_TAG=linux-x86_64)
cargo build --target aarch64-linux-android --bin microbench_flash_attn --release
adb push target/aarch64-linux-android/release/microbench_flash_attn /data/local/tmp/

# 측정 (~3-5분, thermal-light)
adb shell '/data/local/tmp/microbench_flash_attn 2>&1' | tail -30
# 기대: Repeat28+Mask 우리 7.5x μs vs llama.cpp 7.5x μs (TIE)
```

### 5.2 production decode 벤치 (B-4 revert 효과 확인)

```bash
cargo build --target aarch64-linux-android --bin generate --release
adb push target/aarch64-linux-android/release/generate /data/local/tmp/

# 4 ctx, 240s cooldown
for ctx in p_short p1k p2k p4k; do
    adb shell "cd /data/local/tmp && ./generate \
      --model-path llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
      --backend opencl --prompt-file ${ctx}.txt --num-tokens 31 --max-seq-len 8192" \
      | grep "Decode:"
    sleep 240
done
# 기대: p1k ~33ms, p2k ~40ms, p4k ~53ms
```

### 5.3 정확도 회귀 (CPU vs OpenCL bit comparison)

```bash
adb shell '/data/local/tmp/generate \
  --model-path llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --backend cpu --prompt "The capital of France is" --num-tokens 20 --temperature 0.0'
# 기대 출력: "The capital of France is Paris. It has a population of about 2 million people..."

adb shell '/data/local/tmp/generate \
  --model-path llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --backend opencl --prompt "The capital of France is" --num-tokens 20 --temperature 0.0'
# 기대: 위와 완전 동일 토큰 시퀀스
```

## 6. 메모리 시스템 갱신 권장

`.claude/projects/-home-go-Workspace-llm-rs2/memory/` 에 다음 메모리 추가/갱신:

- **feedback**: "Adreno에서 sub_group_reduce_max/add 사용 금지 — SLM tree-reduce가 33-55% 빠름. cross-run로 검증."
- **feedback**: "engine 간 OpenCL profile-events 비교 신뢰 불가 — driver-specific 패널티 차이가 큼. wall-clock만 사용."
- **feedback**: "kernel 비교는 vendor-and-compare 패턴 사용 — llama.cpp 소스를 vendor 폴더에 넣고 우리 ocl harness에서 직접 컴파일/측정."
- **reference**: "microbench_flash_attn (engine/src/bin/) — flash attn isolation bench. cross-engine 비교 가능."

## 7. 커밋 히스토리 (본 sprint)

```
f14d98b docs(perf): complete remaining validation tasks for B-4 revert
e6956ea perf(opencl): revert B-4 sub_group_reduce on Q1 — recover 5.13 μs/n_kv
36b76ad perf(opencl): cross-run microbench — llama.cpp Q1 33-55% faster than ours
d02a724 docs(perf): Option A1 — Phase A re-audit reveals Q1 kernel divergence + phantom baseline
f47bf15 perf(opencl): Option A — production-condition microbench decomposes attn slope
f7ef6df docs(perf): resolve attention 13.23 vs TOTAL 12.45 paradox via op-slope audit
f442782 perf(opencl): isolate KV stride hypothesis via standalone microbench (REJECT)
```

7 커밋, +2,000 / -200 LOC (대부분 분석 문서). production 코드 변경: `engine/kernels/flash_attn_f32_f16.cl` Q1 부분만 (~70줄), `engine/src/bin/microbench_flash_attn.rs` 신규 (~540줄).
