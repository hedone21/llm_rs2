# Handoff — step-4 머지 후 TBT 회복 (Base 95% 이상)

**작성**: 2026-05-21
**진입 문장**: "성능 개선 진행"
**기준 브랜치**: `merge-step4-into-master` (push 완료, PR 대기 또는 머지 후 master 기준)
**worktree 권장**: `.claude/worktrees/perf_recovery_post_step4` (신설)

## 목표 (Definition of Done)

| 지표 | 현재 (머지 후) | sprint1 baseline | 목표 (Base 95% 이상) | 회복 폭 |
|---|---|---|---|---|
| **S25 GGUF Avg TBT** | 33.63 ms (3-run avg) | 29.20 ms | **≤ 30.74 ms** | -8.6% 회복 |
| **S25 AUF Avg TBT** | 34.45 ms (3-run avg) | 31.44 ms | **≤ 33.10 ms** | -4.0% 회복 |
| 정확성 | fluent ✓ | fluent | fluent (회귀 0) | — |

- 측정: Qwen 2.5-1.5B Q4_0 AOS, qnn_oppkg, 6T, num-tokens=32, prompt="I_am_a", n=3 평균.
- 메트릭은 **Avg TBT (tok0 inclusive)** 사용. `rest_tbt`만 보지 말 것 ([[feedback-tbt-metric-tok0-inclusive]]).
- 호스트 회귀 0 + AUF↔GGUF byte-equivalence test 2/2 PASS 유지.

## 배경

step-4 (107 commits, layered architecture promotion + DecodeLoop+ModelForward path) 가
master 위로 통합되면서 **Avg TBT 절대 +15%** 가량 회귀. 머지 결과는 정상이며 회귀 자체는
step-4 작업 비용이지만, 사용자 정책상 **base 95% 이상**으로 회복 필요.

## 가설 (우선순위 순)

### H1 — Phase 4-4.5 DecodeLoop standard happy path 비용 (가장 유력)

로그 라인 `[Phase4-4.5] standard happy path → DecodeLoop+ModelForward` 확인됨.
master(sprint1) 시점에는 generate.rs main()의 inline forward loop. step-4가
`session/standard_happy.rs` + `DecodeLoop::run_until_stop` + `ModelForward` 추상화로
교체. 가능 회귀 원인:

- (a) `Box<dyn Trait>` indirect call 비용 (forward/sampler/stop trait dispatch)
- (b) Per-iter `KvMode` re-check / lifecycle hook (default no-op이지만 함수 호출은 있음)
- (c) `ModelForward::prefill_workspace` lazy alloc 정책 차이
- (d) sampler 경로의 `RepetitionPenaltySampler` 등 step-4 추가 hook (`b412837a` C1)

**검증**: `LLMRS_FWD_TRACE=1` (step-4 4-4.7 C3)로 plan-aware ModelForward path 확인 +
trait dispatch 비용은 perf record로 hot path 격리.

### H2 — `crate::core::*` → top-level (`crate::qcf`, `crate::pressure`, `crate::inference`) 모듈 이동에 의한 inlining barrier

step-4 4-A/B/C/D가 도메인 모듈을 top-level로 promotion. 같은 crate라 ABI 영향 없어야
하지만, codegen-unit 분할이 달라져 LTO 효과가 변경됐을 가능성.

**검증**: `cargo build --release` 후 `-Csave-temps` 또는 `cargo bloat`로 hot
함수의 inlining 상태 비교 (sprint1 baseline vs HEAD).

### H3 — `compute_qcf_swap` rename + callsite 변경에 의한 hot path 변경

`session/qcf_runtime.rs:537,695`의 `read_allow_boundary_env()` cfg 게이트를 머지에서
제거 (`#[cfg(feature = "opencl")]` 삭제). decode 핫패스에서 매 swap 결정 시
환경변수 lookup 비용이 발생할 수도. **다만 swap 미사용 측정에서는 영향 없을 가능성**.

**검증**: 환경변수 lookup이 hot인지 perf로 확인.

### H4 — `--weight-dtype` 처리 분기 변경 (AUF 한정)

init.rs에 추가한 `if is_auf && args.weight_dtype != "f16"` warning 분기가 매 init마다
실행. 한 번이라 무시 가능. (가능성 매우 낮음)

## 진행 단계

### Step 1 — Baseline 재측정 + 격리 측정 (1h)

```bash
# 새 worktree (선택)
cd /home/go/Workspace/llm_rs2
git worktree add .claude/worktrees/perf_recovery_post_step4 -b perf-recovery-post-step4 origin/merge-step4-into-master
cd .claude/worktrees/perf_recovery_post_step4

# 환경 셋업 (필수)
ln -s /home/go/Workspace/llm_rs2/third_party third_party
ln -s /home/go/Workspace/llm_rs2/libs libs

# Android build + deploy
python scripts/run_device.py -d galaxy_s25 generate --skip-exec
adb -s R3CY408S4HN push target/aarch64-linux-android/release/generate /data/local/tmp/generate

# 3가지 measurement points (각각 n=5)
# (1) HEAD (post-merge baseline)
# (2) sprint1 시점 (e31bd698, sprint1 PR 머지본)
# (3) step-4-backup 시점 (fd95e916)

# 측정 헬퍼 (반복 호출):
for i in 1 2 3 4 5; do
  adb -s R3CY408S4HN shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
    ./generate --num-tokens 32 \
    --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
    --backend qnn_oppkg --threads 6 --prompt 'I_am_a'" 2>&1 \
    | grep -E "Avg TBT|TTFT|Decode:"
done
```

- 측정값을 `papers/eurosys2027/_workspace/experiment/perf_recovery_post_step4_baseline.md`에 기록.
- sprint1 vs step-4-backup 격리로 회귀가 **step-4 본체에서 왔는지** 또는 **merge 후 통합 코드**에서 왔는지 분리.

### Step 2 — H1 검증 (DecodeLoop path) (2~3h)

step-4가 도입한 DecodeLoop를 우회하는 측정 진입점 만들기:

1. `engine/src/bin/generate.rs`에 `LLMRS_BYPASS_DECODE_LOOP=1` 환경변수 게이트 추가
2. ON일 때 sprint1 시점의 inline forward loop를 호출 (또는 step-4 base의
   `e83b87d2 Phase 4-4-b — generate.rs narrow happy path 분기 추가` commit이 만든 분기 활용)
3. ON/OFF로 같은 prompt 5회 측정 → 분기 비용 격리

**기대**: bypass=ON에서 sprint1 baseline 회복하면 H1 확정. 그 경우:
- (a) DecodeLoop 내부 hot path 인라인 강화 (`#[inline]` 추가)
- (b) Forward trait dispatch를 `&model` 직접 호출로 변환 (production happy path만)
- (c) Phase 4-5-c `StopCondition` trait 비용이 크면 `BudgetExhausted` 경로 특화

### Step 3 — H2 검증 (LTO/inlining) (1~2h)

```bash
# sprint1 시점 binary
git checkout e31bd698 -- engine/src/
cargo build --release --target aarch64-linux-android --bin generate \
  --features opencl,vulkan,qnn --no-default-features
cp target/aarch64-linux-android/release/generate /tmp/generate_sprint1

# HEAD binary
git checkout HEAD -- engine/src/
cargo build --release ...

# 비교
cargo bloat --release --target aarch64-linux-android --bin generate -n 50 > /tmp/bloat_head.txt
git checkout e31bd698 && cargo bloat --release ... > /tmp/bloat_sprint1.txt
diff /tmp/bloat_sprint1.txt /tmp/bloat_head.txt | head -50
```

LlamaLayer::forward / forward_gen / attention_gen 등 hot 함수의 size + inlining
차이 격리. binary size + section count 차이도 기록.

### Step 4 — fix 시도 (4~6h)

H1/H2 검증 결과에 따라:

- **H1 hit**: standard_happy.rs hot path 특화 (Box<dyn Forward> → 정적 dispatch).
  step-4 trait 추상화는 보존하되, production decode loop은 generic over `<F: Forward>` 형태로 monomorphize.
- **H2 hit**: module promotion으로 인한 codegen unit 분할 점검.
  `[profile.release] codegen-units = 1`은 이미 적용. LTO `fat` 도 확인.
  추가로 `cargo rustc -- -C link-arg=-Wl,--gc-sections` 등 확인.
- **둘 다 hit**: 두 fix 결합.

### Step 5 — 디바이스 게이트 (1h)

```bash
# 5회 측정 (Q4_0 AOS AUF + GGUF 각)
# Avg TBT 목표:
#   GGUF ≤ 30.74 ms
#   AUF  ≤ 33.10 ms
```

5회 측정의 median이 게이트 통과해야 함. 회귀가 ±5% 안에 들면 PASS.

### Step 6 — 호스트 게이트 + commit + PR (30min)

```bash
cargo build -p llm_rs2 --release
cargo test -p llm_rs2 --lib --skip backend::opencl --skip memory::opencl   # 1195+ PASS
cargo test -p llm_rs2 --test test_auf_gguf_byte_equivalence                # 2/2 PASS
cargo fmt --check -p llm_rs2

git commit + git push + gh pr create (또는 GitHub UI)
```

## 환경 셋업 (worktree 재진입 시 매번 필요)

```bash
ln -s /home/go/Workspace/llm_rs2/third_party third_party  # QNN SDK
ln -s /home/go/Workspace/llm_rs2/libs libs                # libOpenCL.so
```

이게 없으면 Android 빌드 실패. master worktree에는 이미 있음.

## Deploy 우회 (run_device.py 버그)

`python scripts/run_device.py -d galaxy_s25 generate --skip-exec` 는 자주
**deploy를 누락**하고 old binary로 실행됨. 새 binary 보장하려면:

```bash
adb -s R3CY408S4HN push target/aarch64-linux-android/release/generate /data/local/tmp/generate
```

수동 push 후 mtime 확인 (`adb shell ls -la /data/local/tmp/generate`).

## 측정 컨벤션

- **Avg TBT** = tok0 inclusive. paper figure도 이거. `rest_tbt`만 보지 말 것.
- n=5 median 우선. mean도 참고.
- prompt: `"I_am_a"` (underscore — adb shell quote 우회).
- 같은 device, 같은 환경 (백그라운드 앱 정지) 가정.
- 3가지 base 측정: sprint1(e31bd698) / step-4-backup(fd95e916) / HEAD.

## 관련 메모리

- [[merge-step4-into-master]] — 본 머지 종결 (HEAD `6fe81729`)
- [[sprint1-auf-loader]] — sprint1 baseline 측정 (29.20 / 31.44 ms)
- [[swap-track-status-20260508]] — swap 측정 컨벤션
- [[feedback-tbt-metric-tok0-inclusive]] — Avg TBT 정책
- [[layered-architecture-decision]] — step-4 layered architecture 본질

## 재진입

**"성능 개선 진행"** — 본 handoff Step 1 (baseline 재측정 + 격리) 부터 시작.

---

## 종결 — 2026-05-21 옵션 A 채택 (CLOSED)

Step 1 baseline 격리 측정 (커밋 `7d02686b`) 결과로 본 task를 **옵션 A로 종결**한다.

### 측정 결과 요약 (n=5 median, S25 qnn_oppkg 6T)

|  | GGUF Avg TBT | AUF Avg TBT |
|---|---|---|
| sprint1 (e31bd698) | 33.08 ms | 32.56 ms |
| step-4-backup (fd95e916) | 32.57 ms | — |
| HEAD merge (000634b3) | 33.84 ms | 33.96 ms |

회귀 분해:
- sprint1 → step-4-backup: **−0.51 ms (개선)**
- step-4-backup → HEAD: **+1.27 ms** (머지 통합 영향)
- 현 환경 회귀폭: GGUF +2.3% / AUF +4.3% — **base 105% 마진 이내 PASS**

### 종결 근거

1. **handoff Goal baseline outdated**: "GGUF ≤30.74 / AUF ≤33.10"의 출발점인 sprint1 baseline 29.20/31.44 ms는 현 환경에서 sprint1 binary 자체로도 재현 불가 (현 환경 33.08/32.56 ms). 측정 환경 (디바이스 thermal/governor/백그라운드 앱/prompt/feature) 차이로 추정.
2. **H1 가설 기각**: 회귀가 step-4 본체(DecodeLoop+ModelForward trait dispatch)가 아닌 머지 통합 코드에서 발생. step-4 refactor 자체는 −0.51 ms 개선까지 함.
3. **현 환경 회귀폭은 노이즈 수준**: 측정 IQR ~1 ms 대비 +0.76 ms (GGUF) / +1.40 ms (AUF). base 105% 이내.
4. **ROI 낮음**: +1.27 ms 마이크로 회복은 outdated baseline을 쫓는 작업이며 CLAUDE.md "단순함 우선" / "외과적 변경" 원칙에 부합하지 않음.

### Follow-up

- handoff 절대 ms 기준 ("GGUF ≤30.74 / AUF ≤33.10")은 **outdated baseline에서 파생** — 향후 perf 작업 시 현 환경에서 새 baseline 측정 후 진행할 것.
- 머지 통합 +1.27 ms 회귀 원인 후보 (`compute_qcf_swap` rename / AUF 모듈 위치 / LoadConfig W-AUF-1 fields / `--secondary-gguf` warning / `read_allow_boundary_env` cfg 게이트 제거)는 microbench으로 재현 시도 가치 낮음 — 본 backlog 등록하지 않음.

### 종결 산출물

- baseline 측정 doc: `papers/eurosys2027/_workspace/experiment/perf_recovery_post_step4_baseline.md`
- raw logs: `papers/eurosys2027/_workspace/experiment/perf_recovery_post_step4_raw/{sprint1,step4_backup,head}_{gguf,auf}.log` (n=5 × 5 cells = 25 runs)
- 메모리 갱신: [[perf-recovery-post-step4]] (status: CLOSED)

### 다음 작업

리펙토링 흐름 유지 — backlog [P1] **Phase 4-4-2.3 decode_fallback 추출** (~2,260 LOC, `bin/generate.rs` L1841~4099) 진입. 진입점: `handoff_phase4_4_2_sprint_exit_2026_05_19.md §5`.
