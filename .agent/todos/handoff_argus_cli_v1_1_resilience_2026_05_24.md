# Handoff: argus-cli v1-1 — resilience default-on

**작성**: 2026-05-24
**HEAD**: `14b3de7a refactor(cli): argus-cli v1-1 resilience default-on`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "argus-cli v1-2 prompt-batch 진행" (또는 v1-3 swap / v1-4 profile / v1-5 KIVI / v1-6 partition)

---

## TL;DR

argus-cli v1 흡수 sprint 1순위. resilience 가 default-on 으로 전환되었고
`--no-resilience` opt-out flag 가 추가되었다. legacy `generate` 의 default-off
정책은 그대로 유지 (no_resilience flag 무시). reject 12종 중 1항목 해제.
호스트 CPU + S25 Adreno OpenCL 32 토큰 bit-identical 회귀 0.

---

## 진행 상태

| 항목 | 상태 |
|---|---|
| `--no-resilience` flag 추가 | `session/cli/mod.rs:481`. clap default=false. |
| argus-cli main effective 결정 | `bin/argus_cli.rs:32` `args.enable_resilience = !args.no_resilience` |
| reject 항목 제거 | `bin/argus_cli.rs::reject_unsupported_modes_v0` 에서 enable_resilience bail 제거 |
| 모듈 헤더 갱신 | argus-cli "v0" 라벨 제거, v1 sub-sprint 6종 로드맵 명시 |
| 호스트 빌드 | `cargo build --release --bin argus_cli` Finished in 41.04s |
| 호스트 sanity | default vs --no-resilience 동일 출력 (generated=32 first=12095 final_pos=36). Decode 112.02 vs 111.05 ms/tok. |
| cargo fmt / clippy | clean (warnings 0) |
| spec inv_layer | 8/8 PASS (15.74s) |
| layer_lint new_violations | 0 |
| S25 Adreno OpenCL | default vs --no-resilience 모두 "The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into". generated=32 first=12095 final_pos=36. TBT 33.01 / 32.86 (Δ 0.46%). §13.8-B baseline 32.85 대비 +0.5% 이내. |

### 핵심 변경

`engine/src/session/cli/mod.rs:474-487`:
```rust
/// Enable resilience manager for adaptive inference.
/// Legacy generate 기준 flag. argus-cli v1+ 는 default-on 정책이며,
/// 비활성화는 [`Self::no_resilience`] (`--no-resilience`) 를 사용한다.
#[arg(long, default_value_t = false)]
pub enable_resilience: bool,

/// Disable resilience manager (argus-cli v1+ opt-out).
/// argus-cli v1 에서는 resilience 가 default-on 이므로 비활성화하려면
/// 이 flag 를 명시. legacy `generate` binary 는 이 flag 를 무시
/// (default-off 정책 유지) — argus-cli main 에서만 [`Self::enable_resilience`]
/// 를 effective 결정한다.
#[arg(long, default_value_t = false)]
pub no_resilience: bool,
```

`engine/src/bin/argus_cli.rs:27-38`:
```rust
fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = Args::parse();

    // v1-1: resilience default-on. `--no-resilience` 가 명시되면 effective=false,
    // 그 외에는 effective=true (legacy `--enable-resilience` flag 도 그대로 효과).
    // SessionInitCtx / prefill / batch 등 호출지는 모두 `args.enable_resilience`
    // 만 참조하므로 진입 직후 1회만 갱신하면 일관된다.
    args.enable_resilience = !args.no_resilience;

    reject_unsupported_modes_v0(&args)?;
```

---

## 다음 작업 (v1 sub-sprint 5 갈래 — 우선순위 순)

### v1-2: `--prompt-batch` 흡수 (예상 0.5~1일)
- reject 1줄 제거: `bin/argus_cli.rs:158-160`.
- 분기: `if let Some(path) = &args.prompt_batch { run_prompt_batch(...) }` 추가.
- `session::batch::run_prompt_batch` 이미 분리되어 있음 (`session/batch/runner.rs`).
- 게이트: 호스트 CPU + S25 OpenCL N=4 prompt 파일 출력 확인.

### v1-3: weight swap 8종 흡수 (예상 1~2일)
- reject 1블록 제거 (`secondary_gguf` / `force_swap_ratio` / `swap_incremental_per_tick` / `swap_intra_forward` / `swap_layer_immediate` / `swap_phase_aware`).
- `session::decode_fallback::swap_dispatch` 위임 (이미 분리).
- happy path guard 에 swap 옵션 미포함 확인 필요 (`is_standard_happy_path` 참조).
- 게이트: S25 OpenCL `--secondary-gguf` + sync swap 32 토큰 bit-identical.

### v1-4: `--profile` / `--profile-events` (예상 0.5일)
- reject 1줄 제거.
- happy path guard 에 `!args.profile && !args.profile_events` 가 있음 → profile 모드는 별 path 진입. `is_standard_happy_path` 분기 후 profile 전용 path 호출.

### v1-5: KIVI / Offload `--kv-mode` (예상 1일)
- reject 1줄 제거 (`KvMode::Standard` 가드).
- `session::*` 내 KIVI/Offload path 분기 호출 확인 필요.

### v1-6: `--tensor-partition > 0` (예상 0.5일)
- reject 1줄 제거.
- happy path guard 에 `args.tensor_partition == 0.0` 가 있음 → partition 모드는 별 path.

---

## Landmines / 미해결

### 1. legacy `generate` binary 와의 미세 차이
- argus-cli 는 default-on, legacy 는 default-off. 사용자가 두 binary 의 동일 동작을 기대하면 surprise.
- mitigation: `--no-resilience` 명시 시 legacy 와 동치. 모듈 헤더 코멘트에 정책 차이 명시.

### 2. session::cli::Args 내 enable_resilience + no_resilience 공존
- 두 flag 가 동시에 켜질 수 있음 (`--enable-resilience --no-resilience`). 현재 argus-cli 는 `args.enable_resilience = !args.no_resilience` 로 no_resilience 우선.
- 미래 binary (chat/bench/eval) 가 다른 정책을 채택하면 일관성 깨질 가능. cli/chat 은 default-on, bench/eval 은 default-off (handoff 합의) → 각 binary main 에서 동일 패턴으로 처리.

### 3. PR 미생성
- §13.8-B sprint (6 commits) + v1-1 (1 commit) 합쳐 master 대비 7 commits 미머지.
- gh CLI auth 미설정 → 직접 PR 생성 필요 또는 `! gh auth login`.

### 4. argus-chat WIP stash@{0}
- chat 트랙 v0 작업이 stash 보관 상태. v1-1~v1-6 완료 후 또는 별 트랙으로 재개.
- handoff `handoff_argus_cli_v0_2026_05_24.md` 의 B 트랙 항목 참조.

### 5. `--prompt-batch` 와 default-on resilience 상호작용 미검증
- v1-2 진입 시 prompt-batch 모드에서 resilience default-on 이 무리 없는지 확인 필요. `session/batch/runner.rs:304` 가 `args.enable_resilience && process_len > 256` 으로 gating 되어 있어 안전해 보이지만 디바이스 실측 필요.

---

## 진입 명령 (다음 세션)

```
"argus-cli v1-2 prompt-batch 진행"
"argus-cli v1-3 swap 진행"
"argus-cli v1-4 profile 진행"
"argus-cli v1-5 KIVI 진행"
"argus-cli v1-6 partition 진행"
```

### 즉시 재현 명령 (v1-1 검증)

```bash
# 호스트 CPU
./target/release/argus_cli \
  --model-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/tokenizer.json \
  --prompt "The capital of France is" --num-tokens 32 --greedy --backend cpu --kv-type f16
# default → resilience on, --no-resilience → off. 둘 다 동일 출력.

# S25 Adreno OpenCL
python scripts/run_device.py -d galaxy_s25 argus_cli \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --prompt-file /data/local/tmp/prompts/capital.txt --num-tokens 32 --greedy --backend opencl --kv-type f16
```
