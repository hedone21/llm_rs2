# S-subcmd Design — clap Subcommand 도입 (2026-05-19)

**Base HEAD**: `a283b35a` (S-cleanup batch 3 직후, KiviArgs awqe env 이전)
**선행**: `sprep_args_matrix_2026_05_18.md`
**사용자 결정 (2026-05-19)**: 옵션 A (clap subcommand) 채택, (나) 진행 — EvictionCmd + KvMode 동시 도입

---

## 목표

cli.rs Args의 polic-specific flag 폭발을 막기 위해 clap subcommand 패턴 도입.

- **EvictionCmd**: 6 variant subcommand. 23 field → enum + sub-args struct
- **KvMode**: ValueEnum 단일 flag + `conflicts_with`로 KIVI/Offload sub-args 격리

clap의 단일 subcommand 제약 (한 binary에 `#[clap(subcommand)]` 한 번만)을 따라 정책 다양성이 큰 Eviction만 subcommand, KvMode는 ValueEnum + flatten으로 처리.

---

## EvictionCmd — 사양

```rust
// session/cli/eviction.rs (신규)
use clap::{Args, Subcommand};

#[derive(Subcommand, Debug, Clone)]
pub enum EvictionCmd {
    /// No eviction (default). KV cache grows without bound up to --max-seq-len.
    None,
    /// Sliding window — keep most recent N tokens.
    Sliding(SlidingArgs),
    /// StreamingLLM — keep `sink` initial tokens + recent window.
    Streaming(StreamingArgs),
    /// H2O — heavy hitter selection (Round 14 default for Llama 3.2 1B).
    H2o(H2oArgs),
    /// H2O+ per-head GQA-aware variant.
    H2oPlus(H2oArgs),       // 동일 args 구조 사용
    /// D2O — Dynamic Discriminative Operations (arXiv 2406.13035).
    D2o(D2oArgs),
}

#[derive(Args, Debug, Clone)]
pub struct SlidingArgs {
    /// Tokens to retain (default 1024).
    #[arg(long, default_value_t = 1024)]
    pub window: usize,
}

#[derive(Args, Debug, Clone)]
pub struct StreamingArgs {
    /// Attention sink tokens to preserve (default 4).
    #[arg(long, default_value_t = 4)]
    pub sink: usize,
    /// Recent window size. 0 = auto (budget − sink).
    #[arg(long, default_value_t = 0)]
    pub recent_window: usize,
}

#[derive(Args, Debug, Clone)]
pub struct H2oArgs {
    /// Fraction of tokens kept as heavy hitters (0.0–1.0). 0.0 = Sliding equivalent.
    #[arg(long, default_value_t = 0.5)]
    pub keep_ratio: f32,
    /// Final transformer layers to track (0 = all).
    #[arg(long, default_value_t = 0)]
    pub tracked_layers: usize,
    /// EMA decay factor per step (0.0 = no decay).
    #[arg(long, default_value_t = 0.0)]
    pub decay: f32,
    /// Disable time-normalized scoring (use raw SUM).
    #[arg(long, default_value_t = false)]
    pub raw_scores: bool,
}

#[derive(Args, Debug, Clone)]
pub struct D2oArgs {
    /// Heavy-hitter keep ratio (paper default 0.75).
    #[arg(long, default_value_t = 0.75)]
    pub keep_ratio: f32,
    /// EMA smoothing β for τ_t (paper Eq.10, default 0.7).
    #[arg(long, default_value_t = 0.7)]
    pub ema_beta: f32,
    /// Eq.11 normalisation constant `e` (paper default 0.1).
    #[arg(long, default_value_t = 0.1)]
    pub merge_e: f32,
    /// Enable layer-level dynamic allocation (per-layer variance).
    #[arg(long, default_value_t = false)]
    pub layer_alloc: bool,
    /// Protected layer indices (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub protected_layers: Option<Vec<usize>>,
}
```

### 공통 eviction args (variant 무관, flatten으로 BaseArgs에 유지)

다음 7 field는 모든 variant에서 의미 있어 `EvictionCommonArgs`로 별도 struct (BaseArgs에 flatten):

```rust
#[derive(Args, Debug, Clone)]
pub struct EvictionCommonArgs {
    /// KV cache budget in tokens. 0 = unlimited.
    #[arg(long, default_value_t = 0)]
    pub kv_budget: usize,
    /// Budget as ratio of prompt length (overrides --kv-budget).
    #[arg(long, default_value_t = 0.0)]
    pub kv_budget_ratio: f32,
    /// Prefix tokens protected from eviction.
    #[arg(long)]
    pub protected_prefix: Option<usize>,
    /// Memory threshold in MB below which eviction triggers.
    #[arg(long, default_value_t = 256)]
    pub memory_threshold_mb: usize,
    /// Cache fraction retained after eviction (0.1–0.99).
    #[arg(long, default_value_t = 0.75)]
    pub eviction_target_ratio: f32,
    /// Initial KV cache capacity (0 = auto).
    #[arg(long, default_value_t = 0)]
    pub initial_kv_capacity: usize,
    /// Eviction lower bound (tokens).
    #[arg(long, default_value_t = 256)]
    pub min_kv_cache: usize,
}
```

### 삭제 후보 (subcmd 이관 중 정리)

| field | 처리 |
|---|---|
| `h2o_debug` | **삭제** — env `LLMRS_H2O_DEBUG`로 이전 (KIVI awqe 패턴) |

---

## KvMode — 사양

clap subcommand 단일 제약 때문에 ValueEnum + sub-args flatten 패턴:

```rust
// session/cli/kv_mode.rs (신규)
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum KvMode {
    /// Standard F32/F16/Q4_0 KV cache (default).
    Standard,
    /// KIVI Q2/Q4/Q8 quantization with FP residual buffer.
    Kivi,
    /// KV cache offload (raw / disk).
    Offload,
}

#[derive(Args, Debug, Clone)]
pub struct KvModeArgs {
    #[arg(long, value_enum, default_value_t = KvMode::Standard)]
    pub kv_mode: KvMode,

    // ── KIVI sub-args (only valid when --kv-mode kivi) ──
    /// KIVI bit-width.
    #[arg(long, default_value_t = 2, requires_ifs = [("kivi", "kv_mode")])]
    pub kivi_bits: u8,
    /// KIVI residual buffer size (multiple of 32).
    #[arg(long, default_value_t = 32)]
    pub kivi_residual_size: usize,
    /// Enable runtime KV quant transition (F16→Q2/Q4/Q8 via signal).
    #[arg(long, default_value_t = false)]
    pub kv_dynamic_quant: bool,

    // ── Offload sub-args (only valid when --kv-mode offload) ──
    /// Offload mode (raw / disk).
    #[arg(long, default_value = "raw")]
    pub offload_mode: String,
    /// Disk offload directory.
    #[arg(long, default_value = "")]
    pub offload_path: String,
    /// Adaptive prefetch ceiling.
    #[arg(long, default_value_t = 128)]
    pub max_prefetch_depth: usize,
}
```

**현 `--kivi` boolean + `--kv-offload <none|raw|disk>` 패턴 폐기**. 단일 `--kv-mode {standard|kivi|offload}`로 통합. 검증은 init.rs에서 `match kv_mode { Standard => ..., Kivi { bits, ... } => ..., Offload { mode, ... } => ... }` 패턴.

---

## Args struct 통합 (최종)

```rust
// session/cli.rs
#[derive(Parser, Debug, Clone)]
pub struct Args {
    // ── 기존 평면 flag (BaseArgs 후보) ──
    pub model_path: String,
    pub prompt: String,
    pub backend: String,
    ...
    // ── KvMode (flatten) ──
    #[clap(flatten)]
    pub kv: KvModeArgs,
    // ── EvictionCommon (flatten, variant 무관) ──
    #[clap(flatten)]
    pub eviction_common: EvictionCommonArgs,
    // ── EvictionCmd (subcommand, optional — default None) ──
    #[command(subcommand)]
    pub eviction: Option<EvictionCmd>,
}
```

**무 subcommand 시 `EvictionCmd::None` 등가** (Option None ≡ EvictionCmd::None).

### 호출 패턴

```bash
# Production
generate -m qwen.gguf -p "Hello" eviction h2o --keep-ratio 0.5
generate -m qwen.gguf --kv-budget 1024 eviction sliding --window 1024
generate -m qwen.gguf --kv-mode kivi --kivi-bits 2 --kivi-residual-size 32

# Default (no eviction, standard KV)
generate -m qwen.gguf -p "Hello"
```

---

## 마이그레이션 매핑 (이전 → 이후)

| 이전 | 이후 |
|---|---|
| `--eviction-policy none` | (subcommand 생략) 또는 `eviction none` |
| `--eviction-policy sliding --eviction-window 1024` | `eviction sliding --window 1024` |
| `--eviction-policy streaming --sink-size 4 --streaming-window 0` | `eviction streaming --sink 4 --recent-window 0` |
| `--eviction-policy h2o --h2o-keep-ratio 0.5 --h2o-decay 0.0` | `eviction h2o --keep-ratio 0.5 --decay 0.0` |
| `--eviction-policy h2o_plus --h2o-keep-ratio 0.5` | `eviction h2o-plus --keep-ratio 0.5` |
| `--eviction-policy d2o --d2o-keep-ratio 0.75 --d2o-ema-beta 0.7` | `eviction d2o --keep-ratio 0.75 --ema-beta 0.7` |
| `--kv-budget 1024 --protected-prefix 4` | (변화 없음, flatten) |
| `--kivi --kivi-bits 2` | `--kv-mode kivi --kivi-bits 2` |
| `--kv-offload disk --offload-path /tmp` | `--kv-mode offload --offload-mode disk --offload-path /tmp` |
| `--h2o-debug` | env `LLMRS_H2O_DEBUG=1` |

### 영향받는 사용처

| 위치 | 변경 |
|---|---|
| `engine/src/session/init.rs` | eviction_policy / kivi / kv_offload 분기를 enum match로 |
| `engine/src/bin/generate.rs` | 동일 |
| `engine/src/session/{chat,ppl,batch,eval}/runner.rs` | `args.eviction_policy.as_str()` 호출처를 `args.eviction_kind()` helper로 |
| `scripts/run_device.py` | CLI invocation string 갱신 (해당 없음, generate가 받는 argv 그대로 전달) |
| `verify/scenarios/*.yaml` | scenario CLI 문자열 갱신 |
| `docs/*.md`, `CLAUDE.md` | 예시 갱신 |
| `engine/tests/spec/*.rs` | parse 테스트 추가 |

### scripts/verify 마이그레이션 헬퍼

`bin/generate`가 받는 argv는 사용자가 입력하므로 일괄 sed 적용:

```bash
# 단순 매핑 (sed)
s/--eviction-policy sliding --eviction-window /eviction sliding --window /
s/--eviction-policy h2o --h2o-keep-ratio /eviction h2o --keep-ratio /
s/--eviction-policy d2o --d2o-keep-ratio /eviction d2o --keep-ratio /
s/--eviction-policy streaming --sink-size /eviction streaming --sink /
s/--kivi --kivi-bits /--kv-mode kivi --kivi-bits /
s/--kv-offload disk /--kv-mode offload --offload-mode disk /
s/--kv-offload raw /--kv-mode offload --offload-mode raw /
```

복잡한 케이스 (multi-arg, conditional)는 PR3에서 수작업.

---

## Commit 분해

| Commit | 작업 | LOC 영향 |
|---|---|---|
| **C1** | `session/cli/eviction.rs` 신설 (EvictionCmd enum + variant args) + `EvictionCommonArgs` | +200 |
| **C2** | cli.rs Args에 EvictionCmd subcommand 통합. 23 field 제거. `args.eviction_kind()` 호환 helper 제공 | net -100 |
| **C3** | init.rs / lib (chat/ppl/batch/eval) 호출처 마이그레이션 | ~50 변경 |
| **C4** | `session/cli/kv_mode.rs` 신설 (KvMode ValueEnum + sub-args) | +120 |
| **C5** | cli.rs에 KvModeArgs flatten. `args.kivi` / `args.kv_offload` 제거. helper 함수 (`is_kivi_mode()`, `is_offload_mode()`) | net -30 |
| **C6** | init.rs / generate.rs / chat REPL의 KIVI/Offload 분기 마이그레이션 | ~80 변경 |
| **C7** | `h2o_debug` → env `LLMRS_H2O_DEBUG` 이전 | ~5 변경 |
| **C8** | scripts/run_device.py / verify/scenarios / docs 일괄 sed | ~30 파일 |
| **C9** | spec test 추가 (EvictionCmd parse / KvMode parse) | +100 |
| **C10** | handoff doc 갱신 + sprep_args_matrix doc 정정 | doc |

검증 게이트:
- 각 C1~C9 후 `cargo check --workspace --bins` PASS
- C3 / C6 후 `cargo test -p llm_rs2 --lib` 회귀 ≤ master flaky
- C9 후 spec test 신규 6건 PASS
- C8 마지막 — Galaxy S25 smoke (chat 1 turn + ppl 32 tok)

---

## Risk + 완화

| R | 항목 | 완화 |
|---|---|---|
| R1 | clap subcommand가 `chat` 같은 다른 binary flag와 conflict | chat이 별도 binary 분리되기 전까지 generate에서만 도입. chat binary는 S-1 이후 자체 EvictionCmd 가짐 |
| R2 | `EvictionCmd::None` (None variant) vs Option<EvictionCmd>::None 이중성 | helper `args.eviction_or_none() -> EvictionCmd`로 정규화 |
| R3 | 기존 shell history / sweep scripts 일괄 마이그레이션 누락 | C8 sed 패턴 + scripts/run_benchmark_suite.py 검색으로 catch-all |
| R4 | KvMode `requires_ifs` clap 구문 호환성 (clap 4 derive) | clap 4.5+ 지원. cargo check로 즉시 확인 |
| R5 | `--h2o-debug` 사용자 환경 / hosts.toml 등에 하드코딩 | env var은 hosts.toml에 LLMRS_H2O_DEBUG로 설정 가능 |
| R6 | spec test 회귀 | C9 parse test로 enum variant 6 + KvMode 3 변종 모두 커버 |

---

## 진입

다음 step: **C1 진행** — `session/cli/eviction.rs` 신설. EvictionCmd enum + 4 sub-args struct 작성. 그 후 C2에서 cli.rs Args 통합.

S-1 (BaseArgs 추출) 진입은 본 S-subcmd 완료 후 (C10까지). BaseArgs는 KvModeArgs / EvictionCommonArgs / EvictionCmd를 자연스럽게 흡수.
