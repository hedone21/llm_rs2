# Handoff: §13.8-B Backend feature unification sprint 종결

**작성**: 2026-05-24
**HEAD**: `a7122009 style: cargo fmt — B-1 wrapper method 1 line 정리`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "argus-cli v1 진행" 또는 "argus-chat 진행" 또는 "argus-bench 진행" 또는 "argus-eval 진행"

---

## TL;DR

B-1~B-5 5 sub-sprint 5 commits로 backend feature broken state 일괄 해결.
`--no-default-features` / `+cuda-embedded` / `+qnn` 3 broken combination 회복 +
`feature-matrix.yml` GitHub Actions 워크플로우 + `feature_matrix.sh` 로컬
스크립트 도입. 9 feature 조합 9/9 PASS. 신규 broken feature combination 머지
영구 차단. 호스트 CPU + S25 Adreno OpenCL 32 토큰 bit-identical 회귀 0.

---

## 진행 상태

### 5 sub-sprint commit chain

| sub-sprint | commit | 핵심 |
|---|---|---|
| **B-1** | `3afcc47b` | `TransformerModel::map_weights_for_host_access` cfg-free wrapper. opencl off / non-OpenCL backend면 Ok(0) 반환. 호출지 2곳 (init.rs:700, swap_dispatch.rs:443) cfg gate 제거. → `--no-default-features` 빌드 회복 |
| **B-2** | (skip, 추가 fix 불필요) | features matrix 4 조합 (`cuda`/`vulkan`/`resilience`/`opencl,resilience`) 모두 이미 PASS 확인. wrapper 추가 본질 격상은 ROI 낮아 skip |
| **B-3** | `e2f5feff` | swap_executor.rs:314 의 `crate::buffer::cuda_mmap_alias_buffer::CudaMmapRegistration` stale path → `crate::memory::cuda::mmap::CudaMmapRegistration`. → `--features cuda-embedded` 빌드 회복 |
| **B-4** | `532e65d8` | qnn_oppkg build.rs + engine/build.rs 양쪽에 SDK 경로 resolution 3단계 fallback (env QNN_SDK_ROOT → workspace_root/third_party → git worktree `.git` 파싱). → `--features qnn` 빌드 회복 (worktree symlink 부재 환경에서도) |
| **B-5** | `9ba98ea7` | `.github/workflows/feature-matrix.yml` (9 matrix entries + cargo cache) + `.agent/skills/developing/scripts/feature_matrix.sh` 로컬 즉시 검증. → 신규 회귀 영구 차단 |
| **fmt** | `a7122009` | cargo fmt 1 line 정리 |

### feature matrix 9/9 PASS (호스트)

```
[default                   ] ✅ PASS
[--no-default-features     ] ✅ PASS  (B-1)
[+opencl                   ] ✅ PASS
[+opencl,resilience        ] ✅ PASS
[+cuda (PC)                ] ✅ PASS
[+cuda-embedded (Jetson)   ] ✅ PASS  (B-3)
[+qnn                      ] ✅ PASS  (B-4 worktree fallback)
[+vulkan                   ] ✅ PASS
[+resilience               ] ✅ PASS
```

### 디바이스 게이트

- **S25 Adreno OpenCL**: Qwen2.5-1.5b Q4_0 32 토큰 greedy
  - 직전 argus-cli v0 baseline: `Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into`
  - B-5 후: 같은 출력, generated=32 first=12095 run=31 final_pos=36 동일
  - TBT: 32.85 ms (직전 32.78 대비 Δ +0.2%, 측정 노이즈)
  - devices.toml `features = ["opencl", "vulkan", "qnn"]` 그대로 — 임시 조정 불필요 (B-4 효과)
- **호스트 CPU**: Qwen2.5-1.5b Q4_0 32 토큰: 출력 일치, generated 동일

### 게이트
- cargo fmt: clean
- cargo clippy --release --bin argus_cli: clean (warnings 0)
- spec inv_layer: 8/8 PASS
- layer_lint new_violations=0
- 9 feature matrix: 9/9 PASS (호스트), full features S25 빌드 PASS

---

## 다음 작업 (4 갈래 — argus 분할 sprint 연속)

### A. argus-cli v1 — production resilience + KIVI/Offload + swap + prompt-batch 흡수

- v0 reject 12종 점진 해제. `--no-resilience` flag (default-on resilience), KIVI/Offload KV mode, swap 8종, prompt-batch, tensor-partition, profile/profile-events.
- 호출 path: 이미 session::* 모듈 분리 완료.
- 비용: 1~2일. S25 + Jetson + CPU 디바이스 게이트.

### B. argus-chat 신설

- session::chat::repl::run_chat_repl_v2 위임 (이미 존재). `--chat-socket` UDS + `--chat-tcp` listener 통합.
- multi-turn KV pos 보존 (4-5-g `c1a4b481` 검증된 path).
- 비용: 0.5~1일.

### C. argus-bench 신설

- 신규 `engine/src/session/bench/` (runner + metrics + histogram + thermal isolation).
- 6 신규 옵션 (`--bench-iterations`, `--bench-warmup`, `--bench-cold-fire`, `--bench-output`, `--bench-metrics`, `--bench-thermal-isolation`).
- 비용: 1~2일.

### D. argus-eval 신설

- clap subcommand 4종: `experiment` / `ppl` / `ll` / `dump`.
- 신규 `engine/src/session/experiment/` (mpsc + schedule). 기존 ppl/eval/dump 모듈 재사용.
- 비용: 2~3일.

---

## Landmines / 미해결

### 1. CI matrix 첫 실행 결과 미검증
- `.github/workflows/feature-matrix.yml` 도입 후 첫 PR 머지/push 시 GHA 실제 실행 결과 미확인. 호스트 ubuntu-latest runner 와 worktree 호스트 환경 (NDK / OpenCL ICD)이 다름.
- 처음 실행에서 OpenCL ICD apt install 정도가 fail 후보. 사용자 PR push 후 GHA 결과 확인 필요.

### 2. legacy_generate Cargo.toml entry 임시 보존
- argus-cli sprint 이후 chat/bench/eval 작업 시 baseline 비교 위해 legacy_generate 빌드 entry 보존.
- 4 sub-sprint 완료 시점 `git rm engine/legacy/generate.rs` + entry 제거 일괄 처리 예정.

### 3. INV-LAYER-005 baseline JSON stale
- `engine/tests/spec/inv_layer_baseline.json` 의 V-30 27건 (bin/generate.rs) entry 는 src/bin 제외로 검사 대상에서 사라졌지만 JSON 갱신 미적용. silent ignore라 영향 0, 단 cleanup 필요.

### 4. QNN_SDK_ROOT 환경변수 문서화
- B-4 에서 추가된 `QNN_SDK_ROOT` env 는 README / ARCHITECTURE.md 에 미문서화. 다음 sprint 또는 backlog 항목.

### 5. CI 9 matrix 빌드 시간 예측 미확정
- cargo cache 적용했지만 첫 PR 시 cache miss로 ~10~15분 추정. 매 PR ~2~3분 추정 (cache hit). 실측 후 fail-fast 또는 matrix 축소 결정.

### 6. cuda-embedded build broken state 의 broader cause
- B-3 단 1줄 path fix로 회복 — 그러나 PR 머지 시점부터 잡히지 않은 것은 CI matrix 미도입 + cuda-embedded 빌드를 호스트에서 안 돌렸기 때문.
- B-5 도입으로 향후 재발은 차단되었으나, 직전 Jetson gate 실패 시 (handoff_inv_layer_L_hot_path_subtrait_2026_05_24.md) 본 fix 가 없어 sprint 무관 broken state 로 잘못 분류된 사례.

---

## 진입 명령 (다음 세션)

```
"argus-cli v1 진행"      # production 옵션 흡수 (resilience + KIVI/Offload + swap + prompt-batch)
"argus-chat 진행"        # session::chat::repl + UDS/TCP listener
"argus-bench 진행"       # 신규 session::bench + TBT histogram + cold-fire
"argus-eval 진행"        # clap subcommand 4종 (experiment/ppl/ll/dump)
```

### 즉시 재현 명령 (검증용)

```bash
# 로컬 feature matrix 검증
./.agent/skills/developing/scripts/feature_matrix.sh argus_cli

# 9 조합 개별 검증
for feat in "" "--no-default-features" "--no-default-features --features cuda-embedded" \
            "--no-default-features --features qnn"; do
  cargo build --release --bin argus_cli $feat
done

# S25 Adreno bit-identical
python scripts/run_device.py -d galaxy_s25 argus_cli \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --prompt "The capital of France is" --num-tokens 32 \
  --greedy --backend opencl --kv-type f16
```
