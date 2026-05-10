# precision_swap: OpenCL backend에서 SOA AUF Q4_0 swap 후 다음 forward에서 "B is not OpenCL buffer"

**Date filed**: 2026-05-01
**Filed by**: PACT 2026 Fig 6 / §5.2 E2E (papers repo, fig6_e2e_run.sh, policy_argus_mobile.lua)
**Scope**: `engine/src/backend/opencl/mod.rs` (matmul_q4_0 path), `engine/src/models/weights/swap_executor.rs::materialise_auf_soa_weight`, `engine/src/buffer/noshuffle_weight_buffer.rs`
**Severity**: high — Manager-driven runtime weight swap의 §3.4.3 narrative 검증 불가. swap 자체는 성공하지만 직후 forward가 항상 실패하므로 OpenCL backend에서 precision_swap이 사실상 no-op + crash.

## Summary

S25 / Qwen 2.5 1.5B / Adreno OpenCL 환경에서 `--secondary-gguf <Q4_0.auf> --secondary-dtype q4_0` 으로 띄운 generate 프로세스에 Manager가 `SwapWeights { ratio: 0.9, target_dtype: Q4_0 }` directive를 보내면, swap_executor 단계는 **정상 종료** (`[WeightSwap] OK: ratio=0.90, swapped=25/28, qcf_swap=0.8929, latency=937~1670ms`) 하지만 직후의 디코드 forward에서 `Error: B is not OpenCL buffer` 로 프로세스가 즉시 종료된다.

`get_cl_mem` (engine/src/backend/opencl/mod.rs:144–161) 은 `NoshuffleWeightBuffer` 를 downcast 후 `d_buf` 를 반환하도록 핸들링하고 있음에도 (line 154–159), swap 후 weight tensor가 이 경로로 풀리지 않는다. 즉 swap이 만든 tensor의 backing이 `NoshuffleWeightBuffer` 가 아니거나, downcast가 실패하는 상황으로 추정된다.

`switch_hw cpu` (post-swap CPU migration) 도 같은 디바이스/run에서 별개의 segfault를 유발하며 (`[Switch] Resilience: GPU→CPU at token 450 → KV Migrate → Switched to CPU → Segmentation fault`), 본 이슈와 별개이지만 swap 시나리오에서 동시에 노출된다. 본 이슈는 GPU에 머물러도 발생하므로 CPU migration과 무관하게 좁혀진다.

## Reproduction

```bash
# 디바이스: Galaxy S25 (Snapdragon 8 Elite / Adreno 830), Android 15
# llm_rs2 HEAD: 12efc90 "feat(manager,engine): support precision_swap alias and expose swap_weights capability"
# 바이너리 md5 (host): a673ced2f36e9228dbe7e5d66fa5f2f9 generate
# AUF: qwen2.5-1.5b-q4_0.auf (1.25 GiB, --variants adreno_soa, --include-lm-head auto)

# Manager + Engine를 별도로 띄우고, Manager에서 SwapWeights를 발화시키는 시나리오.
# 가장 단순한 재현: --force-swap-ratio 로 prefill 직전 swap을 강제해 본다.

adb shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp ./generate \
    -m models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    -b opencl \
    --secondary-gguf models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.auf \
    --secondary-dtype q4_0 \
    --quantize-lm-head q4_0 \
    --force-swap-ratio 0.9 \
    --max-seq-len 4096 \
    --prompt 'Hello world.' \
    -n 50 --greedy --ignore-eos"
```

본 이슈는 PACT 2026 fig6 측정 중 `precision_swap` Lua 액션을 통해 노출됐다 (Manager → Engine resilience transport). 위 재현 명령은 동등한 swap 경로를 단일 프로세스에서 강제로 트리거한다.

## Reference run (PACT 2026 Fig 6 from papers repo)

```bash
# papers repo
FIG6_FULL_MODEL_QUANT=f16 \
FIG6_FULL_POLICY=policy_argus_mobile.lua \
ALLOW_CHARGING=1 SKIP_INTRO=1 \
  pact2026/experiments/scripts/fig6_e2e_run.sh \
  --device R3CY408S4HN --scenario pubg --ctx 4K --config argus_full --run-id 1
```

결과 디렉토리:
- `pact2026/experiments/results/fig6_e2e/pubg/4K/argus_full/R3CY408S4HN_run1_20260501_091412/`
- `pact2026/experiments/results/fig6_e2e/pubg/4K/argus_full/R3CY408S5SB_run1_20260501_090616/` (post-swap CPU migration → segfault 동시 노출)

## Evidence

### (1) 실패 매트릭스

| # | secondary | secondary-dtype | force-swap-ratio | swap trigger | swap result | next forward | 결과 |
|---|-----------|-----------------|------------------|---------------|--------------|---------------|------|
| A | `q4_0.auf` | `q4_0` (명시) | (Manager 동적) | `precision_swap` Lua 액션 (tok≈2770) | ✅ `OK swapped=25/28 latency=1670ms` | ❌ `Error: B is not OpenCL buffer` | swap 직후 종료 |
| B | `q4_0.auf` | `q4_0` | (Manager 동적) | `precision_swap` Lua 액션 (tok≈1764) | ✅ `OK latency=937ms` | ❌ post-swap `switch_hw cpu` → SEGV (별개 이슈) | tok≈1722에서 종료 |
| C | (없음) | — | — | (swap 미발화) | — | — | ✅ 정상 (4K F16 baseline) |
| D | `q4_0.auf` | `auto` (기본) | (Manager 동적) | swap 시도 | ❌ `secondary mmap: layer 1 missing tensor 'attn_q.weight'` | — | secondary open 단계에서 실패 (별개 이슈, 아래 §6 참조) |

→ **`secondary-dtype q4_0` 가 명시되어 swap_executor가 정상 완료한 케이스 (A, B) 에서만 본 이슈 노출**. 즉 Engine이 weight를 실제로 `NoshuffleWeightBuffer` 로 교체한 후의 forward 경로에서 cl_mem downcast가 실패한다.

### (2) Engine 측 evidence

#### Case A — papers `R3CY408S4HN_run1_20260501_091412` `llm_events_*.log` 끝부분

```
[Resilience] Directive seq=149: SwapWeights { ratio: 0.9, target_dtype: Q4_0 }
[WeightSwap] OK: ratio=0.90, swapped=25/28, qcf_swap=0.8929, latency=1670ms
[WeightSwap] stages: prefault=1242.6ms mmap_permute=424.0ms arc_swap=0.0ms madvise=0.5ms soa_reconvert=3.8ms gen_bump=0.0ms
Error: B is not OpenCL buffer
[LLM] end ts=1777594755.354526425
```

decode 직전까지는 정상 진행 (tok 2770까지 F16 forward), `WeightSwap OK` 메시지 (25/28 layers, latency 1670ms = prefault 1242 + mmap_permute 424 + soa_reconvert 4 + 기타 0) 가 찍힌 뒤 **다음 forward call** 에서 즉시 `Error: B is not OpenCL buffer`. soa_reconvert 4ms는 noshuffle SOA registry 재등록 (engine/src/backend/opencl/mod.rs:3980 `register_pre_converted_soa_tensor`) 단계가 동작했음을 시사한다.

#### Case B — papers `R3CY408S5SB_run1_20260501_090616`

```
[Resilience] Directive seq=90: SwapWeights { ratio: 0.9, target_dtype: Q4_0 }
[Resilience] Directive seq=90: SwitchHw { device: "cpu" }
[WeightSwap] OK: ratio=0.90, swapped=25/28, qcf_swap=0.8929, latency=937ms
[WeightSwap] stages: prefault=597.1ms mmap_permute=339.4ms arc_swap=0.0ms madvise=0.4ms soa_reconvert=0.4ms gen_bump=0.0ms
[ScoreDiag] cache_pos=450, prefix=66, decode_steps=1779
...
[Switch] Resilience: GPU→CPU at token 450
[KV Migrate] 28 layers migrated (UMA zero-copy re-tag)
[Switch] Resilience: Switched to CPU.
Segmentation fault 
```

여기서는 swap **직후** GPU에서 한 번의 forward (`cache_pos=450, decode_steps=1779`) 가 이상 없이 진행되었다 — 즉 case A의 "B is not OpenCL buffer" 와는 다른 결과. 차이점은 case A는 `secondary-dtype q4_0` + `--quantize-lm-head q4_0` + `--resilience-prealloc-switch` 조합, case B는 동일 조합에 추가로 swap 직후 `switch_hw cpu` directive 가 들어가 KV migration → segfault 로 흐른 것이다. 본 이슈는 case A 경로에 한정한다 (case B의 SEGV는 post-swap CPU migration 별개 이슈).

### (3) 에러 발생 위치 (코드)

`engine/src/backend/opencl/mod.rs:2052–2068` — `OpenClBackend::matmul_q4_0`:

```rust
pub fn matmul_q4_0(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    ...
    let a_buf =
        get_cl_mem(a.buffer().as_ref()).map_err(|_| anyhow!("A is not OpenCL buffer"))?;
    let b_buf =                                                                     // ← 여기서 실패
        get_cl_mem(b.buffer().as_ref()).map_err(|_| anyhow!("B is not OpenCL buffer"))?;
    let out_buf =
        get_cl_mem(out.buffer().as_ref()).map_err(|_| anyhow!("Out is not OpenCL buffer"))?;
    ...
    if m == 1 {
        let b_key = b_buf.as_ptr() as usize;
        if let Some(entry) = self.lookup_noshuffle_soa(b_key)
            && let Some(ref q_img) = entry.q_img
        {
            return self.matmul_q4_0_noshuffle(...);
        }
    }
    // 일반 Q4_0 GEMV 폴백 ...
}
```

`b` 는 swap 후의 weight tensor (decoder layer의 attn_q/k/v/o, ffn_gate/up/down 중 하나). 정상 경로에서 b.buffer() 는 `NoshuffleWeightBuffer` 를 downcast해서 `d_buf()` 가 반환되어야 하지만 (line 154–159), 실제로는 `Buffer is not an OpenCL buffer type` 으로 error 분기됨.

### (4) 추정 원인 (가설)

`materialise_auf_soa_weight` (engine/src/models/weights/swap_executor.rs:668–726) 가 반환하는 tensor의 buffer trait 이 `NoshuffleWeightBuffer` 가 아니거나, `Tensor::buffer()` → `&dyn Buffer` 의 vtable 이 downcast 가능한 형태로 노출되지 않은 가능성:

1. **Backing trait object 일관성** — `alloc_pre_converted_soa_tensor` 가 `NoshuffleWeightBuffer::new(...)` 로 만든 buffer를 `Box<dyn Buffer>` 로 감쌀 때 `as_any` 메서드가 wrap된 트레잇 객체로 흘러가서 downcast가 실패하는 경우.
2. **swap_executor 가 NoshuffleWeightBuffer 가 아닌 fallback 텐서를 만든 경우** — `materialise_auf_soa_weight` 의 `Ok(None)` 반환 조건 (line 691–711: split_pre_converted_soa 실패, 사이즈 검증 실패, alloc_pre_converted_soa_tensor None) 중 하나가 발화하여 caller 가 `materialise_tensor` 폴백으로 빠진 경우. 이때 buffer는 다른 종류 (CL buffer 또는 host tensor) 로 만들어지며, 그 종류가 `get_cl_mem` 에서 인식 못 하는 변종이면 본 에러가 난다.
3. **register_pre_converted_soa_tensor 시점의 Arc clone 누락** — swap이 새 weight를 ModelRef에 install 할 때 NoshuffleWeightBuffer 의 Arc 원본이 weight tensor 의 backing 으로 가지 않고 사이드 registry 에만 들어가, weight tensor 본체는 placeholder buffer를 가진 상태가 되는 경우.

(1) 또는 (2) 가 가장 가능성 높음. swap_executor의 분기별 로깅을 켜서 어느 path가 실제로 사용됐는지 (auf_soa OK vs aos fallback vs materialise_tensor fallback) 확인하면 좁혀짐.

### (5) Pre-existing test 가 cover 못 한 이유

`engine/tests/spec/test_eng_dat_096_auf_secondary.rs` 는 secondary mmap open + tensor index round-trip 만 테스트하고, 실제 backend forward를 호출하지 않는다. `test_qcf_swap_dump.rs` 는 dump pipeline이 목표라 forward가 들어가지만 NVIDIA/CPU backend 에서 주로 검증되어 OpenCL 경로는 미커버다.

`engine/tests/spec/test_inv_135_136_lm_head_auf.rs` 는 lm_head AUF 경로만 본다.

본 이슈는 **OpenCL backend × NoshuffleWeightBuffer × SwapWeights post-execute forward** 의 cross-path 에서 노출되며, 현 테스트 매트릭스에 직접 대응하는 케이스가 없다.

## 별개 이슈 — secondary-dtype auto 가 single-dtype AUF에서 F32 선택

이번 디버깅 중 발견한 부수적 문제를 함께 기록한다 (별도 이슈 분리 권장).

`engine/src/models/weights/secondary_mmap.rs:748–778` 의 `SecondaryDtypeChoice::Auto` 분기에서, v0.1.x single-dtype AUF (META.default_dtype 미설정) 의 경우 `available_dtypes.iter().next()` 가 BTreeSet 정렬 기준으로 가장 작은 u32 dtype 을 반환한다.

`TensorDType::F32 = 0, F16 = 1, Q4_0 = 3` 이므로 norm 텐서가 F32 로 들어있는 일반적인 AUF (decoder weights Q4_0 + per-layer norm F32 + lm_head Q4_0 = 약 197 Q4_0 + 57 F32 + 1 F16 분포) 에서 Auto가 **F32 를 선택**하게 된다. 그 결과 dtype 필터 (line 845–849) 에서 Q4_0 weights 가 모두 걸러지고 layer 1의 attn_q.weight 가 missing 으로 보고된다 (`secondary mmap: layer 1 missing tensor 'attn_q.weight'`).

회피책 (현재): `--secondary-dtype q4_0` 명시.
근본 fix: Auto 모드가 norm dtype 이 아닌 weight dtype 을 선택하도록, 또는 Q4_0 / F16 만 후보로 두도록 (F32 는 norm 전용 dtype 이므로 secondary swap target 으로는 부적합) 변경.

USAGE.md §2.13.1 line 1238 의 "기본 `auto` — AUF META `default_dtype` 우선, 없으면 TENSOR_INDEX first-match" 설명 자체는 사실에 부합하지만, "first-match" 가 의도와 다르게 norm dtype 으로 떨어지는 점이 문제. v0.2 multi-dtype AUF 빌드 (`auf-tool build --dtypes q4_0,f16 --default-dtype q4_0`) 시에는 META.default_dtype 이 우선이라 본 문제 회피.

## Workaround (PACT 2026 측정용)

본 이슈가 수정되기 전까지 papers/policy_argus_mobile.lua 에서 다음과 같이 운영:

1. `precision_swap` 액션은 emit 하지 않거나 (테스트용 비활성), 발화하더라도 swap 후 forward가 실패하므로 §5.2 main figure 측정 시 swap 트리거 임계값을 매우 보수적으로 두어 사실상 발화 안 되게 한다.
2. `switch_hw cpu` (post-swap CPU migration) 액션은 별개의 segfault 가 있어 마찬가지로 비활성.
3. `--secondary-gguf` 자체는 그대로 두어 Engine capability 에 `swap_weights` 가 등록되도록 유지 (Manager 측 narrative 무결성).

## Suggested fix path

1. **Buffer downcast 검증**: swap_executor 가 만든 weight tensor 의 `buffer().as_any().type_id()` 를 직접 확인하는 unit test 추가 (`tests/spec/test_swap_buffer_type_after_auf_soa.rs`). NoshuffleWeightBuffer 가 아니면 즉시 fail.
2. **Forward path E2E**: SwapWeights 명령 실행 후 1-token decode 까지 진행하는 integration test (engine/tests/integration/test_swap_then_decode_opencl.rs). 현 매트릭스 공백을 메운다.
3. **Auto dtype heuristic fix**: secondary_mmap.rs:Auto 분기에서 F32 후보를 자동 제외 또는 우선순위 (Q4_0 > F16 > F32) 부여.
4. (선택) `force-swap-ratio` 강제 swap 단일 프로세스 재현 명령을 docs/auf_tool_guide.md 에 회귀 검증 절차로 추가.
