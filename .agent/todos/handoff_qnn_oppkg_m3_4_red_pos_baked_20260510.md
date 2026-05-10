# Handoff: M3.4 RED — pos baked architectural blocker

**Date**: 2026-05-10
**HEAD**: `90617cc` (M3.4 14-node body + device gate RED, push됨)
**관련 파일**: `papers/eurosys2027/_workspace/experiment/m3_4_passgate.md` (상세 측정 + root cause)

---

## 0. 한 줄 요약

M3.4 메인 게이트 RED. graphFinalize 28× ~1.36s GREEN, 그러나 prefill 진입 시 segfault. Root cause 정밀 격리 결과 본질은 **pos baked architectural blocker** — M2.H builder가 `start_pos`/`write_pos`를 `QNN_PARAMTYPE_SCALAR`로 graph build 시점 hardcoded → multi-token decode에서 pos 변화 처리 불가. **사용자 architectural decision 필요**.

---

## 1. 측정 결과 요약

### GREEN
- runtime init / model load / lm_head derivation: PASS
- graphFinalize 28× total ~1360 ms (예상 ~33s 대비 24× 빠름, INV-167 1500 ms budget 내)
- layer 0 cold ~1196 ms, layer 1~27 warm ~6.7 ms (driver cache hit)

### RED
- prefill (seq_len=5) 진입 직후 segfault
- 정확성 / TBT / VmRSS 측정 불가

### 결정적 발견 (본 세션 후속 분석)
M2.H microbench `engine/src/bin/microbench_qnn_qwen_layer.rs:1721-1741`:
```rust
let mk_rope_params = |start: i32, th: f32| {
    Qnn_Param_t {
        paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,  // ← BAKED at graph build time
        name: pn_start_pos.as_ptr(),
        __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
            scalarParam: Qnn_Scalar_t {
                __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { int32Value: start },
            },
        },
    }
};
```

M2.H는 single-token (pos=0) 검증이라 이 한계 노출 X. production multi-token decode 시 graph가 pos=0으로 baked → RoPE/KvScatter가 모든 token에서 pos=0 → 결과 garbage (segfault 별개로 정확성부터 fail).

---

## 2. Architectural Decision Required

| ID | 옵션 | 추정 | M3 timeline +α | 위험 / 비고 |
|---|---|---|---|---|
| **D-D** | M2 ops 수정 — `start_pos`/`write_pos`를 SCALAR param → input tensor. `crates/qnn_oppkg/src/ops/rope.rs` + `kv_scatter.rs` + `kernel_rope_simple_oop` / `kernel_kv_scatter_*` `.cl` kernel 갱신. CLAUDE.md D7 결정에 따라 가능. | 5-7일 | +1.5주 | M2 검증 재실행 필요. paper evidence 강함. |
| **D-E** | M3 scope 재정의 — single-token (pos=0) 검증 한정 + multi-token decode는 OpenCL fallback | 2-3일 | +0.5주 | M3 메인 게이트 의미 약화. paper evidence 약함. |
| **D-A/B/C** | segfault만 fix (이전 보고) — pos baked 문제 미해결로 정확성 RED 유지 | 2-5일 | 의미 없음 | 정확성 게이트 PASS 불가 |

**총 M3 timeline (기존 4주 + α)**:
- D-D: +1.5주 = ~5.5주
- D-E: +0.5주 = ~4.5주 (단 게이트 약화)

전체 M3+M4 plan (5~7주 예상)은 D-D 시 6.5~7.5주 → D-E 시 5.5~6.5주.

---

## 3. 이전 세션 산출물

### Code (HEAD `90617cc`)
- `engine/src/backend/qnn_oppkg/layer_graph.rs` (+1010): 14-node body 본격 이식 + execute path
- `engine/src/backend/qnn_oppkg/weight_pack.rs` (신규 +120): GGUF Q4_0 AOS → SOA repack
- `engine/src/backend/qnn_oppkg/mod.rs` (+60): fallback wire-up + 12 trait method 위임
- `engine/src/bin/generate.rs` (+20): qnn_oppkg primary일 때 OpenCL secondary memory 위임

### Spec/Arch
- `papers/eurosys2027/_workspace/experiment/m3_4_passgate.md`: 측정 + root cause 본문
- `arch/30-engine.md` §18: M3 컴포넌트 설계 (M3.0)
- `spec/30-engine.md` 부록 C: ENG-QNN-201~240 (M3.0)
- `spec/41-invariants.md` §3.24: INV-166~180 (M3.0)

---

## 4. 다음 세션 entry point

### D-D 채택 시
1. M2 ops 수정 시작 — `crates/qnn_oppkg/src/ops/rope.rs::DESCRIPTOR` build_layout 수정 — `Qnn_Param_t` SCALAR 제거 + input tensor (`pos: I32, [1]`) 추가
2. KvScatter 동일
3. `engine/kernels/simple_ops.cl::kernel_rope_simple_oop` + KvScatter kernel signature 갱신 (pos를 buffer arg로)
4. M2.H 14-node graph builder도 동일하게 수정 — `microbench_qnn_qwen_layer.rs` 재실행으로 single-token correctness GREEN 재검증
5. `engine/src/backend/qnn_oppkg/layer_graph.rs::AndroidLayerGraph::execute`에 pos buffer write 추가
6. M3.4 device gate 재시도

### D-E 채택 시
1. transformer.rs `forward_into` 분기 — prefill (seq_len > 1)일 때 OpenCL primary path, decode (seq_len == 1, pos == 0) 한 번만 qnn_oppkg fast path
2. 다른 pos에 대해서는 OpenCL fallback
3. M3 게이트를 "decode 첫 토큰 정확성" 한정으로 약화
4. paper section을 single-token TBT proof of concept으로 재작성

---

## 5. Plan re-scope 사용자 결정 사항

본 결정은 plan §M3.4 RED 분기 ("사용자 호출 (scope 재정의)") 그대로. 핵심 질문:

1. **D-D vs D-E**: 1.5주 추가 투자해서 architectural fix할지, 0.5주로 scope 약화하고 빠르게 M4로 진입할지
2. M2 ops/.cl kernel 수정에 대한 회귀 위험 수용 여부
3. paper deadline (PACT2026 / EuroSys2027) 와 M3 강도 trade-off

---

## 6. 다음 세션 시작 절차

```bash
cd /home/go/Workspace/llm_rs2
git log --oneline -5         # 90617cc 확인
cat papers/eurosys2027/_workspace/experiment/m3_4_passgate.md
cat .agent/todos/handoff_qnn_oppkg_m3_4_red_pos_baked_20260510.md  # 본 문서
adb devices                   # R3CY408S4HN
```

사용자 결정 (D-D / D-E) 후 다음 단계 위임 — Senior Implementer 또는 Architect.

---

**End of Handoff**

self-contained: 다음 세션은 본 문서 + m3_4_passgate.md 만으로 시작 가능.
