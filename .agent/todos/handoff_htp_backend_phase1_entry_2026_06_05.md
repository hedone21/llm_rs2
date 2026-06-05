# Handoff: HTP backend S4+S5+Y 완료 (host) → S25 device 검증 진입

**작성**: 2026-06-05 (갱신: S4+S5+Y 완료 + 적대검토 반영)
**HEAD**: `eab1d65e fix(htp): Y — matmul_transposed prefill+decode 모두 NPU dispatch (prefill garbage 수정)`
**HTP 커밋**: `d331d01b`(S4+S5 골격) + `eab1d65e`(Y fix) — master 히스토리, **미push**
**브랜치**: `master` (사용자가 d331d01b 위에 ADR-0003 `8c23a72a` + technique-api M1 `136f7cdd` 병렬 적재)
**작성자**: 메인 세션 (오케스트레이터)

**다음 세션 진입 문장**: "HTP S4+S5+Y device 검증 — argus_cli 로 S25 deploy-test (OOM + token-id 정합)"

---

## TL;DR

HTP FastRPC backend 의 **decode weight matmul NPU dispatch 완전 경로(S4+S5+Y) host-완결**. `--backend htp` 가 Q4_0 weight matmul(QKV/Wo/gate/up/down + Q4_0 lm_head)을 **prefill·decode 모두 HTP NPU 로 dispatch**한다. 멈춘 이유 = **correctness 는 S25 android 디바이스 측정 필수**(host 는 `HtpFastrpcHost::new` 가 Err). host 검증 가능한 컴파일·게이트·repack·dispatchable 유닛테스트 전부 통과 + 적대검토 BLOCKING 0.

**핵심 결론**: q4x4x2 weight repack + M≥1 NPU dispatch = host GREEN. 가장 큰 device 리스크 = **rpcmem ~703MB OOM** 과 **M>1 GEMM 의 DSP 지원 여부**(둘 다 loud-fail). paper angle = quality-aware heterogeneous placement.

---

## 진행 상태 — 이번 세션 완료

| 단계 | 작업 | 상태 | Commit |
|---|---|---|---|
| S1 | `impl Buffer for RpcmemBuffer` + dtype 필드 | ✅ | `c391e40c` |
| S2+S3 | `HtpFastrpcMemory` + init.rs htp arm (`--backend htp` 부팅) | ✅ | `427f85eb` |
| S4+S5 | repack 모듈 승격 + copy_weight_from q4x4x2 + matmul override | ✅ | `d331d01b` |
| **Y** | **prefill+decode 모두 NPU(M≥1) — prefill garbage 수정** | ✅ | `eab1d65e` |

**Y 가 고친 버그(적대검토 발견)**: d331d01b 는 NPU dispatch 를 M==1(decode)만 했는데 copy_weight_from 은 모든 Q4_0 weight 를 q4x4x2 repack → prefill(M>1)이 cpu fallback 으로 q4x4x2 를 standard block_q4_0 로 읽어 **garbage**. multi-token 프롬프트 첫 토큰부터 오답. **1-token 프롬프트면 prefill=decode(M=1)라 버그가 가려짐** ← device 게이트 설계 시 주의.

**검증 게이트(전부 통과)**: `cargo check`(기본+htp)·lib clippy `-D warnings`·fmt clean, lib htp 유닛테스트 8/8(repack 2 + dispatchable 6), NDK `aarch64-linux-android` cross-check PASS(android dispatch 컴파일, htp 에러 0). 적대검토 2회 PASS(BLOCKING 0).

**설계 결정(확정)**:
- **Y = M≥1 dispatch**: `htp_matmul_dispatchable` 가 `(n,k,m)` 반환, `via_htp` 가 ne_x=[k,m,..]/ne_y=[n,m,..]. M==1 이면 microbench proven GEMV 와 byte-identical.
- **pivot = weight q4x4x2 인가**: matmul override 가 weight buffer 가 RpcmemBuffer 면(=q4x4x2) NPU dispatch 아니면 **loud bail** — cpu fallback 절대 금지(silent garbage 봉쇄).
- **copy_weight_from**: Q4_0 2D(K%256==0)만 rpcmem+repack, **F32/F16/그외 전부 cpu_companion** (F16 lm_head ~445MB rpcmem 낭비 회피 = OOM 완화).

---

## 다음 작업 — S25 device 검증 (D→1→2 순서)

**device bin = `argus_cli`** (single-prompt). ⚠️ handoff 구버전/MEMORY 의 `legacy_generate` 는 **존재하지 않음** — `argus_cli` 가 `--backend htp` arm 에 도달하는 유일 경로(`bin_setup::build_inference_ctx`).

```bash
# 빌드 (run_device.py 가 NDK env 주입)
python scripts/run_device.py -d s25 --features htp_fastrpc --bin argus_cli
#  또는 수동: source android.source && cargo build --release \
#    --target aarch64-linux-android --features htp_fastrpc --bin argus_cli -p llm_rs2

# [0순위] dispatch 메커니즘 격리 — microbench 먼저 (prof_usecs>0 증거)
adb -s <S25> push target/aarch64-linux-android/release/microbench_htp_matmul /data/local/tmp/
adb -s <S25> shell "cd /data/local/tmp && LD_LIBRARY_PATH=.:/vendor/lib64 \
    ./microbench_htp_matmul --shape mm_qkv"   # diag: prof_usecs=... , max_abs_err<5e-2

# [1순위] OOM + 부팅 — 전 weight rpcmem 상주 alloc 성공하는가
adb -s <S25> shell "cd /data/local/tmp && LD_LIBRARY_PATH=.:/vendor/lib64 \
    ./argus_cli --backend htp --model-path qwen.gguf --tokenizer-path tok.json \
      --prompt 'The capital of France is' --num-tokens 16 --greedy 2>&1 | tee htp.log"
#   기대: '[htp] matmul_transposed: NPU dispatch 활성 (Q4_0 q4x4x2, M≥1)' 1회 출력
#   RED: 'rpcmem_alloc returned NULL' / 'fastrpc_mmap rc=' → OOM
#   RED: 'q4x4x2 Q4_0 weight matmul 은 NPU dispatch 필요' bail → M>1 미지원 또는 act 비-rpcmem
#   RED: DSP status 에러 → M>1 GEMM DSP 미지원

# [2순위] token-id 정합 — CPU baseline 과 비교 (멀티토큰 프롬프트 필수!)
adb -s <S25> shell "... ./argus_cli --backend cpu --model-path qwen.gguf \
      --tokenizer-path tok.json --prompt '...' --num-tokens 16 --greedy | tee cpu.log"
# 비교: 첫 토큰 일치 + 발산 시점 기록. **logit max_abs_err<1e-2 게이트는 폐기**
#       (single matmul 5e-2 와 모순) → token-id 일치 + top-5 overlap 으로.
```

**위임**: `senior-implementer`(device 수정 필요 시) + `tester`(deploy-test). 권한 `engine/src/backend/htp_fastrpc.rs`.

---

## Landmines / 미해결

- **★ OOM (최우선, RED ~55-70%)**: copy_weight_from 이 Q4_0 weight(projection 28L×7 ≈ **703MB**)를 rpcmem system heap(DMA-BUF, 수백MB 추정)에 전면 상주. lm_head 가 Q4_0 저장이면 추가. **완화 O3(선제 권장)**: 이미 F16 은 cpu 로 뺐음. O4: 큰 weight(ffn) host-heap hybrid(~6줄). RED 시 어느 alloc 에서 NULL/mmap-fail 인지 캡처.
- **★ M>1 GEMM DSP 미검증**: microbench 는 M==1 GEMV 만 GREEN. Y 의 prefill(M>1) dispatch 는 device-미검증. DSP 미지원 시 status 에러(loud, garbage 아님) → M==1 게이트 복원 + decode-only(W'/V) 후퇴. llama.cpp HTP 는 general matmul 하므로 지원 가능성 높음.
- **★ token-id 게이트 모순**: "첫 logit max_abs_err<1e-2"는 single matmul 5e-2 와 모순(196 matmul 누적인데 더 엄격) → **token-id 일치 + top-5 overlap** 으로 재설계 필수. 첫 토큰 불일치=버그, 중반 발산=quant 누적(허용).
- **★ 1-token 프롬프트 함정**: prefill seq_len=1 이면 M=1 이라 prefill garbage(이미 Y 가 고침)와 무관하게 게이트가 약해짐. **반드시 멀티토큰 프롬프트**로 검증.
- **q4x4x2 ↔ standard 비호환**: 동일 바이트를 DSP=q4x4x2(맞게)·CPU=standard(틀리게) 읽음. Y override 가 q4x4x2 weight 의 cpu fallback 을 loud bail 로 봉쇄. Q/K inverse-permute 는 row 단위라 q4x4x2(K축 grouping)와 무충돌.
- **weight-swap 비호환(scope 밖)**: `pressure/weights/phase_aware_swap.rs` 등이 weight 를 standard Q4_0 로 직접 읽음 → q4x4x2 repack 과 상호 배타. resilience swap 활성 시 주의(현 device 게이트는 swap off).
- **prof_usecs 미노출**: via_htp 가 rsp.prof_usecs 를 버림. "NPU 실행" 증거는 microbench(prof 출력) + `htp_dispatch_log_once` eprintln 으로 간접 확보. 필요 시 via_htp 에 prof 누적 ~3줄 추가.
- **NPU 절대속도 < GPU < CPU** (32.4<37.6<63.8 tok/s). "빠르다" 금지 — 협력+품질 서사.

---

## 참고 링크

- 메모리: `[[htp-production-backend-feasibility]]`, `[[htp-f16-matmul-measurement-artifact]]`
- 코드: `engine/src/backend/htp_fastrpc.rs` (matmul_transposed/via_htp/dispatchable :405~, copy_weight_from), `.../repack.rs` (q4x4x2), `.../buffer.rs`(RpcmemBuffer), `.../memory.rs`(HtpFastrpcMemory)
- GREEN reference: `engine/microbench/htp_matmul.rs::run_htp` (M==1 S25 GREEN)
- 설계: `arch/htp_fastrpc.md`, `spec/htp_fastrpc.md` (INV-HTP-FRPC-001~005)
