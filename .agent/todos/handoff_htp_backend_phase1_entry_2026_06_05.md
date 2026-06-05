# Handoff: HTP backend 학술 재평가 완료 → Phase 1 (engine backend wire-up) 진입

**작성**: 2026-06-05
**HEAD**: `259226f2 fix(microbench): htp matmul bin --shape 파싱 — F16/Q4_0 라벨 버그 수정`
**브랜치 / Worktree**: `master`
**작성자**: 메인 세션 (오케스트레이터)

**다음 세션 진입 문장**: "HTP backend Phase 1 진입 — RpcmemBuffer 에 Buffer trait impl 부터"

---

## TL;DR

이번 세션: (1) full_matrix `ours.htp` "F16 7.8ms 고정"이 **측정 하네스 라벨 버그**임을 확정·수정·산출물 정정 완료(`259226f2`). (2) "engine 에 HTP backend 추가" 를 **학술 목적**(OpenCL+HTP heterogeneous, 느림 무관)으로 재평가 — **타당** 판정. 다음 = **Phase 1: microbench 에서 이미 GREEN 인 NPU op dispatch 를 Backend trait 에 wire-up** 하여 `--backend htp` 로 1 token 생성. 멈춘 이유 = 분석·로드맵 확정, 구현은 다음 세션(별도 sprint).

**핵심 결론 한 줄**: device 기반(NPU dispatch)은 이미 GREEN(MUL_MAT/ADD/MUL/FLASH_ATTN correctness PASS), paper angle = **quality-aware heterogeneous placement**(QCF 기반, HeteroLLM 이 회피한 빈자리).

---

## 진행 상태

### 이번 세션 완료

| 작업 | 상태 | Commit / 산출물 |
|---|---|---|
| F16 "7.8ms 고정" = 라벨 버그 규명 (워크플로우 4-reader + device 검증) | ✅ | `prof_usecs` N-선형 202→585→6722 실측 |
| `htp_matmul_f16.rs` + `htp_matmul.rs` `--shape` 파싱 수정 | ✅ | `259226f2` (fmt/clippy clean) |
| S25 재측정 (단발 n=100) | ✅ | 아래 표 |
| 산출물 정정 (report.md/aggregated.csv/summary.json) | ✅ | csv 손상→raw 재집계 복구→line-edit 재정정 (untracked) |
| HTP backend 학술 타당성 재평가 (workflow ×2) | ✅ | 본 handoff + 메모리 2건 |

### 측정 / 검증 결과 (2026-06-05 S25 재측정, 전부 correctness PASS)

| op (Qwen2.5-1.5B) | ours.htp(F16) | ours.htp(Q4_0) | ours.gpu(OpenCL Q4_0) | NPU dispatch |
|---|---|---|---|---|
| MUL_MAT mm_qkv | 0.211 ms | 0.129 ms | 0.127 | ✅ GREEN |
| MUL_MAT mm_ffn | 0.593 ms | 0.258 ms | 0.188 | ✅ GREEN |
| MUL_MAT mm_lmh | 7.78 ms | 2.681 ms | 1.445 | ✅ GREEN |
| FLASH_ATTN_EXT / MUL / ADD (F16) | 0.348 / 0.105 / 0.096 | — | — | ✅ GREEN |

- **NPU op dispatch GREEN** 은 `dde08248`(DspQueueBuffer 24B layout 정정 → NPU compute GREEN)에서 해결됨. rc=14(EUNABLETOLOAD) 는 과거 이슈, **현재 미해결 아님**.
- 절대수치는 **단발 n=100** — paper 인용 전 정식 13-round 재측정 필수.

---

## 다음 작업 — Phase 1: engine backend wire-up (목적: correctness, 느림 무관)

### 액션 (S1~S5, 예상 3~4 사람-주)

1. **S1 `impl Buffer for RpcmemBuffer`** (`engine/src/backend/htp_fastrpc/buffer.rs`) → 검증: alloc→write→readback **bit-identical**. 레퍼런스 = `engine/src/memory/rpcmem/kv_buffer.rs:110`(`RpcmemKvBuffer`, 9-메서드). `as_mut_ptr(&self)` vs RpcmemBuffer `(&mut self)` 시그니처 비대칭은 interior-mut 으로 해소.
2. **S2 HTP `Memory` trait impl** (신규 `htp_fastrpc/memory.rs`) → 검증: `Buffer::size()` 일치.
3. **S3 `init.rs` 에 `"htp"` arm 추가** (opencl arm 복제) → 검증: `--backend htp` 부팅 + ModelLoadStart 도달, `bail!` 안 침.
4. **S4 `matmul`/`matmul_transposed` override** (`htp_fastrpc.rs:231-236`, 현재 cpu_companion 위임) — RpcmemBuffer backing 시 real HTP dispatch, else cpu_companion. 골격 = `rms_norm_via_htp:130`(현재 `#[allow(dead_code)]`) → 검증: 단일 op vs CPU `max_abs_err < 1e-3`.
5. **S5 end-to-end 1 token** → 검증: greedy(temp=0) **CPU/OpenCL/HTP token-id 일치** + 첫 토큰 logit `max_abs_err < 1e-2`.

**Phase 1 게이트 모델 = Q4_0 weight matmul HTP**. RMSNorm/RoPE/Softmax/SiLU/GET_ROWS 5 op 은 DSP 가 **F16 만 NO_SUPPORT, F32 는 지원**(`act-ops.c:799-808 execute_op_activations_f32` 등 F32 case 실연산, F16 default NO_SUPPORT). 우리 single-op packet 이 이미 F32 activation 을 보내므로 **cpu_companion(CPU) 이 아니라 F32 로 HTP dispatch** 가 권장 — NPU↔CPU 왕복 회피. figure 의 ours.htp norm ✗ 는 'F16 셀 미측정' 일 뿐 NPU 무능력이 아님(et.htp 가 같은 Hexagon 에서 QNN native op `op_rms_norm.py RmsNormVisitor` 로 도는 게 증거). **단 S25 F32 norm dispatch correctness 는 미검증** — Phase 1 에서 op별 F32 dispatch + vs CPU max_abs_err 게이트 필요. (2026-06-05 정정: 구 'cpu_companion' 방침 폐기.)

### 위임 prompt (초안)

> **에이전트**: `senior-implementer` (htp_fastrpc + rpcmem + dspqueue)
> **모델**: `opus`
> **권한**: `engine/src/backend/htp_fastrpc/`, `engine/src/session/init.rs`, `engine/src/backend/htp_fastrpc.rs`

```
Phase 1 S1 구현: engine/src/backend/htp_fastrpc/buffer.rs 의 RpcmemBuffer 에
Buffer trait 을 impl 한다. 레퍼런스는 engine/src/memory/rpcmem/kv_buffer.rs:110
(RpcmemKvBuffer)의 9-메서드 패턴. as_mut_ptr(&self) 시그니처 비대칭은 interior-mut
으로 해소. 검증: alloc→write→readback bit-identical 유닛테스트. firmware 수정 금지.
완료 후 /sanity-check.
```

---

## Landmines / 미해결 / 안 가본 길

- **★ 최대 리스크 (novelty 게이트)**: "1B 에서 QCF 가 op-level placement 판별력(ΔPPL 분리)을 내는가" 미검증. 1B 는 attention score BOS 지배(BOS=3002.7 vs gen avg 32.6, Round 1-15)라 8B 와 다름 → RED 가능성. **Phase 2c(QCF placement) 진입 전 device 측정 게이트로 박을 것.** RED 면 angle 을 정적 op 분산으로 후퇴(novelty 약화) 또는 모델 키우기.
- **GPU∥HTP 동시 실행 실이득 미확인**: synth 가 인용한 "0.510 true parallel" 은 **죽은 raw QNN 경로**(libQnnHtp.so, Q20 에서 `err=0x36b1` FAIL) 측정. **FastRPC 경로로 재현된 적 없음.** Phase 1 직후 단일 microbench(같은 DMA-BUF fd 를 KGSL+CDSP SMMU 동시 매핑 coherent?)로 GREEN/RED 판정 필요. DDR 경합(둘 다 memory-bound)으로 단순 partition 은 OpenCL-only 회귀 가능 → throughput 아니라 **latency hiding/energy/quality 프레임**이 정직.
- **시도했지만 정정된 가설**: workflow synth 가 "FastRPC = stock S25 GREEN, device 리스크 RED→GREEN, 3~4주" 라 했으나, adversarial 이 "q22 리포트상 NPU compute 0회, rc=14 미해결" 로 반박. → **둘 다 부분 오류**: adversarial 은 `dde08248`(NPU compute GREEN) 누락(틀림), synth 는 "handshake GREEN" 을 "compute GREEN" 으로 확대(부정확). **정답 = NPU op dispatch 는 현재 GREEN(실측), 단 engine backend wire-up 은 미완.**
- **NPU 절대속도 < GPU < CPU**: llama.cpp 실측 NPU 32.4 < OpenCL 37.6 < CPU 63.8 tok/s (Qwen2.5-1.5B Q4_0). "빠르다" 주장 금지 — 협력+품질관리 서사.
- **경로 2개 혼동 주의**: production 대상 = `htp_fastrpc` feature(FastRPC IDL, `libcdsprpc.so`, QNN SDK 의존 0). 구 `qnn` feature(QNN OpPackage SDK, deprecated, `htp_graph_reuse.rs:22`)와 혼동 금지. graph-mode 는 FastRPC 경로엔 **부재**(synth 의 "graph-mode 대비책" 은 잘못된 toolchain).
- **paper baseline**: OpenCL-only / CPU-only / llama.cpp `-dev GPUOpenCL,HTP0`(단 layer-split 이지 텐서 동시분할 아님, lcpp.htp 일부 `device_init_failed`) / HeteroLLM(재현 불가, 설계 차별표로).

---

## 참고 링크

- 메모리: `[[htp-production-backend-feasibility]]`, `[[htp-f16-matmul-measurement-artifact]]`
- 설계: `arch/htp_fastrpc.md`, `spec/htp_fastrpc.md` (INV-HTP-FRPC-001~005, β/γ/δ defer)
- 측정: `papers/eurosys2027/_workspace/experiment/microbench_full_matrix_2026_05_28/{report.md,aggregated.csv}`
- tensor partition(Phase 2a 재활용): `arch/tensor_partition.md`, `engine/src/backend/.../plan.rs::PartitionStep`
- 선행 sprint: `handoff_qnn_microbench_phase_e_complete_2026_05_26.md`, `handoff_q22_option_d_continue_2026_05_26.md`
- workflow 결과 원본: `/tmp/claude-1000/.../tasks/{wjhbuod8b,w2mwz2ypi}.output`
