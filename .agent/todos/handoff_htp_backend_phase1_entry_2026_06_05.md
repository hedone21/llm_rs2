# Handoff: HTP backend Phase 1 S1–S3 완료 (`--backend htp` 부팅 wire-up) → S4 (NPU dispatch, device) 진입

**작성**: 2026-06-05 (갱신: S1–S3 완료 반영)
**HEAD**: `427f85eb feat(htp): S2+S3 — HtpFastrpcMemory + init.rs htp arm (--backend htp 부팅)`
**브랜치 / Worktree**: `master`
**작성자**: 메인 세션 (오케스트레이터)

**다음 세션 진입 문장**: "HTP backend S4 진입 — matmul HTP dispatch wire-up (S25 deploy-test 게이트)"

---

## TL;DR

HTP FastRPC backend Phase 1 의 **host-완결 가능 구간(S1–S3) 완료**. `--backend htp` 가 이제 engine match 에서 **구성·부팅**된다(전 op cpu_companion passthrough MVP). 다음 = **S4: 이미 microbench 에서 GREEN 인 NPU matmul dispatch 를 `Backend::matmul` override 에 wire-up** + **S5: end-to-end 1 token 정합성**. **멈춘 이유 = S4/S5 는 S25 android 디바이스 측정 필수** (host 는 `HtpFastrpcHost::new` 가 Err — 설계대로). host 로 검증 가능한 컴파일·구성 게이트는 전부 통과.

**핵심 결론 한 줄**: 버퍼·메모리·init arm = host 컴파일 GREEN + 적대 검토 PASS, paper angle = **quality-aware heterogeneous placement**(QCF, HeteroLLM 이 회피한 빈자리).

---

## 진행 상태 — 이번 세션 완료 (S1–S3)

| 단계 | 작업 | 상태 | Commit |
|---|---|---|---|
| 정찰 | Buffer/Memory trait·RpcmemBuffer·init.rs·feature gating ground-truth (workflow 6-reader) | ✅ | — |
| **S1** | `impl Buffer for RpcmemBuffer` + `dtype` 필드 + alloc 시그니처 확장 (microbench 37 호출처 갱신) | ✅ | `c391e40c` |
| **S2** | `HtpFastrpcMemory` (`impl Memory` → `RpcmemBuffer::alloc`, host Arc 공유) 신규 `htp_fastrpc/memory.rs` | ✅ | `427f85eb` |
| **S3** | `init.rs` cfg-gated `"htp"` arm (cpu 골격, secondary 없음) + bail/cli help 갱신 | ✅ | `427f85eb` |

**검증 게이트 (전부 통과)**: `cargo check -p llm_rs2 --features htp_fastrpc` + 기본 feature(회귀 0) + lib clippy(`-D warnings`) + fmt clean, NDK `aarch64-linux-android` cross-check 회귀 0, 정적 단언 `assert_buf::<RpcmemBuffer>()` 로 trait 완전성 강제. S1·S2+S3 각각 적대 검토(soundness + build/regression) **2/2 PASS**.

**설계 결정 (확정)**: cl_mem=None(OpenCL alias 미보유) · is_gpu_buffer=false(NPU, OpenCL mis-routing 회피, trait default) · as_mut_ptr(&self)=raw ptr 직접 반환(interior-mut 불요) · sync_device=no-op(DSP coherency 는 dispatch flag=S4 레벨) · used_memory=increment-only(OpenCLMemory 와 동일 패턴, MVP 근사) · htp memory=rpcmem 전면 할당(사용자 결정, OOM 은 device gate).

---

## 다음 작업 — S4 / S5 (목적: NPU 가속·정합성, **S25 device 필수**)

1. **S4 `matmul`/`matmul_transposed` override** (`engine/src/backend/htp_fastrpc.rs:231-237`, 현재 cpu_companion 위임). weight/x/y buffer 를 `Tensor::buffer().as_any().downcast::<RpcmemBuffer>()` 로 확인 → RpcmemBuffer backing 이면 `dsp_buf()` descriptor + `dspqueue_write/read` 로 real HTP dispatch, else cpu_companion fallback. 골격 = `rms_norm_via_htp`(htp_fastrpc.rs:132, `#[cfg(target_os="android")]` + `#[allow(dead_code)]`). → **검증(S25)**: 단일 matmul vs CPU `max_abs_err < 1e-3`, `prof_usecs` > 0(DSP 실행 증거).
2. **S5 end-to-end 1 token** (`--backend htp --model-path <qwen.gguf>`). → **검증(S25)**: greedy(temp=0) **CPU/OpenCL/HTP token-id 일치** + 첫 토큰 logit `max_abs_err < 1e-2`.
3. **(S4 직후) rpcmem heap OOM 측정**: Qwen ~900MB weight 를 rpcmem system heap(수백MB)에 전면 할당 시 alloc 실패 여부. RED 면 → 큰 weight 는 host-heap, dispatch 텐서만 rpcmem 하는 hybrid 로 후퇴.

**위임 대상**: `senior-implementer` (htp_fastrpc + dspqueue), **권한**: `engine/src/backend/htp_fastrpc.rs`. S4 코드는 host 컴파일 가능하나 correctness 는 device-only → 작성 후 `/deploy-test` (device bin = `legacy_generate`).

---

## Landmines / 미해결

- **★ S4/S5 는 device-only**: `HtpFastrpcHost::new` 가 non-android Err(host.rs:436), FFI 로딩 `new_internal` android-cfg. host 컴파일/구성까지만 가능. `--backend htp` 런타임·bit-identical·토큰정합 전부 S25 deploy-test 필요.
- **★ rpcmem 900MB OOM (미평가)**: S2 가 weight 전면 rpcmem 할당(사용자 결정). system heap 수백MB 초과 위험. S4 직후 측정 게이트 필수.
- **★ novelty 게이트 (Phase 2c)**: "1B 에서 QCF op-level placement 판별력(ΔPPL 분리)" 미검증. 1B BOS 지배(BOS=3002.7 vs gen 32.6)라 RED 가능. Phase 2c 진입 전 device 게이트로 박을 것.
- **used_memory increment-only**: free 시 미감소(단조 증가). 단 production 소비자 0건(grep: galloc 테스트만) → 현 영향 무. 정확 회계 필요 시 RpcmemBuffer Drop back-ref 후속.
- **PRE-EXISTING clippy 차단**: `engine/microbench/htp_rope.rs:190,202,217` (rope_ref: too_many_arguments/needless_range_loop/manual_memcpy) 가 `cargo clippy --bins` 를 막음. git stash 로 본 작업 무관 확정. lib clippy 는 clean. 별도 `chore:` 로 `#[allow]` 부착 또는 정리 가능.
- **android readback 테스트 미실행**: `test_buffer_trait_readback_bit_identical` (buffer.rs, android-gated) 는 S25 deploy-test 에서만 RUN.
- **GPU∥HTP 동시 실행 실이득 미확인**: "0.510 true parallel" 은 죽은 raw QNN 경로 측정. FastRPC 경로 재현 필요(Phase 2a). DDR 경합으로 단순 partition 은 OpenCL-only 회귀 가능 → latency hiding/energy/quality 프레임이 정직.
- **NPU 절대속도 < GPU < CPU** (32.4 < 37.6 < 63.8 tok/s, Qwen Q4_0). "빠르다" 주장 금지 — 협력+품질 서사.
- **미push**: 본 세션 commit (`c391e40c`, `427f85eb`) + 이전 handoff commit (`0c3fc98a`, `cb19bed0`) 로컬만. 사용자 확인 후 push.

---

## 참고 링크

- 메모리: `[[htp-production-backend-feasibility]]`, `[[htp-f16-matmul-measurement-artifact]]`
- 설계: `arch/htp_fastrpc.md`, `spec/htp_fastrpc.md` (INV-HTP-FRPC-001~005)
- 코드: S1 `engine/src/backend/htp_fastrpc/buffer.rs` (impl Buffer) · S2 `.../htp_fastrpc/memory.rs` · S3 `engine/src/session/init.rs:251` (htp arm)
- S4 골격: `engine/src/backend/htp_fastrpc.rs:132 rms_norm_via_htp`, `:231-237 matmul`
- tensor partition(Phase 2a 재활용): `arch/tensor_partition.md`
