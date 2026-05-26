# HTP FastRPC Backend — PoC Specification

> **TL;DR**: `htp_fastrpc` 는 Qualcomm Hexagon Tensor Processor 를 FastRPC IDL transport 로 직접 구동하는 신규 backend 다. QNN SDK(`libQnnHtp*`) 의존이 0건이며, llama.cpp ggml-hexagon 의 HVX skel 코드를 차용하여 cdsprpc 만으로 NPU 연산을 호출한다. 본 sprint (Q-2.2-α PoC) 의 scope 는 **rms_norm 1개 op** 한정이며, 나머지 60+ method 는 `cpu_companion` 위임 또는 default 유지. 5개 INV (`INV-HTP-FRPC-001 ~ 005`) 가 FastRPC handle 라이프사이클, rpcmem 라이프타임, op req IDL contract, error mapping, weight upload 정합성을 통제한다.
>
> **상태**: PoC scope. rmsnorm 1 op 만 진짜 HTP 호출 → GREEN 시 β (matmul/attention) 진입.

## 1. Purpose and Scope

본 문서는 `htp_fastrpc` backend 의 PoC 단계 invariant 와 의존 contract 를 정의한다.

**이 파일이 명세하는 것**:

- FastRPC handle/session 의 lifecycle 규칙 (open/close 시점, hot-path 진입 금지)
- rpcmem-backed Buffer 의 ownership/lifetime 규칙 (Tensor wrapper 가 owner, 중복 free 금지)
- DSP 측 HVX skel 에 보낼 op request packet 의 schema (PoC: rmsnorm 단일 packet)
- AEEResult → engine::error::Error mapping 규칙
- Hot path 진입 전 weight upload 정합성 (rpcmem region 사전 충전)

**이 파일이 명세하지 않는 것**:

- 실제 HVX intrinsic 의 수치 알고리즘 → `arch/htp_fastrpc.md` + llama.cpp `ggml/src/ggml-hexagon/htp/unary-ops.c::rms_norm`
- Backend trait method 별 sprint 단계 (α/β/γ/δ) → `papers/eurosys2027/_workspace/experiment/qnn_q22_dryrun_fastrpc_2026_05_26/backend_interface_matrix.md`
- 4-step dry-run 시퀀스 (이미 검증 완료) → `papers/.../report.md`
- QNN OpPackage backend (`qnn_oppkg`) 와의 차이 → `arch/htp_fastrpc.md` §3

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **HTP** | Hexagon Tensor Processor — Qualcomm Snapdragon SoC 의 NPU. Hexagon DSP + HVX (Hexagon Vector Extension) 1024-bit SIMD. |
| **FastRPC** | Qualcomm 의 host(ARM) ↔ DSP RPC framework. `libcdsprpc.so` (stock S25 `/vendor/lib64/`) 가 transport 를 제공한다. |
| **HVX skel** | DSP 측에서 실행되는 native 코드. `cargo` 가 아닌 Hexagon SDK 로 별도 빌드되어 `libllmrs_htp_v79.so` 형태로 디바이스에 push 된다. |
| **rpcmem** | FastRPC 가 인식하는 DMA-BUF heap 기반 메모리. `rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, ...)` 으로 할당하며, host/DSP 양쪽에서 zero-copy 로 접근 가능하다. |
| **remote handle** | `remote_handle64_open(uri, ...)` 가 반환하는 DSP-side stub instance 의 host-side 식별자 (u64). |
| **session** | FastRPC `effective_domain_id` (CDSP=7) + `session_id` 의 조합. `FASTRPC_RESERVE_NEW_SESSION` 으로 reserve. |
| **op req packet** | Host 가 DSP 로 보낼 IDL-encoded 구조체. op_id + input/output pointer + shape + epsilon 등 op 별 인자를 담는다. |
| **cpu_companion** | Backend trait 의 method 중 NPU 가 미구현한 것을 위임하는 CPU backend 인스턴스. `Backend::cpu_companion(&self) -> Option<&CpuBackend>` 로 노출된다. |

## 3. Specification

### 3.1 Invariants

| ID | 카테고리 | 검증 | Status |
|----|---------|------|--------|
| INV-HTP-FRPC-001 | Safety | runtime, test | PoC scope |
| INV-HTP-FRPC-002 | Safety | runtime, test | PoC scope |
| INV-HTP-FRPC-003 | Correctness | runtime, test | PoC scope |
| INV-HTP-FRPC-004 | Correctness | runtime, test | PoC scope |
| INV-HTP-FRPC-005 | Correctness | runtime, test | PoC scope |

본 카탈로그 행은 `spec/41-invariants.md` §3.27 에도 동기화한다 (sprint α 진입 시점 add).

---

#### INV-HTP-FRPC-001 — FastRPC handle/session lifecycle

**Statement**: `htp_fastrpc` backend 는 process lifetime 동안 FastRPC handle + dspqueue 를 **정확히 1회 setup** 하고 backend Drop 시점에 1회 teardown 한다. 시퀀스 (Path B, llama.cpp ggml-hexagon 차용): `dlopen libcdsprpc.so` → 17 symbol dlsym → `remote_session_control(FASTRPC_RESERVE_NEW_SESSION)` → `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, domain, enable=1)` → `remote_handle64_open(uri="file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke&_modver=1.0", &handle)` → `dspqueue_create(domain, 0, 128KB req, 64KB resp, NULL cb, NULL err_cb, NULL ctx, &queue)` → `dspqueue_export(queue, &queue_id)` → `htp_iface_start(handle, sess_id, queue_id, n_hvx)` → backend ready. Hot path (`Backend::rms_norm` 등 op 호출) 에서 open/close 반복은 금지된다.

**Why**: `remote_handle64_open` 은 ~수십 ms 의 비용을 가지며 DSP 측 PD (Protected Domain) 생성과 skel binary 매핑이 동반된다. 매 forward 마다 반복하면 token 당 latency 가 sub-ms 단위에서 수십 ms 단위로 폭증한다. 또한 unsigned PD 활성화는 session 단위로 1회만 의미가 있다.

**Verification**: `engine/tests/spec/test_inv_htp_frpc_001.rs` (후속 sprint 에서 추가) — backend instance 의 `new()` ~ `drop()` 사이에 `remote_handle64_open` 호출 카운트 == 1, `remote_handle64_close` == 1 임을 확인. PoC scope 에서는 microbench `htp_fastrpc_rmsnorm_correctness` 의 logcat 에 `remote_handle64_open` line 이 정확히 1회 등장함을 manual 검증.

**Implementation**: `engine/src/backend/htp_fastrpc/host.rs::HtpFastRpcHost::new()` (`dlopen` + `dlsym` 17 symbol + `reserve_session` + `unsigned_pd_enable` + `handle_open`), `HtpFastRpcHost::drop()` (`handle_close`).

**Status**: PoC scope. rmsnorm 1 op 진입 시점에 GREEN.

---

#### INV-HTP-FRPC-002 — rpcmem buffer lifetime + single-owner

**Statement**: HTP backend 가 노출하는 모든 Tensor buffer 의 backing memory 는 `rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size)` 로 할당되며, 정확히 **1개의 Rust wrapper struct** (`RpcmemBuffer`) 가 ownership 을 갖는다. wrapper 의 `Drop` 시 `rpcmem_free(ptr)` 가 정확히 1회 호출된다. 동일 rpcmem fd 또는 동일 ptr 의 이중 free, 또는 wrapper 누락으로 인한 leak 은 금지된다.

**Why**: rpcmem 은 DMA-BUF heap 위의 limited resource (S25 stock 에서 system heap 기본 ~수백 MB) 이며, 누수 시 추론 도중 alloc 실패로 backend 가 죽는다. 이중 free 는 kernel-side fd refcount 와 user-space 매핑 사이의 race 를 만들어 random crash 의 원인이 된다. 또한 weight buffer 가 ArcSwap 등으로 swap 될 때 stale ptr 을 free 하면 활성 forward 가 use-after-free 한다.

**Verification**: `engine/tests/spec/test_inv_htp_frpc_002.rs` — `RpcmemBuffer::new(size).drop()` 100 회 반복 후 `/proc/self/status::VmRSS` slope < 1 KB/iter. 별도로 `RpcmemBuffer` 가 `Clone` derive 를 갖지 않음을 컴파일 타임에 강제 (`assert_not_impl!(RpcmemBuffer: Clone)`).

**Implementation**: `engine/src/backend/htp_fastrpc/buffer.rs::RpcmemBuffer { ptr: *mut u8, fd: i32, size: usize, host: Arc<HtpFastRpcHost> }`, `impl Drop for RpcmemBuffer { fn drop(&mut self) { unsafe { (self.host.rpcmem_free)(self.ptr) } } }`. Tensor wrapper (`Backend::Buffer` associated type) 는 `Arc<RpcmemBuffer>` 를 보유하여 multi-owner read 는 허용하되 단 1개 owner 만 free 한다.

**Status**: PoC scope. rmsnorm 의 input/output Tensor 2개에 한정해 적용. β 단계의 weight upload 시점에 weight tensor 들에도 확장.

---

#### INV-HTP-FRPC-003 — Op request packet contract (dspqueue transport)

**Statement**: Host → DSP 의 op 호출 transport 는 **`dspqueue_write(queue, flags, n_bufs, bufs[], sizeof(req), &req, timeout)`** 다. IDL method call (`remote_handle64_invoke`) 은 사용하지 않는다 — `htp_iface.idl` 은 lifecycle 만 정의 (`start/stop/enable_etm/disable_etm`). Path B 차용 정책상 packet schema 는 llama.cpp `htp-msg.h::struct htp_general_req` 와 byte-identical 이어야 한다 (size ~400+ B 고정, op enum + op_params[16] + flags + src0..src4 + dst). PoC rmsnorm 의 핵심 필드: `op = HTP_OP_RMS_NORM (= 6)`, `op_params[0] = epsilon (float)`, `src0 = input tensor (F32)`, `dst = output tensor (F32)`, `n_bufs = 2` (src0, dst). src1~src4 + 미사용 buffer 는 zero-init.

```rust
// engine/src/backend/htp_fastrpc/idl.rs — llama.cpp htp-msg.h 동기화
#[repr(C)]
pub struct HtpTensor {
    pub data: u32,       // rpcmem buffer offset (DSP 측에서 ptr 로 patch)
    pub type_: u32,      // HTP_TYPE_F32 = 0
    pub ne: [u32; 4],    // shape
    pub nb: [u32; 4],    // stride (bytes)
}

#[repr(C)]
pub struct HtpGeneralReq {
    pub op: u32,             // HTP_OP_RMS_NORM = 6 (llama.cpp htp_op enum)
    pub op_params: [i32; 16], // op_params[0] = epsilon (memcpy float)
    pub flags: u32,
    pub src0: HtpTensor,
    pub src1: HtpTensor,
    pub src2: HtpTensor,
    pub src3: HtpTensor,
    pub src4: HtpTensor,
    pub dst:  HtpTensor,
}
```

**Why**: Host Rust `#[repr(C)]` 와 DSP C struct 의 byte layout 이 1 bit 라도 어긋나면 garbage 가 silently 처리된다 (DSP 측 OOB read/write 가 crash 가 아닌 wrong-answer 로 나타남 — HTP 의 보호 도메인은 PD 단위지 packet 단위가 아니다). Path B 차용 정책상 llama.cpp 의 `libggml-htp-v79.so` skel 이 본 schema 를 expect 하므로 우리 host 는 schema 의 변경 권한이 없다 — llama.cpp upstream 따라간다.

**Verification**: `engine/tests/spec/test_inv_htp_frpc_003.rs` (후속 sprint) — `mem::size_of::<HtpGeneralReq>()` 가 llama.cpp htp-msg.h 의 `sizeof(htp_general_req)` 와 동일함을 const assert. host 측 magic constant (`HTP_OP_RMS_NORM = 6`, `HTP_TYPE_F32 = 0`) 를 llama.cpp header 에서 직접 인용 (`#[allow]` 가 아닌 explicit `const`).

**Implementation**: `engine/src/backend/htp_fastrpc/idl.rs::{HtpGeneralReq, HtpTensor, HTP_OP_RMS_NORM}` (host side, llama.cpp htp-msg.h:118-142 직접 매핑). DSP side 는 llama.cpp `libggml-htp-v79.so` 사전 빌드 그대로 차용 (Path B). transport 함수 = `engine/src/backend/htp_fastrpc/host.rs::dspqueue_write_packet`.

**Status**: PoC scope. rmsnorm 호출 시 `HtpGeneralReq` packet 작성 + `dspqueue_write` + `dspqueue_read` flush 한 cycle. β 단계에서 op = HTP_OP_MUL_MAT (4) 등으로 packet 동일 reuse — schema 자체는 그대로, op 값만 변경.

---

#### INV-HTP-FRPC-004 — Error mapping

**Statement**: FastRPC API 의 모든 return code (`AEEResult` 또는 `int32_t`) 는 다음 mapping 으로 `engine::error::Error` 에 wrap 된다. `AEE_SUCCESS == 0` 이외의 모든 값은 `Error::BackendDeviceError { backend: "htp_fastrpc", code: <raw>, msg: <human-readable> }` 으로 변환된다.

| Raw code | 해석 | Engine error |
|----------|------|--------------|
| `0` (`AEE_SUCCESS`) | 성공 | `Ok(())` |
| `0xa1b2c3d4` 등 magic | SDK 내부 sentinel, 실패 | `BackendDeviceError` |
| `0x80000000+` | AEE_E* 계열 | `BackendDeviceError` |
| 음수 | POSIX errno negation | `BackendDeviceError` |

추가로 logcat 의 `E adsprpc remote.c:NN ...` 라인은 진단 path 로 명시되어 있으며, Q-2.1 dry-run RED 의 root cause 가 이 패턴이었다 (`Control stubbed routine`). PoC 의 error 보고는 raw code 만 wrap 하되 후속 sprint 에서 logcat tag 추출 helper 를 add 가능.

**Why**: AEEResult 는 약 200+ variant 의 enum 이며 일관된 mapping 이 없으면 caller 가 `match` 의 모든 가지를 처리해야 한다. 단일 variant 로 wrap 하되 raw code 를 보존하면 caller 는 fallback 결정 (예: cpu_companion 위임) 을 코드 값 기반으로 내릴 수 있다.

**Verification**: `engine/tests/spec/test_inv_htp_frpc_004.rs` — mock FastRPC binding 에 raw code 를 주입하고 mapped Error variant 가 일치함을 확인. Q-2.1 의 `0x36b1` (deviceCreate err) 같은 known signature 는 single-line const 로 식별 (`pub const KNOWN_QNN_DEVICE_CREATE_FAIL: u32 = 0x36b1`).

**Implementation**: `engine/src/backend/htp_fastrpc/host.rs::map_aee_err(code: i32) -> Error`. Host module 모든 FFI call site 에서 `let rc = unsafe { (host.fn_ptr)(args) }; if rc != 0 { return Err(map_aee_err(rc)); }` 패턴.

**Status**: PoC scope. rmsnorm 호출 시 발생 가능한 8개 known error code 를 known constant 로 등록.

---

#### INV-HTP-FRPC-005 — Weight upload precondition

**Statement**: Hot path (`Backend::rms_norm` 등 forward op 호출) 진입 전에, 해당 op 가 참조하는 모든 weight tensor (PoC: rmsnorm 의 `gamma`) 는 **이미 rpcmem region 에 upload 완료** 되어 있어야 한다. weight upload path 는 `Backend::write_buffer(dst: &Tensor, src: &[u8])` 이며, write 완료의 ACK 는 `Backend::synchronize()` 가 보장한다. swap (LayerSlot generation 변경 등) 으로 weight ptr 이 변경되면 INV-HTP-FRPC-002 의 lifetime 규칙이 우선 적용 — old weight 의 RpcmemBuffer 가 drop 되기 전에 new weight 가 alloc + upload + ptr 갱신 완료.

**Why**: rpcmem 영역의 host 쪽 `memcpy` 는 DSP cache coherence 를 직접 보장하지 않으며, DSP 측 HVX 코드가 weight 를 읽기 전에 `fastrpc_mmap` 의 cache invalidate 가 필요하다. `Backend::synchronize` 가 이 fence 를 호출하며, fence 누락 시 DSP 가 stale weight (예: 직전 model 의 가중치) 를 silently 사용한다. 또한 hot path 도중 weight 가 교체되면 forward 도중 garbage 출력 발생.

**Verification**: `engine/tests/spec/test_inv_htp_frpc_005.rs` (후속 sprint) — load → write_buffer (gamma) → synchronize → rms_norm 호출 의 순서를 강제하는 typestate 또는 runtime assert. PoC scope 에서는 manual 검증 (microbench 가 명시적으로 4 단계 분리).

**Implementation**: `engine/src/backend/htp_fastrpc/mod.rs::HtpFastRpcBackend::write_buffer()` + `::synchronize()`. swap 경로는 본 sprint scope 외이나 INV-121 (per-token snapshot) 정신을 그대로 차용한다.

**Status**: PoC scope. rmsnorm gamma weight 1개에 한정. β 단계에서 layer weight 전체 + LayerSlot snapshot 통합 시 본 INV 가 INV-121 과 cross-reference 된다.

### 3.2 Cross-references

| INV | 교차 참조 | 관계 |
|-----|----------|------|
| INV-HTP-FRPC-001 | INV-151, INV-180 (qnn_oppkg cdylib isolation) | 본 backend 는 cdylib 의존이 0 이므로 INV-151 의 cargo edge 격리 의무는 자동 충족 |
| INV-HTP-FRPC-002 | INV-125 (SecondaryMmap lifetime), INV-143 (borrow buffer mmap lifetime) | 본 INV 는 rpcmem heap 한정. INV-125/143 의 mmap lifetime 패턴을 rpcmem 측에 적용 |
| INV-HTP-FRPC-003 | MSG-* (protocol message schema) | 본 IDL 은 host↔DSP 전용. Manager↔Engine 의 IPC schema (MSG-*) 와 직교 |
| INV-HTP-FRPC-004 | INV-012 (Backend trait 추상화) | error variant 를 single `BackendDeviceError` 로 wrap 하여 Backend trait 의 추상화 경계 보존 |
| INV-HTP-FRPC-005 | INV-121 (per-token snapshot), INV-122 (swap correctness gate) | 본 sprint scope 외 이나 β 단계에서 swap 경로 통합 시 cross |

## 4. Constraints

### 4.1 PoC scope 제한

- 본 sprint (Q-2.2-α PoC) 의 backend impl 은 **rms_norm 1개 op 만** 진짜 HTP 호출. 나머지 60+ `Backend` trait method 는 `cpu_companion` 으로 위임하거나 default 반환 유지. method 별 sprint 단계는 `papers/eurosys2027/_workspace/experiment/qnn_q22_dryrun_fastrpc_2026_05_26/backend_interface_matrix.md` §3-3 참조.
- **Path B (skel 차용)** 채택. llama.cpp `ggml/src/ggml-hexagon/` 전체를 Hexagon SDK Docker 안에서 자체 빌드하여 `libggml-htp-v79.so` (DSP skel) + `libggml-hexagon.so` (host shim, 참조용) 산출물을 확보. 본 프로젝트는 **DSP skel binary 만 차용 (라이선스 MIT, attribution 포함)** 하고 host 측은 자체 Rust binding 작성. Hexagon SDK / qaic IDL 컴파일러 / HVX 자체 빌드는 본 sprint scope 외 (β 진입 시점 재검토).
- `feature = "htp_fastrpc"` cargo feature gate 는 신설되며 default OFF. 다른 backend feature (`opencl`, `cuda`, `qnn`) 와 mutually exclusive 아님 — 공존 가능하며 runtime `--backend` flag 로 선택.
- Target = `aarch64-linux-android` 만. host 빌드는 stub (link 가능, 호출 시 `Error::BackendUnavailable`).

### 4.2 Future extension 영역

- β: matmul/attention/silu_mul 등 P0 hot-path op 추가 → INV-HTP-FRPC-003 packet schema 가 op 별로 확장 (3-003-matmul, 3-003-attn).
- γ: weight swap path 통합 → INV-HTP-FRPC-005 가 INV-121/122 와 본격 cross-reference.
- δ: async transfer (`enqueue_write_async`) + dspqueue → INV-HTP-FRPC-001 의 handle lifetime 이 multi-session 으로 확장 가능.

## 5. Examples (non-normative)

### 예시 1: rmsnorm 1회 호출 시 INV-HTP-FRPC-001~005 활성 라인

```
[startup]                                                        // INV-HTP-FRPC-001
  let host = HtpFastRpcHost::new()?;                             // dlopen + reserve_session + handle_open + dspqueue_create + htp_iface_start (1회)

[model load]                                                     // INV-HTP-FRPC-002, 005
  let gamma_buf = RpcmemBuffer::alloc(host.clone(), gamma_size)?; // rpcmem_alloc
  backend.write_buffer(&gamma_tensor, &gamma_bytes)?;            // host memcpy → rpcmem region (cache invalidate 포함)
  backend.synchronize()?;                                        // dspqueue flush, DSP cache fence

[forward — hot path]                                             // INV-HTP-FRPC-003, 004
  let mut req = HtpGeneralReq::zeroed();
  req.op = HTP_OP_RMS_NORM; // = 6
  unsafe { core::ptr::write(req.op_params.as_mut_ptr() as *mut f32, 1e-5_f32); } // op_params[0] = eps
  req.src0 = htp_tensor_init(&x_tensor); // {data=0, type=F32, ne=[dim,1,1,1], nb=[4,..]}
  req.dst  = htp_tensor_init(&y_tensor);
  let bufs = [dspq_buf(&x_buf, CPU_WR_DSP_RD), dspq_buf(&y_buf, DSP_WR_CPU_RD)];
  let rc = unsafe { (host.dspqueue_write)(host.queue, 0, 2, bufs.as_ptr(),
                                          mem::size_of::<HtpGeneralReq>() as u32,
                                          &req as *const _ as *const u8,
                                          DSPQUEUE_TIMEOUT) };
  if rc != 0 { return Err(map_aee_err(rc)); }
  // flush: dspqueue_read blocking
  let mut rsp = HtpGeneralRsp::zeroed();
  let rc = unsafe { (host.dspqueue_read)(host.queue, &mut flags, MAX_BUFS, &mut n_bufs,
                                         bufs_out.as_mut_ptr(),
                                         mem::size_of::<HtpGeneralRsp>() as u32, &mut rsp_size,
                                         &mut rsp as *mut _ as *mut u8, DSPQUEUE_TIMEOUT) };
  if rc != 0 || rsp.status != HTP_STATUS_OK { return Err(map_aee_err(rc)); }

[shutdown]                                                       // INV-HTP-FRPC-001, 002
  // backend drop:
  //   gamma_buf drop → rpcmem_free
  //   queue drop → dspqueue_close
  //   host drop → htp_iface_stop → handle_close → dlclose
```

### 예시 2: schema lock + llama.cpp upstream 추적 (INV-HTP-FRPC-003)

```rust
// engine/src/backend/htp_fastrpc/idl.rs — llama.cpp htp-msg.h:118-142 동기화
const _: () = assert!(std::mem::size_of::<HtpTensor>() == 4 + 4 + 4*4 + 4*4); // 40 B
const _: () = assert!(std::mem::size_of::<HtpGeneralReq>() % 8 == 0);

// llama.cpp htp_op enum 직접 인용 — schema 변경 권한 없음
pub const HTP_OP_MUL:         u32 = 0;
pub const HTP_OP_ADD:         u32 = 1;
pub const HTP_OP_MUL_MAT:     u32 = 4;
pub const HTP_OP_RMS_NORM:    u32 = 6;  // ← PoC 대상
pub const HTP_TYPE_F32:       u32 = 0;
pub const HTP_TYPE_F16:       u32 = 1;
```

llama.cpp `htp-msg.h` 변경 시 본 Rust 측 enum 동기화 의무. CI 게이트로 `xxhash` 비교 등 추가 가능 (γ 단계).
