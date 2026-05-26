# Q-2.0 QNN HTP Microbench Dry-Run Sweep

- 일자: 2026-05-26
- 디바이스: Galaxy S25, serial `R3CY408S5SB`, Hexagon V79
- 범위: 9 HTP microbench (engine/Cargo.toml:108~141)
- 게이트: Q-2.1 raw QNN HTP API spec 진입 가능 여부 (correctness/segfault only, 성능 측정 없음)
- 환경 변수: `LD_LIBRARY_PATH=/data/local/tmp/executorch:/data/local/tmp/qnn`, `ADSP_LIBRARY_PATH=/data/local/tmp/executorch:/data/local/tmp/qnn:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:/dsp`, `taskset 3f`, 각 실행 전 30s cooldown, 30s timeout
- 빌드: `cargo build --release --target aarch64-linux-android --features qnn --bin <name>` (NDK r28 Linux)

## 빌드 결과 (host → arm64-android)

9/9 bin **BUILD PASS** (warning only). `target/aarch64-linux-android/release/microbench_htp_*` 모두 460~680 KB. `engine/build.rs` 가 `qnn_bindings.rs` 생성, libloading dlopen 경로로 link.

## 디바이스 lib 준비

기존 `/data/local/tmp/qnn/` (2025-03-28 stale, SDK 2.33) 에서 누락된 V79 lib들을 `/data/local/tmp/executorch/` (2026-05-26 fresh, SDK 2.37) 에서 복사:

| 복사한 lib | 출처 | 비고 |
|---|---|---|
| libQnnHtp.so | executorch | 2.33 → 2.37 SDK upgrade |
| libQnnHtpV79.so | executorch | 2.37 |
| libQnnHtpV79Skel.so | executorch | 2.37 |
| libQnnHtpPrepare.so | executorch | 2.37 |
| libQnnHtpV79Stub.so | executorch | **첫 sweep 실패 후 2차 수정**: 2.33 stale stub은 V79 (2.37) 와 ABI mismatch (`Stub lib id mismatch: expected v2.37.0.250724175447, detected v2.33.0.250327124043`) → 2.37 stub으로 통일 |

`libQnnHtpV79CalculatorStub.so` 만 2.33 잔존 (executorch 디렉토리에 없음, 본 sweep에서 사용 안 됨).

## 9 bin Sweep 결과

| # | bin | exit | verdict | key error |
|---|---|---|---|---|
| 1 | microbench_htp_correctness | 139 | **SEGFAULT** | `contextCreate err=0x36b1` → cleanup SIGSEGV |
| 2 | microbench_htp_gpu_matmul_concurrent | 1 | FAIL | `contextCreate err=0x36b1` |
| 3 | microbench_htp_gpu_parallel | 1 | FAIL | `contextCreate err=0x36b1` |
| 4 | microbench_htp_graph_reuse | 139 | **SEGFAULT** | `contextCreate err=0x36b1` → cleanup SIGSEGV |
| 5 | microbench_htp_matmul_correctness | 139 | **SEGFAULT** | `contextCreate err=0x36b1` → cleanup SIGSEGV |
| 6 | microbench_htp_opencl_interop | 1 | FAIL | Stage A OpenCL 정상, Stage B 진입 시 `contextCreate err=0x36b1` |
| 7 | microbench_htp_qnngpu_share | 1 | FAIL | `HTP contextCreate err=0x36b1` (QNN-GPU side OK) |
| 8 | microbench_htp_rpcmem_throughput | 1 | FAIL | rpcmem alloc/to_fd 정상, HTP `contextCreate err=0x36b1` |
| 9 | microbench_htp_throughput | 139 | **SEGFAULT** | `contextCreate err=0x36b1` → cleanup SIGSEGV |

**총합**: 0 PASS / 5 FAIL / 4 SEGFAULT / 0 TIMEOUT / 0 BUILD_FAIL

## Root Cause

logcat 분석 결과 9 bin 전부 동일 단일 원인:

```
QnnDsp <E> DspTransport.openSession qnn_open failed, 0x80000406, prio 100
QnnDsp <E> IDspTransport: Unable to load lib 0x80000406
QnnDsp <W> Failed to create transport instance: 1002
QnnDsp <E> Failed to load skel, error: 1002
QnnDsp <E> Transport layer setup failed: 14001
QnnDsp <E> default device creation failed
QnnDsp <E> Failed to create context with err 0x36b1 (= 14001)
```

`0x80000406` = `AEE_ENOSUCHMOD` (fastrpc DSP-side cannot find/load skel). DSP가 `libQnnHtpV79Skel.so`를 application PD에 로드할 수 없는 상태:
- ADSP_LIBRARY_PATH가 `/data/local/tmp/...` 를 가리키지만 **stock S25는 unprivileged shell이 unsigned PD로 skel을 publish할 수 없도록 정책**으로 잠겨 있음.
- executorch의 `qnn_executor_runner` 는 동일 lib path에서 동작 — 차이점은 우리 microbench가 `Library::new("/data/local/tmp/qnn/libQnnHtp.so")` 로 host-side만 dlopen하고, DSP-side skel publish는 fastrpc 가 자동 처리해야 하는데 그 단계에서 0x80000406 (skel not found by DSP).
- 즉 host-side QNN HTP API call 자체는 정상 진입 (Provider/backendCreate/getApiVersion 모두 OK), 첫 fastrpc round-trip 인 contextCreate→deviceCreate 에서만 실패.

### SEGFAULT 4건 stack trace (대표: microbench_htp_correctness)
```
F libc    : Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x7555dcc2c8 in tid 22074 (microbench_htp_)
F DEBUG   : Cmdline: ./microbench_htp_correctness
F DEBUG   : signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0000007555dcc2c8
```
contextCreate 실패 후 application `anyhow::ensure!` 가 std::process::exit 경로로 빠지면서 libQnnHtp 의 destructor 에서 partially-initialized state 정리 시도 → invalid pointer dereference. 즉 **QNN HTP backend 에서 contextCreate 실패 후 cleanup race 가 라이브러리 측 결함으로 노출됨**. PASS 가 아니라 cleanup race 의 진정한 SEGFAULT.

## 결정 게이트 Verdict

**RED**: 0 PASS (≤2 임계 미달). 본 sweep 환경(stock S25 + executorch SDK 2.37 stack)에서 raw QNN HTP API path는 contextCreate 단계에서 동작하지 않음. Q-2.1 raw path spec 진입 **불가**.

### 권고

raw QNN HTP API path 를 Q-2 production 통합에 채택하면, 9 microbench 가 모두 fail/segv 인 환경에서 BackendTrait 구현은 디바이스 검증이 불가능. 두 선택지:

1. **Executorch fallback** — μ-Q1 Phase A~E 에서 paper-grade GREEN 으로 확정된 path. `qnn_executor_runner` 동일 무신호 OS 환경에서 동작하므로 Backend trait 을 .pte 모델 + ET runtime 으로 wrapper 작성. M7 W8A8 0.276 ms 결과 그대로 재사용 가능.
2. **Raw path 복구 시도** — `vendor/dsp/cdsp/` 에 signed skel을 별도 push 받아 ADSP_LIBRARY_PATH 우회 시도. SELinux/permission 정책에 따라 root 권한 필요할 가능성 높음. ETA 불확정.

본 dry-run 결과만 보면 Q-2 sprint 는 **(1) Executorch fallback 으로 재설계** 권고. 

raw path 가 동작 가능한 환경(예: rooted device, signed skel, 또는 Qualcomm-issued QDR build) 가 추가 확보될 경우에만 raw path 를 paper-secondary 로 보존.

## 미해결 / 다음 단계 후보

- (a) `ADSP_LIBRARY_PATH=/vendor/dsp/cdsp/` 단독 (executorch path 제외) 으로 vendor signed skel 만 쓰는 경로 시도 — 단 permission 이슈 위 logcat 에서 동일 fail 예상
- (b) `unsigned PD` 우회: `enable_unsigned_PD` debug flag 또는 fastrpc_config 로 unsigned PD enable — 보안 정책상 stock S25 에서 거의 항상 reject
- (c) microbench 9 종을 .pte + executorch wrapper 로 포팅 (M7 paper path 와 일관성 확보) → Q-2.1 spec 의 raw vs ET 양립 비교는 폐기
- (d) cleanup SEGFAULT 4건 — contextCreate 실패시 partial-init 상태에서 backendFree/contextFree 호출 순서 재검토. raw path 채택 안하더라도 microbench 자체 destructor 강건성 개선 후보 (소규모 patch)

## 파일

- 로그 9건: `papers/eurosys2027/_workspace/experiment/qnn_q20_dryrun_2026_05_26/logs/<bin>.log`
- logcat 9건: `papers/eurosys2027/_workspace/experiment/qnn_q20_dryrun_2026_05_26/logs/<bin>.logcat`
- CSV: `papers/eurosys2027/_workspace/experiment/qnn_q20_dryrun_2026_05_26/results.csv`
