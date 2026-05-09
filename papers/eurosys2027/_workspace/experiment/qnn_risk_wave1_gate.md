# Phase R Wave 1: Fail-fast Gate (R-A1 + R-F1 + R-F2)

**작성일**: 2026-05-09
**Plan**: `.agent/todos/plan_qnn_gpu_risk_assessment_2026-05-09.md`
**기준 commit**: `c469387` (plan)

---

## 0. 결론

**Wave 1 = GREEN** → Wave 2 (Core feasibility) 진입.

| Risk | 결과 | 근거 |
|------|------|------|
| R-A1 kernel ↔ QNN op 매칭 | **GREEN** | 17 핵심 op / 14개 prebuilt + 3개 composition (RoPE/attention/SiLU). 100% 표현 가능 |
| R-F1 SDK license | **GREEN** | device push 모델 (lib only). app embed 없음 |
| R-F2 runtime 호환 | **GREEN** | SDK 2.33 lib을 `/data/local/tmp/qnn/` push. R-Y opt 3 PoC 검증 완료 |

---

## 1. R-A1: Kernel 매핑

상세: `qnn_kernel_mapping.md`

- Production OpenCL 21개 .cl 파일 + simple_ops.cl 안 29개 fused op = ~50개 kernel 변형
- 핵심 op 17개 (matmul/rmsnorm/softmax/silu/rope/attention/cast/concat/gather/dequantize/elementwise) 모두 표현 가능
- prebuilt 부재 op (RoPE, attention, SiLU) → graph composition으로 해결, ORT QNN EP PR #23136 패턴 reference
- 매칭률 ≥9개 기준 충족 (14개 prebuilt 매칭, 6~8 / <6 threshold 모두 통과)

후속 risk (Wave 2~3 이관): R-A2, R-A3, R-A4, R-B2

---

## 2. R-F1: SDK License

### 검증 방법
- `third_party/qnn_sdk_2.33/` 구조 확인 (lib + include만 추출)
- Qualcomm AI Engine Direct SDK의 일반적 사용 모델 확인

### 사용 모델 (production target)
- **Embed 안 함**: SDK lib(.so)을 device `/data/local/tmp/qnn/` 에만 push
- **재배포 안 함**: app 패키지 또는 git repo에 SDK source/binary 포함하지 않음 (`third_party/`는 user-local only)
- **End-user**: 디바이스 소유자가 SDK 별도 설치 (또는 본 프로젝트 setup script로 push)

### Production 영향
- Cargo.toml의 `qnn = ["libloading", "bindgen"]` feature는 빌드 시 SDK 경로만 참조 (런타임 dlopen)
- bindgen은 build.rs에서 header 파싱하여 Rust binding 생성, header 자체는 binary에 포함 안 됨
- 빌드 산출물 (.so)에 QNN 코드 안 들어감 → 재배포 자유 (binding wrapper만)
- 디바이스 lib는 별도 설치/push 단계 필요 (production deployment 시 설치 매뉴얼 필요)

### 결정
**GREEN** — 본 프로젝트 사용 모델은 라이선스 친화적. 단:
- production deployment 시 user-side SDK 설치 가이드 필요 (R-D 마이그레이션 plan 항목으로 이관)
- llama.cpp QNN backend WIP PR (#12063)도 동일 사용 모델

---

## 3. R-F2: Runtime 버전 호환

### 사실
- Galaxy S25 vendor lib (`/vendor/lib64/libQnnHtp.so`): API 2.20.0 (이전 측정)
- 우리 SDK: 2.33 (API 2.25.0)
- vendor api와 SDK api 버전 차이 → vendor lib만 사용 시 SDK 신규 op/feature 사용 불가

### PoC 검증 (이미 완료)
- R-Y opt 3 (`microbench_htp_qnngpu_share.rs`): SDK 2.33 lib을 `/data/local/tmp/qnn/`에 push 후 dlopen
- 5개 핵심 lib (`libQnnHtp.so`, `libQnnHtpV79Stub.so`, `libQnnHtpPrepare.so`, `libQnnSystem.so`, `libQnnHtpV79Skel.so`) + `libQnnGpu.so` push로 동작
- HTP + QNN-GPU shared rpcmem zero-copy 1024/1024 정확성 ✓

### Production 시나리오
- App 또는 deployment 단계에서 SDK lib을 device `/data/local/tmp/qnn/` 또는 internal storage push
- App startup 시 LD_LIBRARY_PATH 또는 dlopen 절대 경로로 SDK 2.33 lib 사용
- **vendor lib을 사용하지 않음** → vendor api 2.20 영향 받지 않음

### 잠재 risk
- Android sandbox가 `/data/local/tmp/qnn/` 경로 접근 차단할 가능성 (rooted device 또는 app private dir만 가능할 수 있음)
- **PoC는 rooted/dev 환경에서 검증됨** (S25 ADB push), production app sandbox에서는 internal app dir로 push 후 LD_LIBRARY_PATH 조정 필요
- 이 부분은 마이그레이션 plan 진입 후 deployment 항목으로 처리 (R-D)

### 결정
**GREEN** (조건부) — PoC 환경에서 SDK 2.33 사용 가능. production app 패키징은 마이그레이션 plan에서 별도 검토 필요.

---

## 4. Wave 2 진입 조건 점검

| 조건 | 상태 |
|------|------|
| Wave 1 모든 risk PASS | ✓ |
| Production code 변경 0 lines | ✓ (분석만 수행) |
| 신규 microbench 0개 | ✓ (Wave 2에서 R-B1, R-G1용 추가 예정) |

---

## 5. Wave 2 작업 항목 (다음 단계)

| Risk | 신규 microbench | 측정 대상 |
|------|----------------|----------|
| R-A2 | 없음 (분석) | RoPE/SiLU/attention composition 비용 분석 |
| R-B1 | `microbench_qnngpu_matmul_tbt.rs` | QNN-GPU MatMul vs OpenCL `mul_mv_f16_f32` TBT |
| R-G1 | `microbench_qnngpu_correctness.rs` | FFN layer (RmsNorm + matmul + SwiGLU + matmul) 정확성 |
| R-C1/2/3 | 없음 (read-only) | KV cache dynamic shape, weight tensor mapping, workspace |

총 2개 신규 microbench bin (engine/src/bin/), 2~3일.

---

**End of Wave 1 Report**
