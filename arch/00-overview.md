# System Overview -- Architecture

> spec/00-overview.md의 구현 상세.

## 코드 매핑

### 3.1 System Context [SYS-001 ~ SYS-009]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-001 | `Cargo.toml` (workspace members) | `members = ["engine", "shared", "manager"]` — 3 크레이트 분리 | Engine과 Manager는 별도 바이너리 |
| SYS-005 | `engine/Cargo.toml`, `manager/Cargo.toml` | 양측 `llm_shared` 의존 | 직접 상호 의존 없음 |
| SYS-006 | `engine/src/bin/generate.rs` | Prefill: `model.forward()`, Decode: `model.forward_into()` | |
| SYS-007 | `shared/src/lib.rs` | `ManagerMessage`, `EngineMessage` enum 정의 | serde derive |
| INV-001 | `Cargo.toml` | Engine ⊥ Manager (workspace members, 직접 의존 없음) | |

### 3.2 System Goals [SYS-010 ~ SYS-019]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-011 | `manager/src/pipeline.rs` | `HierarchicalPolicy` — PI + Supervisory + Selector | |
| SYS-012 | `engine/src/core/qcf/` | `QcfMetric`, `QcfConfig`, `DegradationEstimator` | QcfEstimate는 shared에 미정의 (미구현) |
| SYS-013 | `engine/src/backend/` | `cpu/`, `opencl/` 모듈 | Backend trait |
| SYS-019 | `manager/src/relief/` | `ReliefEstimator` (온라인 선형 회귀) | |

### 3.3 Target Platform [SYS-020 ~ SYS-029]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-020 | `.cargo/config.toml` | ARM64 타겟 설정 (NEON+dotprod) | |
| SYS-022 | `engine/src/backend/cpu/neon.rs` | NEON SIMD: `vdotq_s32` 등 | ARM64 전용 |
| SYS-023 | `engine/src/backend/cpu/x86.rs` | x86 AVX2+FMA 경로 | |
| SYS-025 | `engine/kernels/*.cl` | OpenCL 커널 ~80개 | `opencl` feature gate |
| SYS-026 | `engine/src/backend/opencl/mod.rs` | `CL_MEM_ALLOC_HOST_PTR` | UMA SoC zero-copy |
| INV-002 | `engine/src/backend/cpu/neon.rs` | `#[cfg(target_arch = "aarch64")]` 게이트 | |

### 3.4 Supported Models [SYS-030 ~ SYS-039]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-031 | `engine/src/models/llama/llama_model.rs` | Safetensors 로딩 | `config.json`, `tokenizer.json` |
| SYS-034 | `engine/src/core/` | `BlockQ4_0` 구조체 (18 bytes/block) | GGML 호환 |

### 3.5 Quality Assurance [SYS-040 ~ SYS-049]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-040 | `engine/src/core/qcf/` | `QcfMetric` 구조체 | |
| SYS-043 | `engine/src/core/qcf/` | `DegradationEstimator` — 피스와이즈 선형 + EMA | |
| INV-004 | `engine/src/core/qcf/` | lossy action 실행 시 QcfMetric 생성 | |

### 3.6 Fail-Safety [SYS-050 ~ SYS-059]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-050 | `engine/src/resilience/executor.rs` | Manager 미연결 시 `poll()` no-op | fail-open |
| SYS-051 | `manager/src/main.rs` | Engine 미연결 시 `emit()` skip | |
| SYS-055 | `engine/src/resilience/` | Strategy별 Emergency 대응 | D-Bus 경로 |
| INV-005 | `engine/src/resilience/executor.rs` | Resilience 실패 시 추론 지속 | |

### 3.7 Implementation [SYS-060 ~ SYS-069]

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|
| SYS-061 | `Cargo.toml` | `members = ["engine", "shared", "manager"]` | |
| SYS-062 | `engine/Cargo.toml:45-48` | `default = ["opencl"]`, `resilience = ["zbus"]` | |
| SYS-062 | `manager/Cargo.toml:18-20` | `default = ["dbus"]`, `dbus = ["zbus"]` | |
| SYS-063 | `Cargo.toml:5-9` | `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`, `panic = "abort"` | |
| SYS-064 | 전체 코드 | `std::thread` + `mpsc::channel` only | async 런타임 미사용 |
| SYS-065 | `shared/src/lib.rs` | `serde_json` derive | JSON 전용 |

## 크레이트 구조

| 크레이트 | 패키지명 | Cargo.toml 위치 | 바이너리 |
|---------|---------|----------------|---------|
| engine | `llm_rs2` | `engine/Cargo.toml` | `generate`, `test_backend` |
| shared | `llm_shared` | `shared/Cargo.toml` | (라이브러리) |
| manager | `llm_manager` | `manager/Cargo.toml` | `llm_manager`, `mock_engine`, `mock_manager` |

## Feature Gate

| Feature | 크레이트 | 기본 | 조건 컴파일 | 외부 의존성 |
|---------|---------|------|-----------|-----------|
| `opencl` | engine | 활성 | `#[cfg(feature = "opencl")]` | `ocl` crate |
| `resilience` | engine | 비활성 | `#[cfg(feature = "resilience")]` | `zbus` crate |
| `dbus` | manager | 활성 | `#[cfg(feature = "dbus")]` | `zbus` crate |

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
| (해당 없음 — 이 문서는 시스템 수준 개요이며 config는 개별 컴포넌트에서 정의) | | | |

## CLI

### Engine (`generate`)

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| `--model` | 모델 디렉토리 경로 | SYS-031 |
| `--enable-resilience` | Resilience 매니저 활성화 | SYS-050 |
| `--resilience-transport` | Resilience 전송: `dbus`, `unix:<path>`, `tcp:<host:port>` | SYS-084 |

### Manager (`llm_manager`)

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
| `-c, --config` | TOML 설정 파일 경로 (기본: `/etc/llm-manager/config.toml`) | SYS-024 |
| `-t, --transport` | 전송: `dbus`, `unix:<path>`, `tcp:<host:port>` (기본: `dbus`) | SYS-089 |
| `--client-timeout` | Unix socket 클라이언트 대기 타임아웃(초) (기본: 60) | SEQ-020 |
| `--policy-config` | 정책 설정 TOML 경로 | SYS-088 |

## 빌드 프로파일

| 설정 | 값 | Cargo.toml 위치 |
|------|---|----------------|
| LTO | `fat` | `Cargo.toml:6` |
| codegen-units | `1` | `Cargo.toml:7` |
| panic | `abort` | `Cargo.toml:8` |
| opt-level | `3` | `Cargo.toml:9` |