# 25. 트러블슈팅 가이드

> llm.rs 개발 및 실행 중 발생할 수 있는 주요 문제와 해결 방법을 정리한 가이드입니다.

[이전: 24. Resilience 시스템 사용 가이드](24_resilience_usage_guide.md) | [다음: 26. API 레퍼런스](26_api_reference.md)

---

## Quick Reference Index

에러 메시지의 일부분으로 해당 섹션을 빠르게 찾을 수 있습니다.

| 에러 메시지 (substring) | 섹션 |
|--------------------------|------|
| `could not find` / `NDK` / `ndk` | [1.1](#11-android-ndk-not-found) |
| `unresolved import` / `zbus` | [1.2](#12-feature-gate-mismatches) |
| `unrecognized target feature` / `neon` / `avx2` | [1.3](#13-arm64x86-target-feature-오류) |
| `linker` / `cc not found` / `ld: error` | [1.4](#14-cross-compilation-linker-에러) |
| `No OpenCL devices found` | [2.1](#21-opencl-device-not-found) |
| `Failed to compile` / `kernel` / `Falling back to dummy` | [2.2](#22-opencl-kernel-compilation-failure) |
| `out of memory` / `alloc` / `OOM` | [2.3](#23-memory-exhaustion-oom) |
| `MemTotal not found` / `MemAvailable not found` / `/proc/meminfo` | [2.4](#24-procmeminfo-read-failure) |
| `Connection refused` / `D-Bus listener exited` | [3.1](#31-d-bus-connection-refused) |
| `Failed to create proxy` / `org.llm.Manager1` | [3.2](#32-resilience-manager-not-registered) |
| `Invalid level` / `from_dbus_str` | [3.3](#33-invalid-d-bus-signal-format) |
| `Buffer does not support CPU pointer` / `null buffer` | [3.4](#34-gpu-buffer-eviction-failure) |
| `Unsupported dtype in safetensors` / `Error finding tensor` | [4.1](#41-model-loading-실패) |
| 토큰 생성 속도 저하 (느린 추론) | [4.2](#42-slow-inference-performance) |

---

## 1. Build Errors

### 1.1 Android NDK not found

**증상**

Android 타겟으로 크로스 컴파일 시 linker 또는 `ar` 바이너리를 찾을 수 없다는 에러가 발생합니다.

```
error: linker `/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang` not found
```

**원인**

`.cargo/config.toml`에 NDK 경로가 `/opt/android-ndk/`로 하드코딩되어 있습니다:

```toml
# .cargo/config.toml
[target.aarch64-linux-android]
linker = "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang"
ar = "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
```

반면 `android.source`는 다른 경로(`/home/go/.opt/android-ndk-r27d`)를 설정하며, API 레벨도 다릅니다 (config.toml은 `android35`, android.source는 `android21`).

**해결**

1. `source android.source`를 **반드시** 먼저 실행하여 환경 변수를 설정합니다:

```bash
source android.source
cargo build --target aarch64-linux-android --release --bin generate
```

2. `android.source`가 설정하는 `CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER` 환경 변수가 `.cargo/config.toml`의 `linker` 설정을 **오버라이드**합니다. 환경 변수가 cargo config보다 우선순위가 높으므로 `source android.source` 후에는 config.toml의 하드코딩 경로가 무시됩니다.

3. NDK가 시스템에 설치되어 있지 않다면, `android.source`의 `NDK_HOME` 경로를 자신의 NDK 설치 경로로 수정하세요.

---

### 1.2 Feature gate mismatches

**증상**

`resilience` 관련 코드를 사용하는데 `zbus` crate를 찾을 수 없다는 컴파일 에러가 발생합니다.

```
error[E0432]: unresolved import `zbus`
```

또는 `--enable-resilience` 플래그를 사용했는데 아무런 효과가 없습니다.

**원인**

`Cargo.toml`의 feature 정의를 보면:

```toml
[features]
default = ["opencl"]
opencl = ["ocl"]
resilience = ["zbus"]
```

`opencl`은 default feature이므로 별도 지정 없이 활성화되지만, `resilience`는 default가 **아닙니다**. `resilience` feature 없이 빌드하면 `zbus` 의존성이 포함되지 않고, D-Bus 관련 코드가 `#[cfg(feature = "resilience")]` 게이트로 제거됩니다.

**해결**

Resilience 기능이 필요한 경우 `--features resilience`를 명시적으로 추가합니다:

```bash
# resilience 포함 빌드
cargo build --release --bin generate --features resilience

# mock_manager는 resilience feature가 필수
cargo build --release --bin mock_manager --features resilience
```

호스트에서 GPU 없이 개발할 때 `opencl` feature를 끄고 싶다면:

```bash
cargo check --no-default-features --features resilience
```

---

### 1.3 ARM64/x86 target feature 오류

**증상**

컴파일 시 인식할 수 없는 target feature 에러가 발생하거나, 런타임에 SIGILL (illegal instruction) 크래시가 발생합니다.

```
error: the target feature `dotprod` is not currently enabled
```

**원인**

`.cargo/config.toml`에서 타겟별로 특정 SIMD instruction set을 요구합니다:

```toml
# ARM64 (Android)
[target.aarch64-linux-android]
rustflags = ["-C", "target-feature=+neon,+dotprod", ...]

# x86_64 (호스트)
[target.x86_64-linux]
rustflags = ["-C", "target-feature=+avx2,+fma"]
```

- **ARM64**: NEON + dotprod가 필수. ARMv8.0만 지원하는 구형 디바이스에서는 dotprod를 사용할 수 없습니다.
- **x86_64**: AVX2 + FMA가 필수. 2013년 이전 CPU (Haswell 미만)에서는 지원하지 않습니다.

**해결**

1. **ARM64 디바이스 확인**: `adb shell cat /proc/cpuinfo`에서 `asimddp` (dotprod) feature가 있는지 확인합니다. 없다면 `.cargo/config.toml`에서 `+dotprod`를 제거하되, 양자화 matmul 성능이 저하될 수 있습니다.

2. **x86_64 호스트 확인**: `lscpu | grep avx2`로 AVX2 지원 여부를 확인합니다. 미지원 CPU라면 `.cargo/config.toml`에서 해당 줄을 주석 처리하세요. SIMD 최적화 없이 scalar 경로로 동작합니다.

3. **호스트에서 개발 시**: x86_64 호스트 빌드는 `cargo check` 또는 `cargo test`를 사용합니다. ARM64 전용 코드는 `#[cfg(target_arch = "aarch64")]`로 게이팅되어 있으므로 호스트 빌드에는 영향을 미치지 않습니다.

---

### 1.4 Cross-compilation linker 에러

**증상**

Android 타겟 빌드 시 linker를 찾을 수 없거나, C library 링킹 에러가 발생합니다.

```
error: linker `cc` not found
```

또는:

```
error: linking with `aarch64-linux-android21-clang` failed
ld: error: unable to find library -lgcc
```

**원인**

`android.source`에서 설정하는 환경 변수들이 누락되었습니다:

```bash
export CC_aarch64_linux_android=$TOOLCHAIN/bin/aarch64-linux-android21-clang
export CXX_aarch64_linux_android=$TOOLCHAIN/bin/aarch64-linux-android21-clang++
export AR_aarch64_linux_android=$TOOLCHAIN/bin/llvm-ar
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$TOOLCHAIN/bin/aarch64-linux-android21-clang
```

`source android.source` 없이 빌드하면 cargo가 시스템 기본 `cc`를 linker로 사용하려 하므로 크로스 컴파일이 실패합니다.

**해결**

1. **반드시** 빌드 전에 `source android.source`를 실행합니다:

```bash
source android.source
cargo build --target aarch64-linux-android --release --bin generate
```

2. `$NDK_HOME` 경로가 실제 NDK 설치 위치를 가리키는지 확인합니다:

```bash
ls $NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/
```

3. 새 터미널을 열 때마다 `source android.source`를 재실행해야 합니다. 환경 변수는 현재 셸 세션에만 적용됩니다.

---

## 2. Runtime Errors

### 2.1 OpenCL device not found

**증상**

프로그램 시작 시 OpenCL 디바이스를 찾을 수 없다는 에러와 함께 패닉이 발생합니다.

```
thread 'main' panicked at 'No OpenCL devices found'
```

**원인**

`src/backend/opencl/mod.rs:88-93`에서 OpenCL 디바이스 탐색 로직을 보면:

```rust
let device = Device::list(platform, Some(flags::DEVICE_TYPE_GPU))?
    .into_iter()
    .next()
    .unwrap_or_else(|| Device::first(platform).expect("No OpenCL devices found"));
```

먼저 GPU 타입 디바이스를 찾고, 없으면 플랫폼의 첫 번째 디바이스로 폴백합니다. 두 경우 모두 실패하면 패닉이 발생합니다. 원인은 다음 중 하나입니다:

- OpenCL 런타임(ICD)이 설치되어 있지 않음
- GPU 드라이버가 OpenCL을 지원하지 않음
- Android 디바이스에서 OpenCL 공유 라이브러리(`libOpenCL.so`)가 없음

**해결**

1. **호스트 개발 시**: GPU가 없는 환경에서는 `--backend cpu` 플래그를 사용하여 CPU 백엔드로 실행합니다.

2. **Android 디바이스**: `adb shell ls /system/vendor/lib64/libOpenCL.so`로 OpenCL 라이브러리 존재 여부를 확인합니다. Qualcomm Adreno GPU가 탑재된 디바이스에서는 대부분 지원됩니다.

3. **호스트에서 OpenCL 설치**:
```bash
# Ubuntu/Debian
sudo apt install ocl-icd-opencl-dev intel-opencl-icd  # Intel GPU
# 또는
sudo apt install nvidia-opencl-dev                      # NVIDIA GPU
```

4. 디바이스 목록 확인: `clinfo` 명령으로 시스템에서 인식하는 OpenCL 디바이스를 확인할 수 있습니다.

---

### 2.2 OpenCL kernel compilation failure

**증상**

프로그램이 시작은 되지만, stderr에 WARN 메시지가 출력되고 일부 연산이 올바르게 동작하지 않습니다.

```
WARN: Failed to compile matmul kernel: <error>. Falling back to dummy.
WARN: Failed to compile Q4_0 kernel: <error>. Using dummy.
```

**원인**

`src/backend/opencl/mod.rs:106-149`에서 kernel 컴파일 실패 시 graceful fallback 처리를 합니다:

```rust
let program = match Program::builder()
    .devices(device)
    .src(matmul_src)
    .cmplr_opt(CL_FAST_MATH_OPTS)
    .build(&context)
{
    Ok(p) => p,
    Err(e) => {
        eprintln!("WARN: Failed to compile matmul kernel: {}. Falling back to dummy.", e);
        Program::builder()
            .devices(device)
            .src("__kernel void kernel_mul_mat_f32_f32() {}")
            .build(&context)?
    }
};
```

dummy kernel로 대체되므로 프로그램은 크래시하지 않지만, 해당 연산의 결과가 올바르지 않습니다. 주요 원인:

- OpenCL 드라이버가 특정 OpenCL extension을 지원하지 않음
- 드라이버 버전이 오래되어 CL 2.0+ 기능 미지원
- kernel 소스의 `#pragma` 지시어와 드라이버 비호환

**해결**

1. WARN 메시지가 출력되면 해당 kernel의 연산이 정상 작동하지 않으므로, **`--backend cpu`로 전환**하여 CPU 백엔드를 사용하세요.

2. GPU 드라이버를 최신 버전으로 업데이트합니다. Android의 경우 시스템 업데이트를 확인하세요.

3. `.cl` kernel 파일은 고도로 최적화되어 있으므로 **직접 수정하지 마세요** (CLAUDE.md 제약사항 참조). 드라이버 호환성 문제가 지속되면 이슈를 등록하세요.

---

### 2.3 Memory exhaustion (OOM)

**증상**

추론 도중 메모리 할당 실패로 프로그램이 크래시합니다.

```
memory allocation of XXXXX bytes failed
```

또는 Android에서 프로세스가 OOM killer에 의해 종료됩니다.

**원인**

`Galloc` allocator (`src/memory/galloc.rs`)는 내부적으로 `SharedBuffer::new()`를 호출하여 메모리를 할당합니다. 현재 구현에서는 `used_memory()`가 항상 0을 반환하며, **명시적인 OOM 복구 메커니즘이 없습니다**:

```rust
impl Memory for Galloc {
    fn alloc(&self, size: usize, dtype: DType) -> Result<Arc<dyn Buffer>> {
        let buf = SharedBuffer::new(size, dtype);
        Ok(Arc::new(buf))
    }

    fn used_memory(&self) -> usize {
        0 // Tracking not implemented
    }
}
```

Llama 3.2 1B 모델 기준 대략적인 메모리 사용량:
- 모델 가중치 (Q4_0): ~600MB
- KV cache (긴 시퀀스): 시퀀스 길이에 비례하여 증가
- 활성화 텐서 (workspace): ~수십 MB

**해결**

1. **KV cache eviction 활성화**: 긴 시퀀스를 생성할 때는 sliding window eviction을 사용하여 KV cache 크기를 제한합니다:

```bash
./generate --model-path models/llama3.2-1b \
  --prompt "Hello" -n 500 \
  --eviction-policy sliding --eviction-window 256
```

2. **Resilience Manager 활용**: `--enable-resilience` 플래그로 D-Bus 메모리 압박 신호에 따른 자동 eviction을 활성화합니다 (feature: `resilience` 필요).

3. **생성 토큰 수 제한**: `-n` 옵션으로 생성 토큰 수를 줄입니다.

4. **Android 디바이스**: RAM이 4GB 미만인 디바이스에서는 Q4_0 양자화와 짧은 시퀀스(`-n 128` 이하)를 권장합니다.

---

### 2.4 /proc/meminfo read failure

**증상**

시스템 메모리 통계를 읽을 때 에러가 발생합니다.

```
Error: MemTotal not found
```

또는:

```
Error: MemAvailable not found
```

**원인**

`src/core/sys_monitor.rs:63-77`에서 `LinuxSystemMonitor`는 `/proc/meminfo`를 파싱하여 메모리 정보를 얻습니다:

```rust
fn mem_stats(&self) -> Result<MemoryStats> {
    let content = std::fs::read_to_string("/proc/meminfo")?;
    Self::parse_meminfo(&content)
}
```

`parse_meminfo()`는 `MemTotal`, `MemAvailable`, `MemFree` 필드를 찾는데, 다음 상황에서 실패합니다:

- Linux가 아닌 플랫폼 (macOS, Windows)에서 실행
- 컨테이너 환경에서 `/proc`이 마운트되지 않음
- 매우 오래된 커널 (2.6.x)에서 `MemAvailable` 미지원

**해결**

1. **비-Linux 플랫폼**: `LinuxSystemMonitor`는 이름 그대로 Linux 전용입니다. macOS/Windows에서는 메모리 모니터링 기능을 사용하지 마세요.

2. **Docker/컨테이너**: `/proc`를 마운트하거나, 컨테이너 실행 시 `--privileged` 플래그를 사용합니다.

3. **오래된 커널**: `MemAvailable`은 Linux 3.14+에서 지원됩니다. 이전 커널에서는 `MemFree + Buffers + Cached`로 근사값을 계산하도록 코드 수정이 필요합니다.

---

## 3. D-Bus / Resilience Errors

> **참고**: Resilience 관련 에러 처리의 기본 원칙과 상세 내용은 [docs/24 - 6. 에러 처리 & 트러블슈팅](24_resilience_usage_guide.md#6-에러-처리--트러블슈팅)을 참조하세요.

### 3.1 D-Bus connection refused

**증상**

프로그램 시작 시 D-Bus 연결 실패 로그가 출력되지만 추론은 정상 진행됩니다.

```
[Resilience] D-Bus listener exited: Connection refused. LLM continues without resilience.
```

**원인**

`src/resilience/dbus_listener.rs:37-84`에서 D-Bus listener는 **fail-open** 설계를 따릅니다. `run()` 메서드가 System Bus 연결에 실패하면 에러를 반환하고, `spawn()` 메서드가 이를 잡아서 경고만 출력합니다:

```rust
fn run(&self) -> anyhow::Result<()> {
    let conn = zbus::blocking::Connection::system()?;
    // ...
}
```

D-Bus System Bus에 접근할 수 없는 주된 이유:
- `dbus-daemon`이 실행되고 있지 않음
- 사용자 권한으로 System Bus 접근이 거부됨
- Android 환경에서 D-Bus가 설치되지 않음

> **상세 내용**: [docs/24 섹션 6.1 - D-Bus 연결 실패](24_resilience_usage_guide.md#61-d-bus-연결-실패) 참조.

**해결**

1. Fail-open 설계이므로 **추론 자체에는 영향이 없습니다**. Resilience 기능 없이 정상 동작합니다.

2. Resilience가 필요하다면 D-Bus System Bus 상태를 확인합니다:

```bash
# D-Bus daemon 실행 확인
systemctl status dbus

# System Bus 접근 테스트
dbus-send --system --dest=org.freedesktop.DBus \
  /org/freedesktop/DBus org.freedesktop.DBus.ListNames
```

3. 권한 문제라면 D-Bus policy 파일을 추가합니다 (`/etc/dbus-1/system.d/org.llm.Manager1.conf`).

4. 로그 레벨을 `debug`로 올려 상세 에러를 확인합니다:

```bash
RUST_LOG=debug ./generate --enable-resilience ...
```

---

### 3.2 Resilience Manager not registered

**증상**

D-Bus 연결은 성공하지만 proxy 생성 단계에서 에러가 발생합니다.

```
[Resilience] D-Bus listener exited: Failed to create proxy for org.llm.Manager1. LLM continues without resilience.
```

**원인**

`dbus_listener.rs:42`에서 `org.llm.Manager1` 서비스에 대한 proxy를 생성하려 하지만, 해당 서비스가 System Bus에 등록되어 있지 않습니다:

```rust
let proxy = zbus::blocking::Proxy::new(
    &conn, MANAGER_DEST, MANAGER_PATH, MANAGER_IFACE
)?;
```

Manager 서비스(`org.llm.Manager1`)가 실행되고 있지 않으면 proxy 생성이 실패합니다.

**해결**

1. **테스트 환경**: mock_manager를 먼저 실행합니다:

```bash
# 터미널 1: Manager 시작
cargo build --release --bin mock_manager --features resilience
./target/release/mock_manager

# 터미널 2: 추론 시작
./target/release/generate --enable-resilience ...
```

2. **프로덕션 환경**: Resilience Manager 서비스가 systemd 유닛으로 등록되어 있는지 확인합니다.

3. Manager 없이도 fail-open으로 추론은 정상 작동합니다. Manager가 나중에 시작되더라도 현재 세션에서는 감지되지 않으므로, 추론을 재시작해야 합니다.

---

### 3.3 Invalid D-Bus signal format

**증상**

D-Bus 신호 수신 후 파싱 실패 경고가 로그에 출력됩니다.

```
WARN: Failed to parse signal MemoryPressure: Invalid level: Normal
```

**원인**

`Level::from_dbus_str()` (`src/resilience/signal.rs:43-50`)은 **대소문자를 구분(case-sensitive)** 합니다:

```rust
pub fn from_dbus_str(s: &str) -> Option<Self> {
    match s {
        "normal" => Some(Level::Normal),
        "warning" => Some(Level::Warning),
        "critical" => Some(Level::Critical),
        "emergency" => Some(Level::Emergency),
        _ => None,
    }
}
```

유효한 값은 소문자만 허용됩니다: `"normal"`, `"warning"`, `"critical"`, `"emergency"`. 대문자(`"Normal"`)나 대문자 전체(`"CRITICAL"`)는 `None`을 반환하여 파싱이 실패합니다. 테스트 코드에서도 이를 명시적으로 확인합니다:

```rust
assert_eq!(Level::from_dbus_str("Normal"), None); // case-sensitive
```

동일한 패턴이 `RecommendedBackend::from_dbus_str()`, `ComputeReason::from_dbus_str()`, `EnergyReason::from_dbus_str()`에도 적용됩니다.

**해결**

1. D-Bus 신호를 전송하는 Manager 측에서 level 문자열을 **반드시 소문자**로 전송해야 합니다.

2. `busctl`이나 `dbus-send`로 테스트할 때 대소문자에 주의합니다:

```bash
# 올바른 예
dbus-send --system --type=signal \
  /org/llm/Manager1 org.llm.Manager1.MemoryPressure \
  string:"critical" uint64:1073741824 uint64:536870912

# 잘못된 예 - "Critical"은 파싱 실패
dbus-send --system --type=signal \
  /org/llm/Manager1 org.llm.Manager1.MemoryPressure \
  string:"Critical" uint64:1073741824 uint64:536870912
```

3. 파싱 실패 시 해당 신호만 무시되고 추론은 계속됩니다. 다만 해당 유형의 시스템 압박에 대한 대응이 누락됩니다.

---

### 3.4 GPU buffer eviction failure

**증상**

Resilience에 의한 KV cache eviction 시 에러가 발생합니다.

```
[Resilience] Eviction error: Buffer does not support CPU pointer access
```

**원인**

GPU 전용 버퍼에 대해 `prune_prefix()`를 호출하면, CPU 포인터를 통한 데이터 접근이 불가능하여 실패합니다. OpenCL 백엔드에서 `CL_MEM_ALLOC_HOST_PTR` 플래그 없이 할당된 버퍼는 CPU에서 직접 접근할 수 없으며, eviction 시 KV cache 데이터를 재배치하려면 CPU 포인터가 필요합니다.

> **상세 내용**: [docs/24 섹션 6.2 - Eviction 에러](24_resilience_usage_guide.md#62-eviction-에러) 참조.

**해결**

1. **CPU 백엔드 사용**: `--backend cpu`로 실행하면 모든 버퍼가 CPU 메모리에 있으므로 eviction이 항상 성공합니다.

2. **Zero-copy 모드**: ARM SoC에서 `CL_MEM_ALLOC_HOST_PTR`을 사용하는 zero-copy 버퍼를 사용하면 GPU 버퍼도 CPU에서 접근할 수 있습니다.

3. null buffer pointer가 발생하는 경우: KV cache가 초기화되기 전에 eviction이 트리거되었을 가능성이 있습니다. 최소 1회 이상의 forward pass가 완료된 후에 Resilience 신호가 수신되어야 합니다.

---

## 4. Device & Performance Issues

### 4.1 Model loading 실패

**증상**

모델 파일을 로딩할 때 에러가 발생합니다.

```
Error: Unsupported dtype in safetensors: I32
```

또는:

```
Error finding tensor: model.layers.0.self_attn.q_proj.weight
```

**원인**

`src/models/llama/llama_model.rs:74-122`에서 Safetensors 파일의 텐서를 로드할 때, 지원하는 dtype은 3가지뿐입니다:

```rust
match tensor_view.dtype() {
    safetensors::Dtype::F32 => { /* 직접 복사 */ },
    safetensors::Dtype::BF16 => { /* bf16 → f32 변환 */ },
    safetensors::Dtype::F16 => { /* f16 → f32 변환 */ },
    _ => {
        return Err(anyhow!(
            "Unsupported dtype in safetensors: {:?}",
            tensor_view.dtype()
        ));
    }
}
```

지원 dtype: **F32**, **BF16**, **F16**. 그 외 dtype (I32, I64, U8 등)은 지원하지 않습니다.

텐서 이름을 찾지 못하는 경우는 HuggingFace 모델의 가중치 이름 규칙이 예상과 다를 때 발생합니다.

**해결**

1. **모델 형식 확인**: HuggingFace에서 모델을 다운로드할 때 Safetensors 형식인지 확인합니다. GGUF, PyTorch `.bin`, ONNX 등 다른 형식은 지원하지 않습니다.

2. **dtype 확인**: Python으로 모델의 dtype을 확인합니다:

```python
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    for name in f.keys():
        tensor = f.get_tensor(name)
        print(f"{name}: {tensor.dtype}")
```

3. **지원 모델**: Llama 3.2 계열 모델 (1B, 3B)이 지원됩니다. 다른 아키텍처(Mistral, GPT 등)는 가중치 이름 매핑이 다르므로 로딩에 실패합니다.

4. **텐서 이름 불일치**: 모델이 여러 shard로 분할된 경우 (`model-00001-of-00002.safetensors` 등), 현재 구현은 단일 파일만 지원할 수 있습니다. 모델 문서에서 파일 구조를 확인하세요.

---

### 4.2 Slow inference performance

**증상**

토큰 생성 속도가 기대보다 현저히 느립니다 (예: 1B 모델에서 1 tok/s 미만).

**원인**

성능 저하의 원인은 다양하며, 여러 요소가 복합적으로 작용할 수 있습니다.

**해결**

다음 체크리스트를 순서대로 확인하세요:

1. **`--release` 모드 빌드 확인**: Debug 빌드는 최적화가 없어 10배 이상 느릴 수 있습니다.

```bash
# 반드시 --release로 빌드
cargo build --release --bin generate
# debug 빌드가 아닌지 확인
file target/release/generate  # "not stripped" 확인
```

2. **SIMD target feature 활성화 확인**: `.cargo/config.toml`에서 NEON+dotprod (ARM64) 또는 AVX2+FMA (x86_64)가 설정되어 있는지 확인합니다. SIMD 없이는 scalar 연산으로 폴백되어 4-8배 느립니다.

3. **백엔드 선택 확인**: 디바이스에 OpenCL 지원 GPU가 있다면 `--backend opencl`을 사용합니다. GPU가 없는 환경에서 opencl 백엔드를 지정하면 오히려 오버헤드만 발생합니다.

```bash
# GPU 사용
./generate --backend opencl --model-path models/llama3.2-1b --prompt "Hello" -n 128

# CPU 전용
./generate --backend cpu --model-path models/llama3.2-1b --prompt "Hello" -n 128
```

4. **KV cache dtype 확인**: Q8_0 KV cache는 F32 대비 메모리를 절약하지만, 양자화/역양자화 오버헤드가 있습니다. 메모리가 충분하다면 F32 KV cache가 더 빠를 수 있습니다.

5. **Thermal throttling 확인**: 모바일 디바이스에서 장시간 추론 시 열 제한(thermal throttling)으로 CPU/GPU 클럭이 낮아질 수 있습니다.

```bash
# Android에서 CPU 주파수 확인
adb shell cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# 열 상태 확인
adb shell cat /sys/class/thermal/thermal_zone0/temp
```

6. **프로파일링**: 병목 지점을 정확히 파악하려면 내장 프로파일링 도구를 사용합니다:

```bash
# Android 디바이스 프로파일링
python scripts/android_profile.py

# 결과 시각화
python scripts/visualize_profile.py
```

7. **Micro benchmark**: 특정 연산의 성능을 개별적으로 측정합니다:

```bash
# 디바이스에서 개별 연산 벤치마크
./.agent/skills/testing/scripts/run_android.sh micro_bench
```

---

## 부록: 로그 레벨 설정

모든 카테고리의 문제를 디버깅할 때 `RUST_LOG` 환경 변수로 로그 레벨을 조절할 수 있습니다:

```bash
# 전체 debug 로그
RUST_LOG=debug ./target/release/generate ...

# 특정 모듈만 debug
RUST_LOG=llm_rs2::resilience=debug,llm_rs2::backend::opencl=info ./target/release/generate ...

# 로그 레벨: error < warn < info < debug < trace
```

| 레벨 | 출력 내용 |
|------|-----------|
| `error` | 복구 불가능한 에러 |
| `warn` | D-Bus 파싱 실패, kernel 컴파일 fallback 등 |
| `info` | 초기화 정보, 디바이스 이름, 리스너 시작/종료 |
| `debug` | 수신된 D-Bus 신호 상세, 액션 처리, 텐서 로딩 |
| `trace` | 연산별 타이밍, 버퍼 할당/해제 |
