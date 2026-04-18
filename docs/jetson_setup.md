# Jetson 보드 빌드/배포 셋업

호스트(Linux/Mac)에서 Jetson 보드(JetPack 5.1.x = Ubuntu 20.04 / glibc 2.31)로 크로스 컴파일·배포·추론하기 위한 일회성 셋업 가이드.

`cargo-zigbuild`가 호스트의 최신 glibc(예: Arch 2.41)와 보드 glibc(2.31) 간 ABI 호환을 처리한다. `run_device.py` + `devices.toml` + `hosts.toml`이 워크플로를 자동화한다.

## 호스트 설치 항목

| 패키지 | 설치 명령 | 용도 |
|---|---|---|
| **zig** | `sudo pacman -S zig` (Arch) / `brew install zig` (mac) / `apt install zig` (Ubuntu — 또는 ziglang.org에서 직접) | cargo-zigbuild이 링커로 사용 |
| **cargo-zigbuild** | `cargo install cargo-zigbuild` (`~/.cargo/bin`을 PATH에 포함) | glibc 버전 지정 cross-link |
| **rust target** | `rustup target add aarch64-unknown-linux-gnu` | aarch64 cross-compile target |
| **openssh + sshpass** | 대부분 기본 설치됨 (Arch: `sudo pacman -S openssh sshpass`) | SSH 배포 + ssh-copy-id 자동화 |

> `aarch64-linux-gnu-gcc` 같은 별도 크로스 툴체인은 **불필요** — zig가 다 처리한다.

## SSH 키 등록 (보드에 1회)

`SshConnection`은 비밀번호 인증을 지원하지 않으므로 ssh key 등록이 필수.

```bash
# 보드 비밀번호로 키 자동 등록 (Jetson 기본: nvidia/nvidia)
sshpass -p <password> ssh-copy-id -o StrictHostKeyChecking=accept-new -p <port> <user>@<host>

# 검증 (비번 묻지 않아야 함)
ssh -p <port> -o BatchMode=yes <user>@<host> 'uname -m'
```

## hosts.toml 생성

```bash
python scripts/device_registry.py bootstrap-host
```

> Jetson용 zigbuild 경로는 hosts.toml의 toolchain entry가 **필요 없다** — `devices.toml`의 `zig_target` 필드가 모든 걸 처리.
> hosts.toml은 Android NDK 같은 기존 토체인용으로만 유지.

## devices.toml 등록 (이미 존재)

```toml
[devices.jetson]
name = "Jetson Orin"

[devices.jetson.connection]
type = "ssh"
host = "<board-ip>"
user = "nvidia"
port = 4121

[devices.jetson.build]
target = "aarch64-unknown-linux-gnu"
zig_target = "aarch64-unknown-linux-gnu.2.31"   # JetPack 5.1.x = glibc 2.31
binary_dir = "target/aarch64-unknown-linux-gnu/release"
features = ["cuda"]
default_features = false

[devices.jetson.paths]
work_dir = "/home/nvidia/llm_rs2"
model_dir = "/home/nvidia/llm_rs2/models/llama3.2-1b"
eval_dir = "/home/nvidia/llm_rs2/eval"
lib_dir = "/usr/local/cuda/lib64"
```

다른 보드(예: JetPack 6 = Ubuntu 22.04)일 경우 `zig_target = "aarch64-unknown-linux-gnu.2.35"`로 조정.

## 보드 측 사전 셋업 (1회)

```bash
ssh -p 4121 nvidia@<board-ip>
mkdir -p /home/nvidia/llm_rs2/{models,eval}
# 모델 weight 전송 (호스트에서)
scp -P 4121 -r models/llama3.2-1b nvidia@<board-ip>:/home/nvidia/llm_rs2/models/
```

## 빌드 + 배포 + 추론

```bash
# CPU
python scripts/run_device.py -d jetson generate -b cpu --prompt "'Hello'" -n 50

# CUDA (GPU)
python scripts/run_device.py -d jetson generate -b cuda --prompt "'Hello'" -n 50

# 빌드만
python scripts/run_device.py -d jetson --skip-deploy --skip-exec generate

# 배포만 (이미 빌드된 바이너리)
python scripts/run_device.py -d jetson --skip-build --skip-exec generate
```

> SSH로 전달되는 `--prompt` 인자는 셸 토큰화를 거치므로 공백이 있으면 `"'…'"`로 감싼다.

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `version 'GLIBC_2.39' not found` | zigbuild 미사용 (일반 cargo build) → `zig_target` 필드 확인 |
| `cargo-zigbuild: command not found` | `~/.cargo/bin`이 PATH에 없음 → `export PATH="$HOME/.cargo/bin:$PATH"` |
| `Permission denied (publickey)` | ssh key 미등록 → 위의 `ssh-copy-id` 단계 |
| `/usr/local/cuda/lib64/libcudart.so: not found` | 보드에 CUDA가 없음 → `dpkg -l | grep cuda`로 JetPack 확인 |
| `cuda kernel compile error sm_XX` | cudarc cuda-11040 vs 보드 CUDA 버전 mismatch — `engine/Cargo.toml`의 `cudarc` features 조정 |

## 검증된 환경 (2026-04-18)

- 호스트: Arch Linux (glibc 2.41), zig 0.15.2, cargo-zigbuild 0.22.2
- 보드: Jetson AGX Xavier (sm_72, JetPack 5.1.2 = R35.5, Ubuntu 20.04, glibc 2.31, CUDA 11.8)
- 모델: Llama 3.2 1B (F16 safetensors)
- 결과:
  - CPU: Decode 16.7 tok/s, Prefill 39.5 tok/s
  - CUDA: Decode 26.0 tok/s, Prefill 140.6 tok/s
