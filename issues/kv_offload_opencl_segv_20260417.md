# KV offload: OpenCL backend에서 offload 버퍼 초기화 직후 SIGSEGV / EFAULT

**Date filed**: 2026-04-17
**Filed by**: PACT 2026 Fig 2 / §2.2 KV offload panel smoke (papers repo, action_profile_v2 계열)
**Scope**: `engine/src/core/kv_cache/offload/*` (raw / disk), `generate` CLI의 `--kv-offload` + `--backend opencl` 조합
**Severity**: high — Fig 2의 R2C3 KV offload panel을 S25 OpenCL 환경에서 측정할 수 없음. 논문의 "4 action × no free lunch" 서술 중 offload 케이스가 공백.

## Summary

S25 / Qwen 2.5 1.5B / Adreno OpenCL 환경에서 `--kv-offload raw` 또는 `--kv-offload disk` 를 `--backend opencl` 과 함께 지정하면, 모델 로드 + offload 버퍼 초기화 로그 (`[Offload] mode=..., dtype=F16, layers=28, token_bytes=512, max_seq=...`) 가 찍힌 직후 prefill이 시작되기 전에 프로세스가 종료된다. raw 모드는 libc `__memmove_aarch64_nt` 안에서 NULL deref (`SIGSEGV SEGV_MAPERR fault addr 0x0`), disk 모드는 Rust 상위 레이어에서 `Error: Bad address (os error 14)` (EFAULT) 으로 잡힌다. **동일 플래그로 `--backend cpu` 는 정상 동작**하므로 OpenCL 경로 전용 버그로 좁혀진다. Prompt 길이 / `--max-seq-len` 값과 무관하다 (1555 tok @ 8192, 9 tok @ 512 모두 실패).

## Reproduction

```bash
# 디바이스: Galaxy S25 (Snapdragon 8 Elite / Adreno), Android
# 바이너리: /data/local/tmp/generate  (2026-04-17 13:29 빌드, md5 c4194a776c1affe0e713f621d81d6e36)
# llm_rs2 HEAD (host): 27510e4 "perf(partition): gate dual-map on actual use + GPU-only fast path"

# (A) FAIL — OpenCL + raw offload (long prompt)
adb shell "/data/local/tmp/generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b --backend opencl --weight-dtype f16 \
  --max-seq-len 8192 --kv-type f16 \
  --kv-layout seq --kv-offload raw \
  --prompt-file /data/local/tmp/llm_rs2/eval/q1_medium_prompt.txt \
  -n 100 --ignore-eos --greedy"

# (B) FAIL — OpenCL + disk offload (same prompt)
adb shell "/data/local/tmp/generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b --backend opencl --weight-dtype f16 \
  --max-seq-len 8192 --kv-type f16 \
  --kv-layout seq --kv-offload disk --offload-path /data/local/tmp/kv_offload_files \
  --prompt-file /data/local/tmp/llm_rs2/eval/q1_medium_prompt.txt \
  -n 20 --ignore-eos --greedy"

# (C) FAIL — OpenCL + raw offload (짧은 prompt, max-seq-len 512)
adb shell "/data/local/tmp/generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b --backend opencl --weight-dtype f16 \
  --max-seq-len 512 --kv-type f16 \
  --kv-layout seq --kv-offload raw \
  --prompt-file /data/local/tmp/kv_smoke/short_prompt.txt \
  -n 20 --ignore-eos --greedy"
# short_prompt.txt: "Hello world. This is a short test."

# (D) PASS — CPU + raw offload (동일 플래그, backend만 cpu)
adb shell "/data/local/tmp/generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b --backend cpu --weight-dtype f16 \
  --max-seq-len 4096 --kv-type f16 \
  --kv-layout seq --kv-offload raw \
  --prompt-file /data/local/tmp/llm_rs2/eval/q1_medium_prompt.txt \
  -n 20 --ignore-eos --greedy"

# (E) PASS — OpenCL + no offload (baseline)
adb shell "/data/local/tmp/generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b --backend opencl --weight-dtype f16 \
  --max-seq-len 8192 --kv-type f16 \
  --prompt-file /data/local/tmp/llm_rs2/eval/q1_medium_prompt.txt \
  -n 100 --ignore-eos --greedy"
```

## Evidence

### (1) 실패/성공 매트릭스

| # | backend | kv-offload | kv-layout | max-seq-len | prompt tok | 결과 |
|---|---------|------------|-----------|-------------|------------|------|
| A | opencl  | raw        | seq       | 8192        | 1555       | ❌ SIGSEGV (libc memmove) |
| B | opencl  | disk       | seq       | 8192        | 1555       | ❌ `Error: Bad address (os error 14)` (EFAULT) |
| C | opencl  | raw        | seq       | 512         | 9          | ❌ SIGSEGV (A와 동일 위치) |
| D | cpu     | raw        | seq       | 4096        | 1555       | ✅ 정상 (20 tok 완주, prefill/decode 진행) |
| E | opencl  | none       | head      | 8192        | 1555       | ✅ 정상 (100 tok, TTFT 10.9 s, avg TBT 72 ms, RSS 3565 MB) |

→ **`backend=opencl` + `kv-offload ∈ {raw, disk}` 조합이 유일한 실패 변수**. prompt 크기·max-seq-len 독립, `--kv-layout seq` 자체는 무관 (offload 필수 조건일 뿐).

### (2) 공통 실패 지점 — offload 초기화 직후

세 실패 케이스 모두 로그의 마지막 줄이 동일한 구조:

```
KV cache type: F16, layout: SeqMajor (initial capacity: 128 tokens, 65536B per layer, max: 512)
[Offload] mode=raw, dtype=F16, layers=28, token_bytes=512, max_seq=512
```

이후 `[Prefill] prefill_chunk_size auto-selected: ...` 또는 `Prefill: ... ms` 메시지가 **한 줄도 찍히기 전에** 프로세스가 죽는다. 즉 offload 오브젝트 construction은 성공 (print 통과) 했지만, 그 직후 첫 사용 시점 (prefill 진입 또는 offload와 연결된 GPU buffer의 초기 memcpy) 에서 NULL pointer 참조.

### (3) Tombstone (case A, short prompt도 동일 프레임)

`logcat -v threadtime` 에서:

```
04-17 22:11:34.243 F libc   : Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR),
                              fault addr 0x0 in tid 18681 (generate), pid 18681 (generate)
04-17 22:11:34.294 F DEBUG  : Cmdline: /data/local/tmp/generate --model-path .../qwen2.5-1.5b
                              --backend opencl --weight-dtype f16 --max-seq-len 8192
                              --kv-type f16 --kv-layout seq --kv-offload raw
                              --prompt-file .../q1_medium_prompt.txt -n 100
                              --ignore-eos --greedy --tbt-log ...
04-17 22:11:34.294 F DEBUG  : signal 11 (SIGSEGV), code 1 (SEGV_MAPERR),
                              fault addr 0x0000000000000000
04-17 22:11:34.294 F DEBUG  :   #00 pc 0000000000098a50
     /apex/com.android.runtime/lib64/bionic/libc.so (__memmove_aarch64_nt+592)
04-17 22:11:34.294 F DEBUG  :   #01 pc 00000000003125c4 /data/local/tmp/generate
04-17 22:11:34.294 F DEBUG  :   #02 pc 00000000003e2ac0 /data/local/tmp/generate
04-17 22:11:34.294 F DEBUG  :   #03 pc 000000000024a1ec /data/local/tmp/generate
04-17 22:11:34.294 F DEBUG  :   #04 pc 0000000000253894 /data/local/tmp/generate
04-17 22:11:34.295 F DEBUG  :   #05 pc 00000000002704b4 /data/local/tmp/generate
04-17 22:11:34.295 F DEBUG  :   #06 pc 0000000000280b50 /data/local/tmp/generate
04-17 22:11:34.295 F DEBUG  :   #07 pc 00000000001fef20 /data/local/tmp/generate
04-17 22:11:34.295 F DEBUG  :   #08 pc 00000000002a37c8 /data/local/tmp/generate
04-17 22:11:34.295 F DEBUG  :   #09 pc 000000000006a714
     /apex/com.android.runtime/lib64/bionic/libc.so (__libc_init+124)
```

- 톱 프레임이 **`memmove`** 이고 fault addr = 0x0 → `memmove(dst=NULL, src, n)` 또는 `memmove(dst, src=NULL, n)` 형태의 NULL deref.
- 디바이스 바이너리는 strip 되어 있어 symbol 이름은 없지만 `/data/tombstones/tombstone_30{.pb}` 에 proto 덤프 존재 (host에서 `addr2line` 또는 unstripped 빌드 재수행으로 복원 가능).

### (4) disk 변형은 Rust 쪽에서 EFAULT 로 잡힘

case B의 stderr:

```
[Offload] mode=disk, path=/data/local/tmp/kv_offload_files, dtype=F16,
          layers=28, token_bytes=512, max_seq=8192
Error: Bad address (os error 14)
```

`os error 14 == EFAULT` 은 `write(2) / mmap(2)` 계열 syscall 이 잘못된 포인터를 받은 경우 반환된다. raw case의 NULL memmove 와 같은 원인 (NULL buffer가 syscall에 전달) 을 disk 경로에서는 error-chain 으로 잡아 graceful exit 한 것으로 보임.

## Hypothesis (root cause 후보)

1. **Offload 버퍼와 OpenCL device buffer 사이의 binding 누락** — CPU backend에서는 host 포인터 = offload 대상 버퍼가 동일 주소공간이라 `memmove` 가 자연스럽게 동작한다. OpenCL 경로에서는 KV cache가 OpenCL `cl_mem` (또는 SVM / import된 host buffer) 으로 존재하는데, offload store가 그 `cl_mem` 을 host-accessible pointer 로 (map 혹은 mirror host buffer) 해석하지 못하고 raw null pointer 를 들고 있을 가능성.
   - CPU backend OK + offload 초기화 로그 후 첫 접근에서 NULL → offload 측의 destination/source buffer 가 OpenCL 경로에서만 미초기화.
2. **`--kv-layout seq` 전환 시 GPU 측 allocation 이 SeqMajor 용으로 안 잡힘** — 로그에 `layout: SeqMajor (initial capacity: 128 tokens, ...)` 까지만 있고 OpenCL device allocation (e.g. `Allocated cl_mem for KV head/seq ...`) 류 메시지가 없음. Head-major 경로에만 GPU buffer 생성 코드가 있고 Seq-major + offload 분기에서 누락됐을 수 있음.
3. **Offload `write_layer / read_layer` stub** — offload trait 이 구현돼 있지만 OpenCL-specific path 에서 backing store 가 `None` 또는 dangling 인 채로 반환되어 첫 호출 (prefill에서 첫 layer의 KV 를 기록할 때) 에 NULL 을 dst 로 memmove.

hypothesis 1이 가장 유력. CPU backend에서는 동일 offload trait 구현이 돌아가므로 offload 로직 자체는 맞고, OpenCL backend와의 접합부 (KV cache 초기화 시 cl_mem ↔ offload slab pairing) 가 빠진 것으로 보인다.

## Suggested fix

- **단기 (blocker 해제)**: `engine/src/core/kv_cache/mod.rs` (혹은 backend 별 KV cache 초기화) 에서 `kv_offload != None` + `backend == OpenCL` 경로에 명시적 assertion 추가 후, NULL dst / src 의 실제 원인 식별. offload store 생성 시 OpenCL backend 용 host-mirror buffer (clCreateBuffer + `CL_MEM_ALLOC_HOST_PTR` 등) 를 같이 만들고, offload 측의 `backing_ptr()` 가 이 mirror 를 가리키도록 wiring.
- **중기**: seq-major + offload + OpenCL 조합에 대해 integration test (host CI 에서 돌 수 있는 tiny model, 10 tokens 정도) 를 `tests/` 에 추가. 지금은 CPU 경로만 exercise 되고 있어 regression 발견이 늦어짐.
- **Paper-side**: fix 전까지 Fig 2 R2C3 KV offload panel 은 **hatched placeholder** 유지. fix 후 `action_profile_v2_run.sh` 에 `kv_offload_raw` case 를 추가하여 baseline_gpu (no offload) 와 동일 240 s protocol 로 재측정.

## Related

- **Smoke 데이터**: `/data/local/tmp/kv_smoke/{baseline,offload,cpu_offload,gpu_disk,gpu_short}_gen.log` (local copies at `/tmp/*_gen.log`)
- **Fig 2 / §2.2 합의**: `papers/memory/project_fig2_design.md` — "KV offload panel pending, `--kv-offload raw --kv-layout seq --kv-type f16` (USAGE.md §2.9) 로 action_profile_v2_run.sh 에 새 case 필요"
- **USAGE.md §2.9**: `/docs/USAGE.md` L859~L906 — offload 모드 / 전제조건 / 예제 (예제는 `--backend opencl` + llama3.2-1b 기준인데 이 조합이 실제로는 실패함 → **USAGE 예제도 최소 smoke 검증 필요**)
- **관련 최근 커밋**: `27510e4 perf(partition): gate dual-map on actual use + GPU-only fast path`, `56d9c51 fix(kv-cache): shrink buffers after eviction + zero-copy KIVI get_view` — offload 경로 변경 여부 확인 필요.

## Paper impact

- Fig 2 (cross-domain effects) R2C3 panel 공백 → §2.2 내러티브의 "4 action × no free lunch" 중 KV offload 케이스 측정 불가 상태.
- abstract 4/23, paper 4/30 마감 (2026-04-17 6일 연장) 안에 fix + S25 재측정 가능 시 inline panel 로 복구, 아니면 appendix 로 이동 또는 다른 action (예: H2O keep-50 dose-response) 으로 대체.

## Resolution (2026-04-17)

**Status**: FIXED — S25 재현 매트릭스 7/7 PASS.

두 단계로 원인 분석·수정됨.

1. **Iter 1 — NULL pointer on GPU-only buffer**: `OpenCLBuffer::as_ptr()` / unmapped `UnifiedBuffer::as_ptr()` 는 `ptr::null()` 을 반환하는데 `OffloadKVCache::update()` 가 이를 `slice::from_raw_parts` 로 감싸 `Vec::extend_from_slice` (raw) / `File::write_all` (disk) 에 전달 → `__memmove_aarch64_nt(NULL, ...)` SIGSEGV 또는 `write(fd, NULL, n)` EFAULT. GPU-only tensor 는 `backend.read_buffer()` 로 host staging buffer 로 복사 후 전달하도록 수정. `set_gpu_backend(backend, memory)` API 추가, `run_offload` 에서 `backend.is_gpu()` 시 주입. `get_view()` GPU 경로에서 staged KV bytes 를 reusable device buffer 로 업로드해 `attention_gen` (cl_mem 필수) 도 동작.

2. **Iter 2 — `write_buffer` size-strict assertion**: iter 1 에서 GPU KV device buffer 를 `max_seq_len × token_bytes` 로 alloc 하고 매 토큰 재사용했으나, tensor shape 을 current prefix 로 줄여 감싸도 `Tensor::size() == buffer.size()` 라 여전히 max_bytes. `Backend::write_buffer` 의 `assert_eq!(src.len(), t.size())` 가 panic → `write_buffer: size mismatch (e.g. 796160 vs 4194304)` SIGABRT. `Backend::write_buffer_range(t, src, dst_offset)` default trait 메서드 추가, `OpenCLBackend` 에서 `clEnqueueWriteBuffer(dst_mem, blocking=true, offset=dst_offset, ...)` 으로 override. `OffloadKVCache::get_view()` 에서 `write_buffer_range(..., 0)` 로 교체해 prefix 영역만 업로드.

**변경 파일**:
- `engine/src/core/backend.rs` — `write_buffer_range` default impl
- `engine/src/backend/opencl/mod.rs` — `write_buffer_range` override (partial `clEnqueueWriteBuffer`)
- `engine/src/core/offload/mod.rs` — `OffloadKVCache::{new, set_gpu_backend, update, get_view}` GPU-aware
- `engine/src/bin/generate.rs` — `run_offload()` 에서 GPU backend wiring

`.cl` 커널 수정 없음.

**검증** (S25 R3CY408S5SB, Qwen 2.5 1.5B, binary md5 `a394de8882c45846b4cf95610d153819`):

| Case | Backend | Offload | Prompt / max_seq | Before | After | TTFT | Avg TBT | Tok/s |
|------|---------|---------|------------------|--------|-------|------|---------|-------|
| A | opencl | raw f16  | 1555 / 8192 | SIGSEGV    | **PASS** | 12774 ms | 149.8 ms | 6.7  |
| B | opencl | disk f16 | 1555 / 8192 | EFAULT     | **PASS** | 13792 ms | 159.3 ms | 6.3  |
| C | opencl | raw f16  | 9 / 512     | SIGSEGV    | **PASS** | 409 ms   | 68.6 ms  | 14.6 |
| D | cpu    | raw f16  | 1555 / 4096 | PASS       | PASS    | 26267 ms | 91.6 ms  | 10.9 |
| E | opencl | none     | 1555 / 8192 | PASS       | PASS    | 12145 ms | 72.2 ms  | 13.9 |
| F | opencl | disk f16 | 9 / 512     | (신규)     | **PASS** | 530 ms   | 89.3 ms  | 11.2 |
| G | opencl | raw f32  | 9 / 512     | (신규)     | **PASS** | 458 ms   | 81.9 ms  | 12.2 |

**후속 작업 (별도 이슈로 권장)**:
- OpenCL + offload 경로 decode TBT 가 no-offload 대비 ~2× (73→150 ms) — host↔GPU 전체-prefix 재업로드 오버헤드. 증분 upload (seq append 만) 또는 `get_view` caching 으로 튜닝 여지.
- `OpenCLBuffer::as_ptr() = null` invariant 를 타입시스템 수준으로 강제 (panic/debug_assert) — 유사 NULL memmove 지뢰 제거.
- Seq-major + offload + OpenCL 조합에 대한 host CI integration test (tiny model, 10 tok) 추가 — 회귀 방지.
