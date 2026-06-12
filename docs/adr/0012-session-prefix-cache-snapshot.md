# ADR-0012: 세션 KV persistence (prefix cache) Tier 1 — format snapshot/restore capability + 세션 prefix 저장/복원

## Status

Proposed

## Date

2026-06-12

## Context

LLM 추론에서 동일한 system prompt 또는 고정 prefix를 매 실행마다 prefill하는 비용이 존재한다.
Llama 3.2 1B (16 layers, kv_heads=8, head_dim=64) 기준으로 512 token prefill 은 호스트 CPU에서 수백 ms 수준이며, 반복 실행 시 TTFT(Time To First Token)를 지배한다.

KV 캐시는 prefix 토큰열의 **순수 함수**이며, RoPE 위치는 절대 좌표로 baked된다. 따라서 "prefill 직후·eviction 전" 상태의 KV를 디스크에 저장하고 다음 실행에서 복원하면 동일 결과를 얻을 수 있다 — position 매핑이나 별도 RoPE 재계산 없이.

단, grow-on-demand 구조 때문에 capacity는 실행마다 다를 수 있다. 이 경우에도 재현성을 보장하려면 capacity 패딩을 제거한 packed 형태로 직렬화해야 한다.

## Decision

본 ADR은 다음 6가지 핵심 결정을 확립한다.

### D1 — `SnapshotRestore` capability opt-in (base trait 불변)

`KVCacheFormat` base trait(6-method)은 변경하지 않는다(`INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`). snapshot/restore 기능은 별도 `SnapshotRestore` capability trait으로 분리하여 opt-in으로 제공한다.

- `StandardFormat`(F32/F16/Q4_0): `SnapshotRestore` 구현
- KIVI format, opaque format: `SnapshotRestore` 미구현 → no-cache 폴백(정확성 안전, 단지 가속 없음)
- 이 설계로 snapshot-aware format만 비용을 지불한다 (ISP — Interface Segregation Principle)

### D2 — 직렬화 형식: capacity 패딩 제거 packed-form

```
magic: [u8; 8] = b"ARGUSKV1"
version: u32   = 1
header_len: u32
model_hash: [u8; 32]
format_id: u32
tokenizer_hash: [u8; 32]
kv_heads: u32
head_dim: u32
n_layers: u32
token_count: u32
token_ids: [u32; token_count]
payload: (layer-major) layer 0 K, layer 0 V, layer 1 K, layer 1 V, ...
```

각 layer의 K/V payload는 per-head `[0..token_count)` 범위만 추출한다 (capacity 패딩 제거). grow-on-demand로 capacity가 실행마다 달라지므로 패딩 제거가 cross-run 재현성의 핵심이다.

### D3 — hash 산출원: `auf::source_hash::compute_source_hash` 재사용

model/tokenizer 파일의 hash 계산에 기존 `compute_source_hash` (sha256(size‖mtime‖head_8MB‖tail_8MB)) 를 재사용한다. 파일 내용 변경을 보수적으로 감지하며 별도 crate 추가 없이 구현 가능하다.

### D4 — CLI: `--save-prefix-cache` / `--prefix-cache` 2-flag (의도 명시)

단일 flag(`--prefix-cache`) 대신 저장과 복원을 분리하여 의도를 명시한다.

- `--save-prefix-cache <path>`: prefill 직후 KV를 path에 저장
- `--prefix-cache <path>`: 실행 시작 시 path에서 KV 복원 시도 (miss 시 fresh prefill)
- 두 flag 모두 None = prefix cache 코드 미진입 (분기 1회)
- `is_standard_happy_path` 호환: eviction-전 저장이라 eviction-off happy path와 직교 정합

### D5 — GPU 경로: `backend.read_buffer()` / `write_buffer()` coherent 강제 (ARM UMA — as_ptr 금지)

ARM UMA에서 `as_ptr()` 직접 접근은 stale cache를 읽을 수 있다 (INV-191, INV-171 참조). snapshot/restore의 device 버퍼 접근은 반드시 `backend.read_buffer()` / `backend.write_buffer()` 경유로 한다. CPU backend의 `read_buffer` default impl은 memcpy라 오버헤드 없다.

### D6 — 부분 일치: 접두 일치 + 잔여 prefill (중간 divergence 재일치 = miss)

저장된 `token_count`개의 token_ids가 현재 prompt의 정확한 접두(`prompt[0..token_count]`)와 일치할 때만 복원한다.

- `token_count == prompt.len()`: prefill 완전 skip
- `token_count < prompt.len()`: `prefill(prompt[token_count..], start_pos=token_count)` 잔여 prefill
- 중간 divergence 후 재일치 (부분 내부 일치): miss → fresh prefill (Tier 3 배제)

### D7 — last-token logits 스냅샷 포함 (full restore bit-exact 의 원천, 2026-06-12 구현 중 추가)

스냅샷 v2 부터 헤더에 `logits_len`/`last_logits[f32×vocab]` 를 포함하고, full restore 는 re-forward 없이 저장된 logits 로 decode 를 시작한다.

- **근거 (host 실측 진단)**: snapshot/restore 의 KV 는 byte-perfect 였음에도(복원 KV re-save ≡ fresh save, `cmp` IDENTICAL) full restore 출력이 fresh 와 분기 — 원인은 마지막 토큰 logits 의 **GEMM 배치 경로 차이**(fresh = M=n batch prefill, 구 restore = M=1 `forward_gen` re-forward)가 만드는 ±ulp 차이의 greedy 분기. 재계산을 제거하고 저장값을 쓰면 KV(byte-identical) + 시작 logits(byte-identical) + decode 경로(동일) = token-id bit-exact. llama.cpp `llama_state_save_file` 동일 접근.
- **부수 수정**: restore 경로는 prefill 의 `sampler.observe_token` 부수효과(repetition penalty ring buffer)도 건너뛰므로 `seed_sampler_history(tokens)` 로 동등 상태를 재구성한다 (기본 `repetition_penalty=1.1` 이라 `--greedy` 에서도 발현).
- **비용**: 1B vocab 128256 × 4B ≈ 513KB/스냅샷 — prefix 500tok KV(4.6~16MB) 대비 수용 가능.
- **partial restore 는 bit-exact 비보장** (잔여 prefill 의 M 이 fresh 와 다름 — 수치적 동등, INV-191 비고 참조).

## Consequences

### 긍정적 영향

- 동일 system prompt 반복 실행 시 TTFT 대폭 단축 (prefill 완전 skip)
- base trait 무변으로 기존 format(KIVI, opaque) 무영향
- 파일 없음/무효화 = Ok(None) 이라 panic 없이 silent fallback

### 부정적 영향

- 디스크 공간 사용: `n_layers × 2 × kv_heads × token_count × head_dim × dtype_size` bytes (Llama 3.2 1B, F16, 512 token ≈ 16 × 2 × 8 × 512 × 64 × 2 = 16 MB)
- file I/O가 실패할 경우 (권한/디스크 풀) 경고 후 무시 (save 실패 = 다음 실행에 cache miss)

### 위험

- Q4_0 dtype의 element↔block 변환 오류 가능성 (Known Bug #4 SIGSEGV와 동형 함정) → 테스트로 커버 (INV-191)
- 파일 손상/부분 기록 → atomic write (tmp→rename) 로 방어

## Risks

- **R1**: GPU 경로에서 as_ptr 실수 → ARM UMA stale 읽기 → byte-not-identical. Source-grep INV-191 테스트로 감지.
- **R2**: Q4_0 element/block 단위 혼용 → OOB 접근. shrink_to_fit SIGSEGV 동형 함정. 테스트: INV-191 cross-capacity Q4_0 케이스.
- **R3**: token_ids 부분 일치 로직 오류 → 잘못된 KV 사용 → 토큰 불일치. 테스트: INV-190 token_ids divergence.

## References

- `spec/30-engine.md` §3.7 (ENG-080~085)
- `spec/33-engine-data.md` §3.25 (ENG-DAT-110)
- `spec/41-invariants.md` §3.29 (INV-189~191)
- `arch/30-engine.md` §19
- `engine/src/auf/source_hash.rs` (hash 산출원)
- `engine/src/kv/kv_cache.rs` Known Bug #4 (Q4_0 shrink_to_fit SIGSEGV)
