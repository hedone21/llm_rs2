# Handoff: Format 축 production e2e 완주 (GATE-C deferred (a))

**작성**: 2026-06-09
**HEAD**: `8728bcd7 docs(handoff): GATE-C 멀티-vtable bundle ABI (ADR-0010) V1~V5 완주 handoff`
**브랜치**: master (미푸시)
**작성자**: 메인 세션 (production e2e 수동 검증)

**다음 세션 진입 문장**: **"Format 축 production e2e (host+device) 완주 — Android dlopen 가능 확정(S25). 다음은 (c) Backend v3 축 grill(BackendCapability 메서드 미확정) 또는 opaque generic-floor decode 페널티(device 1.34x) 최적화 중 택1."**

---

## TL;DR

GATE-C handoff 의 deferred (a) production e2e + (b) device 를 실모델로 완주. `argus_cli --load-plugin <번들 .so> --kv-format bundle_fmt` 가 **host(x86 CPU) + device(Galaxy S25, Android 16, SELinux Enforcing) 양쪽에서 실제 토큰을 생성**(zero-compile `.so` 가 opaque descriptor-driven KV 로 추론 구동). **★ADR-0009 D6 landmine 해소: Android dlopen 가능**(adb shell `/data/local/tmp` = `shell_data_file` 도메인, avc denial 0). **코드 무변경**(검증 전용, 모델 의존이라 커밋 테스트 아님 — handoff 명시대로 수동 검증·수치 보고). **멈춘 이유**: production e2e 목표(opaque 동적 구동 + 토큰 생성) host+device 충족 + 보강 검증 green → format 축 완주.

---

## 진행 상태 / 측정 (qwen2.5-1.5b q4_0, CPU, greedy, n_tok=48~64)

| # | 검증 | 결과 |
|---|---|---|
| 핵심 | `--load-plugin libexample_bundle.so --kv-format bundle_fmt` | ✅ opaque KV alloc + **coherent 토큰 생성**. stderr 확정: `KV format: bundle_fmt (opaque)` + `[DecodeLoop] kv storage = OPAQUE (descriptor-driven)` |
| ① | 결정성 (opaque ×2) | ✅ byte-identical (divergence 는 안정적 compute-path 특성, 비결정성 아님) |
| ② | fail-fast (no `--load-plugin`) | ✅ `Error: Unknown --kv-format 'bundle_fmt' (… --load-plugin 확인)` exit 1 → bundle_fmt 는 **순수 동적**(정적 등록 0, dlopen 이 유일 출처) |
| ③ | 2번째 모델 (qwen2.5-1.5b-**instruct**, 다른 weights) | ✅ typed/opaque 양쪽 coherent (둘 다 "Charles Babbage" 언급, 10단어 공유 후 갈림) |
| ④ | **전 opaque 경로 동일성**: synth_q4(정적 force-link) == example_kv_format(동적 단일축 .so) == bundle_fmt(동적 번들 .so) | ✅ **셋 다 byte-identical 출력** — 등록 출처(정적/동적/번들)는 compute 에 투명(source-agnostic make_format) |
| ⑤ | 정확도 (f16 reference) | typed-q4 & opaque-q4 **둘 다 f16 과 8단어(=프롬프트)만 공유 후 첫 생성 토큰부터 갈림**(q4 KV 양자화 열화, "180s" vs f16 "1950s") → opaque floor 는 typed-q4 와 **동일 정확도**(열화 아님) |
| ⑥ | decode 성능 (n=3 median) | typed **112.26** vs opaque **110.84** ms/tok → **comparable(노이즈 내)**. generic floor 의 decode 페널티 이 config 엔 없음 |

**typed-q4 vs opaque-q4 토큰 비-동일성(~21토큰째 갈림)은 EXPECTED & 양성**: 두 attention compute 경로가 다름 — opaque=generic floor(`standard_format.rs:374` KV 전체 `dequant_to_f32_tensor`→**F32** `attention_gen`), typed=`backend.attention_gen` 의 dtype-aware Q4_0 arm(`backend/cpu/common.rs:701-818` 블록 **인라인 dequant + NEON FMA**). 저장은 byte-identical(`encode_via_descriptor` == `BlockQ4_0::quantize`, `dtype_layout.rs:251`+테스트 :420), f32 reduction 순서만 다름(FMA-vs-mul-add) → greedy argmax 가 ~21토큰 후 flip. **버그 아님**: 결정적·coherent·f16 대비 typed 와 동일 정확도. ④ 가 강한 불변식(전 opaque 경로 상호 byte-identical).

---

## Device 검증 — Galaxy S25 (SM-S931N, Android 16, SELinux **Enforcing**), CPU backend

| # | 검증 | 결과 |
|---|---|---|
| ★ | **Android dlopen** (`--load-plugin /data/local/tmp/libexample_bundle.so --kv-format bundle_fmt`) | ✅ `KV format: bundle_fmt (opaque)` + `[DecodeLoop] kv storage = OPAQUE` + **coherent 토큰 생성** + **avc denial 0** → ADR-0009 D6 landmine 해소 |
| ① | 결정성 (opaque ×2) | ✅ identical |
| ② | **동적 == 정적**: dlopen bundle_fmt == force-link 정적 synth_q4 | ✅ **byte-identical 출력** — dlopen plugin 이 엔진 정적 opaque 와 동일 floor 구동(등록 출처 투명, device 에서도) |
| ③ | fail-fast (no `--load-plugin`) | ✅ `Unknown --kv-format 'bundle_fmt'` exit 1 |
| ④ | decode 성능 (n=3 median, warm) | typed Q4_0(NEON) **55.43** vs opaque generic floor **74.01** ms/tok = **1.34x** |

**★ device perf 발견 — opaque generic floor 가 ARM 에서 유의미하게 느림(1.34x)**: host x86 에선 typed≈opaque(110.84 vs 112.26, 무시) 였으나 device 는 1.34x. 원인 = typed Q4_0 는 **NEON 인라인-dequant attention 커널**(빠름), opaque 는 generic floor(KV 전체 `dequant_to_f32_tensor`→F32 `attention_gen`, ADR-0005 D5 "exact 느릴 뿐") — ARM 에선 특화 NEON 커널 우위가 커서 floor 페널티가 실측됨. **thermal 민감**(cold 단일런 typed 30.35 → warm n=3 55.43). production opaque format 의 fast-path(NEON dequant-attention)는 ADR-0007 D4 per-format 인코더 확장 영역. **feasibility 는 증명**(coherent·결정적·정적과 동일), perf 는 후속 최적화.

**빌드 절차(device)**: argus_cli=`run_device.py -d galaxy_s25 --skip-exec argus_cli`(features opencl,vulkan,qnn,htp_fastrpc). plugin cdylib=수동 NDK env(`android.source` 는 macOS 경로라 **부적합** — hosts.toml `android-ndk` 툴체인 `/opt/android-ndk/.../linux-x86_64/bin/aarch64-linux-android21-clang` 를 CC/AR/LINKER 로 export) + `cargo build --release --target aarch64-linux-android -p example-bundle --features plugin-cdylib` → `adb push libexample_bundle.so /data/local/tmp`.

---

## 다음 작업 (택1)

1. **Backend v3 축** (c): `BackendCapability` trait 메서드 미확정(`name()` 1개) → **설계 grill 선행**. V1~V2 인프라(봉투/슬라이스/dispatcher/export_plugin!) 재사용하나 BackendVTableAbi + dual-wiring 매크로 + DynBackendCap 선행(ADR-0010 §3 C2). GPU device-only.
2. **opaque generic-floor decode 최적화**: device 1.34x 페널티 해소 — NEON dequant-attention fast path(ADR-0007 D4 per-format 인코더/attention 확장). 북극성 plugin 의 production 실용성 향상.
3. **device GPU/rpcmem 경로**: 본 검증은 `-b cpu`(dlopen feasibility 격리). opaque KV 의 `opencl --opencl-rpcmem` 경로(standard_format opaque arm → `backend.attention_gen` GPU f32 flash)는 미검증.

---

## Landmines / 미해결

- **모델 파일 제약(opaque KV 무관)**: opaque e2e 모델 선택 시 — (1) **혼합 양자화 GGUF 불가**: `llama3.2-3b-q4_0.gguf` 는 Q6_K 텐서 포함 → `Error: GGUF: Q6_K type is not supported`. (2) **Q4_0 embedding 불가**: `llama3.2-3b-q4_0-pure.gguf` 는 token_embd 가 Q4_0 → `Error: gather: unsupported src dtype Q4_0`(opaque KV alloc 후 첫 embedding lookup 에서 실패, KV 와 무관). qwen2.5 계열(project `convert_safetensors_to_gguf.py`)은 embed **F16** 유지라 동작. **device 모델 준비 시 동일 제약**.
- **opaque ≠ typed bit-identical**: 위 ⑥ 설명. 회귀 가드는 ④(전 opaque 경로 동일성) + `gate_c_plugin_bundle` floor byte-identity(mf_q4 round-trip). 실모델 token-identity 게이트는 **의도적으로 미작성**(모델 의존, handoff scope).
- **resilience default-on**: smoke 는 `--no-resilience` 사용(manager 소켓 불요). production 추론은 default-on 이면 transport graceful fallback(`resilience_init.rs:8` warn+Ok(None)) — 검증엔 명시 off 가 깔끔.
- **engine/Cargo.toml drift**(score_readback `[[bin]]`)는 이번에도 **미커밋 유지**. 이 작업 무관.
- `.so` 빌드: `cargo build --release -p example-bundle --features plugin-cdylib` → `target/release/libexample_bundle.so`(v2 엔트리 `register_kv_{stage,format}s_v2` 존재). 단일축 = `-p example-kv-format`.

---

## 참조
- SSOT: `docs/adr/0010-…`(GATE-C ABI) · `docs/adr/0008-…`(opaque KV) · `docs/adr/0005-…`(generic floor D3/D4/D5).
- 코드 앵커: `engine/src/session/bin_setup.rs:37,100-151`(W1/W2 배선 — register_dynamic_plugins + 이름기반 typed/opaque 분기) · `engine/src/pressure/standard_format.rs:356-405`(opaque attention floor) · `engine/src/format/dtype_layout.rs:270`(encode_via_descriptor) · `engine/src/bin/argus_cli.rs`(엔트리).
- 선행 handoff: `handoff_gate_c_v3_multivtable_2026_06_09.md`(deferred (a) 가 본 작업).
