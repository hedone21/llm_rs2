# Gemma 3 4B Support — Final Status (Phase 4 complete)

Date: 2026-04-14T11:48:05+09:00
Branch: feat/gemma3-4b-support
Base: master

## Commit chain (master..HEAD)
```
68f0bed docs(gemma3-4b): Phase 3 Task 10 — OpenCL validation status
61b4e18 docs(gemma3-4b): Phase 3 Task 8 — no crash on original 2-shard path
5926c45 fix(config): scope Gemma3 text_config defaults to Gemma3 arch only
c4e9f64 test(gemma3-4b): integration smoke for multimodal config + mapper
e1e7252 feat(config): apply Gemma3 text_config defaults for multimodal wrappers
5b07e36 refactor(loader): extract missing_tensor_err helper (DRY)
c388aa2 feat(loader): use prefix-aware mapper factory; diagnostic error for missing tensors
b6e014d feat(mappers): prefix-aware WeightMapper factory for multimodal wrappers
9201724 fix(config): guard nested multimodal wrapper; surface arch name on missing text_config
da9981a feat(config): add weight_prefix and text_config flatten for multimodal wrappers
361fedc chore(gemma3-4b): baseline — workaround config + crash on 2nd eval
```

## Verified
- Unit: 8/8 models::config::tests including 5 new tests
- Mapper: 5/5 models::mappers::tests including 2 new tests
- Lib (skip convert, unified_buffer): 875/0
- Integration: gemma3_4b_loading 2/2 PASS on original 2-shard + wrapper config
- CPU eval-ll on original path: 40/40 numeric PASS (447s)
- OpenCL eval-ll (PoCL, CPU-emulated): 40/40 numeric PASS (655s). 단일 프롬프트 정상 생성. 엔진 inter-question 상태 관리 정확성 확인.
- OpenCL eval-ll (NVIDIA CUDA, RTX 3090 Ti): exit 0, 40 항목 처리, Q1–Q4 NLL은 CPU와 일치, **Q5 이후 NaN**. 단일 프롬프트 blank (EOS 즉시). 원인: NVIDIA OpenCL 1.2 JIT가 Adreno 특화 커널 9개 컴파일 실패 후 fallback 실행 시 상태 오염. **4B 전용이 아닌 pre-existing 호스트 드라이버 상호작용 이슈** (1B도 Q5+ `CL_OUT_OF_RESOURCES`). 별도 이슈 `issues/host_opencl_nvidia_fallback_20260414.md` / 브랜치 `fix/host-opencl-nvidia-fallback` 예정. — Task 10 raw data: `notes/gemma3_4b_task10_status.md`
- 1B regression: Llama/Gemma3/Qwen2 all generate sensible text
- cargo fmt clean; no new clippy errors; 4 minor pre-existing style warnings unchanged

## Known non-blocking
- NVIDIA host OpenCL: Q5+ NaN on eval-ll, blank single-prompt (pre-existing host-driver JIT fallback issue, not 4B-specific; tracked in `issues/host_opencl_nvidia_fallback_20260414.md`)
- head_dim=256 flash_attn kernel variant missing → prefill attention falls back to CPU on OpenCL (future optimization)
- test_f32_to_f16_roundtrip SIGABRT (pre-existing unrelated)
- unified_buffer tests fail on non-OpenCL host (pre-existing unrelated)
