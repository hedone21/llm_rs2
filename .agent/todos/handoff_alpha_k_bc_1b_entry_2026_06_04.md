# Handoff: α-K BC step-① → ①-b (forward_into prefill batch entry)

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `df867a4e docs(todo): BC roadmap Step 1 — sub-순서 정정 + C3 완료 반영`
**브랜치**: `master` (origin push 미실행 — 사용자 요청 시)
**다음 세션 진입 문장**: **"BC ①-b 진행"**

> 전체 roadmap = `.agent/todos/roadmap_alpha_k_bc_completion_2026_06_04.md` (5 step). 설계 SSOT = `arch/pipeline_stage_design_v2.md` §9.1-BC1'(step-① 재분해) + §9.1 "⚠️ α-K (3p)/(4) 방향"(BC 확정). ADR = `docs/adr/0001-kv-dispatch-paradigm.md` §8.3/§6.5. 트랙 메모리 = [[project-pipeline-alpha-k]].

---

## TL;DR
α-K (4) `KVCacheOps` 완전 폐기로 가는 **BC 완주** 결정됨(사용자 2026-06-04, B-5 legacy frozen 충돌을 "legacy disposable"로 해소). step-① 의 **①-a(phantom/이미 완료) + C3(write_kv_batch GPU batch scatter, host 완료 `2e6b50fb`)** 종결. 다음 = **①-b: `forward_into_fmt` 에 multi-token prefill batch entry 추가 + `write_kv_batch` 배선**. **왜 멈췄나**: ①-b 는 `forward_gen_fmt` 의 `seq_len==1` 하드코딩 해소(multi-token QKV/RoPE/attention 필요분 점진)라 **Architect 설계 라운드 + Senior 구현 + S25 device 게이트**가 필요한 fresh substantial 증분 → 세션 경계 분리.

---

## 진행 상태 (검증된 게이트)
| 증분 | 상태 | commit | 게이트 |
|---|---|---|---|
| BC 방향 결정 (B-5 해소, ④-a vtable 0) | ✅ | `b0be9ce4`·`1b6a234a` | 사용자 확정 |
| BC roadmap + step-① 설계 | ✅ | `1e5ca5d9`·`df867a4e` | — |
| ①-a (write_kv/attention_into decode seam) | ✅ phantom | (3a/3b `5ea8ad47`/`3bc03e59` + 3c-fwd `c2b05aff`) | 이미 device PASS |
| **C3 (write_kv_batch GPU prefill batch scatter)** | ✅ host | `2e6b50fb` | **host**: build + `standard_format` 12/12 + fmt + clippy clean (메인 재검증). **device bit-identical = ①-b 후** |
| **①-b (prefill batch entry, C1)** | ★다음 | — | device |
| ①-c eval flip / ①-d 비-decode 잔여 | 대기 | — | host(①-b follow-on) |

(3p)=B-1·offload=B-3·eval=B-4·legacy 폐기=Step 5 는 roadmap 참조. **순서 = forward_into flip(root) → eval/ppl/batch/warmup(follow-on)** — eval-first 아님(아래 R6).

---

## 다음 작업 (①-b)
1. **Architect 설계** → `forward_into_fmt`(transformer.rs:~2015, 현 `debug_assert_eq!(seq_len,1)`)에 multi-token prefill batch 경로 cut. `forward_gen_fmt`(forward_gen_fmt.rs, seq_len=1 하드코딩) multi-token 확장 형태 — **필요분만 점진**(C2 prefill 부가기능 collector/partition/chunk 추측성 선반영 금지, CLAUDE.md §2). `write_kv_batch`(C3 완료) 배선.
2. **Senior 구현** → `ModelForward::prefill`(model_forward.rs:355)·`run_chunked_prefill`(session/prefill.rs) prefill 호출처를 fmt batch entry 로 전환. 검증: host build/test/fmt/clippy.
3. **S25/Jetson device 게이트** (Tester/deploy-test) → `--no-gpu-plan` prefill bit-identical (F16/Q4_0/**F32-device-only** carve-out) + avg_tbt Δ≤+3%(prefill TTFT 회귀 부재). **C3 의 GPU batch scatter(F32/F16)도 이 게이트에서 함께 검증**(①-b 가 write_kv_batch 를 live 발화시키므로).

---

## Landmines / 미해결 (R6)
- **C3 GPU scatter 경로 host 미검증** — host GPU 부재로 `write_kv_batch` 의 GPU scatter 분기(F32/F16) 자체는 미실행(host 테스트는 CpuBackend cast 경로만 커버). ①-b 배선 후 **device 게이트가 첫 실증** — bit-identical 깨지면 C3 의 batch scatter dst_off/advance_pos 회계부터 의심.
- **`forward_gen_fmt` seq_len==1 하드코딩** — q/k_rope shape `[batch,1,n_heads,head_dim]`, 단일-토큰 LayerWorkspace. multi-token 확장이 ①-b 의 본체. dead branch(partition/fused) 생략 유지.
- **bit-identical carve-out (3c-fwd 교훈)** — F16/Q4_0/F32-device-only만 bit-identical. **F32+host-mapped 는 not bit-identical**(별도 사전 이슈).
- **의존 방향 역전** — eval(①-c)/ppl/batch/warmup(①-d)은 forward_into flip(①-b) **이후** thin follow-on. eval-first 로 착수 금지(eval 의 `C`=forward_into cache 타입, SSOT §9.1-BC1' ★반증 1~3).
- **device 게이트 = S25 연결 필요** (legacy_generate 가 현 device-gate bin; argus_cli 이주는 Step 4).
- **cargo authoritative** — subagent 자기보고/IDE 진단 불신, 메인 세션이 `cargo build/test/clippy` 직접 재검증 후 커밋.
- **커밋 금지 untracked**: `.antigravitycli/`·`.claude/scheduled_tasks.lock`·`papers/.../microbench_*`·`.agent/todos/handoff_microbench_*`·`arch/pipeline/`. 명시 파일만 add(`git add -A` 금지).

---

## 자기점검
- 진입 문장 한 줄 시작 가능? ✓ "BC ①-b 진행"
- 왜 멈췄나? ✓ ①-b = forward_gen_fmt multi-token 확장 = Architect 설계+Senior+device 라운드, 세션 경계
- 최대 landmine? ✓ C3 device 미검증(①-b 게이트가 첫 실증) + forward_gen_fmt seq_len==1
- 검증 게이트 수치/명령? ✓ host(12/12 등) + device(--no-gpu-plan bit-identical + Δ≤+3%)
- 길이 적정? ✓ 상세는 roadmap/SSOT 링크
