# Handoff — Sprint 1 W-AUF-1 완료 / 다음 단계 W-AUF-1B + W-AUF-2 (2026-05-20)

## Sprint 1 W-AUF-1 — 완료 ✅

`.auf` primary loader 도입 sprint. 6 commits로 C1~C6 모두 종결. 코드/CLI/문서 일관 정렬.

| Commit | 내용 |
|---|---|
| `6adfefbf` | C1 — `AufSource` + `PrimaryFormat` detection |
| `390c31d6` | C2 — secondary_mmap AUF 분리 + `AufViewBuffer` + `load_tensor` zero-copy |
| `fe6c7681` | (chore) cargo fmt 누적분 정리 |
| `15dd27af` | C3 — 3-way dispatch + `resolve_secondary` stub |
| `b2a6a3a7` | C4 — `--primary-variant`/`--primary-dtype`/`--no-self-secondary` + `--secondary-gguf` deprecation warning |
| `f1628858` | C5 — `auf_tool build --tokenizer-config` + `generate --eos-token-id`/`--bos-token-id` |
| (this commit) | C6 — 문서 동기화 (USAGE.md, auf_tool_guide.md, ARCHITECTURE.md, arch/, spec/, AGENTS.md) |

## 게이트 요약

- `cargo build -p llm_rs2` PASS (lib + bins)
- `cargo test -p llm_rs2 --lib --no-default-features` 1154 PASS, 회귀 0
- 신규 unit test 누적 — AUF 관련 106 (C2) + dispatch 2 (C3) + CLI parser 6 (C4) + tokenizer config 6 (C5) = 신규 20건 PASS
- `cargo fmt` clean, 호출처 동작 변화 0건 (re-export로 import path 보존)
- OpenCL feature 켠 1163/1181 (병렬 race로 인한 18건 flaky는 본 sprint와 무관, 단독 실행 시 PASS)

## W-AUF-1B (auf_tool multi-dtype mode) — 완료 ✅ (2026-05-20)

**Retrospective**: auf_tool multi-dtype 인프라는 본 sprint 시작 전에 이미 master에
존재했음. W-AUF-0 inspection 결과 (모든 AUF에 `CAPABILITY_BIT_MULTI_DTYPE=0`)는
누구도 `--dtypes`로 빌드하지 않았기 때문이며, 코드/spec 부재가 아니다. 관련 commits:

- `a87e8b64 feat(auf): introduce v0.2 multi-dtype capability bit + META default_dtype`
- `8b35a911 feat(auf): writer dtype-aware payload + auf_tool --dtypes/--default-dtype`
- `97457f2f fix(auf): correct multi-dtype 1-D tensor handling (ISSUE-E-1)`

CLI 옵션 매핑:
- plan의 `--multi-dtype` ↔ 실제 `--dtypes` (이미 구현)
- plan의 `--default-dtype` ↔ 동일 (이미 구현)
- `CAPABILITY_BIT_MULTI_DTYPE` set + `format_minor = 2`: 자동 (multi_dtype_enabled flag)

**실측 빌드 결과** (commit이 첨부할 수 없는 4.2 GiB 산출물):

```bash
target/release/auf_tool build \
  --input  models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
  --tokenizer        models/qwen2.5-1.5b/tokenizer.json \
  --tokenizer-config models/qwen2.5-1.5b/tokenizer_config.json \
  --output models/qwen2.5-1.5b/qwen2.5-1.5b-multi-dtype.auf \
  --variants cpu_aos \
  --dtypes   q4_0,f16
```

- 출력 파일: `models/qwen2.5-1.5b/qwen2.5-1.5b-multi-dtype.auf` (4.24 GiB)
- format: **v0.2.1**, `capability_opt = 0x0c` (LM_HEAD_PRECOMPUTED_Q4_0 + MULTI_DTYPE_VARIANTS)
- TENSOR_INDEX: 453 dtype entries (255 base tensor × 평균 ~1.78 dtype)
- META.default_dtype = Q4_0
- TOKENIZER: eos=151643 (W-AUF-1 C5의 `--tokenizer-config` 효과로 자동 채움), bos=-1 (Qwen2.5 tokenizer_config에 bos_token=null)
- `auf_tool verify`: 19/19 게이트 PASS (INV-137 shape consistency + INV-138 default_dtype 포함)

본 파일은 **W-AUF-2 self-secondary 자동 활성 검증의 1순위 인프라**.

## 남은 작업 — W-AUF-2 (self-secondary 자동 활성)

**진입 문장**: "W-AUF-2 진행"
**전제**: W-AUF-1B 완료 (multi-dtype AUF 빌드 + 실측 산출물 확보 ✅)
**위험**: M (resolve_secondary stub → 본격 구현 + RpcMem alias 호환)

내용:
1. `SecondaryMmap::from_auf_self_secondary(view, primary_variant_tag, primary_dtype, config)` 신설 — primary가 AUF일 때 같은 view를 secondary로 재포장 (mmap 1회).
2. `resolve_secondary` stub 본격 구현:
   - explicit `secondary_source` 우선
   - AUF primary + multi-dtype/variant capability + `!disable_self_secondary` → self-secondary 자동 활성
3. `AufSource::has_swap_candidate()` 정확화 (multi-dtype 또는 variant 기준)
4. **R8 — Eager-Flattened Adapter 패턴 적용**: `SecondaryWeightsBacking` trait + `GgufBacking`/`AufBacking` 구현. RpcmemSecondaryStore가 trait abstraction에만 의존. SOLID 5원칙 만족. 게이트: Phase 6.5 baseline 대비 S25 swap latency ≤ 5% 변동.

자세한 sub-step: plan 파일 § "W-AUF-2.3 sub-step (Eager-Flattened Adapter)" — 2.3.1~2.3.7.

## 디바이스 검증 (W-AUF-1 → W-AUF-2 진입 전 권장)

호스트 테스트는 1154 PASS이지만 AUF primary path의 실제 디바이스 inference는 미검증.

권장 sanity check (W-AUF-1B 진입 전 또는 직후):
```bash
# Llama 3.2-3B AUF (현 가용)로 AUF primary 정답성 확인
python scripts/run_device.py -d s25 generate -- \
  --model-path /data/local/tmp/llama3.2-3b-q4_0-aos.auf \
  --backend qnn_oppkg \
  --primary-variant cpu-aos \
  --primary-dtype q4_0 \
  --gen 32 \
  --tokenizer-path /data/local/tmp/llama3.2-3b.tokenizer.json
```

기대: 32 토큰 정답 출력 (GGUF primary 대비 bit-identical 또는 의미 동치). Qwen 2.5는 본 sprint v0.1에선 bias 미지원 (`tensor_id_to_auf`이 `LayerBias`를 None 반환) → W-AUF-1B에서 AUF v0.2 도입 시 검증.

## 정책 안내 (문서 동기화 완료)

- 정식 entry: `--model-path foo.auf`
- AUF 빌드: `auf_tool build --tokenizer-config tokenizer_config.json ...`
- `--secondary-gguf`: deprecated alias, stderr 경고 1회. 향후 제거.
- 새 CLI flags: `--primary-variant` / `--primary-dtype` / `--no-self-secondary` / `--eos-token-id` / `--bos-token-id`

상세는 `docs/USAGE.md` §2.13, `docs/auf_tool_guide.md` §2.1/§3.1, `AGENTS.md` "Weight asset 경로 정책" 절 참조.

## Plan 파일

`/home/go/.claude/plans/proud-strolling-whale.md` — W-AUF-1 commit 분할(C1~C6)은 사실상 완결. W-AUF-1B + W-AUF-2 + W-AUF-2.3 (R8) 섹션은 그대로 후속 진입점.

## Worktree

`/home/go/Workspace/llm_rs2/.claude/worktrees/sprint1_auf_loader` (브랜치 `worktree-sprint1_auf_loader`). 본 sprint 종결 후 master에 ff-merge 가능. 다음 sprint는 별도 worktree 권장.
