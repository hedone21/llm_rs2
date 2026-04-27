# AUF v0.2 — Multi-Quant Layer Variants (Single-File Mixed-Precision Swap)

> **목표**: 한 개의 AUF 파일에 동일 layer/tensor에 대한 **두 dtype variant**(예: Q4_0 + F16)를 동시 보관하여, 외부 두 벌 GGUF 없이 layer 단위 동적 weight swap을 수행한다.
> **단방향 swap 가정**: F16 → Q4 (또는 더 압축된 dtype)만. 역 swap은 본 스코프 외. 이 가정 덕에 `SwapExecutor` 인터페이스는 unchanged.
> **연관 브랜치**: `feat/weight` (v0.1.1까지 master 머지 완료, HEAD 근방).
> **작성일**: 2026-04-27.
> **연관 TODO**: [feat_weight_swap.md](feat_weight_swap.md) (Phase 1~5 완료된 동적 swap 인프라 위에 형식 확장).
>
> **Cross-link 요약**:
> - v0.1.x 인프라: `LayerSlot`, `SecondaryMmap`, `TransformerWeights`, `SwapExecutor`, `WeightSwapHandler` 등은 **변경 없음**.
> - v0.2는 **포맷 레이어만** 확장 (TensorIndex entry-level dtype 분기). 런타임 swap 실행 경로는 secondary AUF view에서 **dtype을 골라 mmap**하는 thin wrapper만 추가.
> - 호환성 정책: format_major=0 유지 (기존 reader 호환), format_minor bump (0.1.x → 0.2), capability_optional bit 3 = `MULTI_DTYPE_VARIANTS` 신설.
> - 작업량 추정: **9.5~14.5일** (Sprint A→A'→B→C→D→E 순차, A' 0.5일 추가).

---

## 의사결정 default (확정)

| ID | 옵션 | 결정 | 근거 요약 |
|----|------|------|----------|
| **Q1** | section_tag에 dtype 포함 vs entry-level dtype 필드 | **B (entry-level dtype 필드)** | section_tag 24B 한계 회피. TensorIndex entry 단위로 dtype 분기. |
| **Q2** | lm_head multi-dtype 허용 vs Q4_0 single 고정 | **B (lm_head도 multi-dtype 후보 적용)** | 사용자 결정 변경 (2026-04-27, 옵션 B). Adreno SOA variant 안에서는 dtype 무관하게 AOS layout 강제 (INV-135 v2 의무 그대로 유지). |
| **Q5** | format_major bump vs capability_optional 신설 | **B (capability_optional 신설)** | format_major=0 유지로 v0.1.x reader 호환. bit 3 = `MULTI_DTYPE_VARIANTS`. |

---

## 핵심 제약사항

1. **단방향 swap**: F16→Q4만. 역방향 경로는 본 스코프 외.
2. **lm_head 정책 (사용자 결정 변경, 2026-04-27)**: lm_head도 multi-dtype 후보 적용 (옵션 B). 단 Adreno SOA variant 안에서는 dtype에 무관하게 **AOS 18B/block layout 강제** (INV-135 v2 의무 그대로 유지). 즉, dtype 다양화는 허용하되 layout 다양화는 금지.
3. **Backward compatibility**: v0.1.1 reader는 v0.2 AUF의 single-dtype subset 부분(예: lm_head)만 정상 로드 가능해야 함. capability_optional bit 3 미지원 시 multi-dtype layer는 reader에서 거부.
4. **Forward compatibility**: v0.2 reader는 v0.1.1 AUF (single dtype only)도 정상 로드 가능해야 함.
5. **schema_version=1 유지**: TensorIndex schema breaking change 금지. dtype 필드는 호환적 확장.
6. **section_tag 24B 한계**: tag 자체 변경 없음 (entry-level 분리 방식).
7. **SwapExecutor 인터페이스 unchanged**: 단방향 가정 덕에 기존 v0.1.x 인프라 재사용.

---

## Phase 구조 요약

| Sprint | 내용 | 예상 기간 | 담당 | 의존성 |
|--------|------|----------|------|--------|
| **A** | Spec 정의 + 호환성 정책 | 2~3d | Architect | — |
| **A'** | lm_head 정책 변경 spec 수정 (옵션 B 반영) | 0.5d | Architect (재위임) | A |
| **B** | section/TensorIndex schema 확장 | 2~3d | Implementer | A' |
| **C** | Writer / auf_tool 확장 (dequant→requant) | 3~4d | Senior Implementer (핫패스), Implementer (CLI/glue) | B |
| **D** | Reader / Secondary mmap 통합 | 1~2d | Implementer | C |
| **E** | 호환성 검증 + 디바이스 측정 | 1~2d | Tester | D |

**진행 순서 강제**: A → A' → B → C → D → E. 병렬화 없음.

---

## [P1] Sprint A — Spec 정의 + 호환성 정책

- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (v0.1.1 머지 완료 기반)
- **담당**: Architect
- **예상 기간**: 2~3일
- **Description**: AUF v0.2 multi-dtype variant 포맷의 spec/arch 문서를 정의하고, v0.1.x ↔ v0.2 양방향 호환성 정책을 명세한다.

### Sub-task 체크리스트

- [ ] `spec/33-engine-data.md §3.22` multi-dtype TensorIndex entry schema 추가
  - schema_version=1 호환적 확장 (필드 추가만, 기존 필드 의미 변경 금지)
  - dtype field, payload_offset/payload_size dtype별 분기 명세
- [ ] `spec/32-engine-algorithms.md §3.12.17` writer/reader dtype 분기 알고리즘 명세
  - writer: dequant→requant 파이프라인 (F16 GGUF 입력 → Q4_0 entry 추가, Q4 GGUF 입력 → F16 entry 추가)
  - reader: AufView dtype select → secondary mmap 분기
- [ ] `spec/41-invariants.md §3.16~3.17` 새 INV 1~3개 추가
  - 예: INV-multi-dtype-001 "동일 (layer_idx, tensor_kind) 쌍에 대해 dtype variant entry는 최대 N개"
  - INV-multi-dtype-002 (Sprint A 초안) "lm_head는 Q4_0 single entry 강제" → **Sprint A'에서 갱신 예정** (lm_head도 multi-dtype 허용, 단 Adreno SOA variant에서 AOS layout 강제)
  - INV-multi-dtype-003 "multi-dtype entry의 payload_size는 dtype별 element_count × bytes_per_element 일치"
- [ ] `arch/auf_format.md §1.3 / §2.5b` update + Mermaid 다이어그램 갱신
  - TensorIndex entry layout 다이어그램에 dtype field 추가
  - multi-dtype payload section 배치 시각화
- [ ] format_minor bump (예: 0.1.2 → 0.2 또는 0.2.0)
- [ ] capability_optional bit 3 = `MULTI_DTYPE_VARIANTS` 신설
  - `READER_KNOWN_CAPABILITIES`에 추가 예정 명세 (구현은 Sprint B)
- [ ] v0.1.x ↔ v0.2 양방향 호환 동작 명세
  - **v0.1.1 reader × v0.2 AUF**: capability_optional MULTI_DTYPE_VARIANTS 미지원 → multi-dtype layer는 reject (또는 첫 dtype만 노출 — 결정 필요)
  - **v0.2 reader × v0.1.1 AUF**: dtype 필드 부재 시 default dtype (section_tag 기반 추론)으로 동작

### Acceptance Criteria

- spec/arch 문서 수정 완료 + 내부 cross-reference 일관성 (ENG-DAT-XXX, ENG-ALG-XXX, INV-XXX ID 할당)
- v0.1.x reader가 v0.2 AUF의 single-dtype subset을 어떻게 처리하는지 명확히 기술
- format_minor bump 정책 + capability bit allocation table 업데이트
- 9~14일 추정의 Sprint A 부분(2~3일)이 spec 명세로 마감

---

## [P1] Sprint A' — lm_head 정책 변경 spec 수정 (옵션 B 반영)

- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Sprint A 완료
- **담당**: Architect (재위임)
- **예상 기간**: 0.5일
- **Description**: 사용자 결정 변경(2026-04-27, 옵션 B)에 따라 Sprint A에서 작성한 lm_head 관련 spec/arch 항목을 갱신한다. lm_head도 multi-dtype 후보로 적용하되, Adreno SOA variant 내에서는 dtype 무관하게 AOS 18B/block layout을 강제한다 (INV-135 v2 의무 보존).

### Sub-task 체크리스트

- [ ] `spec/41-invariants.md` INV-multi-dtype-002 갱신
  - 기존: "lm_head는 Q4_0 single entry 강제"
  - 변경: "lm_head는 multi-dtype variant 허용. 단 variant_tag = adreno_soa인 경우 dtype에 무관하게 AOS 18B/block layout 강제 (INV-135 v2 cross-reference)"
- [ ] `spec/33-engine-data.md §3.22` lm_head 처리 절 갱신
  - lm_head entry도 일반 layer weight와 동일한 dtype 분기 파이프라인 적용
  - Adreno SOA variant 내 layout 강제 조건 명시
- [ ] `spec/32-engine-algorithms.md §3.12.17` writer 알고리즘 절 갱신
  - lm_head를 multi-dtype 변환 대상에 포함
  - Adreno SOA payload 구성 시 lm_head dtype-별 entry 모두 AOS bytes로 강제
- [ ] `arch/auf_format.md` 다이어그램/본문 lm_head 예외 표현 제거 또는 layout 제약으로 재표현
- [ ] ENG-DAT-C16 (또는 lm_head 관련) cross-reference 갱신
  - 기존 "lm_head Q4_0 single" 표기 → "lm_head multi-dtype + Adreno SOA AOS 강제" 표기

### Acceptance Criteria

- INV-multi-dtype-002 갱신 완료, INV-135 v2와 cross-link 일관
- spec/33, spec/32, arch/auf_format.md lm_head 관련 본문이 옵션 B 반영
- Sprint B 진입 시 schema 변경 범위 (lm_head도 dtype 분기 entry 보유 가능) 확정
- ENG-DAT-XXX cross-reference 일관성 유지

### Notes

- 사용자 결정 plan 파일: `/home/go/.claude/plans/zazzy-herding-bonbon.md`
- INV-135 v2 (Adreno SOA에서 lm_head AOS 18B/block 강제)는 layout 차원 의무이므로, dtype 다양화와는 직교한다. 다만 동일 dtype × 다른 layout 조합은 본 스코프 외(여전히 Adreno SOA = AOS 단일 layout).

---

## [P1] Sprint B — section / TensorIndex schema 확장

- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Sprint A' 완료 (옵션 B 반영된 spec 기반)
- **담당**: Implementer
- **예상 기간**: 2~3일
- **Description**: TensorIndex entry에 dtype 필드를 추가하고, header reader의 capability_optional 처리를 확장한다. Writer/Reader 양쪽 host-side unit test 통과.

### Sub-task 체크리스트

- [ ] section_tag 24B 한계 검토 (entry-level 분리 방식이라 tag는 그대로 OK 확인)
- [ ] `tensor_index.rs` entry struct에 dtype field 추가
  - schema_version=1 유지 (호환적 확장)
  - serialize/deserialize round-trip 보장
- [ ] `header.rs` `READER_KNOWN_CAPABILITIES`에 `MULTI_DTYPE_VARIANTS` (bit 3) 추가
- [ ] `header.rs` `has_multi_dtype()` helper 추가
- [ ] 호스트 단위 테스트
  - `round_trip` (TensorIndex serialize→deserialize, dtype 필드 보존)
  - `find_lm_head_entry` (lm_head multi-dtype entry lookup 정상; Adreno SOA variant 내에서는 dtype 무관하게 AOS layout entry로 lookup)
  - `multi_dtype_variant_lookup` ((layer_idx, tensor_kind, dtype) 3-tuple lookup 정상)
  - `v01_compat_read` (v0.1.1 AUF 파일을 v0.2 reader로 읽기 회귀)
  - `capability_unknown_reject` (bit 3 set이지만 reader unknown 시 reject)

### Acceptance Criteria

- 신규 unit test 5건 이상 통과
- v0.1.1 회귀 테스트 모두 통과 (기존 AUF 파일 정상 read)
- cargo fmt + clippy clean
- format_minor bump 반영

---

## [P1] Sprint C — Writer / auf_tool 확장 (dequant→requant 파이프라인)

- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Sprint B 완료
- **담당**: Senior Implementer (dequant→requant 핫패스), Implementer (CLI/glue)
- **예상 기간**: 3~4일 (가장 무거운 sprint)
- **Description**: auf_tool에 `--dtypes` 옵션을 추가하고, F16 GGUF 입력에서 Q4_0 entry를 추가하거나 Q4 GGUF 입력에서 F16 entry를 추가하는 dequant→requant 변환 파이프라인을 구현한다.

### Sub-task 체크리스트

- [ ] `BuildArgs`에 `--dtypes q4_0,f16` 옵션 추가 + parsing
  - 다중 dtype 지정 시 default primary dtype 결정 정책 (예: 첫 번째)
- [ ] `extract_weight_blobs`를 dtype별로 호출 가능하도록 분리
  - F16 GGUF → Q4_0 추가 entry: F16 dequant → Q4_0 quant 파이프라인 (Senior Implementer)
  - Q4 GGUF → F16 추가 entry: Q4_0 dequant → F16 변환 (Senior Implementer)
  - **lm_head도 layer weight와 동일한 dtype 변환 파이프라인 적용** (사용자 결정 변경, 옵션 B). 단 Adreno SOA variant 내에서는 모든 dtype entry의 bytes를 **AOS 18B/block layout으로 강제** (INV-135 v2). 다른 variant (e.g., generic)에서는 dtype-default layout 사용.
- [ ] `build_variant_payload(blobs, variant_tag, dtype)` 시그니처 확장
  - dtype별 payload section 분리 또는 통합 (포맷 결정은 Sprint A/A' 명세 따름)
  - **Adreno SOA variant + lm_head**: dtype 무관하게 AOS bytes로 강제 직렬화. variant_tag 검사 후 lm_head tensor_kind에 한해 layout override.
- [ ] `build_tensor_index` 다중 dtype payload offset 추적 로직
  - 동일 (layer_idx, tensor_kind) 쌍에 대해 dtype별 payload_offset/payload_size entry
- [ ] auf_tool `info` / `verify` 서브커맨드 multi-dtype 출력 지원
  - 각 layer별 사용 가능한 dtype variant 표시
  - capability_optional MULTI_DTYPE_VARIANTS 비트 set 여부 표시
- [ ] integration test
  - `auf_tool build --variants adreno_soa --dtypes q4_0,f16 --output mixed.auf` 정상 동작
  - `auf_tool verify mixed.auf` 검증 통과
  - `auf_tool info mixed.auf` 다중 dtype 출력 확인

### Acceptance Criteria

- F16 GGUF 입력 → multi-dtype mixed.auf 생성 정상
- Q4_0 GGUF 입력 → multi-dtype mixed.auf 생성 정상
- lm_head multi-dtype entry 정상 생성 (INV-multi-dtype-002 v2 만족), Adreno SOA variant 내 lm_head는 dtype과 무관하게 AOS 18B/block bytes (INV-135 v2 만족)
- auf_tool info/verify multi-dtype 표시 정확
- dequant→requant numerical 검증 (host CPU 단위 비교 토큰 일치)
- cargo fmt + clippy clean

### Notes

- Senior Implementer는 dequant→requant 핫패스에서 NEON/SIMD 최적화 검토 (변환 자체가 무거우므로 host-side 단발 변환은 단순 구현 우선, 디바이스 측 변환은 본 스코프 외).
- Q4_0 단방향 변환은 lossy. dequant→requant round-trip은 비결정적이지 않아야 함 (블록 단위 결정성 보장).

---

## [P1] Sprint D — Reader / Secondary mmap 통합

- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Sprint C 완료
- **담당**: Implementer
- **예상 기간**: 1~2일 (단방향 가정으로 단순화)
- **Description**: AufView에 dtype-aware weights_range를 추가하고, generate.rs CLI에 `--secondary-dtype` 플래그를 도입한다. SwapExecutor는 unchanged 확인.

### Sub-task 체크리스트

- [ ] `AufView::weights_range(backend, dtype)` 분기 구현
  - 기존 `weights_range(backend)` API는 backward-compat용으로 첫 dtype 또는 default dtype 반환
- [ ] `build_auf_secondary_from_view(view, primary_config, dtype_select)` 구현
  - primary와 다른 dtype 1개 자동 선택 (auto mode)
  - 또는 CLI override (`--secondary-dtype q4_0|f16`)
- [ ] **SwapExecutor unchanged 확인** (인터페이스 변경 없음 — 단방향 가정 덕)
  - 기존 swap 경로는 그대로 secondary view에서 mmap range 가져오는 식
- [ ] `generate.rs` `--secondary-dtype <auto|q4_0|f16>` 플래그 추가
  - 기본값 `auto` (primary와 다른 첫 dtype 자동 선택)
- [ ] secondary mmap zero-copy 동작 확인 (UMA 경로)

### Acceptance Criteria

- primary=F16 + secondary=Q4_0 (mixed.auf 1개 파일에서) 정상 swap
- SwapExecutor 코드 변경 0줄 (인터페이스 unchanged 검증)
- `--secondary-dtype auto` / `q4_0` / `f16` 모두 host smoke 정상
- v0.1.1 AUF (single dtype) + v0.2 reader 회귀 정상 (Sprint B에서 이미 검증된 경로 재확인)
- cargo fmt + clippy clean

---

## [P1] Sprint E — 호환성 검증 + 디바이스 측정

- **Status**: TODO
- **Sprint**: current
- **Dependencies**: Sprint D 완료
- **담당**: Tester
- **예상 기간**: 1~2일
- **Description**: v0.1.1 ↔ v0.2 양방향 호환성 fuzz/회귀 테스트, S25 디바이스 mixed-dtype dynamic swap 측정, 운영 문서 갱신.

### Sub-task 체크리스트

- [ ] **v0.1.1 reader × v0.2 AUF 호환성 fuzz test**
  - capability_optional MULTI_DTYPE_VARIANTS 미지원 시 적절히 reject 또는 single dtype subset만 노출
  - lm_head Q4_0 single entry는 정상 read
- [ ] **v0.2 reader × v0.1.1 AUF 회귀 test**
  - 기존 v0.1.1 AUF 파일이 v0.2 reader에서 정상 동작 (forward compat)
- [ ] **S25 디바이스 측정**: mixed.auf 1개로 layer 0~7=Q4 + layer 8~15=F16 dynamic swap
  - INV-122 v2.1 임계값 만족 확인
  - swap latency / TBT / PSS 측정
  - 정확성 (token match) 회귀 없음
  - **mixed.auf의 lm_head F16 entry도 정상 lookup되는지 확인** (Adreno SOA variant 내, AOS 18B/block layout 검증). lm_head F16/Q4_0 entry 양쪽 모두 read 가능 + 출력 token 정확성 동일.
- [ ] `convert_to_auf.sh` `--dtypes` 옵션 추가
- [ ] docs 갱신
  - `docs/USAGE.md §2.13` multi-dtype AUF 사용법
  - `docs/auf_tool_guide.md` `--dtypes` 옵션 + info/verify 다중 dtype 출력 예시
  - `docs/auf_format_changelog.md` v0.2 변경사항 (format_minor bump, capability bit 3, schema_version=1 유지)
- [ ] notify-send "llm.rs" "AUF v0.2 multi-quant complete"

### Acceptance Criteria (Definition of Done)

- v0.1.1 reader × v0.2 AUF 호환성 PASS
- v0.2 reader × v0.1.1 AUF 회귀 PASS
- S25 디바이스 mixed-dtype dynamic swap 측정 PASS (INV-122 v2.1 임계값 충족, 정확성 회귀 없음)
- convert_to_auf.sh `--dtypes` 옵션 동작
- docs 3종 갱신 완료
- 전체 cargo fmt + clippy clean
- 신규/회귀 unit + integration test 모두 PASS

---

## Definition of Done (전체)

다음 모든 조건이 충족되어야 v0.2 multi-quant AUF 작업 종결로 간주한다.

1. **Spec/Arch**: spec/33, spec/32, spec/41, arch/auf_format.md 갱신 완료. INV ID 할당 + cross-link 일관성.
2. **포맷 호환성**:
   - format_major=0 유지 (기존 reader 호환)
   - format_minor bump (예: 0.2.0)
   - capability_optional bit 3 = MULTI_DTYPE_VARIANTS 신설
   - schema_version=1 유지 (TensorIndex 호환적 확장만)
3. **양방향 호환**: v0.1.1 reader × v0.2 AUF / v0.2 reader × v0.1.1 AUF 모두 검증.
4. **lm_head 정책 (옵션 B, 사용자 결정 2026-04-27)**: lm_head도 multi-dtype 후보 적용. 단 Adreno SOA variant 안에서는 dtype에 무관하게 **AOS 18B/block layout 강제** (INV-135 v2 의무 그대로 유지, OpenCL image 한계 회피). dtype 다양화 ≠ layout 다양화.
5. **SwapExecutor unchanged**: 런타임 인프라 코드 변경 0줄 (단방향 가정 검증).
6. **Writer**: auf_tool `--dtypes q4_0,f16` 옵션 동작. dequant→requant 파이프라인 결정성 보장.
7. **Reader**: AufView dtype-aware. generate.rs `--secondary-dtype` 플래그 동작.
8. **디바이스 측정 PASS**: S25에서 mixed.auf 1개로 layer 0~7=Q4 + layer 8~15=F16 dynamic swap 정상. INV-122 v2.1 임계값 충족, 정확성 회귀 없음.
9. **문서**: docs/USAGE.md §2.13, docs/auf_tool_guide.md, docs/auf_format_changelog.md 갱신.
10. **품질 게이트**: cargo fmt + clippy + 전체 cargo test PASS.
11. **커밋 컨벤션**: Conventional Commits, sprint별 별도 커밋 권장.

---

## 참고 / Cross-link

- v0.1.x 인프라(`LayerSlot`, `SecondaryMmap`, `TransformerWeights`, `SwapExecutor`, `WeightSwapHandler`, AUF v0.1 self-contained format)는 [feat_weight_swap.md](feat_weight_swap.md) 참고.
- AUF v0.1 format 명세 + section table 구조는 `arch/auf_format.md` (Sprint A에서 §1.3/§2.5b update 예정).
- INV-135 v2 (lm_head AOS 18B/block, OpenCL image 한계) 근거: Phase 6 Sprint G-1-F fix.
- 단방향 swap 가정의 정당성: 메모리 압력은 일방향(F16→Q4)이며, 압력 해제 시 reload는 본 프로젝트 스코프 외(향후 별도 sprint).
- **사용자 결정 plan 파일**: `/home/go/.claude/plans/zazzy-herding-bonbon.md`
- **lm_head 정책 변경 시점**: 2026-04-27 (옵션 A "Q4_0 single 유지" → 옵션 B "multi-dtype 후보 적용 + Adreno SOA AOS layout 강제"). Sprint A 직후 Sprint A'에서 spec 갱신 처리.

## Open Question (결정 필요)

- **lm_head Q4_0 single vs multi-dtype**: **[해결]** 사용자 결정 (2026-04-27, 옵션 B): multi-dtype 적용. Adreno SOA에서는 dtype 무관하게 AOS layout 강제 (INV-135 v2).
- (그 외 open question은 현재 없음 — Q1/Q2/Q5 모두 default 표에서 결정 완료.)
