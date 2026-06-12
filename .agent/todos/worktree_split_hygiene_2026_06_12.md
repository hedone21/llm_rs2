# 워크트리 분리 위생 — 공통 회귀 게이트 + rename 상호 배제 규칙

**작성**: 2026-06-12 (PM 초안 + 메인 세션 실측 교정)
**상위 스프린트**: `sprint_kv_roadmap_item34_2026_06_12.md` P5 (분리 위생 3종)
**진입 handoff**: `handoff_kv_roadmap_item34_entry_2026_06_12.md`
**목적**: KV 로드맵 항목 3+4 ✅ 완료 = "KV 구조 확정" 게이트 도달. 직후 사용자가 별도 워크트리에서 대형 리팩토링을 병렬 시작한다. 양 브랜치(메인 KV/QCF 트랙 + 리팩토링 워크트리)가 안전하게 머지되도록 **분기 전 공통 기준을 문서로 고정**하는 것이 본 문서.

---

## TL;DR

분리 위생 3종 중 ②③을 본 문서로 고정한다(① origin 푸시는 메인 세션 수행).

- **②** 양 브랜치 머지 판정 = (a) host lib + (b) α-K frozen 3-dtype byte-identical + (c) S25 verify 28/30. 세 항 모두 충족해야 머지.
- **③** 대형 `git mv`/rename은 리팩토링 워크트리 전담. 메인은 KV 표면 동결(QCF_kv 라운드는 예외로 진행 가능). 충돌 시 구조=워크트리·내용=메인.

---

## ① origin 푸시 — 메인 세션 수행 (본 문서 범위 밖)

본 스프린트 미푸시 commit(QCF 세션분 + 측정 스프린트분 + 항목 3 `783bcadd`/`a98cd679` + 항목 4 ADR `70729062` + 본 문서)을 origin/master로 푸시 = 공통 베이스 앵커. 확인: `git log origin/master --oneline | head`.

---

## ② 공통 회귀 게이트 (양 브랜치 머지 판정 기준 — 3항)

세 항 **모두** GREEN이어야 머지 가능. 어느 한 항이라도 신규 실패 시 머지 차단(known-fail 예외는 각 항에 명시).

### (a) host lib — host 단독, GPU 무관

```bash
cargo test -p llm_rs2 --lib          # 신규 실패 0
cargo test -p technique-api          # 20/0
cargo fmt --all --check              # clean
cargo clippy --workspace -- -D warnings   # clean
```

**baseline (2026-06-12 갱신 — 결정적 실패 2건 수정 후)**:
- `cargo test -p llm_rs2 --lib`: 비-OpenCL **결정적 실패 0** (구 baseline 실패 2건 `experiment_schedule_parse_roundtrip`/`protected_prefix_score_based_defaults_to_4`는 2026-06-12 fixture 수정으로 해소). 카운트 제외 2종: ① OpenCL 환경 실패(`backend::opencl::*` ~21 + `memory::opencl::unified` SIGABRT — 호스트 POCL 한계, backlog P2-chore (b); 실행 시 `-- --skip backend::opencl --skip memory::opencl` 권장) ② `kv_cache` RSS flaky 2건(`test_prune_prefix_calls_release_unused_pages`/`test_release_unused_pages_rss_reduction` — 병렬 교란 간헐 FAIL, 격리 `--test-threads 1` PASS면 무시).
- `cargo test -p technique-api`: **20/0**.

**판정**: 비-OpenCL 결정적 **실패 0**. 카운트 제외 2종(OpenCL 환경 + RSS flaky 격리-PASS)은 양 브랜치 공통.

### (b) α-K frozen 3-dtype byte-identical (S25) — forward happy path 무회귀

**정본 절차**(실행 명령·모델): `frozen_baseline_alpha_k_5f_2026_06_05.md`.

**⚠ 재앵커 (2026-06-12, P3 발견)**: frozen 문서의 sig md5 3종(`304f4ada`/`684d01d9`/`1cfba273`)은 **추출 방식이 재현 불가**(추출 스크립트 미기록) — md5 사전계산값 의존을 폐기하고 아래로 고정한다.

- **비교 대상 (영속 보존본, repo 커밋)**: `.agent/measurements/frozen_baseline_alpha_k_5f/blA_argus_{f16,f32,q4}.out`
  - 무결성 md5 (whole-file): f16 `0cef6d9dfd4440d282f635b0d8ec71b5` / f32 `229533b53d1454b3c285c67355f43d53` / q4 `c301321bcfb8e9c3b6ab46ee0c12c542`
  - 사본 원천: device `/data/local/tmp/blA_argus_*.out` (2026-06-05 frozen 캡처 보존본, 2026-06-12 P3에서 회수)
- **비교 방식**: **결정론 라인 직접 byte 비교** — 생성 텍스트 라인 + generated summary 라인만 추출해 diff. (timing/프로파일 라인은 비결정이라 제외)

```bash
# 1. 새 실행 (S25 galaxy_s25, opencl --opencl-rpcmem, 6T) — frozen 문서 명령 그대로, dtype만 교체
#    → /tmp/g3_new_{f16,f32,q4}.out

# 2. 결정론 라인 추출 + 비교 (실측 검증된 앵커, 2026-06-12)
for d in f16 f32 q4; do
  grep -E '^The history|^\[Phase4-4\.5\] generated=' \
    .agent/measurements/frozen_baseline_alpha_k_5f/blA_argus_${d}.out > /tmp/base_${d}.det
  grep -E '^The history|^\[Phase4-4\.5\] generated=' /tmp/g3_new_${d}.out > /tmp/new_${d}.det
  diff /tmp/base_${d}.det /tmp/new_${d}.det && echo "$d: BYTE-IDENTICAL" || echo "$d: MISMATCH(머지 차단)"
done
```
앵커 설명(실측 포맷): 생성 텍스트 = `The history of computing began with the invention...`로 시작하는 1줄(프롬프트 에코는 `Prompt: `접두라 미매치), summary = `[Phase4-4.5] generated=32 (first=279 + run=31) stopped_by=... final_pos=37`.

- **tbt 게이트**: median Decode ms/tok Δ ≤ **+3%**. frozen baseline median = f16 **54.22** / f32 **54.04** / q4 **53.79** ms/tok → 상한 f16≤55.8 / f32≤55.7 / q4≤55.4.
- **참고 (2026-06-12 P3 G3 실측)**: avg_tbt median = f16 **54.08** / f32 **54.02** / q4 **53.52** (frozen 대비 전부 음수 Δ — 회귀 없음).

### (c) S25 verify 매트릭스

```bash
python verify/verify.py --device galaxy_s25 --model f16,q4   # 28/30 PASS
```

- **판정**: **28/30 PASS**. known-fail 2 = QCF_kv 3층 결함(backlog QCF_kv 항목, 설계 라운드 대기). 이 2건은 **양 브랜치 공통 known-fail** → 머지 차단 아님(신규로 늘면 차단).

---

## ③ rename 상호 배제 규칙

양 브랜치가 동일 파일을 `git mv`하면 머지 충돌이 폭발한다(rename detection 실패 → 양쪽 신규 파일로 인식). 분기 기간 중 아래 규칙으로 충돌 표면을 제거한다.

### R1. 대형 rename/이동은 리팩토링 워크트리 전담
- 모듈 이동, no-`mod.rs` sweep 잔여분(manager/src 8 + manager/tests 1 + crates/qnn_oppkg 2) 등 대형 `git mv`·파일 이동 = **리팩토링 워크트리 전담**.
- 메인(KV/QCF 트랙)은 분기 기간 중 **`git mv`·파일 이동 금지**. 단 **신규 파일 추가는 허용**(이동이 아니므로 충돌 무위험).

### R2. 메인 트랙 KV 표면 동결
분기 기간 중 메인은 아래 KV 표면을 **무변경**(리팩토링이 구조를 옮기는 동안 의미 안정 보장):
- plan ABI: `KVCachePlan` / `PlanAbi`
- format trait: `KVCacheFormat`
- technique-api 표면 (`TensorKind`/`StageCtx`/`TensorHandle` 포함)
- → 항목 2(K/V 비대칭 merge)·항목 5(persistence)·항목 4-impl(read-plan 구현) 전부 **리팩토링 머지 후 착수** (backlog 표기 존재).

### R3. 예외 — QCF_kv 설계 라운드는 메인 진행 가능
- QCF_kv 설계 라운드(`qcf_kv.rs` / estimator / manager policy lua)는 **KV 표면 밖**이라 메인에서 진행 가능.
- 이 영역은 R2 동결 대상이 아님 — 명시적으로 허용.

### R4. 충돌 발생 시 우선순위
- **구조(파일 위치) 정본 = 리팩토링 워크트리.**
- **내용(로직) 정본 = 메인.**
- 즉 머지 시 파일이 이동했으면 리팩토링 위치를 채택하고, 그 위치에 메인의 로직 변경을 재적용한다.

---

## 종결 조건

- ② 게이트 3항 문서화 ✅ (본 문서).
- ③ rename 배제 규칙 문서화 ✅ (본 문서).
- ① origin 푸시 = 메인 세션 수행.
- 세 항 완료 시 = 사용자 병렬 리팩토링 분기 준비 완료 = "KV 구조 확정" 게이트 종결.
