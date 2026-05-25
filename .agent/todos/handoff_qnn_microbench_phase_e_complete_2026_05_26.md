# Handoff: μ-Q1 sprint 종결 — QNN microbench Phase A~E 완료

**작성**: 2026-05-26
**HEAD**: `ec2e5d8e feat(microbench): Phase D Executorch HTP path + single-shot 측정` (이후 phase D.4 v2 + Phase E 통합 commit 예정)
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "Q-2 진입 — HTP backend trait `engine/src/backend/qnn_htp/` 신설"

---

## TL;DR

μ-Q1 sprint **completed**. 4-cell paper-grade conclusion 확보:
- **M7 (HTP W8A8) 0.276 ms** = OpenCL F16 raw 의 0.47× (2.13× 빠름)
- **M6b (HTP F16) 0.456 ms** = OpenCL F16 raw 의 0.77× (1.29× 빠름)
- M3 (OpenCL F16 raw) 0.589 ms — baseline
- M4 (QNN-GPU OpPackage) 1.529 ms — worst (Phase R reproducible)

→ 결정 게이트 (M7/M3 = 47%) 통과 → **Q-2 backend trait 신설 권장**.

미완:
- M5 (OpenCL Q4_0 latency 측정 bin) — 별 sprint
- M1/M1b/M2 (HTP raw via libloading) — #195 segfault risk 미확정. Executorch path 가 동작했으므로 device-side OK, raw API path 만 검증 필요.

---

## 진행 상태

| Task | 상태 | Commit / 위치 |
|---|---|---|
| A.1 `microbench_qnn_matrix.py` 작성 | ✅ | `a76e4a79` |
| A.2 S25 zone naming (8 zone: cpu+gpuss+nsphvx+nsphmx+ddr) | ✅ | (인-script) |
| A.3 dry-run protocol 검증 (M3 CV 1.71%) | ✅ | `7cdd69cf` |
| C M3/M4 production rounds=10 측정 | ✅ | `708d1b52` |
| D.1 Executorch venv + QNN SDK 2.37 auto-download | ✅ | `ec2e5d8e` |
| D.2 `matmul_{f16,w8a8}.pte` 빌드 | ✅ | `ec2e5d8e` |
| D.3 `qnn_executor_runner` Android arm64 build + S25 push | ✅ | `ec2e5d8e` |
| D.4 production rounds=10 measurement | ✅ | (next commit) |
| E fair-pair report.md 작성 | ✅ | `qnn_microbench_phase_e_report.md` |

### 측정 데이터 위치

- **Phase C**: `papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_c_2026_05_26/`
- **Phase D single-shot**: `papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_d/`
- **Phase D production v2**: `papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_d_prod_v2/`
- **Phase E 통합 보고서**: `papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_e_report.md`

각 디렉토리에 `raw/`, `aggregated.csv`, `thermal_log.csv`, `report.md`, `env.json`.

---

## 핵심 결과

```
M3  OpenCL F16 raw        0.589 ms   CV 1.91%   GREEN  (baseline)
M4  QNN-GPU OpPackage F16 1.529 ms   CV 9.0%    YELLOW (2.60× slower)
M6b Executorch HTP F16    0.456 ms   CV 3.14%   GREEN  (1.29× faster)
M7  Executorch HTP W8A8   0.276 ms   CV 7.11%   YELLOW (2.13× faster)
```

CV 모두 < 10%. 모든 trial reproducible (single-shot vs production v2 1~2% 이내).
Peak thermal 37.3°C (Phase C) / 33.4°C (Phase D) — trigger 50°C 충분히 여유.

---

## 다음 액션 (Q-2 진입)

### Q-2 sprint 권장 plan

`engine/src/backend/qnn_htp/` 신설 — Backend trait native integrate.

1. **Q-2.1 spec**: `arch/qnn_htp_backend.md` + Spec ID 발급 (Architect)
2. **Q-2.2 skeleton**: Backend trait impl 골격 (config / capability / dtype matrix)
3. **Q-2.3 MatMul**: 본 매트릭스의 M7 (W8A8) path 를 production 으로 통합
4. **Q-2.4 generate.rs 연결**: `--backend qnn_htp` flag, model load + forward
5. **Q-2.5 device gate**: Qwen 2.5-1.5b end-to-end token gen 정확성 + TBT 측정

Q-2 진입 전 검증:
- Q-2.0 dry-run: `microbench_htp_matmul_correctness` 가 S25 에서 segfault 없는지 확인 (#194/#195)

### 별 sprint 후보

- **M5 latency 측정 bin** (μ-Q1.5): `microbench_qnn_oppkg_matmul_q40_tbt` 신규 작성. Q4_0 production hot path 의 단독 latency 측정. fair-pair Q4_0 vs W8A8 비교 base.
- **M1/M1b/M2 HTP raw libloading path**: Executorch 와 별개의 native QNN HTP API 측정. paper 측 framework overhead 분리 가능.
- **Per-cell paper figure**: bar chart with error bar (CV).

---

## Landmines / 미해결

### Phase D 환경 risk (재현 시)

1. **Arch Linux + GCC 16 + CMake 4 + Python 3.14**: 모두 Executorch 공식 비호환. uv venv 3.12 + `--skip_x86_64` build 로 우회.
2. **QNN SDK 2.37 auto-download 부분 실패 가능**: 한 번은 lib/python 만 staged 됨 (1차 fail). `rm -rf .venv/.../sdk/qnn && python -c "install_qnn_sdk()"` 로 재시도해야 성공.
3. **stderr vs stdout**: Executorch logging 이 stderr 로 — `parse_trial_output` 가 stdout 만 보면 fail. fix 적용 (v2 build).
4. **NDK 29 vs 권장 26c**: 빌드 자체는 성공. 추후 newer NDK 의 deprecation 영향 가능성.

### 매트릭스 측정의 제약

1. **M3/M4 같은 bin 의 dual-result**: 현재 cell 두 개가 같은 bin 을 두 번 호출 (시간 낭비). group_key 의 bin 공유 캐싱 미구현. 50분 → 25분 가능.
2. **M5 (Q4_0) 측정 미완**: production hot path 와 vs W8A8 fair 비교 부재.
3. **M2 (HTP W8A8 raw)**: Executorch 통하지 않는 native QNN API path 측정 부재. paper 측 framework overhead 분석 부재.

### 본 환경 specific 변수

- S25 SoC SM8750 (V79) — Executorch qc_schema.py 에 정식 등록 (id=69)
- QNN SDK 2.37 vs 본 프로젝트 2.33 — 둘 다 V79 지원. BackendOpInfo API 는 2.41+ (warning 만, 동작 OK).
- M7 의 PT2E quantizer = single random sample calibration. paper-grade 가 아니라면 충분, 실제 모델에선 representative calibration set 필요.

---

## 산출물 매핑

| 항목 | 위치 | 비고 |
|---|---|---|
| Matrix runner | `scripts/microbench_qnn_matrix.py` | 8 thermal zone + tukey IQR + round-robin shuffle |
| AOT builder | `papers/.../qnn_microbench_phase_d/build_pte_matmul.py` | nn.Linear 1536×8960 → .pte |
| F16 .pte | `papers/.../qnn_microbench_phase_d/matmul_f16.pte` | 27.5 MB |
| W8A8 .pte | `papers/.../qnn_microbench_phase_d/matmul_w8a8.pte` | 13.2 MB |
| Phase C raw | `papers/.../qnn_microbench_phase_c_2026_05_26/raw/` | M3/M4 × 10 round |
| Phase D raw | `papers/.../qnn_microbench_phase_d_prod_v2/raw/` | M6b/M7 × 10 round |
| 통합 보고서 | `papers/.../qnn_microbench_phase_e_report.md` | fair-pair 분석 + Q-2 entry decision |

외부 자산 (사용자 시스템):
- `/home/go/Workspace/executorch/` — shallow clone, .venv (uv 3.12 + executorch + QNN SDK 2.37) — Q-2 등 후속에서 재사용 가능
- `/data/local/tmp/executorch/` — S25 의 qnn_executor_runner + 7 .so + 2 .pte

---

## 다음 세션 진입 명령

```
"Q-2 진입 — HTP backend trait engine/src/backend/qnn_htp/ 신설"
```

또는 (Q-2 진입 전 미완 측정 보완 권장 시):
```
"μ-Q1.5 — M5 OpenCL Q4_0 latency 측정 bin 작성"
```

---

## 자기점검

- [x] 진입 문장 한 줄만으로 다음 세션이 첫 명령 가능
- [x] 멈춘 이유: sprint cap 도달 + paper-grade conclusion 4건 확보 → Q-2 진입 시점
- [x] 가장 큰 landmine 표면화: Phase D 환경 risk 4건 + 매트릭스 제약 3건 + specific 변수 3건
- [x] 검증 게이트: 측정 결과 (median + CV) + commit SHA + raw 데이터 위치
- [x] 본문 길이 적정: ~750 토큰
