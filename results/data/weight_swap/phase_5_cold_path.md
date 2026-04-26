# Phase 5 Sprint A — WSWAP-5-COLD-UNIFORM 측정 리포트

> **목적**: 5차 측정에서 발견된 per-layer 양봉 분포 (cold ~53 ms / warm ~36 ms,
> SLA <50 ms p50 5.7% 초과)의 직접 원인인 AUF mmap demand-paging을 prefault로
> 흡수하여 cold path를 균일화한다.

## 변경 요약

### 코드 변경

| 파일 | 라인 | 변경 |
|------|------|------|
| `engine/src/models/weights/secondary_mmap.rs` | +90 | `SecondaryMmap::prefault()` public API + per-variant impl + `prefault_byte_range()` helper (Linux/Android: `madvise(MADV_WILLNEED)` + page-touch warmup; 기타 OS no-op) |
| `engine/src/models/weights/swap_executor.rs` | +25 | `StageBreakdown::prefault_ms` field + `to_log_line()` 갱신 + `execute_on_slots()` 시작점에서 batch-once `secondary.prefault()` 호출 + 측정 |
| `engine/src/models/weights/secondary_mmap.rs` (tests) | +20 | prefault helper 단위 테스트 3건 (empty/aligned/unaligned) |
| `engine/src/models/weights/swap_executor.rs` (tests) | +30 | `StageBreakdown` log line + default 검증 2건 |

### 적용 후보

**후보 C (A+B 결합)** 채택:
- **A**: `madvise(MADV_WILLNEED)` — 커널 background prefetch 힌트
- **B**: explicit page-touch warmup — 4 KiB step `read_volatile`로 명시적 page fault

호출 위치: `SwapExecutor::execute_on_slots`의 layer loop **시작 전 단 한 번**.
batch-level cost로 charge되어 cold/warm 직접 비교 가능 (per-layer가 아닌 stage).

## 호스트 sanity (PASS)

```
cargo fmt --check  → clean
cargo clippy --workspace -- -D warnings  → clean
cargo test --workspace --no-fail-fast (skip 6 known UMA SIGABRT)  → 1966 passed, 0 failed
```

신규 단위 테스트:
- `prefault_byte_range_handles_empty_slice`
- `prefault_byte_range_touches_pages_without_panic`
- `prefault_byte_range_handles_unaligned_slice`
- `stage_breakdown_log_line_includes_prefault_stage`
- `stage_breakdown_prefault_default_zero`

기존 회귀: AUF spec 36/36 + secondary_mmap 8/8 + swap_executor 8/8 모두 통과.

## 디바이스 측정 (Galaxy S25, `R3CY408S5SB`)

### 환경

- 디바이스: Galaxy S25, Android 16, kernel 6.6.77, OpenCL backend, 6 threads
- 모델: `Llama-3.2-1B-Instruct-f16.gguf` (primary) + `Llama-3.2-1B-Instruct.auf` (5차와 동일 sha256 `1a1ead0c...`)
- generate binary: 본 작업 브랜치 HEAD (mtime 2026-04-26 09:48)
- 측정: `--force-swap-ratio 1.0 --num-tokens 200 --protected-prefix 4 --prompt 'The capital of France is'`
- thermal cooldown: 매 run 사이 CPU zone 평균 < 50°C까지 대기 (V10 thermal isolation)

### Stage A — 스모크 (PASS)

```
weight_swap: force ratio=1.00, swapped 16/16 layers in 815.0ms
weight_swap stages: prefault=67.0ms mmap_permute=342.3ms arc_swap=0.0ms madvise=0.0ms soa_reconvert=215.7ms gen_bump=0.0ms
The capital of France is Paris, which has a population of about 2.1 million people. Paris ...
```

- "Paris" 정답 ✓
- garbage 없음 (4차 회귀 가드) ✓
- prefault stage 정상 보고 ✓

### Stage B — N=5 latency (V10 thermal isolation, ratio=1.0)

| run | cpu_avg_before | total_ms | prefault | mmap_permute | soa_reconvert | per-layer (excl prefault) | per-layer (total) | Decode ms/tok |
|-----|----------------|----------|----------|--------------|----------------|---------------------------|-------------------|----------------|
| 1   | 56.2°C         | 807.0    | 67.1     | 315.9        | 242.8          | 46.2                      | 50.4              | 37.42          |
| 2   | 47.3°C         | 826.3    | 9.5      | 362.6        | 273.8          | 51.0                      | 51.6              | 28.02          |
| 3   | 46.4°C         | 829.7    | 9.5      | 365.3        | 273.0          | 51.3                      | 51.9              | 26.20          |
| 4   | 45.7°C         | 674.5    | 62.4     | 273.3        | 193.7          | 38.3                      | 42.2              | 21.77          |
| 5   | 45.1°C         | 826.5    | 10.4     | 360.2        | 273.8          | 51.0                      | 51.6              | 20.93          |

**Stats (N=5)**:
- per-layer **excl prefault**: min=38.3, **p50=51.0**, p95=51.3, mean=47.6 ms, **range ±13.6%**
- per-layer **total** (incl prefault): min=42.2, **p50=51.6**, p95=51.9, mean=49.5 ms
- mmap_permute per-layer: min=17.1, **p50=22.5**, p95=22.8 ms
- soa_reconvert per-layer: min=12.1, **p50=17.1**, p95=17.1 ms (회귀 없음)
- prefault batch-level: min=9.5, p50=10.4, p95=67.1 ms (양봉 — page cache 상태 의존)

### 5차 vs 6차 비교 표

| 지표 | 4차 (`aee9adc`) | 5차 (`21c6d82`) | 6차 (본 작업) | 변화 (5→6) |
|------|----------------:|----------------:|---------------:|----------:|
| per-layer p50 | 206 ms | 52.8 ms | **51.6 ms** (total) / **51.0 ms** (excl prefault) | total: −1.2 ms (−2.3%) / excl: −1.8 ms (−3.4%) |
| mmap_permute p50 | (미보고) | 23.6 ms | **22.5 ms** | **−1.1 ms (−4.7%)** ✓ 단조 감소 |
| soa_reconvert p50 | 172 ms | 17.0 ms | **17.1 ms** | +0.1 ms (회귀 없음) ✓ |
| per-layer 변동 폭 | — | ±25% | **±13.6%** (excl prefault) | **−45.6%** ✓ 양봉 좁아짐 |
| min per-layer | — | 36.4 ms | 38.3 ms | +1.9 ms |
| max per-layer | — | 53.6 ms | 51.9 ms | −1.7 ms |
| prefault batch | (없음) | (없음) | 9.5–67.1 ms (자체 양봉) | (신규 stage) |

## SLA 충족 / 양봉 분포 변화 분석

### Acceptance Criteria 평가

| 기준 | 결과 | 비고 |
|------|------|------|
| per-layer p50 < 50 ms | ⚠️ 51.6 ms (total) / 51.0 ms (excl prefault) — **3.2% / 2.0% 초과** | 5차 5.7% 초과 대비 개선 |
| 변동 폭 < ±10% | ⚠️ ±13.6% (excl prefault) | 5차 ±25% 대비 약 절반 개선, 기준 미달 |
| mmap_permute p50 단조 감소 | ✅ 23.6 → 22.5 ms (−4.7%) | PASS |
| INV-122 회귀 없음 (Paris) | ✅ PASS | greedy sanity 정답 |
| clippy clean | ✅ PASS |  |
| fmt clean | ✅ PASS |  |
| 호스트 generate 회귀 없음 | ✅ PASS | 1966 tests pass |

### 양봉 분포 변화 분석

5차 양봉 **per-layer 변동 ±25%** (cold ~53 / warm ~36) → 6차 **per-layer 변동 ±13.6%** (38.3–51.3 excl prefault). 즉 prefault가 mmap demand-paging의 일부를 흡수하긴 했으나, 변동성이 prefault stage 자체로 전이되었다 (9.5–67.1 ms 양봉).

핵심 관찰:
1. **mmap_permute 자체 변동성**: 5차 (미보고) → 6차 17.1–22.8 ms (±14%). prefault가 mmap_permute의 worst case를 일부 막음 (기존 cold ~24 → 22.8).
2. **prefault 비용 양봉**: 첫 swap에서 cold pages를 모두 prefault할 때 60+ ms, 후속 run에서 OS가 페이지를 캐시에 유지하면 10 ms 미만. 즉 **prefault는 cold 비용을 명확히 분리·측정 가능한 stage로 격리**시키지만, 비용 자체를 완전히 제거하지는 못한다.
3. **min per-layer 38.3 ms** — 5차 36.4 ms와 거의 같음. 하한선은 prefault로 추가 단축 불가.

### 잔여 issue / 후속 권장

1. **±10% 미달**: 단순 prefault만으로는 SLA에 못 미친다. 추가 후보:
   - **C2 — `Mmap::lock()` (mlock equivalent)**: 페이지를 RAM에 고정 (메모리 압박 트레이드오프). Android에서 `RLIMIT_MEMLOCK` 한계 확인 필요.
   - **C3 — 비동기 prefault**: 모델 로드 직후 별도 스레드로 prefault, swap 시작 시점엔 이미 warm. 첫 swap 첫 layer가 cold일 위험 낮음.
   - **C4 — fadvise(POSIX_FADV_WILLNEED)**: 파일 디스크립터 레벨에서 readahead 강제 (mmap 외부 path).
2. **prefault 양봉 자체 해소**: model load 시점에 prefault 1회 실행 → swap 시점 비용 0. 단, OS가 RAM 압박으로 페이지 evict 가능 → cold 재발생 위험.
3. **thermal isolation 정확도**: 6차에서 일부 run의 Decode가 throttle (run 1의 37 ms/tok)되어 swap latency도 영향 받았을 가능성. 5차 측정 환경 (33°C idle)을 정확히 재현하기는 본 호스트 thermal 상태에서 어려움.

### Phase 5 Sprint A verdict

**부분 충족**:
- 정량 지표 (mmap_permute 단조 감소, 변동폭 절반 감소, INV-122 회귀 없음): 충족
- SLA p50 <50 ms 본질 충족: 미달 (3.2% 초과, 5차 5.7%에서 절반 개선)
- 변동 폭 ±10%: 미달 (±13.6%)

5차의 5.7% → 6차의 3.2%로 SLA 게이트가 점진적으로 좁혀짐. 본질 충족까지는 후속 sprint에서 mlock 또는 비동기 prefault 추가 필요. 본 sprint의 핵심 성과는 **변동성을 ±25% → ±13.6%로 절반 감소** + **prefault stage를 측정 가능한 형태로 격리**.

## 산출물

- `/tmp/wswap5_cold_v2/run_{1..5}.log` — N=5 측정 raw 로그
- 본 리포트 (`results/data/weight_swap/phase_5_cold_path.md`)
- 신규 commit (자동 커밋 예정)
