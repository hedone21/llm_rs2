# Resilience Verify Harness

llm.rs의 Resilience 기능(SystemSignal → Policy → EngineCommand → Engine 전 경로)을 **실제 바이너리 조합**으로 end-to-end 검증하는 자동화 하네스. On-device(Android/Jetson) + 호스트(x86 CPU) 매트릭스 지원.

자세한 설계 배경은 `arch/resilience_verify_v2.md` 참조.

## 지금 바로 한 번 돌려보기

```bash
# 호스트(CPU) 스모크 테스트 — Throttle 1개만
python resilience_verify/verify.py --device host --model f16 \
    --scenario-filter throttle_smoke --runs 1 --skip-deploy

# S25 전 시나리오 × F16+Q4 (빌드·배포 자동)
python resilience_verify/verify.py --device galaxy_s25 --model f16,q4 --runs 1

# Jetson 단일 시나리오
python resilience_verify/verify.py --device jetson --model q4 \
    --scenario-filter memory_critical_evict --runs 1
```

결과: `resilience_verify/results/<ts>_<device>_<models>/` 아래에 per-scenario verdict.json + baseline/action 원 로그 + 매트릭스 summary.md.

## 무엇을 검증하는가 (4-Layer Gate)

매 scenario × device × model × run 조합을 **4층 assertion**으로 판정.

| Layer | 통과 기준 |
|---|---|
| **functional** | stderr 패턴 매칭 + `stderr_sequence`(순서 강제) + heartbeat 필드 |
| **crash_and_progress** (하드 게이트) | SIGSEGV/panic 없음 + decode 토큰 ≥ 요청의 50% + returncode = 0 |
| **performance** | `avg_tbt_ms`가 baseline 대비 YAML에 지정한 delta_pct 범위 안 |
| **accuracy** | detokenize된 최종 응답 텍스트의 ROUGE-L / BLEU-4 / char similarity |

`crash_and_progress`는 `pass_criteria`와 무관한 **무조건 실패** 게이트. 나머지 셋은 YAML의 `pass_criteria: all | functional_only`로 제어한다 (`functional_only`는 KvQuantDynamic→Q4 같이 출력이 의도적으로 달라지는 경우에만 명시 opt-in + justify 주석 필수).

## 두 가지 주입 경로

| layer | 주입자 | 무엇을 테스트하나 |
|---|---|---|
| `engine_cmd` | `mock_manager`가 EngineCommand 직접 주입 | 엔진 핸들러 단위 검증 (정책은 우회) |
| `signal` | 진짜 `llm_manager` + `ExternalMonitor`로 SystemSignal 주입 | SystemSignal → LuaPolicy → EngineCommand 변환 경로 포함한 end-to-end |

`signal` 경로는 adb forward tcp:19102 → device tcp:9102 터널로 호스트 `signal_client.py`가 device의 ExternalMonitor에 JSONL을 씀.

## 디렉터리 구조

```
resilience_verify/
  README.md          ← 이 파일
  USAGE.md           ← CLI 레퍼런스 + YAML 시나리오 작성법
  verify.py          ← CLI entry (argparse + matrix planner + report emit)
  harness/           ← Python 구현 패키지
    orchestrator.py  — baseline/action 실행, 디바이스별 분기
    assertions.py    — 4층 verifier + aggregate_verdict
    log_parser.py    — stderr/JSONL/heartbeat 파서, crash/sequence 헬퍼
    spec_loader.py   — YAML 시나리오 로더
    fixtures.py      — 프로젝트 경로 / models.toml / 프롬프트 로더
    process_control.py — local subprocess / SSH / ADB 백그라운드 + foreground 실행 헬퍼
    signal_client.py — SystemSignal JSONL TCP/Unix injector (별도 실행 가능한 CLI)
    text_accuracy.py — 토큰 jsonl → detokenized string, ROUGE-L/BLEU-4 비교
    report.py        — summary.md / summary.jsonl / 콘솔 테이블
  scenarios/         ← YAML 시나리오 (14개). 하나당 한 테스트 케이스
    _schema.yaml     — 필드 레퍼런스 (주석으로)
    direct_cmd_*.yaml — layer: engine_cmd
    signal_*.yaml     — layer: signal
    prefill_*.yaml, memory_*.yaml, thermal_*.yaml — 다양한 mid-flow 주입 시나리오
  fixtures/          ← 정적 데이터
    prompts/*.txt    — 프롬프트 프리셋 (short_smoke, medium_qa, long_narrative)
    models.toml      — device × {f16,q4} 별 모델 파일 + tokenizer 경로
    budgets.toml     — device별 timeout/threshold 오버라이드
    manager_config_external_only.toml — llm_manager 결정론 config (모든 실측 monitor off, external TCP만 on)
  results/           ← gitignored. <ts>_<device>_<models>/<scenario>/<model>/r<idx>/ 트리
```

## 시나리오 현황 (14개)

### engine_cmd — mock_manager 직접 주입 (11개)
- `direct_cmd_throttle_smoke` — Throttle{delay_ms=20}
- `direct_cmd_target_tbt` — SetTargetTbt{target_ms=250}
- `direct_cmd_target_tbt_restore` — 250ms pacing → 0ms 복귀
- `direct_cmd_partition_ratio` — partition disable 0.3 → 0.0
- `direct_cmd_partition_ratio_enable` — partition enable 0 → 0.3 *(신규, BS1)*
- `direct_cmd_kvquant_f16_to_q4` — KV dtype F16 → Q4
- `direct_cmd_kvquant_restore` — KvQuantDynamic → RestoreDefaults *(신규)*
- `memory_critical_evict` — KvEvictH2o keep_ratio=0.5
- `thermal_emergency_suspend` — Suspend 미드 디코드
- `prefill_midway_injection` — long prompt + KvStreaming 중간 주입
- `prefill_midway_partition_enable` — long prompt + SetPartitionRatio prefill 중 *(신규, silent no-op 감지)*

### signal — 진짜 llm_manager 경유 (2개)
- `signal_memory_critical` — MemoryPressure{Critical} → KvEvict*
- `signal_thermal_critical_throttle` — ThermalAlert{Critical} → Throttle/SetTargetTbt

## 의존성

- Python ≥ 3.11 (tomllib), `pyyaml`, `tokenizers` (HuggingFace, detokenize용)
- 엔진 측 바이너리: `generate`, `mock_manager`, `llm_manager` (`--features lua`)
- adb(Android) / ssh(Jetson) 연결 + `devices.toml`

## 자주 쓰는 옵션

| 옵션 | 설명 |
|---|---|
| `--device` | devices.toml의 key (`host`, `galaxy_s25`, `jetson`, ...) |
| `--model` | 쉼표 구분 (`f16,q4`). models.toml의 key |
| `--scenario-filter` | 문자열 substring(쉼표 OR) 또는 `all` |
| `--layer` | `engine_cmd | signal | both` (default `both`) |
| `--runs` | run 반복 수 (Tokens/TBT 변동 측정 시 ↑) |
| `--skip-build`, `--skip-deploy` | 빌드·배포 스킵 |
| `--dry-run` | 실제 실행 없이 매트릭스만 출력 |

## 결과 읽기

- `verdict.json` — 4층 상세 + overall_pass
- `summary.md` — 매트릭스 표 (Scenario/Device/Model/Crash/Tokens/ROUGE/BLEU/TBT Δ%)
- `action.stderr`, `manager.stdout`, `signal_client.log` — 디버깅용 원 로그

## Troubleshooting

- **adb unix socket permission denied**: TCP(`127.0.0.1:9100`) 고정이라 기본 경로. 별도 설정 불필요.
- **action.jsonl pull failed**: 디바이스에서 action이 시작 전에 죽었을 때. `action.stderr` / `action.adb.stderr`를 먼저 봐라.
- **signal scenario에서 directive가 안 온다**: ExternalMonitor 바인딩 타이밍. `signal_client.log`가 `connected` 라인에서 멈추면 `--pre-sleep` 값을 늘려라 (기본 8초).
- **functional_only 경고 로그**: 명시 opt-in했으면 무시해도 됨. YAML에 justify 주석 달면 된다.

## 다음 단계

자세한 CLI/스키마/디버깅은 [USAGE.md](USAGE.md). 아키텍처 심화는 [`arch/resilience_verify_v2.md`](../arch/resilience_verify_v2.md).
