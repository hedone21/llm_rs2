# Resilience Verify — Usage Guide

상세 CLI 레퍼런스와 YAML 시나리오 작성법.

## 1. 실행 전 체크리스트

1. 프로젝트 루트에서 실행 (`pwd`가 `llm_rs2`).
2. `devices.toml`에 해당 device 엔트리 존재.
3. `resilience_verify/fixtures/models.toml`에 `[devices.<device_id>.<model_key>]` 엔트리 존재 (model_path + tokenizer_path).
4. adb 연결 또는 SSH 접근 가능 (로컬 host 제외).
5. tokenizers Python 패키지 설치: `pip install tokenizers pyyaml`.

## 2. CLI 옵션 전체

```text
python resilience_verify/verify.py [OPTIONS]
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--device DEV` | `host` | devices.toml의 key. connection.type에 따라 local/ssh/adb 분기 |
| `--model M[,M...]` | `f16` | models.toml의 key. 쉼표로 다중 지정 (`f16,q4`) |
| `--scenario-filter S` | (전체) | scenario.id substring. 쉼표 OR (`memory,thermal`). `all` 또는 `*`은 전체 |
| `--layer L` | `both` | `engine_cmd` \| `signal` \| `both` |
| `--runs N` | `1` | run 반복 수 |
| `--skip-build` | off | cross-compile 스킵 (기존 바이너리 재사용) |
| `--skip-deploy` | off | adb push / scp 스킵 |
| `--output-root DIR` | `resilience_verify/results` | 결과 트리 루트 |
| `--scenarios-dir DIR` | `resilience_verify/scenarios` | YAML 탐색 루트 |
| `--dry-run` | off | 실행 없이 매트릭스만 출력 |

## 3. 대표 사용 예

### Smoke test — 호스트 1개 시나리오
```bash
python resilience_verify/verify.py --device host --model f16 \
  --scenario-filter throttle_smoke --runs 1 --skip-deploy
```

### S25 전체 매트릭스 (F16+Q4, 빌드·배포 자동)
```bash
python resilience_verify/verify.py --device galaxy_s25 --model f16,q4 --runs 1
```

### 특정 계층만 (signal 경로 회귀 확인)
```bash
python resilience_verify/verify.py --device galaxy_s25 --model f16 \
  --layer signal --scenario-filter signal_ --runs 3
```

### 빠른 재실행 (이미 배포돼 있을 때)
```bash
python resilience_verify/verify.py --device galaxy_s25 --model q4 \
  --scenario-filter partition --skip-build --skip-deploy --runs 1
```

### Dry-run으로 매트릭스 계획만 보기
```bash
python resilience_verify/verify.py --device jetson --model f16,q4 --dry-run
```

## 4. 결과 트리

```
resilience_verify/results/<ts>_<device>_<models>/
  summary.md                                ← 매트릭스 표 (Crash/Tokens/ROUGE/BLEU/TBT)
  summary.jsonl                             ← 동일 데이터 한 줄 per 결과
  <scenario>/<model>/r<idx>/
    verdict.json                            ← 4층 assertion 상세 + overall_pass
    baseline.jsonl / baseline.stderr        ← baseline run 원 로그
    action.jsonl   / action.stderr          ← action run 원 로그
    baseline.decoded.txt / action.decoded.txt  ← detokenized 최종 텍스트
    manager.stdout / manager.stderr         ← mock_manager or llm_manager 로그
    heartbeats.json                         ← 파싱된 heartbeat 레코드 배열
    prompt.txt                              ← baseline이 실제 사용한 프롬프트
    scenario.json                           ← (scenario 모드 시) mock_manager가 실행한 commands
    signal_client.log                       ← (signal 경로 시) signal injector 로그
    injection_schedule.json                 ← (signal 경로 시) 실제 주입된 schedule
    manager_config.toml                     ← (signal 경로 시) 배포된 llm_manager 설정
```

### verdict.json 구조 (요약)
```json
{
  "scenario_id": "direct_cmd_partition_ratio",
  "device": "galaxy_s25",
  "model": "f16",
  "backend": "opencl",
  "layer": "engine_cmd",
  "mode": "single",
  "timings": {"baseline_wall_s": ..., "action_wall_s": ...},
  "baseline_returncode": 0, "action_returncode": 0,
  "functional":    {"pass": true/false, "details": {...}},
  "performance":   {"pass": true/false, "details": {"delta_pct": ...}},
  "accuracy":      {"pass": true/false, "details": {"rouge_l_f1": ...}},
  "crash_and_progress": {"pass": true/false, "details": {"crash_hits":[...], "actual_decode_tokens":..., "action_returncode":...}},
  "overall_pass":  true/false,
  "pass_criteria": "all"
}
```

## 5. YAML 시나리오 스키마

`scenarios/*.yaml` 한 파일 = 한 테스트 케이스. 언더스코어로 시작하는 `_schema.yaml` 등은 로더가 무시.

### 최소 필수 필드
```yaml
id: my_scenario                    # 고유 id (파일명과 일치 권장)
layer: engine_cmd                  # engine_cmd | signal
devices: [galaxy_s25, jetson]      # 실행 대상 device key 목록
models: [f16, q4]                  # 실행 대상 model key 목록
baseline:
  prompt: fixtures:short_smoke     # fixtures:<name> 또는 프로젝트 상대/절대 경로
  decode_tokens: 64
  seed: 42
  greedy: true
  backend: auto                    # auto | cpu | opencl | cuda
  prefill_chunk_size: 64
  extra_args: []
action:
  enable_resilience: true
  mock_manager_command: Throttle   # 단일 directive 모드
  mock_manager_params: { delay_ms: 20, wait_secs: 1 }
expected:
  functional:
    engine_commands_any_of: [Throttle]
    stderr_patterns:
      - "\\[Resilience\\] Directive.*Throttle"
  crash_and_progress:
    decode_tokens_min_ratio: 0.5
  performance:
    metric: avg_tbt_ms
    delta_vs_baseline: { min_pct: 5.0, tolerance_pct: 10.0 }
  accuracy:
    min_rouge_l: 0.90
    min_bleu_4: 0.50
    min_char_similarity: 0.85
pass_criteria: all                 # all | functional_only
```

### action 변형

**scenario 모드** (mock_manager에 여러 명령 순차 전송):
```yaml
action:
  enable_resilience: true
  mock_manager_commands:
    - { delay_sec: 0.5, command: KvEvictH2o, params: { keep_ratio: 0.5 } }
    - { delay_sec: 2.0, command: RestoreDefaults }
```

**signal 경로** (llm_manager의 ExternalMonitor에 SystemSignal JSON 주입):
```yaml
layer: signal
action:
  enable_resilience: true
  injection_schedule:
    - delay_sec: 0.8
      signal:
        memory_pressure:
          level: critical
          available_bytes: 30000000
          total_bytes: 8000000000
          reclaim_target_bytes: 100000000
```
`signal.{memory_pressure|thermal_alert|compute_guidance|energy_constraint}` 필드는 `shared/src/lib.rs`의 `SystemSignal` enum과 **snake_case**로 대응.

### expected 확장 필드

**ordered stderr sequence** (v2 신규 — "A 다음에 B가 나와야 한다"):
```yaml
expected:
  functional:
    stderr_sequence:
      - { pattern: "\\[Resilience\\] Directive.*SetPartitionRatio", name: dir_received }
      - { pattern: "\\[Partition\\] ratio=0\\.3", name: activated, after: dir_received, min_occurrences: 1 }
      - { pattern: "\\[Experiment\\] Done", after: activated, min_occurrences: 1 }
```

**heartbeat 필드 체크** (engine_cmd + mock_manager heartbeat 파싱 전용):
```yaml
expected:
  functional:
    heartbeat_checks:
      - { field: active_actions, contains: throttle, required: true }
      - { field: state, transitions: [Running, Suspended, Running] }
      - { field: active_device, transitions_to: cpu }
      - { field: kv_dtype }           # stderr 폴백 (KIVI-Resilience 로그)
      - { field: kv_cache_tokens, decrease_from_peak: true }
```

**crash + progress 세부 조정**:
```yaml
expected:
  crash_and_progress:
    decode_tokens_min_ratio: 0.0     # Suspend처럼 조기 종료가 정상인 경우
    crash_deny_patterns:             # 기본 deny-list 재정의 (보통 그대로 둠)
      - "SIGSEGV"
      - "thread '.*' panicked"
    allow_nonzero_returncode: false
```

**performance 방향**:
```yaml
expected:
  performance:
    metric: avg_tbt_ms
    delta_vs_baseline:
      max_pct: 80.0       # 상한 (e.g. 오버헤드가 80%를 넘으면 안 됨)
      min_pct: 150.0      # 하한 (e.g. pacing이 +150%를 달성해야 함)
      tolerance_pct: 20.0 # 둘 다 이 만큼 느슨
```

### pass_criteria

- `all` (기본) — functional AND performance AND accuracy 모두 통과해야 함
- `functional_only` — performance/accuracy는 기록만, functional만 판정. **크래시/진행 게이트는 여전히 적용**. 의도적으로 출력이 달라지는 경우(SwitchBackend, KvQuantDynamic→Q4, Suspend 등) 전용 + YAML에 justify 주석 필수

## 6. 새 시나리오 추가 3-Step

1. `scenarios/my_new.yaml` 작성 (위 스키마 참조)
2. `python resilience_verify/verify.py --scenario-filter my_new --device host --dry-run`로 시나리오가 로드되는지 확인
3. 실제 device에서 1회 돌려 stderr_sequence 패턴 맞추기 → 결과의 `verdict.json.functional.details.stderr_sequence.steps` 확인 후 `matched_line_nos`에 실제 로그가 잡히도록 패턴 보정

## 7. signal 경로 디버깅

signal 시나리오가 FAIL하면 다음 순서로 확인:

1. `signal_client.log` — `connected` 안 뜨면 adb forward 문제. `pre-sleep`값 ↑ 시도
2. `manager.stdout` — `[ExternalMonitor] TCP client connected from ...` 안 보이면 데이터 전달 안됨. 보이면 `[ExternalMonitor] Injected: ...` 확인
3. 둘 다 있는데 Directive 안 오면 LuaPolicy가 해당 signal에 action을 매핑하지 않음. `manager.stdout`의 `Directive seq=...` 확인
4. Directive는 나왔는데 엔진이 반응 안 하면 → `action.stderr`에 `[Resilience] Directive ...` 라인 확인. 없으면 직렬화/프레임 문제.

## 8. 내부 구조 요약 (깊이 파고 싶을 때)

호출 흐름:
```
verify.py
 └─ orchestrator.run_scenario(spec, device_cfg, model_key, out_dir, run_idx)
    ├─ _run_scenario_local | _run_scenario_ssh | _run_scenario_adb | _run_scenario_adb_signal
    │   ├─ baseline: _generate_cmd → run_foreground_*
    │   └─ action: mock_manager/llm_manager background + signal_client + run_foreground_*
    └─ _finalize_verdict
       ├─ text_accuracy.decode_jsonl_to_text
       ├─ log_parser.load_summary / parse_heartbeats
       ├─ assertions.verify_functional / verify_performance / verify_accuracy / verify_crash_and_progress
       └─ assertions.aggregate_verdict(pass_criteria)
```

assertion 의 세부 규칙은 `harness/assertions.py`의 docstring 참조.

## 9. 유지 보수 메모

- YAML의 `functional_only`는 **반드시** justify 주석(한 줄 이상)을 해당 라인 바로 위에 달고 쓴다. 없으면 의미가 퇴화됨.
- 엔진 로그 포맷이 바뀌면 패턴 깨짐 — `stderr_patterns` / `stderr_sequence` 규칙 업데이트 필요. 로그 포맷 변경 시 가장 먼저 깨지는 건 이쪽 테스트.
- 결과 디렉터리는 `.gitignore`에 이미 포함. 커밋하지 말 것.
- 새 device 추가 시 `devices.toml`, `models.toml` 두 곳 모두 업데이트.
