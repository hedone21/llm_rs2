# AB-4 frozen partition baseline (command-driven SetPartitionRatio, S25)

**캡처**: 2026-06-11, HEAD=`b5184d33`(AB-4 host 완료 직후, device 게이트 run)
**장비**: Galaxy S25 (R3CY408S5SB), `opencl --opencl-rpcmem`, 6T
**모델**: `qwen2.5-1.5b-{f16,q4_0}.gguf` (weight 양축), KV f16 고정
**용도**: AB-4 PartitionStage(OneShot·PreForward) directive 경로의 회귀 anchor. AB-6/AB-2 및 후속
파이프라인 변경 후 동일 시나리오 재실행 시 sig·marker 가 본 문서와 MATCH 해야 한다.

## 시나리오 (command-driven partition — manager directive → PartitionStage)

mock_manager 가 TCP 로 `SetPartitionRatio{ratio:0.3}` directive 를 송신, dispatcher
`submit_partition`(last-applied sticky 게이트) → PartitionStage OneShot 이 decode step 0
PreForward 에서 lazy host-map + re-slice. directive 는 prefill(918 tok, ~5-7s) 중 도착(wait-secs 2)
하여 **decode step 0 에서 결정적으로 적용** — greedy 출력이 run 간 byte-identical (f16 3/3·q4_0 3/3 실측).

```bash
# device /data/local/tmp 에서:
(./mock_manager --tcp 127.0.0.1:9899 --command SetPartitionRatio --ratio 0.3 --wait-secs 2 &)
sleep 1
RUST_LOG=info ./argus_bench -b opencl --opencl-rpcmem \
    -m models/qwen2.5-1.5b/qwen2.5-1.5b-{f16,q4_0}.gguf \
    --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
    --greedy -n 128 --threads 6 --max-seq-len 1536 --kv-type f16 \
    --prompt-file medium_qa.txt --protected-prefix 4 \
    --resilience-transport tcp:127.0.0.1:9899
# medium_qa.txt = verify/fixtures/prompts/medium_qa.txt (918 tokens)
```

## 동결 기준 (n=3, 전 run 일치)

sig recipe: `{ grep -B2 '^TTFT' out | head -2; grep -o 'generated=.*' out; } | md5sum`
(생성텍스트 마지막 2줄 + generated summary — md5 절대값과 함께 **원본 라인 대조가 정본** — α-K/β 선례).

| weight | sig (md5) | 필수 marker (글자단위) | generated summary | Decode ms/tok (median) |
|---|---|---|---|---|
| f16 | `64bce5dc972d6be433dc142736e475e9` | `[Resilience] Directive seq=1: SetPartitionRatio { ratio: 0.3 }` + `[Partition] Lazy-mapped 336 weight tensors for host access` + `[Partition] Re-split 84 weights with ratio 0.30` | `generated=128 (first=16 + run=127) stopped_by=BudgetExhausted final_pos=1045` | **66.26** (65.34/66.26/66.61) |
| q4_0 | `c1af68272552e6407b76acb3ce42bdec` | 동일 (Lazy-mapped 336 / Re-split 84) | 동일 (`first=16`, `final_pos=1045`) | **39.52** (39.52/38.39/41.54) |

- 회계 불변식: `final_pos = 917(prompt) + 128 = 1045` — partition 은 KV/pos 불변 (§5.5.1, eviction 없음).
- verify YAML regex 계약 충족: `\[Partition\] (Lazy-mapped|ratio=0\.3)` 의 Lazy-mapped alt 글자단위 MATCH
  (`direct_cmd_partition_ratio_enable.yaml:26`). `[Experiment] Done` 은 `--experiment-output` 시에만 출력 — 본 시나리오 범위 밖.
- TTFT 참고치: f16 ~5.1s / q4_0 ~6.7s (게이트 아님 — directive 는 prefill 중 도착, 적용은 decode step 0).
- raw 백업: device `/data/local/tmp/ab4_part_{f16,q4_0}_{dir_1..3,static}.out` — 본 문서가 정본.

## static CLI oracle (Stage 경로 ≡ 정적 경로 교차 검증)

동일 ratio 의 `--no-resilience --tensor-partition 0.3` (init.rs 정적 경로, `[Partition] Prepared 84
weights with ratio 0.30`) 출력과 directive 경로 출력이 **양 dtype 모두 sig IDENTICAL** — partition 은
decode-only 라 step 0 적용(directive) == init 적용(static) 등가가 성립하고, 실측으로 bit-exact 확인.
Decode ms/tok 참고: static f16 63.61 / q4_0 38.90.

## 관찰 (게이트 무관, 모니터링용)

- **silent kill 1회 (9 run 중)**: 최초 q4_0 static run 이 prefill 직후 무출력 사망(exit 메시지 없음,
  stdout 버퍼 유실 패턴 — LMK/driver 일과성 의심). 동일 명령 재실행 2회 연속 정상 완주(EXIT=0) +
  sig IDENTICAL 로 결정성 재확인. back-to-back 연속 실행(3s 간격) 시에만 관찰됨.
- directive f16 run 간 Decode 상승 추세(65.3→66.6) = 열누적. q4_0 static(38.90) ≈ directive(39.52)
  — `--tensor-partition` 의 auto zero-copy 가 decode tbt 에 유의미한 차이를 만들지 않음.

## 게이트 사용법 (AB-6/AB-2/후속)

1. 동일 명령 재실행 (n≥1) → sig + 3 marker + generated summary **전부 MATCH**.
2. Decode ms/tok 는 부수 지표 (partition 0.3 은 GPU-only 대비 의도된 감속 — f16 +22%/q4_0 +23%,
   happy-path frozen tbt 게이트와 별도).
3. ratio 변경/disable(`ratio<=0`) 거동은 본 baseline 범위 밖 — host 테스트
   (`partition_changed_ratio_resubmits`/`ratio_zero_disables_partition` 등 12종)가 커버.
