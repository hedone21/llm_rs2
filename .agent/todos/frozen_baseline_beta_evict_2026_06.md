# β-3 frozen eviction baseline (command-driven evict, S25)

**캡처**: 2026-06-10, HEAD=`96113e3e`(β-3 commit B 직후 — v1 (a.5) live 경로 산출)
**장비**: Galaxy S25 (R3CY408S5SB), `opencl --opencl-rpcmem`, 6T
**모델**: `qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf` (F16 weight)
**용도**: β-4(dispatcher cutover)/β-5(pressure 일원화)/β-7(v1 삭제) 게이트의 command-driven eviction 회귀 anchor. **β-4 이후 동일 시나리오 재실행 시 sig·marker 가 본 문서와 MATCH 해야 한다.**

## 시나리오 (command-driven evict — manager directive → (a.5))

mock_manager 가 TCP 로 `KvEvictSliding{keep_ratio:0.5}` directive 를 송신, engine 이 (a.5) 에서
`forward.try_evict`(UER → `CacheManager::force_evict`) 로 mid-decode prune. directive 는
prefill(918 tok, ~5s) 중 도착(wait-secs 2)하여 **decode step 0 에서 결정적으로 적용** —
greedy 출력이 run 간 byte-identical (f16 3/3·q4 3/3 실측).

```bash
# device /data/local/tmp 에서:
(./mock_manager --tcp 127.0.0.1:9899 --command KvEvictSliding --keep-ratio 0.5 --wait-secs 2 &)
sleep 1
RUST_LOG=info ./argus_bench -b opencl --opencl-rpcmem \
    -m models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
    --greedy -n 128 --threads 6 --max-seq-len 1536 --kv-type {f16,q4} \
    --prompt-file medium_qa.txt --protected-prefix 4 \
    --resilience-transport tcp:127.0.0.1:9899 eviction sliding
# medium_qa.txt = verify/fixtures/prompts/medium_qa.txt (918 tokens)
```

**주의**: `[CacheEvent]` 는 `log::info!` — `RUST_LOG=info` 필수 (미설정 시 marker 부재가 정상,
eviction 자체는 final_pos 회계로 확인 가능).

## 동결 기준 (n=3, 전 run 일치)

| KV dtype | sig (생성텍스트 2줄 md5) | CacheEvent marker | generated summary | Decode ms/tok (median) |
|---|---|---|---|---|
| f16 | `c41930a5a1ccfef825ff0e81f8f04e13` | `policy='sliding@Warning', removed=459, new_pos=459` | `generated=128 (first=16 + run=127) stopped_by=BudgetExhausted final_pos=586` | **60.19** (60.13/60.24/60.19) |
| q4  | `84db59fb755c596dca858e6605db0cb5` | `policy='sliding@Warning', removed=459, new_pos=459` | `generated=128 (first=220 + run=127) stopped_by=BudgetExhausted final_pos=586` (first 토큰은 f16 과 상이 — β-4 재검증으로 정밀화 2026-06-10) | **59.56** (59.56/59.62/59.53) |

- sig 추출: `grep -B2 '^TTFT' out | head -2 | md5sum` (생성 텍스트 마지막 2줄 — 본 시나리오 출력은 2줄).
- 필수 marker: `[Resilience] Directive seq=1: KvEvictSliding { keep_ratio: 0.5 }` + `[CacheEvent] Eviction completed` + `[CacheEvent] Budget eviction (forced)`.
- 회계 불변식: `final_pos = 918×0.5(=459) + 127`. removed=459, new_pos=459.
- TTFT 참고치: f16 ~5.0s / q4 ~6.4s (게이트 아님 — prefill 은 evict 전).
- raw 백업: 호스트 `/tmp/b3b_dev/evict_{f16,q4}_{1..3}.out` (세션 한정) — 본 문서가 정본.

## 게이트 사용법 (β-4/5/7)

1. 동일 명령 재실행 (n≥1) → sig md5 + CacheEvent marker(removed/new_pos) + generated summary **전부 MATCH**.
2. β-4: directive 소비가 dispatcher→OneShot EvictionStage 로 교체된 후에도 산출 동일해야 함 (UER 등가는 host 9종+1 이 선행 증명 — `engine/tests/beta3_eviction_stage_equivalence.rs`).
3. Decode ms/tok 는 부수 지표 (eviction 시나리오는 happy-path frozen tbt 게이트와 별도 — 참고 추세만).
4. pressure-driven eviction (β-5 신설) 은 별도 측정 지점 — 본 baseline 은 command-driven 한정.
