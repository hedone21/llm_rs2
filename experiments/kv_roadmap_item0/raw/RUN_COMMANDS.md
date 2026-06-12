# P3 측정 실행 명령 전문 (재현성)

## 환경
- HEAD: bdf744a82318102cbeec8311108a156ff51e84ec (master)
- 빌드: cargo build --release -p llm_rs2 --features rkv (57.58s)
- 모델: models/llama3.2-1b/llama3.2-1b-f16.gguf (F16, 2.3GB)
- 토크나이저: models/llama3.2-1b/tokenizer.json (명시 필수)
- KV type: f32 (설계서 §1.1), backend: cpu, sampling: greedy(temperature 0.0)
- 프롬프트: experiments/kv_roadmap_item0/prompts/ppl0{1..5}L_*.txt (각 303~326 tok, ≥300 tok 충족)

## 측정 1: R-KV (UNMEASURABLE)
# eval_setup.rs match에 rkv arm 부재 → fail-fast. e2e 불가.
./target/release/argus_eval --model-path models/llama3.2-1b/llama3.2-1b-f16.gguf \
  --tokenizer-path models/llama3.2-1b/tokenizer.json \
  --ppl experiments/kv_roadmap_item0/prompts/ppl03L_technical.txt \
  --kv-type f32 --temperature 0.0 -b cpu --kv-budget 200 --protected-prefix 4 \
  eviction rkv --lambda 0.1
# → Error: Unknown eviction policy: 'rkv'. Use: none, sliding, streaming, h2o, h2o_plus, d2o
# 단위 테스트(합성 벡터만): cargo test -p llm_rs2 --features rkv --release rkv  → 11 passed

## 측정 2: A2SF (h2o + --score-decay 스윕, 5도메인)
for f in <each ppl0NL prompt>; do
  # sliding baseline
  ./target/release/argus_eval --model-path <m> --tokenizer-path <t> --ppl $f \
    --kv-type f32 --temperature 0.0 -b cpu --kv-budget 200 --protected-prefix 4 \
    eviction sliding --window 2048
  # h2o decay sweep {0.0, 0.7, 0.8, 0.9}
  for decay in 0.0 0.7 0.8 0.9; do
    ./target/release/argus_eval --model-path <m> --tokenizer-path <t> --ppl $f \
      --kv-type f32 --temperature 0.0 -b cpu --kv-budget 200 --protected-prefix 4 \
      --score-decay $decay eviction h2o --keep-ratio 0.5
  done
done
# 산출: a2sf/emr_compare.csv, a2sf/bos_ratio.json

## 측정 3: head 분산 (--dump-importance 확장, 5도메인) — DEGENERATE (all C_h=0)
for f in <each ppl0NL prompt>; do
  ./target/release/argus_eval --model-path <m> --tokenizer-path <t> \
    --prompt-file $f --kv-type f32 -b cpu --dump-importance > head_variance/${name}_dump.json
done
# 산출: head_variance/*.json (max_min_ratio=1.0, matrix all C_h=0.0)

## 측정 4: Demote 모사 (host 통합 테스트, 엔진 무수정) — NMSE only (PPL/EMR 미산출)
cargo test -p llm_rs2 --test demote_measure -- --nocapture   # dev 프로필(release는 panic=abort 충돌)
# → 7 passed. [Q4vsQ2] / [Demote smoke] 수치 → demote/ppl_compare.csv
