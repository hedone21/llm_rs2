# P3 재측정 실행 명령 전문 (재현성) — 2026-06-12 rerun

## 환경
- HEAD: c702ff83d76c54f8b49e6833bc4c41fe5b7979ca (수정 라운드 7커밋 반영)
- 빌드: cargo build --release -p llm_rs2 --features rkv --bin argus_eval
- 모델: models/llama3.2-1b/llama3.2-1b-f16.gguf (F16, 2.3GB)
- 토크나이저: models/llama3.2-1b/tokenizer.json (명시)
- KV type: f32, backend: cpu, sampling: greedy(temperature 0.0)
- 프롬프트: prompts/ppl0{1..5}L_*.txt — 토큰수 literary=304/encyclopedic=308/technical=326/conversational=327/news=305 (모두 >=300 tok)
- Known Bug 회피: --protected-prefix 4 (Bug 1), --kv-budget 200, sliding은 --window 2048 (Bug 2)
- prefill 고정: --ppl-prefill-tokens 150 (도메인 무관 통일 — 1차 budget 가변 절단 제거)

## ⚠ 프로토콜 편차 (지시 vs 구조 충돌, 보고 대상)
- 지시: "prefill을 프롬프트 전체 길이로 고정"
- 충돌: PPL 모드는 reference 텍스트(=프롬프트)를 prefill+decode로 분할. prefill=전체이면 decode=0 → eviction 미발동(n_evictions=0) → eviction 측정 불가.
- 차선: prefill을 도메인 무관 고정값 150으로 통일. 효과 = (1) 1차 budget(200) 가변 절단 제거, (2) 전 도메인 동일 prefill, (3) decode 154~177 step 확보로 eviction 1건 정상 발동.
- 검증: prefill=전체(304~327) 실행 시 전 도메인 n_evictions=0 확인 (raw/rkv_ppl0NL_*.log).

## 측정 1: R-KV (e2e 완주 — 1차 UNMEASURABLE 해소)
# 1단 (RkvStats 덤프):
ARGUS_RKV_DUMP=1 ./target/release/argus_eval --model-path <M> --tokenizer-path <T> --ppl <prompt> \
  --kv-type f32 --temperature 0.0 -b cpu --kv-budget 200 --protected-prefix 4 --ppl-prefill-tokens 150 \
  eviction rkv --lambda 0.1   # → [RkvStats] layer=L head=H mpc=X fraction=Y (128 lines/run, layer = plan순번 %16 역산)
# 2단 (sliding/h2o/rkv 3-way PPL): 동일 플래그로 eviction {sliding --window 2048 | h2o --keep-ratio 0.5 | rkv --lambda 0.1}

## 측정 2: A2SF (--dump-a2sf + --score-decay 스윕, 5도메인)
for decay in 0.0 0.7 0.8 0.9; do
  ./target/release/argus_eval --model-path <M> --tokenizer-path <T> --ppl <prompt> \
    --kv-type f32 --temperature 0.0 -b cpu --kv-budget 200 --protected-prefix 4 --ppl-prefill-tokens 150 \
    --score-decay $decay --dump-a2sf <out.json> eviction h2o --keep-ratio 0.5
done   # → [A2SF] bos_ratio=X bos_score=Y non_bos_mean=Z hh_topk_len=K + dump JSON

## 측정 3: head 분산 (--dump-importance, 5도메인)
./target/release/argus_eval --model-path <M> --tokenizer-path <T> --prompt-file <prompt> \
  --kv-type f32 -b cpu --dump-importance > <out.json>   # → head_concentration.matrix C_h + max_min_ratio
# 한계: last_step_head_attn() = 마지막 처리 layer만 (JSON layer 0 슬롯에 저장, layer 1-15 = 0). 단일 layer 해상도.

## 측정 4: Demote 실모델 PPL (DEMOTE_TEST_* env, dev 프로필, 5도메인)
DEMOTE_TEST_MODEL=<M> DEMOTE_TEST_TOKENIZER=<T> DEMOTE_TEST_TEXT=<prompt> \
  cargo test -p llm_rs2 --test demote_measure test_demote_vs_sliding_real_model_ppl -- --nocapture
# → [DemotePPL] (a) sliding PPL = X, (b) demote PPL = Y → GO/RED
# 비교 설계: (a) sliding 64토큰 F32 vs (b) demote 256토큰(창 밖 192 Q4 왕복). ~2min/도메인.
# 보조 NMSE: cargo test -p llm_rs2 --test demote_measure -- --nocapture (Q4/Q2 + demote/sliding NMSE)
