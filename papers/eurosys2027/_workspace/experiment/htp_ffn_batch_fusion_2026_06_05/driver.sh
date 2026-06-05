#!/usr/bin/env bash
# HTP FFN gate/up batch fusion — A/B device measurement (Galaxy S25, R3CY408S5SB).
#
# A/B = 한 바이너리, env LLMRS_DISABLE_HTP_FFN_BATCH 토글:
#   OFF(=batch disabled, fallback 개별 matmul_transposed) vs ON(=batch enqueue×2→drain×1).
# 모든 다른 변수(모델/프롬프트/스레드) 고정 → floor recovery 효과만 격리.
#
# 사용: bash driver.sh   (raw 로그 → raw/, 요약 → stdout)
set -u
SERIAL=R3CY408S5SB
DEV=/data/local/tmp
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="$HERE/raw"
mkdir -p "$RAW"

MODEL="models/qwen2.5-1.5b-gguf/qwen2.5-1.5b-q4_0.gguf"
TOK="models/qwen2.5-1.5b-gguf/tokenizer.json"
PROMPT='The capital of France is'
LIBS="LD_LIBRARY_PATH=$DEV:/vendor/lib64:/system/lib64"
ADSP="ADSP_LIBRARY_PATH=$DEV"

# $1=raw_file  $2=prefix_env  $3=backend  $4=ntok
run() {
  adb -s "$SERIAL" shell "cd $DEV && $2 $LIBS taskset 3f ./argus_cli \
    --backend $3 --model-path $MODEL --tokenizer-path $TOK \
    --prompt '$PROMPT' --num-tokens $4 --greedy 2>&1" >"$1" 2>&1
}
decode_ms() { grep -oE 'Decode: [0-9.]+ ms/tok' "$1" | grep -oE '[0-9.]+' | head -1; }
gen_line()  { grep -E 'The capital of France is .+(area|people|million)' "$1" | head -1; }
first_id()  { grep -oE 'first=[0-9]+' "$1" | head -1; }

echo "## 정확성 (16 tok, greedy)"
run "$RAW/correct_htp_on.log"  "$ADSP"                                "htp" 16
run "$RAW/correct_cpu.log"     ""                                     "cpu" 16
run "$RAW/correct_htp_off.log" "LLMRS_DISABLE_HTP_FFN_BATCH=1 $ADSP"  "htp" 16
echo "  HTP batch ON : first=$(first_id "$RAW/correct_htp_on.log")  batch_log=$(grep -c 'gate/up batch dispatch 활성' "$RAW/correct_htp_on.log")"
echo "    text: $(gen_line "$RAW/correct_htp_on.log")"
echo "  CPU          : first=$(first_id "$RAW/correct_cpu.log")"
echo "    text: $(gen_line "$RAW/correct_cpu.log")"
echo "  HTP batch OFF: first=$(first_id "$RAW/correct_htp_off.log")  batch_log=$(grep -c 'gate/up batch dispatch 활성' "$RAW/correct_htp_off.log")"
echo "    text: $(gen_line "$RAW/correct_htp_off.log")"

echo ""
echo "## TBT (64 tok, n=3 median) — Decode ms/tok"
echo -n "  batch OFF: "
for i in 1 2 3; do run "$RAW/tbt_off_$i.log" "LLMRS_DISABLE_HTP_FFN_BATCH=1 $ADSP" "htp" 64; echo -n "$(decode_ms "$RAW/tbt_off_$i.log") "; done; echo
echo -n "  batch ON : "
for i in 1 2 3; do run "$RAW/tbt_on_$i.log" "$ADSP" "htp" 64; echo -n "$(decode_ms "$RAW/tbt_on_$i.log") "; done; echo
