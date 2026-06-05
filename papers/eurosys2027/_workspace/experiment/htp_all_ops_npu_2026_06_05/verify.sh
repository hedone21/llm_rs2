#!/usr/bin/env bash
# HTP all-ops-NPU 정확성 검증 (Galaxy S25 R3CY408S5SB).
#
# baseline(all CPU ops) 토큰열 기준으로 각 op 의 NPU env 를 (1) 개별 isolation
# (2) 전체 동시 로 켜며 token-id 16/16 일치(정확성 보존) + decode TBT 확인.
# 개별 isolation 이 핵심 — 누적은 첫 깨진 op 가 이후를 가린다.
# env: LLMRS_HTP_NPU_{SILU,ROPE,RMSNORM,ADD,ATTN}=1.
#
# 사용: bash verify.sh [model] [tok]   (기본 q4)
set -u
SERIAL=R3CY408S5SB; DEV=/data/local/tmp
HERE="$(cd "$(dirname "$0")" && pwd)"; RAW="$HERE/raw"; mkdir -p "$RAW"
MODEL="${1:-models/qwen2.5-1.5b-gguf/qwen2.5-1.5b-q4_0.gguf}"
TOK="${2:-models/qwen2.5-1.5b-gguf/tokenizer.json}"
PROMPT='The capital of France is'
LIBS="LD_LIBRARY_PATH=$DEV:/vendor/lib64:/system/lib64"; ADSP="ADSP_LIBRARY_PATH=$DEV"
OPS="${OPS:-SILU ROPE RMSNORM ADD ATTN}"   # 전 op (attention 포함)

run() {  # $1=label  $2=prefix_env  $3=ntok
  adb -s "$SERIAL" shell "cd $DEV && $2 $LIBS $ADSP taskset 3f ./argus_cli \
    --backend htp --model-path $MODEL --tokenizer-path $TOK \
    --prompt '$PROMPT' --num-tokens $3 --greedy 2>&1" >"$RAW/$1.log" 2>&1
}
gen() { grep -E '^The capital of France is' "$RAW/$1.log" | head -1; }
dms() { grep -oE 'Decode: [0-9.]+ ms/tok' "$RAW/$1.log" | grep -oE '[0-9.]+' | head -1; }
chk() { if diff -q <(gen "$1") <(gen baseline) >/dev/null 2>&1 && [ -n "$(gen baseline)" ]; then echo "16/16 OK "; else echo "MISMATCH"; fi; }

echo "## baseline (all CPU, env none)"
run baseline "" 16
echo "  text: $(gen baseline)"
echo "  decode $(dms baseline) ms/tok"
echo ""
echo "## 개별 isolation (각 op 단독 enable)"
for op in $OPS; do
  lc=$(echo "$op" | tr 'A-Z' 'a-z')
  run "iso_$lc" "LLMRS_HTP_NPU_${op}=1" 16
  printf "  %-8s : %s | %s\n" "$op" "$(chk iso_$lc)" "$(gen iso_$lc)"
done
echo ""
echo "## 전체 동시 enable"
ALL=""; for op in $OPS; do ALL="$ALL LLMRS_HTP_NPU_${op}=1"; done
run all16 "$ALL" 16
echo "  ALL(16tok): $(chk all16) | $(gen all16)"
run all64 "$ALL" 64
echo "  ALL(64tok) decode: $(dms all64) ms/tok  (baseline $(dms baseline))"
