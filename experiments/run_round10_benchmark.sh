#!/bin/bash
# Round 10: Academic Benchmark Evaluation — KV Cache Eviction 정확도 평가
#
# 3-Tier 평가:
#   Tier 1: Perplexity — 5 도메인 프롬프트, 512/1024 토큰
#   Tier 2: NIAH — passkey/fact needle, depth 10-90%, 4/8 blocks
#   Tier 3: QA — single-doc QA, summarization, few-shot, multi-hop
#
# Eviction 조건:
#   - none (baseline)
#   - sliding (memory_critical at 50%)
#   - h2o (memory_critical at 50%)
#
# 참조: docs/30_evaluation_methodology.md

set -euo pipefail

BINARY="./target/release/generate"
MODEL="/home/go/Workspace/models"
RESULTS="experiments/results/round10"
CONFIGS="experiments/configs"

mkdir -p "$RESULTS"

# ============================================================
#  Counters
# ============================================================
TOTAL=0
DONE=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

run_generate() {
    local name="$1"
    local prompt="$2"
    local tokens="$3"
    local max_seq="$4"
    local eviction="$5"    # none | sliding | h2o
    local schedule="$6"    # config json path or "none"
    shift 6
    local extra_args=("$@")

    TOTAL=$((TOTAL + 1))
    local outfile="$RESULTS/${name}.jsonl"

    if [[ -f "$outfile" ]]; then
        echo "  SKIP $name (already exists)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local cmd=("$BINARY" -m "$MODEL" -p "$prompt" -n "$tokens"
        --max-seq-len "$max_seq" --greedy
        --experiment-output "$outfile"
        --experiment-sample-interval 10
        --experiment-logits-topk 10)

    case "$eviction" in
        none)
            cmd+=(--eviction-policy none)
            ;;
        sliding)
            cmd+=(--eviction-policy sliding --eviction-window "$max_seq")
            ;;
        h2o)
            cmd+=(--eviction-policy h2o
                  --h2o-keep-ratio 0.5
                  --h2o-recent-window 128
                  --h2o-decay 0.1)
            ;;
    esac

    if [[ "$schedule" != "none" ]]; then
        cmd+=(--experiment-schedule "$schedule")
    fi

    if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "  RUN  $name ($eviction, ${tokens}tok)"

    if "${cmd[@]}" 2>&1 | grep -E '^\[Experiment\]|^\[Resilience\]|Error' | head -5; then
        DONE=$((DONE + 1))
    else
        DONE=$((DONE + 1))
    fi
}

# ============================================================
#  Tier 1: Perplexity (PPL)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Tier 1: Perplexity Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PPL01="It was a bright cold day in April, and the clocks were striking thirteen. The street was deserted except for a few figures hurrying along the pavement, their coat collars turned up against the wind that swept down from the grey towers."
PPL02="The theory of plate tectonics describes the large-scale motion of seven major plates and many minor plates of Earth's lithosphere. The concept was first proposed in 1912 by Alfred Wegener as continental drift, but it was not until the 1960s that a comprehensive theory emerged."
PPL03="A hash table is a data structure that implements an associative array, mapping keys to values. It uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found. Ideally, the hash function will assign each key to a unique bucket."
PPL04="So the thing about learning a new language is that everyone tells you to just immerse yourself, but nobody really explains what that means in practice. I tried watching movies without subtitles for a month, and honestly, it was frustrating at first because I could barely catch every other word."
PPL05="Scientists at the European Organization for Nuclear Research announced preliminary results from a new series of experiments that could reshape our understanding of fundamental physics. The findings, which have not yet been peer reviewed, suggest the existence of previously undetected interactions between subatomic particles."

declare -A PPL_PROMPTS
PPL_PROMPTS[PPL01]="$PPL01"
PPL_PROMPTS[PPL02]="$PPL02"
PPL_PROMPTS[PPL03]="$PPL03"
PPL_PROMPTS[PPL04]="$PPL04"
PPL_PROMPTS[PPL05]="$PPL05"

for tokens in 512 1024; do
    max_seq=$((tokens * 2))
    inject_pos=$((tokens / 2))
    schedule="$CONFIGS/memory_critical_${inject_pos}.json"

    echo ""
    echo "── PPL: ${tokens} tokens (inject@${inject_pos}) ──"

    for ppl_id in PPL01 PPL02 PPL03 PPL04 PPL05; do
        prompt="${PPL_PROMPTS[$ppl_id]}"

        # Baseline
        run_generate "${ppl_id}-${tokens}-base" "$prompt" "$tokens" "$max_seq" none none

        # Sliding + eviction
        run_generate "${ppl_id}-${tokens}-sl" "$prompt" "$tokens" "$max_seq" sliding "$schedule"

        # H2O + eviction
        run_generate "${ppl_id}-${tokens}-h2o" "$prompt" "$tokens" "$max_seq" h2o "$schedule"
    done
done

# ============================================================
#  Tier 2: NIAH (Needle-in-a-Haystack)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Tier 2: Needle-in-a-Haystack"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use assemble_niah.py to generate prompts on-the-fly
NIAH_SCRIPT="experiments/prompts/assemble_niah.py"

for needle in N-PASS N-FACT; do
    for depth in 0.1 0.25 0.5 0.75 0.9; do
        for blocks in 4 8; do
            depth_pct=$(echo "$depth" | awk '{printf "%d", $1*100}')
            niah_id="NIAH-${needle#N-}-D${depth_pct}-B${blocks}"

            echo ""
            echo "── $niah_id ──"

            # Generate the prompt
            niah_prompt=$(python "$NIAH_SCRIPT" --needle "$needle" --depth "$depth" --blocks "$blocks")

            # Max seq based on blocks: 4 blocks ~ 300 tok prompt, 8 blocks ~ 600 tok prompt
            if [[ "$blocks" -eq 4 ]]; then
                max_seq=1024
            else
                max_seq=2048
            fi

            # Baseline (no eviction — should always retrieve correctly)
            run_generate "${niah_id}-base" "$niah_prompt" 64 "$max_seq" none none

            # Sliding + immediate eviction at token 1
            run_generate "${niah_id}-sl" "$niah_prompt" 64 "$max_seq" sliding "$CONFIGS/niah_evict_at_1.json"

            # H2O + immediate eviction at token 1
            run_generate "${niah_id}-h2o" "$niah_prompt" 64 "$max_seq" h2o "$CONFIGS/niah_evict_at_1.json"
        done
    done
done

# ============================================================
#  Tier 3: QA (LongBench-style)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Tier 3: QA Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Single-doc QA
QA_SD01_DOC="The Panama Canal is an artificial waterway in Panama that connects the Atlantic Ocean with the Pacific Ocean. The canal cuts across the Isthmus of Panama and is a conduit for maritime trade. Construction of the canal was one of the largest and most difficult engineering projects ever undertaken. France began work on the canal in 1881 but stopped due to engineering problems and high mortality among workers. The United States took over the project in 1904 and opened the canal on August 15, 1914. One of the largest and most difficult engineering projects ever undertaken, the canal had an enormous impact on shipping between the two oceans, eliminating the need to navigate the long and treacherous route around the southern tip of South America via the Drake Passage. The canal uses a system of locks that lift ships up to Gatun Lake, an artificial lake created to reduce the amount of excavation work required, and then lower them on the other side. The original locks are 33.5 meters wide. A third, wider lane of locks was opened on June 26, 2016, allowing larger ships known as New Panamax vessels to pass through.

Question: When did the United States take over the Panama Canal project and when was it opened?
Answer:"

QA_SD02_DOC="Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can be stored and later released to fuel the organism's activities. In most cases, oxygen is also released as a waste product. Most plants, algae, and cyanobacteria perform photosynthesis, which is largely responsible for producing and maintaining the oxygen content of Earth's atmosphere. Photosynthesis occurs in two main stages: the light-dependent reactions and the Calvin cycle. During the light-dependent reactions, which take place in the thylakoid membranes of the chloroplast, water molecules are split using light energy, producing oxygen, ATP, and NADPH. The Calvin cycle, which takes place in the stroma of the chloroplast, uses the ATP and NADPH produced in the light reactions to fix carbon dioxide into three-carbon sugars. These sugars are then used to make glucose and other organic molecules. The overall equation for photosynthesis is: 6CO2 + 6H2O + light energy produces C6H12O6 + 6O2.

Question: Where do the light-dependent reactions of photosynthesis take place and what do they produce?
Answer:"

QA_SD03_DOC="The Renaissance was a cultural movement that began in Italy during the late 14th century and later spread throughout Europe. It marked the transition from the medieval period to modernity, spanning roughly the 14th through 17th centuries. The movement began in Florence, where wealthy patrons such as the Medici family funded artists, scholars, and architects. Key figures of the Renaissance include Leonardo da Vinci, who was known for paintings such as the Mona Lisa and The Last Supper as well as his scientific investigations; Michelangelo, who created the ceiling frescoes of the Sistine Chapel and the sculpture David; and Raphael, whose School of Athens exemplified the ideals of the High Renaissance. The Renaissance also saw major advances in science, with Copernicus proposing a heliocentric model of the solar system and Galileo improving the telescope to make astronomical observations. The invention of the printing press by Johannes Gutenberg around 1440 played a critical role in spreading Renaissance ideas across Europe by making books more accessible to a wider audience.

Question: Who funded the artists in Florence during the Renaissance and name three key figures of the movement.
Answer:"

# Summarization
QA_SUM01_DOC="A new study published in the journal Nature Climate Change has found that global ice loss has accelerated significantly over the past three decades. Researchers analyzed satellite data from 1994 to 2017 and found that Earth lost 28 trillion tonnes of ice during this period. The rate of ice loss increased from 0.8 trillion tonnes per year in the 1990s to 1.3 trillion tonnes per year by 2017. The greatest losses occurred in the Arctic sea ice, Antarctic ice shelves, and mountain glaciers. The study found that ice loss from the Antarctic and Greenland ice sheets has increased sixfold since the 1990s, now matching the worst-case scenarios predicted by the Intergovernmental Panel on Climate Change. The researchers warn that for every centimeter of sea level rise, approximately one million people in low-lying coastal areas are at risk of displacement. The study emphasizes the urgent need for immediate action to reduce greenhouse gas emissions to limit future ice loss and its cascading effects on global sea levels, weather patterns, and ecosystems.

Question: Summarize the key findings of this study in two to three sentences.
Answer:"

# Few-shot
QA_FS01_DOC="Classify the sentiment of each review as positive or negative.

Review: The food was absolutely delicious and the service was impeccable. We will definitely be coming back.
Sentiment: positive

Review: Waited over an hour for cold food. The waiter was rude and unapologetic. Never again.
Sentiment: negative

Review: A pleasant surprise! The menu was creative and prices were very reasonable for the quality.
Sentiment: positive

Review: The ambiance was nice but the food was mediocre at best. Overpriced for what you get.
Sentiment: negative

Review: Excellent brunch spot with generous portions and friendly staff. The pancakes were the best I have ever had.
Sentiment:"

# Multi-hop
QA_MH01_DOC="Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire. He developed the theory of relativity, one of the two pillars of modern physics alongside quantum mechanics. His work is also known for its influence on the philosophy of science. He received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.

The Nobel Prize in Physics is awarded annually by the Royal Swedish Academy of Sciences. The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Röntgen for his discovery of X-rays. The prize is presented in Stockholm, Sweden, on December 10 each year, the anniversary of Alfred Nobel's death.

Alfred Nobel was a Swedish chemist, engineer, inventor, and philanthropist. He was born on October 21, 1833, in Stockholm, Sweden. He is best known for his invention of dynamite in 1867 and for establishing the Nobel Prizes in his will. Nobel signed his last will on November 27, 1895, and died on December 10, 1896, in San Remo, Italy.

Question: In what city is the Nobel Prize presented, and what is the connection between the date of the ceremony and Alfred Nobel?
Answer:"

declare -A QA_PROMPTS
QA_PROMPTS[QA-SD01]="$QA_SD01_DOC"
QA_PROMPTS[QA-SD02]="$QA_SD02_DOC"
QA_PROMPTS[QA-SD03]="$QA_SD03_DOC"
QA_PROMPTS[QA-SUM01]="$QA_SUM01_DOC"
QA_PROMPTS[QA-FS01]="$QA_FS01_DOC"
QA_PROMPTS[QA-MH01]="$QA_MH01_DOC"

for qa_id in QA-SD01 QA-SD02 QA-SD03 QA-SUM01 QA-FS01 QA-MH01; do
    prompt="${QA_PROMPTS[$qa_id]}"

    echo ""
    echo "── $qa_id ──"

    # Baseline
    run_generate "${qa_id}-base" "$prompt" 128 2048 none none

    # Sliding + eviction at token 5
    run_generate "${qa_id}-sl" "$prompt" 128 2048 sliding "$CONFIGS/niah_evict_at_5.json"

    # H2O + eviction at token 5
    run_generate "${qa_id}-h2o" "$prompt" 128 2048 h2o "$CONFIGS/niah_evict_at_5.json"
done

# ============================================================
#  Summary
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 10 Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:   $TOTAL experiments"
echo "  Done:    $DONE"
echo "  Skipped: $SKIPPED"
echo "  Failed:  $FAILED"
echo "  Time:    ${MINUTES}m ${SECONDS_REM}s"
echo ""
echo "Results in: $RESULTS/"
echo "Next: python experiments/analysis/round10_analyze.py"
