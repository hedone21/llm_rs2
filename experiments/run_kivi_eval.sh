#!/bin/bash
# KIVI Q2 KV Cache accuracy evaluation
# Compares: Baseline (FP32 KV) vs KIVI (Q2 + FP32 residual)
# Prompts: PPL-01~05, QA-FS-01~03
# Tokens: 256 (sufficient for quality comparison)
set -euo pipefail

BIN="./target/release/generate"
MODEL="models/llama3.2-1b"
OUTDIR="experiments/results/kivi"
TOKENS=256
TOPK=10

# PPL prompts
declare -A PPL_PROMPTS
PPL_PROMPTS[PPL-01]="It was a bright cold day in April, and the clocks were striking thirteen. The street was deserted except for a few figures hurrying along the pavement, their coat collars turned up against the wind that swept down from the grey towers."
PPL_PROMPTS[PPL-02]="The theory of plate tectonics describes the large-scale motion of seven major plates and many minor plates of Earth's lithosphere. The concept was first proposed in 1912 by Alfred Wegener as continental drift, but it was not until the 1960s that a comprehensive theory emerged."
PPL_PROMPTS[PPL-03]="A hash table is a data structure that implements an associative array, mapping keys to values. It uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found. Ideally, the hash function will assign each key to a unique bucket."
PPL_PROMPTS[PPL-04]="So the thing about learning a new language is that everyone tells you to just immerse yourself, but nobody really explains what that means in practice. I tried watching movies without subtitles for a month, and honestly, it was frustrating at first because I could barely catch every other word."
PPL_PROMPTS[PPL-05]="Scientists at the European Organization for Nuclear Research announced preliminary results from a new series of experiments that could reshape our understanding of fundamental physics. The findings, which have not yet been peer reviewed, suggest the existence of previously undetected interactions between subatomic particles."

# Few-shot prompts
declare -A FS_PROMPTS
FS_PROMPTS[QA-FS-01]='Classify the sentiment of each review as positive or negative.

Review: The food was absolutely delicious and the service was impeccable. We will definitely be coming back.
Sentiment: positive

Review: Waited over an hour for cold food. The waiter was rude and unapologetic. Never again.
Sentiment: negative

Review: A pleasant surprise! The menu was creative and prices were very reasonable for the quality.
Sentiment: positive

Review: The ambiance was nice but the food was mediocre at best. Overpriced for what you get.
Sentiment: negative

Review: Excellent brunch spot with generous portions and friendly staff. The pancakes were the best I have ever had.
Sentiment:'

FS_PROMPTS[QA-FS-02]='Determine the category of each news headline.

Headline: Stock Market Hits Record High as Tech Shares Surge
Category: business

Headline: New Species of Deep-Sea Fish Discovered Near Hydrothermal Vents
Category: science

Headline: National Team Advances to World Cup Semifinals After Dramatic Penalty Shootout
Category: sports

Headline: Parliament Passes Landmark Climate Legislation After Months of Debate
Category: politics

Headline: Researchers Develop Novel Vaccine Platform Using mRNA Technology for Malaria Prevention
Category:'

FS_PROMPTS[QA-FS-03]='Translate the following English words to their French equivalents.

English: house
French: maison

English: book
French: livre

English: water
French: eau

English: cat
French: chat

English: garden
French:'

echo "=== KIVI Q2 Accuracy Evaluation ==="
echo "Model: $MODEL"
echo "Tokens: $TOKENS"
echo ""

# Run PPL prompts: baseline + KIVI
for id in PPL-01 PPL-02 PPL-03 PPL-04 PPL-05; do
    echo "--- $id: Baseline ---"
    $BIN -m "$MODEL" -p "${PPL_PROMPTS[$id]}" -n "$TOKENS" --greedy \
        --experiment-output "$OUTDIR/BASE-${id}.jsonl" \
        --experiment-logits-topk "$TOPK" \
        --experiment-sample-interval 0 \
        --backend cpu --kv-type f32 2>&1 | grep -E '^\[|Done'

    echo "--- $id: KIVI ---"
    $BIN -m "$MODEL" -p "${PPL_PROMPTS[$id]}" -n "$TOKENS" --greedy \
        --kivi --kivi-residual-size 32 \
        --experiment-output "$OUTDIR/KIVI-${id}.jsonl" \
        --experiment-logits-topk "$TOPK" \
        --experiment-sample-interval 0 \
        --backend cpu 2>&1 | grep -E '^\[|Done'
    echo ""
done

# Run Few-shot prompts: baseline + KIVI (fewer tokens — answer is short)
for id in QA-FS-01 QA-FS-02 QA-FS-03; do
    echo "--- $id: Baseline ---"
    $BIN -m "$MODEL" -p "${FS_PROMPTS[$id]}" -n 32 --greedy \
        --experiment-output "$OUTDIR/BASE-${id}.jsonl" \
        --experiment-logits-topk "$TOPK" \
        --experiment-sample-interval 0 \
        --backend cpu --kv-type f32 2>&1 | grep -E '^\[|Done'

    echo "--- $id: KIVI ---"
    $BIN -m "$MODEL" -p "${FS_PROMPTS[$id]}" -n 32 --greedy \
        --kivi --kivi-residual-size 32 \
        --experiment-output "$OUTDIR/KIVI-${id}.jsonl" \
        --experiment-logits-topk "$TOPK" \
        --experiment-sample-interval 0 \
        --backend cpu 2>&1 | grep -E '^\[|Done'
    echo ""
done

# Also run longer sequence (512 tokens) for PPL-01 to test extended generation
echo "--- PPL-01-512: Baseline (512 tokens) ---"
$BIN -m "$MODEL" -p "${PPL_PROMPTS[PPL-01]}" -n 512 --greedy \
    --experiment-output "$OUTDIR/BASE-PPL01-512.jsonl" \
    --experiment-logits-topk "$TOPK" \
    --experiment-sample-interval 0 \
    --backend cpu --kv-type f32 2>&1 | grep -E '^\[|Done'

echo "--- PPL-01-512: KIVI (512 tokens) ---"
$BIN -m "$MODEL" -p "${PPL_PROMPTS[PPL-01]}" -n 512 --greedy \
    --kivi --kivi-residual-size 32 \
    --experiment-output "$OUTDIR/KIVI-PPL01-512.jsonl" \
    --experiment-logits-topk "$TOPK" \
    --experiment-sample-interval 0 \
    --backend cpu 2>&1 | grep -E '^\[|Done'

echo ""
echo "=== All experiments complete ==="
echo "Results in: $OUTDIR/"
ls -la "$OUTDIR/"
