#!/usr/bin/env python3
"""Core quality metrics library for experiment analysis.

Computes FDT, EMR, Suffix EMR, ROUGE-L, BLEU-4, and Top-K Overlap
between baseline and experiment token sequences. No external dependencies.
"""

import json
import math
from collections import Counter


def load_jsonl(filepath):
    """Load JSONL file, return (token_records: list[dict], summary: dict).

    Args:
        filepath: Path to the JSONL file.

    Returns:
        Tuple of (token_records, summary). token_records is a list of
        per-token dicts. summary is the final summary dict (with _summary=True).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no summary record is found.
    """
    token_records = []
    summary = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("_summary"):
                summary = record
            else:
                token_records.append(record)

    if summary is None:
        raise ValueError(f"No summary record found in {filepath}")

    return token_records, summary


def compute_fdt(base_tokens, exp_tokens):
    """First Divergent Token: min(i) where exp[i].token_id != base[i].token_id.

    Returns len(min(base, exp)) if all compared positions match.

    Args:
        base_tokens: List of baseline token records.
        exp_tokens: List of experiment token records.

    Returns:
        Integer index of first divergent token.
    """
    min_len = min(len(base_tokens), len(exp_tokens))
    for i in range(min_len):
        if base_tokens[i]["token_id"] != exp_tokens[i]["token_id"]:
            return i
    return min_len


def compute_emr(base_tokens, exp_tokens):
    """Exact Match Rate: count(matching) / min(len(base), len(exp)).

    Args:
        base_tokens: List of baseline token records.
        exp_tokens: List of experiment token records.

    Returns:
        Float between 0.0 and 1.0.
    """
    min_len = min(len(base_tokens), len(exp_tokens))
    if min_len == 0:
        return 1.0
    matches = sum(
        1
        for i in range(min_len)
        if base_tokens[i]["token_id"] == exp_tokens[i]["token_id"]
    )
    return matches / min_len


def compute_suffix_emr(base_tokens, exp_tokens, fdt):
    """Suffix EMR: matches after FDT / tokens after FDT.

    Returns 1.0 if fdt >= min(len(base), len(exp)) (no divergence).

    Args:
        base_tokens: List of baseline token records.
        exp_tokens: List of experiment token records.
        fdt: First Divergent Token index.

    Returns:
        Float between 0.0 and 1.0.
    """
    min_len = min(len(base_tokens), len(exp_tokens))
    if fdt >= min_len:
        return 1.0
    suffix_len = min_len - fdt
    matches = sum(
        1
        for i in range(fdt, min_len)
        if base_tokens[i]["token_id"] == exp_tokens[i]["token_id"]
    )
    return matches / suffix_len


def _lcs_length(a, b):
    """Compute length of Longest Common Subsequence between two lists.

    Uses O(min(m,n)) space optimization.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        Integer length of LCS.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0

    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (len(b) + 1)

    return prev[len(b)]


def compute_rouge_l(base_text, exp_text):
    """ROUGE-L using LCS. Tokenize by whitespace.

    Args:
        base_text: Reference text string.
        exp_text: Hypothesis text string.

    Returns:
        Dict with "precision", "recall", "f1" float values.
    """
    base_words = base_text.split()
    exp_words = exp_text.split()

    if not base_words and not exp_words:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not base_words or not exp_words:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = _lcs_length(base_words, exp_words)

    precision = lcs_len / len(exp_words) if exp_words else 0.0
    recall = lcs_len / len(base_words) if base_words else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def _count_ngrams(words, n):
    """Count n-grams in a word list.

    Args:
        words: List of word strings.
        n: N-gram size.

    Returns:
        Counter of n-gram tuples.
    """
    ngrams = Counter()
    for i in range(len(words) - n + 1):
        ngrams[tuple(words[i : i + n])] += 1
    return ngrams


def compute_bleu4(base_text, exp_text):
    """BLEU-4: geometric mean of 1-4gram precision + brevity penalty.

    Implements the standard BLEU score from Papineni et al. (2002).
    Uses clipped n-gram precision with a single reference.

    Args:
        base_text: Reference text string.
        exp_text: Hypothesis (candidate) text string.

    Returns:
        Float BLEU-4 score between 0.0 and 1.0.
    """
    ref_words = base_text.split()
    hyp_words = exp_text.split()

    if not hyp_words:
        return 0.0
    if not ref_words:
        return 0.0

    # Brevity penalty
    bp = 1.0
    if len(hyp_words) < len(ref_words):
        bp = math.exp(1 - len(ref_words) / len(hyp_words))

    # Clipped n-gram precisions for n=1..4
    log_precisions = []
    for n in range(1, 5):
        ref_ngrams = _count_ngrams(ref_words, n)
        hyp_ngrams = _count_ngrams(hyp_words, n)

        if not hyp_ngrams:
            # If hypothesis has fewer than n words, precision is 0
            return 0.0

        clipped_count = 0
        total_count = 0
        for ngram, count in hyp_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += count

        if total_count == 0:
            return 0.0

        precision = clipped_count / total_count
        if precision == 0:
            return 0.0

        log_precisions.append(math.log(precision))

    # Geometric mean of precisions (uniform weights)
    avg_log_precision = sum(log_precisions) / len(log_precisions)
    bleu = bp * math.exp(avg_log_precision)

    return bleu


def compute_topk_overlap(base_tokens, exp_tokens, k=10):
    """Per-token Top-K overlap: |topK_base intersect topK_exp| / K.

    Each token's top_logits field has [[token_id, logit_value], ...].
    Compares up to min(len(base), len(exp)) positions.

    Args:
        base_tokens: List of baseline token records.
        exp_tokens: List of experiment token records.
        k: Number of top logits to compare.

    Returns:
        List of overlap ratios, one per token position.
    """
    min_len = min(len(base_tokens), len(exp_tokens))
    overlaps = []

    for i in range(min_len):
        base_logits = base_tokens[i].get("top_logits", [])
        exp_logits = exp_tokens[i].get("top_logits", [])

        base_ids = set(entry[0] for entry in base_logits[:k])
        exp_ids = set(entry[0] for entry in exp_logits[:k])

        if not base_ids and not exp_ids:
            overlaps.append(1.0)
        elif not base_ids or not exp_ids:
            overlaps.append(0.0)
        else:
            overlap = len(base_ids & exp_ids) / k
            overlaps.append(overlap)

    return overlaps
