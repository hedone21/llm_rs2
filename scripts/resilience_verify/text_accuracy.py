"""Detokenize experiment JSONL and compute accuracy metrics."""

from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Dict, List

from .log_parser import load_token_records

# Reuse project quality metrics helpers
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS_PATH = _PROJECT_ROOT / "experiments"
if str(_EXPERIMENTS_PATH) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_PATH))

from analysis.quality_metrics import compute_rouge_l, compute_bleu4  # noqa: E402


def _load_tokenizer(tokenizer_path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "`tokenizers` package required. Install with: pip install tokenizers"
        ) from e
    return Tokenizer.from_file(str(tokenizer_path))


def decode_jsonl_to_text(jsonl_path: Path, tokenizer_path: Path) -> str:
    """Collect all token_id values from the JSONL and decode via HF tokenizer."""
    records = load_token_records(jsonl_path)
    ids: List[int] = []
    for r in records:
        tid = r.get("token_id")
        if tid is None:
            continue
        try:
            ids.append(int(tid))
        except (TypeError, ValueError):
            continue
    if not ids:
        return ""
    tok = _load_tokenizer(tokenizer_path)
    try:
        return tok.decode(ids, skip_special_tokens=False)
    except TypeError:
        # older tokenizers versions
        return tok.decode(ids)


def compare_texts(baseline_text: str, action_text: str) -> Dict[str, float]:
    """Return {rouge_l_f1, bleu_4, char_similarity}."""
    rouge = compute_rouge_l(baseline_text, action_text)
    bleu = compute_bleu4(baseline_text, action_text)
    char_sim = difflib.SequenceMatcher(None, baseline_text, action_text).ratio()
    return {
        "rouge_l_f1": float(rouge.get("f1", 0.0)),
        "rouge_l_precision": float(rouge.get("precision", 0.0)),
        "rouge_l_recall": float(rouge.get("recall", 0.0)),
        "bleu_4": float(bleu),
        "char_similarity": float(char_sim),
    }
