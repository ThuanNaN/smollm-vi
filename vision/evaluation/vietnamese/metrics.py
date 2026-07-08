"""Text metrics for Vietnamese VQA evaluation (exact match, token F1)."""

import re
import string
import unicodedata


def normalize_answer(s: str) -> str:
    """NFC-normalize, lowercase, drop punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred: str, refs: list[str]) -> float:
    p = normalize_answer(pred)
    return 1.0 if any(p == normalize_answer(r) for r in refs) else 0.0


def _f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in ref_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, refs: list[str]) -> float:
    """Max token-level F1 of pred against any reference."""
    pred_tokens = normalize_answer(pred).split()
    return max((_f1(pred_tokens, normalize_answer(r).split()) for r in refs),
               default=0.0)
