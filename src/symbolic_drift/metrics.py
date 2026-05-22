"""Symbolic Drift metrics.

This module is intentionally numpy-only at the top level: heavy dependencies
(``torch``, ``transformers``, ``sentence-transformers``) are imported lazily
inside :func:`get_sentence_embedding` so that the package remains importable
in environments without a GPU or even without those packages installed.

Conventions
-----------
A *symbol sequence* is a ``list[str]``; each element is treated as an atomic
unit by edit-distance and histogram metrics, while the DTW metric embeds each
symbol as a sentence.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from typing import Sequence

import numpy as np

# Heavy deps are imported lazily — see _ensure_embedding_model.
_TOKENIZER = None
_MODEL = None
_DEVICE = None
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _ensure_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> tuple:
    """Lazily load the sentence-embedding model and tokenizer."""
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is not None:
        return _TOKENIZER, _MODEL, _DEVICE

    import torch
    from transformers import AutoModel, AutoTokenizer

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _MODEL = AutoModel.from_pretrained(model_name).to(_DEVICE)
    _MODEL.eval()
    return _TOKENIZER, _MODEL, _DEVICE


def _mean_pool(model_output, attention_mask):
    import torch  # local

    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def get_sentence_embedding(sentences: Sequence[str]) -> np.ndarray:
    """Encode each input string with the default sentence-transformers model."""
    if not sentences:
        return np.zeros((0, 384), dtype=np.float32)

    import torch
    import torch.nn.functional as F

    tokenizer, model, device = _ensure_embedding_model()
    encoded = tokenizer(list(sentences), padding=True, truncation=True, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    embeddings = _mean_pool(output, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


# --------------------------------------------------------------------------- #
# Sequence-aware distances (DTW + edit distance)
# --------------------------------------------------------------------------- #

def compute_dtw_distance(
    reference: Sequence[str],
    candidate: Sequence[str],
) -> tuple[float, float, list[tuple[int, int]]]:
    """Cosine-DTW distance between two symbol sequences.

    Returns
    -------
    raw_distance : float
        Sum of cosine distances along the optimal alignment path.
    normalized_distance : float
        Raw distance divided by alignment-path length, in roughly ``[0, 2]``
        (for cosine distance).
    path : list[tuple[int, int]]
        The alignment path produced by ``fastdtw``.
    """
    if not reference or not candidate:
        return float("nan"), float("nan"), []

    from fastdtw import fastdtw
    from scipy.spatial.distance import cosine

    ref_emb = get_sentence_embedding(reference)
    cand_emb = get_sentence_embedding(candidate)
    distance, path = fastdtw(ref_emb, cand_emb, dist=cosine)
    normalized = distance / max(len(path), 1) if not np.isnan(distance) else float("nan")
    return float(distance), float(normalized), path


def symbol_edit_distance(seq_i: Sequence[str], seq_j: Sequence[str]) -> int:
    """Symbol-level Levenshtein distance (each list element is one atomic symbol)."""
    n, m = len(seq_i), len(seq_j)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_i[i - 1] == seq_j[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def compute_normalized_edit_distance(seq_i: Sequence[str], seq_j: Sequence[str]) -> float:
    """Normalized symbol-level edit distance: ``edit / max(|seq_i|, |seq_j|)``."""
    if not seq_i and not seq_j:
        return 0.0
    max_len = max(len(seq_i), len(seq_j))
    if max_len == 0:
        return 0.0
    return symbol_edit_distance(seq_i, seq_j) / max_len


# --------------------------------------------------------------------------- #
# Distributional distances (histogram + Jensen–Shannon)
# --------------------------------------------------------------------------- #

def compute_histogram(sequence: Sequence[str]) -> dict[str, float]:
    """Normalized symbol histogram: counts divided by sequence length."""
    if not sequence:
        return {}
    counts = Counter(sequence)
    total = len(sequence)
    return {k: v / total for k, v in counts.items()}


def kl_divergence(p: dict[str, float], q: dict[str, float], *, epsilon: float = 1e-8) -> float:
    """KL divergence ``D_KL(p || q)`` with epsilon smoothing."""
    keys = set(p) | set(q)
    total = 0.0
    for key in keys:
        pv = p.get(key, 0.0) + epsilon
        qv = q.get(key, 0.0) + epsilon
        total += pv * np.log(pv / qv)
    return float(total)


def jensen_shannon_divergence(
    h_i: dict[str, float],
    h_j: dict[str, float],
    *,
    epsilon: float = 1e-8,
) -> float:
    """``JSDiv(h_i, h_j) = 0.5 D_KL(h_i || m) + 0.5 D_KL(h_j || m)``."""
    keys = set(h_i) | set(h_j)
    m = {key: 0.5 * (h_i.get(key, 0.0) + h_j.get(key, 0.0)) for key in keys}
    return 0.5 * kl_divergence(h_i, m, epsilon=epsilon) + 0.5 * kl_divergence(h_j, m, epsilon=epsilon)


def jensen_shannon_distance(
    h_i: dict[str, float],
    h_j: dict[str, float],
    *,
    epsilon: float = 1e-8,
    normalize: bool = True,
) -> float:
    """Square root of JS divergence, optionally normalized to ``[0, 1]``."""
    js = np.sqrt(jensen_shannon_divergence(h_i, h_j, epsilon=epsilon))
    if normalize:
        js = js / np.sqrt(np.log(2))
    return float(js)


# --------------------------------------------------------------------------- #
# Combined drift, SRI
# --------------------------------------------------------------------------- #

def compute_drift_distance(
    seq_i: Sequence[str],
    seq_j: Sequence[str],
    *,
    alpha: float = 0.5,
    epsilon: float = 1e-8,
) -> float:
    """Convex combination of edit distance and Jensen–Shannon distance.

    ``d(S_i, S_j) = α · d_seq(S_i, S_j) + (1−α) · d_dist(h_i, h_j)``
    """
    d_seq = compute_normalized_edit_distance(seq_i, seq_j)
    d_dist = jensen_shannon_distance(
        compute_histogram(seq_i), compute_histogram(seq_j), epsilon=epsilon, normalize=True
    )
    return alpha * d_seq + (1.0 - alpha) * d_dist


def compute_perturbation_drift(
    anchor: Sequence[str],
    variants: Sequence[Sequence[str]],
    *,
    alpha: float = 0.5,
) -> float:
    """Mean drift from anchor to its perturbed variants."""
    if not variants:
        return 0.0
    return sum(compute_drift_distance(anchor, v, alpha=alpha) for v in variants) / len(variants)


def compute_sri(
    anchor: Sequence[str],
    variants: Sequence[Sequence[str]],
    *,
    alpha: float = 0.5,
) -> float:
    """Symbolic Robustness Index: ``1 − D_pert(Q)``. Higher is more robust."""
    return 1.0 - compute_perturbation_drift(anchor, variants, alpha=alpha)


# --------------------------------------------------------------------------- #
# Symbol extraction from raw model output
# --------------------------------------------------------------------------- #

_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_JSON_LIST_RE = re.compile(r"(\[\s*\{.*?\}\s*\])", re.DOTALL)


def extract_symbols(raw: str | None) -> list[str]:
    """Extract the ordered list of ``symbol`` strings from a mapper response.

    The mapper is prompted to return a JSON array inside ``<answer>...</answer>``,
    e.g. ``[{"step_index": 1, "category": "...", "symbol": "..."}]``. This
    function falls back to scanning for any JSON-array-of-objects if the
    ``<answer>`` block is missing.
    """
    if not isinstance(raw, str):
        return []

    candidates: list[str] = []
    if (m := _ANSWER_BLOCK_RE.search(raw)) is not None:
        candidates.append(m.group(1))
    if (m := _JSON_LIST_RE.search(raw)) is not None:
        candidates.append(m.group(1))

    for payload in candidates:
        cleaned = payload.strip().replace('""', '"')
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        symbols = [
            item["symbol"] for item in data
            if isinstance(item, dict) and "symbol" in item
        ]
        if symbols:
            return symbols

    return []


def normalize_symbol_sequence(sequence) -> list[str]:
    """Coerce arbitrary inputs to ``list[str]`` of stripped, non-empty symbols."""
    if sequence is None:
        return []
    if isinstance(sequence, np.ndarray):
        sequence = sequence.tolist()
    if not isinstance(sequence, list):
        raise TypeError(f"symbol sequence must be list-like, got {type(sequence).__name__}")
    out: list[str] = []
    for item in sequence:
        if item is None:
            continue
        token = str(item).strip()
        if token:
            out.append(token)
    return out


# --------------------------------------------------------------------------- #
# Generic XML/JSON parsers shared by the LLM-judge wrappers in judges.py
# --------------------------------------------------------------------------- #

def parse_json_eval(raw: str | None) -> dict[str, str]:
    """Parse a JSON object out of an LLM-judge response."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {raw!r}")
    body = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    return json.loads(body)


def parse_xml_eval(raw: str | None) -> dict[str, str]:
    """Parse ``<response><answer>...</answer><reasoning>...</reasoning></response>``.

    Tolerant of mismatched tags and stray content.
    """
    text = (raw or "").strip()
    text = re.sub(r"^\s*```(?:xml)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    if (m := re.search(r"<response\b[^>]*>.*</response\s*>", text, flags=re.DOTALL | re.IGNORECASE)):
        text = m.group(0)

    def _extract(tag: str) -> str:
        primary = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}\s*>", text, flags=re.DOTALL | re.IGNORECASE)
        if primary:
            return primary.group(1).strip()
        # Fallback: tag opened but never closed before </response> / EOF.
        fallback = re.search(rf"<{tag}\b[^>]*>(.*?)(</response\s*>|\Z)",
                             text, flags=re.DOTALL | re.IGNORECASE)
        return fallback.group(1).strip() if fallback else ""

    def _strip_tags(s: str) -> str:
        s = html.unescape(s)
        s = re.sub(r"<[^>]+>", " ", s)
        return " ".join(s.split()).strip()

    return {
        "answer": _strip_tags(_extract("answer")),
        "reasoning": _strip_tags(_extract("reasoning")),
    }
