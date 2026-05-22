"""GRPO reward building blocks for SymbolicDrift training.

This module is the entrypoint pointed at by ``custom_reward_function.path`` in
the ``verl`` GRPO launch scripts. It exposes a small set of atomic sub-rewards
that score one (response, reference) pair on a single axis — composition is
left to the user.

Atomic rewards
--------------
* :func:`dtw_reward`              — semantic alignment via cosine-DTW over symbol embeddings.
* :func:`sri_reward`              — single-anchor convex drift (edit + Jensen-Shannon).
* :func:`format_reward`           — adherence to the ``<step_*>...<answer>`` template.
* :func:`anti_distraction_reward` — sharper, position-sensitive variant for robustness training.
* :func:`readability_reward`      — Flesch reading-ease shaped to a target band.
* :func:`answer_lexical_diversity_reward` — discourages degenerate / repetitive answers.

Composing your own reward
-------------------------
``compute_score`` is the function ``verl`` calls. The default below is a thin
``0.5·DTW + 0.5·format`` combo. Replace it (in place, in a fork, or by setting
``custom_reward_function.name`` to one of your own functions) to suit your
training recipe::

    # symbolic_drift/reward.py
    def compute_score(solution_str, ground_truth, *, data_source="symbolic_test", extra_info=None):
        return (
            0.4 * dtw_reward(solution_str, ground_truth)
            + 0.4 * sri_reward(solution_str, ground_truth)
            + 0.2 * format_reward(solution_str)
        )

The mapper LLM and ontology used by DTW / SRI / anti-distraction are loaded
lazily on first use, so importing this module is cheap and side-effect free.
"""
from __future__ import annotations

import logging
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import metrics
from .clients import build_client
from .prompts import SYSTEM_PROMPT, build_mapping_prompt, load_ontology

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lazy ontology / rater singletons.
# --------------------------------------------------------------------------- #

DEFAULT_ONTOLOGY_PATH = Path(
    os.environ.get(
        "SYMBOLIC_ONTOLOGY",
        Path(__file__).resolve().parent.parent.parent / "data" / "ontology.json",
    )
)
DEFAULT_RATER_ALIAS = os.environ.get("SYMBOLIC_DRIFT_RATER_ALIAS", "qwen3_235b")


@lru_cache(maxsize=1)
def _get_ontology() -> str:
    return load_ontology(DEFAULT_ONTOLOGY_PATH)


@lru_cache(maxsize=1)
def _get_rater():
    return build_client(model_alias=DEFAULT_RATER_ALIAS)


def _map_to_symbols(solution_str: str) -> list[str]:
    """Run the mapper LLM on ``solution_str`` and extract its symbol sequence."""
    rater = _get_rater()
    prompt = build_mapping_prompt(reasoning=solution_str, ontology=_get_ontology())
    response = rater.generate(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, temperature=0.0)
    symbols = metrics.extract_symbols(response)
    if not isinstance(symbols, list):
        raise TypeError(f"extract_symbols returned non-list: {symbols!r}")
    return symbols


def _coerce_reference(ground_truth) -> list[str]:
    if not isinstance(ground_truth, list):
        raise TypeError(
            f"ground_truth must be list[str], got {type(ground_truth).__name__}: {ground_truth!r}"
        )
    return ground_truth


# --------------------------------------------------------------------------- #
# Atomic sub-rewards
# --------------------------------------------------------------------------- #

def dtw_reward(solution_str: str, ground_truth) -> float:
    """``1 − normalized_DTW`` between the response's mapped symbols and the reference."""
    response_symbols = _map_to_symbols(solution_str)
    reference = _coerce_reference(ground_truth)
    _, normalized_distance, _ = metrics.compute_dtw_distance(reference, response_symbols)
    if math.isnan(normalized_distance):
        return 0.0
    return 1.0 - normalized_distance


def sri_reward(solution_str: str, ground_truth, *, alpha: float = 0.5) -> float:
    """``1 − (α·edit + (1−α)·JS)`` drift between response and reference."""
    response_symbols = _map_to_symbols(solution_str)
    reference = _coerce_reference(ground_truth)
    drift = metrics.compute_drift_distance(reference, response_symbols, alpha=alpha)
    return 1.0 - drift


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_STEP_RE = re.compile(
    r"(?:<step_?(\d+)>(.*?)(?=<step_\d+>|<answer>|</think>)|"
    r"(?:Step|step)_?(\d+):\s*(.*?)(?=(?:Step|step)_?\d+:|<answer>|</think>))",
    re.IGNORECASE | re.DOTALL,
)


def format_reward(solution_str: str, *, min_step_chars: int = 2) -> float:
    """Reward for adhering to the ``<step_*> ... <answer>`` template.

    Awards 0.25 for any detected step, +0.25 for an ``<answer>`` block, and
    +0.5 if at least one step has non-trivial content.
    """
    answer = _ANSWER_RE.search(solution_str)
    steps: list[tuple[int, str]] = []
    for m in _STEP_RE.finditer(solution_str):
        if m.group(1):  # <step_1>...</step_1>
            steps.append((int(m.group(1)), m.group(2).strip()))
        elif m.group(3):  # Step 1: ... or step_1: ...
            steps.append((int(m.group(3)), m.group(4).strip()))

    if not steps:
        return 0.0

    reward = 0.25
    if answer is not None:
        reward += 0.25
    if any(len(content) >= min_step_chars for _, content in steps):
        reward += 0.5
    return reward


def readability_reward(solution_str: str, *, mu: float = 60.0, sigma: float = 25.0) -> float:
    """Gaussian-shaped reward over Flesch reading ease, peaked at ``mu``."""
    import textstat

    fre = textstat.flesch_reading_ease(solution_str)
    return math.exp(-((fre - mu) ** 2) / (2 * sigma**2))


def answer_lexical_diversity_reward(solution_str: str, *, min_tokens: int = 5,
                                    sigmoid_d0: float = 0.2, sigmoid_k: float = 8.0) -> float:
    """Discourage degenerate / highly repetitive ``<answer>`` blocks."""
    answer_match = _ANSWER_RE.search(solution_str)
    if not answer_match:
        return 0.0
    answer = answer_match.group(1).strip()
    tokens = [t.lower() for t in answer.split() if t.isalpha()]
    if not tokens:
        return 0.0

    diversity = len(set(tokens)) / len(tokens)
    length_scale = min(1.0, len(tokens) / min_tokens)
    raw = diversity * length_scale
    return 1.0 / (1.0 + math.exp(-sigmoid_k * (raw - sigmoid_d0)))


# --------------------------------------------------------------------------- #
# Anti-distraction (a.k.a. perturbation-robust) reward
# --------------------------------------------------------------------------- #

def _safe_dtw_similarity(reference: list[str], candidate: list[str]) -> float:
    if not reference or not candidate:
        return 0.0
    _, normalized_distance, _ = metrics.compute_dtw_distance(reference, candidate)
    if math.isnan(normalized_distance):
        return 0.0
    return float(np.clip(1.0 - normalized_distance, 0.0, 1.0))


def _position_accuracy(reference: list[str], candidate: list[str]) -> float:
    if not reference:
        return 0.0
    return sum(1 for r, c in zip(reference, candidate) if r == c) / len(reference)


def _symbol_edit_similarity(reference: list[str], candidate: list[str]) -> float:
    if not reference and not candidate:
        return 1.0
    if not reference or not candidate:
        return 0.0
    edit = metrics.symbol_edit_distance(reference, candidate)
    return max(0.0, 1.0 - edit / max(len(reference), len(candidate), 1))


def _prefix_match_ratio(reference: list[str], candidate: list[str]) -> float:
    if not reference:
        return 0.0
    matches = 0
    for r, c in zip(reference, candidate):
        if r != c:
            break
        matches += 1
    return matches / len(reference)


def anti_distraction_reward(solution_str: str, ground_truth) -> float:
    """Sharper symbolic reward intended for perturbation robustness.

    Compared to plain DTW this adds:

    * exact position-sensitive symbol matching,
    * discrete symbol edit similarity,
    * prefix consistency to reward early recovery,
    * explicit penalties for unmapped symbols and length drift.

    Returns a value in ``[0, 1]``.
    """
    response_symbols = metrics.normalize_symbol_sequence(_map_to_symbols(solution_str))
    reference = metrics.normalize_symbol_sequence(_coerce_reference(ground_truth))

    if not reference or not response_symbols:
        return 0.0

    position_score = _position_accuracy(reference, response_symbols)
    edit_score     = _symbol_edit_similarity(reference, response_symbols)
    prefix_score   = _prefix_match_ratio(reference, response_symbols)
    dtw_score      = _safe_dtw_similarity(reference, response_symbols)

    unmapped_count = sum(1 for s in response_symbols if s.lower() == "unmapped")
    unmapped_ratio = unmapped_count / max(len(response_symbols), 1)

    length_gap = abs(len(reference) - len(response_symbols))
    length_penalty = min(1.0, length_gap / max(len(reference), 1))

    raw = (
        0.35 * position_score
        + 0.30 * edit_score
        + 0.15 * prefix_score
        + 0.20 * dtw_score
        - 0.25 * unmapped_ratio
        - 0.10 * length_penalty
    )
    return float(np.clip(raw, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Default verl entrypoint. Override / replace freely.
# --------------------------------------------------------------------------- #

def compute_score(
    solution_str: str,
    ground_truth,
    data_source: str = "symbolic_test",
    extra_info=None,
) -> float:
    """Default reward: ``0.5 · DTW + 0.5 · format``.

    Replace this function (or point ``custom_reward_function.name`` at one of
    your own) to compose any combination of the atomic rewards above.
    """
    return 0.5 * dtw_reward(solution_str, ground_truth) + 0.5 * format_reward(solution_str)
