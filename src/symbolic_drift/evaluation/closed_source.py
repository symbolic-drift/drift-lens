"""Inference + scoring on the SymbolicDrift test set with closed-source Bedrock models.

For each row in the test parquet we:

1. Generate a reasoning response from the target model.
2. Map the response to a symbol sequence using the rater model.
3. Compute DTW, SRI, and ROD against the row's ground-truth symbols.

Example
-------
    python -m symbolic_drift.evaluation.closed_source \\
        --model_alias claude_sonnet_4_5 \\
        --test_data ./data/qwen3_test.parquet \\
        --ref_csv   ./data/qwen3_test.csv
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..clients import MODEL_ALIASES, build_client
from ..metrics import (
    compute_corpus_symbol_distribution,
    compute_drift_distance,
    compute_dtw_distance,
    extract_symbols,
    jensen_shannon_distance,
)
from ..prompts import SYSTEM_PROMPT, build_mapping_prompt, load_ontology


DEFAULT_DATA_DIR = Path(os.environ.get("SYMBOLIC_DATA_DIR", "./data"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("SYMBOLIC_OUTPUT_DIR", "./outputs"))
DEFAULT_ONTOLOGY = Path(
    os.environ.get(
        "SYMBOLIC_ONTOLOGY",
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "ontology.json",
    )
)

REF_COLUMNS = [
    "category", "intervention", "question",
    "ori_question", "gt_responses", "gt_mapping_responses",
]


def chunked(iterable: Iterable, size: int) -> Iterable[list]:
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def repeat_rows(rows, repeat_count: int):
    for row in rows:
        for repeat_id in range(repeat_count):
            yield repeat_id, row


def score_response(mapping_response: str, ground_truth: list[str]) -> dict:
    """Compute DTW / SRI / ROD on a single (response_mapping, gt) pair."""
    response_symbols = extract_symbols(mapping_response)
    if not isinstance(ground_truth, list):
        raise TypeError(f"ground_truth must be list[str]: {ground_truth!r}")

    _, normalized_dtw, dtw_path = compute_dtw_distance(ground_truth, response_symbols)
    dtw_score = 1.0 - normalized_dtw if normalized_dtw == normalized_dtw else 0.0

    drift = compute_drift_distance(response_symbols, ground_truth, alpha=0.5)
    sri_score = 1.0 - drift

    p_response = compute_corpus_symbol_distribution([response_symbols])
    p_reference = compute_corpus_symbol_distribution([ground_truth])
    rod_score = 1.0 - jensen_shannon_distance(p_response, p_reference, normalize=True)

    return {
        "response_symbols": response_symbols,
        "dtw_path": dtw_path,
        "dtw_score": dtw_score,
        "sri_score": sri_score,
        "rod_score": rod_score,
    }


def extract_prompt_text(prompt, *, include_system: bool) -> str:
    """Concatenate the system + user content if the model accepts both, else user only."""
    if include_system and len(prompt) > 1:
        return prompt[0]["content"] + prompt[1]["content"]
    return prompt[0]["content"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_alias", default="claude_sonnet_4_5", choices=sorted(MODEL_ALIASES))
    p.add_argument("--model_id", default=None, help="Optional explicit Bedrock model ID override.")
    p.add_argument("--provider", default=None, help="Required when using --model_id directly.")
    p.add_argument("--rater_alias", default="qwen3_235b", choices=sorted(MODEL_ALIASES))

    p.add_argument("--test_data", type=Path, required=True)
    p.add_argument("--ref_csv",   type=Path, required=True)
    p.add_argument("--data_dir",  type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--ontology",   type=Path, default=DEFAULT_ONTOLOGY)

    p.add_argument("--exp_name", default="exp_closed_source")
    p.add_argument("--project_name", default="Symbolic_GRPO-claude_sonnet_45")
    p.add_argument("--include_system_prompt", action=argparse.BooleanOptionalAction, default=True,
                   help="Pass system + user content to the generator. Disable for Gemma-style models.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--parallel_workers", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--repeat", type=int, default=1, help="Repeat each row this many times.")
    p.add_argument("--test", action="store_true", help="Run on 3 rows only.")
    return p.parse_args()


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_data_path = _resolve(args.test_data, args.data_dir)
    ref_csv_path   = _resolve(args.ref_csv, args.data_dir)
    print(f"Reading test data: {test_data_path}")
    print(f"Reading reference: {ref_csv_path}")

    test = pd.read_parquet(test_data_path).reset_index(drop=True)
    ref  = pd.read_csv(ref_csv_path, usecols=REF_COLUMNS).reset_index(drop=True)
    if len(test) != len(ref):
        raise ValueError(f"Row count mismatch: test={len(test)} vs ref={len(ref)}")
    test = pd.concat([test, ref], axis=1)
    if args.test:
        test = test.iloc[:3].copy()

    records = list(repeat_rows(test.to_dict("records"), args.repeat))
    print(f"Inference rows: {len(records)} (repeat={args.repeat})")

    ontology = load_ontology(args.ontology)
    generator = build_client(
        model_alias=None if args.model_id else args.model_alias,
        model_id=args.model_id,
        provider=args.provider,
        max_tokens=args.max_tokens,
        temperature=args.temp,
    )
    rater = build_client(model_alias=args.rater_alias, max_tokens=args.max_tokens, temperature=0.0)

    out_rows: list[dict] = []
    for batch in chunked(records, args.batch_size):
        payloads = [
            {
                "repeat_id": repeat_id,
                "row": row,
                "prompt_text": extract_prompt_text(row["prompt"], include_system=args.include_system_prompt),
                "ground_truth": row["reward_model"]["ground_truth"].tolist(),
            }
            for repeat_id, row in batch
        ]

        with ThreadPoolExecutor(max_workers=min(args.parallel_workers, len(payloads))) as pool:
            responses = list(pool.map(
                lambda p: generator.generate(SYSTEM_PROMPT, p["prompt_text"]),
                payloads,
            ))
            mapping_responses = list(pool.map(
                lambda pair: rater.generate(
                    SYSTEM_PROMPT, build_mapping_prompt(reasoning=pair[1], ontology=ontology),
                ),
                zip(payloads, responses),
            ))

        for payload, response_text, mapping_text in zip(payloads, responses, mapping_responses):
            scores = score_response(mapping_text, payload["ground_truth"])
            row = payload["row"]
            out_rows.append({
                "repeat_id":     payload["repeat_id"],
                "category":      row["category"],
                "intervention":  row["intervention"],
                "question":      row["question"],
                "ori_question":  row["ori_question"],
                "gt_responses":          row["gt_responses"],
                "gt_mapping_responses":  row["gt_mapping_responses"],
                "prompt":         payload["prompt_text"],
                "response":       response_text,
                "mapping_response": mapping_text,
                "ground_truth":   payload["ground_truth"],
                **scores,
            })

    df_out = pd.DataFrame(out_rows)
    print(
        f"Overall  DTW: {df_out['dtw_score'].mean():.4f}  "
        f"SRI: {df_out['sri_score'].mean():.4f}  "
        f"ROD: {df_out['rod_score'].mean():.4f}"
    )

    suffix = generator.model_id.split(".")[-1].replace(":", "_")
    out_path = args.output_dir / f"eval_{args.project_name}_{args.exp_name}_{suffix}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
