"""Build train/test parquet datasets by combining a per-model ground-truth CSV
with the persona-perturbation library.

Inputs
------
* ``--gt_data``    A CSV of (question, response, mapping_response, gt) rows with
                   the "gt" column being a Python-literal string of a list of
                   symbols. Produced by ``symbolic_drift.data.generate_*``.
* ``--questions``  A CSV with a ``raw_question`` column (the anchor questions).

Output
------
A ``(csv, parquet)`` pair under ``--output_dir``:

* The CSV is a flat table with one row per ``(question, intervention)`` pair,
  preserving columns useful for downstream analysis.
* The parquet is the verl-compatible payload (``data_source``, ``prompt``,
  ``reward_model``).

Example
-------
    python -m symbolic_drift.data.prepare \\
        --gt_data ./data/ground_truth_qwen3_selected.csv \\
        --questions ./data/test_raw_questions.csv \\
        --model_name qwen3 \\
        --part test \\
        --output_dir ./data
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

from ..interventions import INTERVENTIONS, PERSONA_CATEGORIES
from ..prompts import build_completeness_prompt


def _flatten_interventions() -> tuple[dict[str, str], list[str]]:
    string_to_category: dict[str, str] = {}
    for category, strings in INTERVENTIONS.items():
        for s in strings:
            string_to_category[s] = category
    return string_to_category, list(string_to_category.keys())


def build_record_with_persona(question: str, intervention: str, gt: list[str]) -> dict:
    """Persona-style record: persona disclosure goes in the system prompt."""
    system = (
        "You are a helpful assistant with access to user memory context.\n\n"
        "## User Memory Context\n"
        f"{intervention}\n\n"
        "Please answer the question using specific format below. \n"
    )
    return {
        "data_source": "symbolic_test",
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": build_completeness_prompt(question)},
        ],
        "reward_model": {"style": "function", "ground_truth": gt},
    }


def build_record_with_prefix(question: str, intervention: str, gt: list[str]) -> dict:
    """Non-persona record: intervention is prepended to the user question itself."""
    system = "You are a helpful assistant.\n\nPlease answer the question using specific format below. \n"
    return {
        "data_source": "symbolic_test",
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": build_completeness_prompt(intervention + question)},
        ],
        "reward_model": {"style": "function", "ground_truth": gt},
    }


def expand(
    raw_questions: list[str],
    intervention_strings: list[str],
    intervention_to_category: dict[str, str],
    q2gt: dict[str, tuple[str, str, list[str]]],
) -> tuple[list[dict], pd.DataFrame]:
    """Cartesian product of questions × interventions, with metadata.

    Returns the training records (for parquet) and a flat metadata frame
    (for CSV).
    """
    records: list[dict] = []
    rows: list[dict] = []

    for question in raw_questions:
        if question not in q2gt:
            print(f"[skip] no ground truth for question: {question[:80]!r}")
            continue
        gt_response, gt_mapping_response, gt = q2gt[question]
        for intervention in intervention_strings:
            category = intervention_to_category[intervention]
            builder = build_record_with_persona if category in PERSONA_CATEGORIES else build_record_with_prefix
            record = builder(question, intervention, gt)
            records.append(record)
            rows.append(
                {
                    "input": record,
                    "reward_model": gt,
                    "category": category,
                    "intervention": intervention,
                    "question": question,
                    "ori_question": question,
                    "gt_responses": gt_response,
                    "gt_mapping_responses": gt_mapping_response,
                    "gts": gt,
                }
            )
    return records, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt_data", required=True, type=Path,
                   help="CSV of (question, response, mapping_response, gt) rows.")
    p.add_argument("--questions", required=True, type=Path,
                   help="CSV with a 'raw_question' column.")
    p.add_argument("--model_name", required=True,
                   help="Model name used in the output filenames (e.g. 'qwen3').")
    p.add_argument("--part", required=True, choices=["train", "test"],
                   help="Split label used in the output filenames.")
    p.add_argument("--output_dir", default=Path("./data"), type=Path,
                   help="Directory to write the .csv and .parquet outputs into.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Heavy import deferred so `--help` works without `datasets` installed.
    import datasets

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ground-truth file: {args.gt_data}")
    df_gt = pd.read_csv(args.gt_data)
    df_gt["gt"] = df_gt["gt"].apply(ast.literal_eval)
    q2gt: dict[str, tuple[str, str, list[str]]] = {
        row["question"]: (row["response"], row["mapping_response"], row["gt"])
        for _, row in df_gt.iterrows()
    }

    print(f"Loading raw questions: {args.questions}")
    raw_questions = pd.read_csv(args.questions)["raw_question"].tolist()
    print(f"  {len(raw_questions)} questions")

    intervention_to_category, intervention_strings = _flatten_interventions()
    print(f"  {len(intervention_strings)} intervention strings across {len(set(intervention_to_category.values()))} categories")

    records, df_flat = expand(raw_questions, intervention_strings, intervention_to_category, q2gt)
    print(f"Expanded to {len(records)} (question, intervention) rows.")

    csv_path = args.output_dir / f"{args.model_name}_{args.part}.csv"
    parquet_path = args.output_dir / f"{args.model_name}_{args.part}.parquet"

    df_flat.to_csv(csv_path, index=False)
    datasets.Dataset.from_list(records).to_parquet(parquet_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {parquet_path}")


if __name__ == "__main__":
    main()
