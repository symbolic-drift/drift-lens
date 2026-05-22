"""Generate baseline reasoning + symbol mappings for the SymbolicDrift questions
using closed-source models via AWS Bedrock.

For each ``(question_index, question)`` row in ``--questions`` we:

1. Sample ``--repeated_times`` reasoning responses from the generator model.
2. Run an instruction-following LLM judge over each response.
3. Map each response to a symbol sequence using the rater model.

The result is a CSV with one row per ``(question_index, repeat_id)``.

Example
-------
    python -m symbolic_drift.data.generate_closed_source \\
        --model_alias claude_sonnet_4_5 \\
        --questions ./data/raw_questions.csv \\
        --output_dir ./outputs
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..clients import MODEL_ALIASES, build_client
from ..prompts import (
    SYSTEM_PROMPT,
    build_completeness_prompt,
    build_instruction_following_judge_prompt,
    build_mapping_prompt,
    load_ontology,
)


DEFAULT_DATA_DIR = Path(os.environ.get("SYMBOLIC_DATA_DIR", "./data"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("SYMBOLIC_OUTPUT_DIR", "./outputs"))
DEFAULT_ONTOLOGY = Path(
    os.environ.get(
        "SYMBOLIC_ONTOLOGY",
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "ontology.json",
    )
)


def chunked(iterable: Iterable, size: int) -> Iterable[list]:
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def make_record(question_index: int, question: str, repeat_id: int) -> dict:
    user_prompt = build_completeness_prompt(question)
    return {
        "question_index": question_index,
        "repeat_id": repeat_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "prompt_text": user_prompt,
    }


def process_one(record: dict, generator, rater, ontology: str) -> tuple:
    response_text = generator.generate(SYSTEM_PROMPT, record["prompt_text"])
    eval_text = rater.generate(
        SYSTEM_PROMPT,
        build_instruction_following_judge_prompt(record["messages"], response_text),
    )
    mapping_text = rater.generate(
        SYSTEM_PROMPT,
        build_mapping_prompt(reasoning=response_text, ontology=ontology),
    )
    return (record["question_index"], record["repeat_id"], response_text, eval_text, mapping_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_alias", default="claude_sonnet_4_5", choices=sorted(MODEL_ALIASES))
    parser.add_argument("--model_id", default=None,
                        help="Optional explicit Bedrock model ID; overrides --model_alias.")
    parser.add_argument("--provider", default=None,
                        help="Required when using --model_id directly.")
    parser.add_argument("--rater_alias", default="qwen3_235b", choices=sorted(MODEL_ALIASES))
    parser.add_argument("--questions", type=Path, required=True,
                        help="CSV with columns 'question_index' and 'question'.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ontology",   type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--repeated_times", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--parallel_workers", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--test", action="store_true",
                        help="Only run on the first 2 questions; useful for smoke-testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.questions)
    questions = df[["question_index", "question"]].values.tolist()
    if args.test:
        questions = questions[:2]
    print(f"Loaded {len(questions)} questions from {args.questions}")

    records: list[dict] = []
    for question_index, question in questions:
        for repeat_id in range(args.repeated_times):
            records.append(make_record(question_index, question, repeat_id))

    ontology = load_ontology(args.ontology)
    generator = build_client(
        model_alias=None if args.model_id else args.model_alias,
        model_id=args.model_id,
        provider=args.provider,
        max_tokens=args.max_tokens,
        temperature=args.temp,
    )
    rater = build_client(model_alias=args.rater_alias, max_tokens=args.max_tokens, temperature=0.0)

    started = time.time()
    results: list[tuple] = []
    for batch in chunked(records, args.batch_size):
        with ThreadPoolExecutor(max_workers=min(args.parallel_workers, len(batch))) as pool:
            results.extend(pool.map(lambda r: process_one(r, generator, rater, ontology), batch))
    print(f"Inference finished in {time.time() - started:.1f}s")

    df_res = pd.DataFrame(
        results,
        columns=["question_index", "repeat_id", "response", "evaluation_response", "mapping_response"],
    )
    df_out = pd.merge(df, df_res, on="question_index", how="inner")

    suffix = generator.model_id.split(".")[-1].replace(":", "_")
    out_path = args.output_dir / f"ground_truth_{suffix}_repeated{args.repeated_times}.csv"
    df_out.to_csv(out_path, index=False, escapechar="\\")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
