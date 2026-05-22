"""Inference + scoring on the SymbolicDrift test set with local vLLM.

For each row in the test parquet we:

1. Generate a reasoning response from the target model with vLLM.
2. Map the response to a symbol sequence using a Bedrock rater.
3. Compute DTW and SRI against the row's ground-truth symbols.

Example
-------
    python -m symbolic_drift.evaluation.vllm_inference \\
        --model Qwen/Qwen3-4B \\
        --test_data ./data/qwen3_test.parquet \\
        --ref_csv   ./data/qwen3_test.csv
"""
from __future__ import annotations

import argparse
import os
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..clients import MODEL_ALIASES, build_client
from ..prompts import SYSTEM_PROMPT, build_mapping_prompt, load_ontology
from .closed_source import REF_COLUMNS, repeat_rows, score_response


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="HF model name or local path for vLLM.")
    p.add_argument("--rater_alias", default="qwen3_235b", choices=sorted(MODEL_ALIASES))

    p.add_argument("--test_data", type=Path, required=True)
    p.add_argument("--ref_csv",   type=Path, required=True)
    p.add_argument("--data_dir",  type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--ontology",   type=Path, default=DEFAULT_ONTOLOGY)

    p.add_argument("--exp_name", default="exp_vllm")
    p.add_argument("--project_name", default="Symbolic_GRPO-vllm")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--include_system_prompt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--test", action="store_true")
    return p.parse_args()


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Heavy import deferred so --help works without vLLM installed.
    from vllm import LLM, SamplingParams

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
    rater = build_client(model_alias=args.rater_alias, max_tokens=args.max_tokens, temperature=0.0)

    sampling = SamplingParams(temperature=float(args.temp), max_tokens=args.max_tokens)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    out_rows: list[dict] = []
    for batch in chunked(records, args.batch_size):
        prompts: list[str] = []
        meta: list[dict] = []
        for repeat_id, row in batch:
            prompt = row["prompt"]
            if args.include_system_prompt and len(prompt) > 1:
                prompt_text = prompt[0]["content"] + prompt[1]["content"]
            else:
                prompt_text = prompt[0]["content"]
            prompts.append(prompt_text)
            meta.append({
                "repeat_id": repeat_id,
                "row": row,
                "prompt_text": prompt_text,
                "ground_truth": row["reward_model"]["ground_truth"].tolist(),
            })

        outputs = llm.generate(prompts, sampling)
        for entry, output in zip(meta, outputs):
            response_text = output.outputs[0].text
            mapping_text = rater.generate(
                SYSTEM_PROMPT, build_mapping_prompt(reasoning=response_text, ontology=ontology),
            )
            scores = score_response(mapping_text, entry["ground_truth"])
            row = entry["row"]
            out_rows.append({
                "repeat_id":      entry["repeat_id"],
                "category":       row["category"],
                "intervention":   row["intervention"],
                "question":       row["question"],
                "ori_question":   row["ori_question"],
                "gt_responses":          row["gt_responses"],
                "gt_mapping_responses":  row["gt_mapping_responses"],
                "prompt":   entry["prompt_text"],
                "response": response_text,
                "mapping_response": mapping_text,
                "ground_truth":     entry["ground_truth"],
                **scores,
            })

    df_out = pd.DataFrame(out_rows)
    print(
        f"Overall  DTW: {df_out['dtw_score'].mean():.4f}  "
        f"SRI: {df_out['sri_score'].mean():.4f}"
    )

    safe_model_name = args.model.rstrip("/").split("/")[-1]
    out_path = args.output_dir / f"eval_{args.project_name}_{args.exp_name}_{safe_model_name}_t{args.temp}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
