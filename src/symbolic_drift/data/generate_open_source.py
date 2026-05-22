"""Generate baseline reasoning + symbol mappings using local open-source models
served by ``vLLM``.

For each ``(question_index, question)`` row in ``--questions`` we:

1. Sample ``--repeated_times`` reasoning responses with vLLM.
2. Run an instruction-following LLM judge (Bedrock rater) over each response.
3. Map each response to a symbol sequence using the same rater.

The result is a CSV with one row per ``(question_index, repeat_id)``.

Example
-------
    python -m symbolic_drift.data.generate_open_source \\
        --model Qwen/Qwen3-4B \\
        --questions ./data/raw_questions.csv \\
        --output_dir ./outputs
"""
from __future__ import annotations

import argparse
import os
import time
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


def build_messages(question: str, *, include_system: bool) -> list[dict]:
    user_prompt = build_completeness_prompt(question)
    if include_system:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    return [{"role": "user", "content": user_prompt}]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="HF model name or local path for vLLM.")
    p.add_argument("--rater_alias", default="qwen3_235b", choices=sorted(MODEL_ALIASES))
    p.add_argument("--questions", type=Path, required=True,
                   help="CSV with columns 'question_index' and 'question'.")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--ontology",   type=Path, default=DEFAULT_ONTOLOGY)
    p.add_argument("--repeated_times", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Prompts per vLLM batch. Defaults to 8 × num_gpus.")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Heavy deps imported here so `--help` and module import don't pull them in.
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_csv(args.questions)
    questions = df[["question_index", "question"]].values.tolist()
    if args.test:
        questions = questions[:2]
    print(f"Loaded {len(questions)} questions from {args.questions}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Gemma's chat template doesn't accept a system role.
    include_system = "gemma" not in args.model.lower()
    records: list[dict] = []
    for question_index, question in questions:
        messages = build_messages(question, include_system=include_system)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        for repeat_id in range(args.repeated_times):
            records.append({
                "question_index": question_index,
                "repeat_id": repeat_id,
                "messages": messages,
                "prompt_text": prompt_text,
            })

    ontology = load_ontology(args.ontology)
    rater = build_client(model_alias=args.rater_alias, max_tokens=args.max_tokens, temperature=0.0)

    n_gpus = max(1, torch.cuda.device_count())
    print(f"Detected GPUs: {n_gpus}")
    llm = LLM(model=args.model, tensor_parallel_size=n_gpus)
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=args.temp)
    batch_size = args.batch_size or 8 * n_gpus

    started = time.time()
    rows: list[tuple] = []
    for batch in chunked(records, batch_size):
        prompts = [r["prompt_text"] for r in batch]
        outputs = llm.generate(prompts, sampling)
        for record, output in zip(batch, outputs):
            response_text = output.outputs[0].text
            eval_text = rater.generate(
                SYSTEM_PROMPT,
                build_instruction_following_judge_prompt(record["messages"], response_text),
            )
            mapping_text = rater.generate(
                SYSTEM_PROMPT,
                build_mapping_prompt(reasoning=response_text, ontology=ontology),
            )
            rows.append((record["question_index"], record["repeat_id"], response_text, eval_text, mapping_text))
    print(f"Inference finished in {time.time() - started:.1f}s")

    df_res = pd.DataFrame(
        rows,
        columns=["question_index", "repeat_id", "response", "evaluation_response", "mapping_response"],
    )
    df_out = pd.merge(df, df_res, on="question_index", how="inner")

    safe_model_name = args.model.replace("/", "_")
    out_path = args.output_dir / f"ground_truth_{safe_model_name}_repeated{args.repeated_times}.csv"
    df_out.to_csv(out_path, index=False, escapechar="\\")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
