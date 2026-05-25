# DriftLens

Measuring how much an LLM's *reasoning trajectory* changes when its prompt is
perturbed — and training models to be robust against it.

This repository contains the reference implementation for the **DriftLens**
benchmark and the post training recipes used in the accompanying paper. Given a
chain-of-thought response, we map each reasoning step to a symbol from a small
ontology, then compare the resulting symbol sequence against a reference using
sequence-aware (DTW, edit-distance) and distributional (Jensen–Shannon) metrics.

The benchmark dataset is hosted on Hugging Face:
**[sata-bench/SymbolicDrift](https://huggingface.co/datasets/sata-bench/SymbolicDrift)**.

---

## What's in here

```
src/symbolic_drift/
├── clients.py            # Unified Bedrock client (Anthropic / OpenAI / DeepSeek / Qwen / ...)
├── prompts.py            # Symbol-mapping prompt + LLM-judge prompts
├── metrics.py            # DTW, edit distance, Jensen–Shannon, SRI
├── reward.py             # GRPO reward function (entrypoint for verl)
├── interventions.py      # Persona-perturbation library
├── judges.py             # Parsers for LLM-judge outputs
├── data/
│   ├── prepare.py                      # Build train/test parquet from raw GT
│   ├── generate_closed_source.py       # Bedrock-based GT generation
│   └── generate_open_source.py         # vLLM-based GT generation
└── evaluation/
    ├── closed_source.py                # Inference + scoring with Bedrock models
    └── vllm_inference.py               # Inference + scoring with local vLLM
```

Top-level:
- `data/ontology.json` — the reasoning-step ontology used by the symbol mapper.
- `scripts/` — reference GRPO launch scripts (`verl`-based).
- `configs/environment.yml` — pinned conda environment used for the experiments.

---

## Install

```bash
# Option A: conda env matching the paper exactly
conda env create -f configs/environment.yml
conda activate symbolic_drift

# Option B: editable pip install (lighter; pulls just core deps)
pip install -e .
pip install -e ".[bedrock]"   # if you need the AWS Bedrock clients
pip install -e ".[vllm]"      # if you need local vLLM inference
```

The training scripts in `scripts/` additionally require [`verl`](https://github.com/volcengine/verl)
and `flash-attn`. Both must be installed from source after the base environment is set up:

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/volcengine/verl && (cd verl && pip install -e .)
```

---

## Quick start

### 1. Compute Symbolic Drift metrics on a pair of symbol sequences

```python
from symbolic_drift.metrics import (
    compute_dtw_distance, compute_drift_distance, compute_sri,
)

reference = ["Protection of people and environment", "Ethical responsibility"]
candidate = ["Protection of people and environment", "Security and stability"]

_, normalized_dtw, _ = compute_dtw_distance(reference, candidate)
drift = compute_drift_distance(reference, candidate, alpha=0.5)
sri   = compute_sri(reference, [candidate])  # robustness of variants vs anchor
```

### 2. Map a free-form reasoning passage to a symbol sequence

```python
import json
from symbolic_drift.clients import BedrockClient
from symbolic_drift.prompts import build_mapping_prompt
from symbolic_drift.metrics import extract_symbols

ontology = json.load(open("data/ontology.json"))
rater = BedrockClient(model_id="qwen.qwen3-235b-a22b-2507-v1:0")

response = rater.generate(
    system_prompt="Please answer the question using specific format below.",
    user_prompt=build_mapping_prompt(reasoning="<the model's CoT here>", ontology=ontology),
)
symbols = extract_symbols(response)
```

### 3. Use the reward function inside `verl`

`symbolic_drift.reward` exposes a small set of atomic rewards plus a
`compute_score` entrypoint that `verl`'s `main_ppo` runner can call directly:

| Function                          | What it scores |
|-----------------------------------|----------------|
| `dtw_reward`                      | semantic alignment via cosine-DTW over symbol embeddings |
| `sri_reward`                      | single-anchor convex drift (edit + Jensen–Shannon) |
| `format_reward`                   | adherence to the `<step_*>...<answer>` template |
| `anti_distraction_reward`         | position + edit + prefix + DTW − unmapped − length-drift penalty |
| `readability_reward`              | Flesch reading-ease shaped to a target band |
| `answer_lexical_diversity_reward` | discourages degenerate / repetitive answers |

The default `compute_score` is a thin `0.5·DTW + 0.5·format` combo. Compose
your own freely — for example:

```python
# src/symbolic_drift/reward.py
def compute_score(solution_str, ground_truth, *, data_source="symbolic_test", extra_info=None):
    return (
        0.4 * dtw_reward(solution_str, ground_truth)
        + 0.4 * sri_reward(solution_str, ground_truth)
        + 0.2 * format_reward(solution_str)
    )
```

See `scripts/train_grpo_qwen3.sh` for a full launch example.

### 4. Run inference + scoring on the SymbolicDrift test set

```bash
# Closed-source models on Bedrock
python -m symbolic_drift.evaluation.closed_source \
    --model_alias claude_sonnet_4_5 \
    --test_data  data/qwen3_test.parquet \
    --ref_csv    data/qwen3_test.csv

# Open-source models with local vLLM
python -m symbolic_drift.evaluation.vllm_inference \
    --model Qwen/Qwen3-4B \
    --test_data data/qwen3_test.parquet \
    --ref_csv   data/qwen3_test.csv
```

Both scripts respect the following environment variables for path resolution:

- `SYMBOLIC_DATA_DIR`   — root for input parquet/csv files (default: `./data`).
- `SYMBOLIC_OUTPUT_DIR` — root for written CSVs (default: `./outputs`).
- `SYMBOLIC_ONTOLOGY`   — path to the ontology JSON (default: `data/ontology.json`).

### 5. AWS credentials

The `BedrockClient` uses the default AWS credential chain (`AWS_PROFILE`,
environment variables, instance profile, ...). It does **not** hardcode a
profile name. Set `AWS_PROFILE` if you need to pick a non-default one.

---

## Citation

A preprint is forthcoming. In the meantime please cite the dataset card:
<https://huggingface.co/datasets/sata-bench/SymbolicDrift>.

## License

MIT — see [LICENSE](LICENSE).
