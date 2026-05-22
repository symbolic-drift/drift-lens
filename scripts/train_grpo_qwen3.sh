#!/usr/bin/env bash
#
# GRPO fine-tuning of Qwen3-4B on the SymbolicDrift training data.
#
# Required environment:
#   * `verl` installed (https://github.com/volcengine/verl)
#   * `flash-attn` installed
#   * `symbolic_drift` package importable (`pip install -e .` from repo root)
#   * Bedrock-capable AWS credentials available via the default chain (the
#     reward function calls a rater LLM on Bedrock).
#
# Override any path / hyperparameter via env vars before invoking, e.g.:
#   TRAIN_FILE=./data/qwen3_train.parquet bash scripts/train_grpo_qwen3.sh
set -euo pipefail
set -x

PROJECT_NAME=${PROJECT_NAME:-Symbolic_GRPO-Qwen3-4B}
EXP_NAME=${EXP_NAME:-exp_001}
GEN_TP=${GEN_TP:-4}

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/data"}
CKPTS_DIR=${CKPTS_DIR:-"${REPO_ROOT}/ckpts/${PROJECT_NAME}/${EXP_NAME}"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/qwen3_train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_DIR}/qwen3_test.parquet"}

# Reward composition. See src/symbolic_drift/reward.py — by default
# `compute_score` is `0.5 * dtw_reward + 0.5 * format_reward`. Edit that file
# (or pass `custom_reward_function.name=...`) to use a different combination.
REWARD_PATH=${REWARD_PATH:-"${REPO_ROOT}/src/symbolic_drift/reward.py"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=8 \
    data.max_prompt_length=640 \
    data.max_response_length=640 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name=compute_score
