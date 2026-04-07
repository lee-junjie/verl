#!/bin/bash
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROJECT_NAME=${PROJECT_NAME:-verl_justrl_moe}
export PROJECT_PATH=/mnt/3fs2/data/junjie.li/rl-parity-test/verl
export SHARED_DATA_ROOT=/mnt/3fs2/data/shared_data/BytedTsinghua-SIA
export TRAIN_DATASET=${TRAIN_DATASET:-$SHARED_DATA_ROOT/DAPO-Math-17k/data/dapo-math-17k.parquet}
export TEST_AIME24=${TEST_AIME24:-$SHARED_DATA_ROOT/AIME-2024/data/aime-2024.parquet}
export TEST_DATASET=${TEST_DATASET:-"['$TEST_AIME24']"}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${PROJECT_NAME}_$(date +%Y%m%d_%H%M%S)_$$}
export VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-8}
export TRAIN_TP=${TRAIN_TP:-8}
export TRAIN_EP=${TRAIN_EP:-8}
export TRAIN_ETP=${TRAIN_ETP:-1}
export INFER_TP=${INFER_TP:-4}
export INFER_EP=${INFER_EP:-4}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
export OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-True}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
export CKPT_PATH=/mnt/3fs2/data/junjie.li/rl-parity-test/verl/checkpoints
export TMP_DIR=${TMP_DIR:-/mnt/3fs2/data/junjie.li/rl-parity-test/verl/tmp/justrl_moe}
mkdir -p "$TMP_DIR" "$CKPT_PATH"
NUM_NODES="${NUM_NODES:-1}"
RUN_LOG="${TMP_DIR}/run_training_${EXPERIMENT_NAME}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export TENSORBOARD_DIR=$TMP_DIR/logs/$PROJECT_NAME/$EXPERIMENT_NAME
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

mkdir -p "$TENSORBOARD_DIR"


cd "$PROJECT_PATH"

python -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.return_raw_chat=True \
    data.train_batch_size=256 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1024 \
    data.max_response_length=15360 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.truncation='error' \
    data.custom_cls.path=$PROJECT_PATH/examples/grpo_trainer/justrl_dataset.py \
    data.custom_cls.name=JustRLRLDataset \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TRAIN_TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$TRAIN_EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$TRAIN_ETP \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.expert_parallel_size=$INFER_EP \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.skip_tokenizer_init=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_new_tokens=31744 \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TRAIN_TP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$TRAIN_EP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$TRAIN_ETP \
    actor_rollout_ref.ref.megatron.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.max_resp_len=15360 \
    trainer.val_before_train=True \
    "trainer.logger=['console','tensorboard']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME"
