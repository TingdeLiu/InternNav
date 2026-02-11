#!/bin/bash

# Local multi-GPU training script for InternVLA-N1-System2
# Usage:
#   Default (2 GPUs: 6,7):  bash ./scripts/train/qwenvl_train/train_system2_qcyl.sh
#   Custom GPUs:            CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/train/qwenvl_train/train_system2_qcyl.sh
#   All 8 GPUs:             CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/train/qwenvl_train/train_system2_qcyl.sh

# Default to GPU 6,7 if not specified
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}

# Calculate number of GPUs from CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using ${NUM_GPUS} GPUs: ${CUDA_VISIBLE_DEVICES}"

# Random port for distributed training
MASTER_PORT=$((RANDOM % 101 + 29500))

# DeepSpeed configuration
deepspeed=scripts/train/qwenvl_train/zero2.json

# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct

# Training hyperparameters
lr=2e-5
vision_tower_lr=5e-6
batch_size=2
grad_accum_steps=1
max_pixels=313600
min_pixels=3136

# Dataset configuration
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30

# Output configuration
run_name=InternVLA-N1-System2
output_dir=checkpoints/${run_name}

torchrun --standalone --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    \
    --output_dir ${output_dir} \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb
