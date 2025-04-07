#!/bin/bash

# --- 配置区 ---

# 1. 模型设置
export MODEL_TYPE='AIDC-AI/Ovis2-2B' # 指定您要微调的基础模型 (ModelScope ID 或本地路径)

# 2. 数据集设置
#    - 使用您在 @register_dataset 中定义的 dataset_name
#    - 提供 train 和 validation JSONL 文件的路径
export CUSTOM_DATASET_NAME='handwritten-math-formula-jsonl' 
export TRAIN_DATA_PATH='./data/processed/fullhand/train.jsonl' # 训练集 JSONL 路径
export VAL_DATA_PATH='./data/processed/fullhand/val.jsonl'   # 验证集 JSONL 路径

# 3. 输出设置
export OUTPUT_DIR='./output/Ovis2-2B-MathFormula-lora' # 指定输出目录

# 4. 训练参数 (根据需要调整)
export NUM_EPOCHS=1                      # 训练轮数
export BATCH_SIZE=1                      # 每个设备的训练批次大小 (根据显存调整)
export EVAL_BATCH_SIZE=1                 # 每个设备的评估批次大小
export LR=1e-4                           # 学习率
export GRAD_ACC_STEPS=16                 # 梯度累积步数 (有效批次大小 = BATCH_SIZE * GRAD_ACC_STEPS * NUM_GPUS)
export MAX_LENGTH=1024                   # 最大序列长度 (需要足够容纳图片编码+提示+较长的LaTeX公式)

# 5. LoRA 参数 (保持或调整)
export LORA_RANK=8
export LORA_ALPHA=32
export LORA_TARGET='all-linear'          # LoRA 应用的模块

# 6. 环境设置
#    - 设置 PYTHONPATH 包含 src 目录，以便 Swift 能找到您的自定义数据集脚本
export PYTHONPATH=${PYTHONPATH}:$(pwd)/src 
#    - 设置可见的 GPU (例如使用第一块 GPU)
export CUDA_VISIBLE_DEVICES=0 

# --- 执行 Swift SFT 命令 ---

swift sft \
    --model_type ${MODEL_TYPE} \
    --dataset ${CUSTOM_DATASET_NAME}=${TRAIN_DATA_PATH} ${CUSTOM_DATASET_NAME}=${VAL_DATA_PATH} \
    --sft_type lora \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_length ${MAX_LENGTH} \
    --learning_rate ${LR} \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_target_modules ${LORA_TARGET} \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --warmup_ratio 0.05 \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --lazy_preprocess true \
    --report_to tensorboard 
    # --deepspeed default-zero2 # 如果需要，启用 DeepSpeed

echo "微调脚本执行完毕。"