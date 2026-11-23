#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# ▼ この3行を追加（クォータ制限のない /dev/shm にモデルキャッシュと一時ファイルを逃がす）
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/tmp
export HF_HUB_ENABLE_HF_TRANSFER=0

INCLUDE_COT=true
CONFIG="dataset/qwen2.5-vl-3B-waymo"
OUTPUT_DIR="./dataset/waymo"

if [ "$INCLUDE_COT" = true ]; then
    echo "Preprocessing with Chain-of-Thought (CoT)..."
    CUDA_VISIBLE_DEVICES=0,1 python tools/preprocessing/cot_sample_generation.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
else
    echo "Preprocessing without Chain-of-Thought (No-CoT)..."
    python tools/preprocessing/nocot_sample_generation.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
fi