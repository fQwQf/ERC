#!/bin/bash
# Multi-GPU Training Launcher
# Usage: bash scripts/launch_multi_gpu.sh [num_gpus] [max_samples] [epochs]

set -e

NUM_GPUS=${1:-8}
MAX_SAMPLES=${2:-30000}
EPOCHS=${3:-3}

echo "=============================================="
echo "Multi-GPU Training Launcher"
echo "=============================================="
echo "GPUs: $NUM_GPUS"
echo "Max samples: $MAX_SAMPLES"
echo "Epochs: $EPOCHS"
echo "=============================================="

# Set environment variables
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# Launch training with torchrun
echo ""
echo "Starting 7B SOTA Training (BS=3, 6 GPUs)..."
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=6 \
    scripts/train_7b_sota.py \
    --max_samples 100000 \
    --epochs 3 \
    --batch_size 3 \
    --grad_acc 14 \
    --lr 2e-5 \
    --output_dir outputs/checkpoints/7b_sota_v2

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="
