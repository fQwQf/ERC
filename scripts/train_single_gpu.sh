#!/bin/bash
# Single GPU Training with Gradient Accumulation
# This avoids multi-GPU memory issues

set -e

MAX_SAMPLES=${1:-30000}
EPOCHS=${2:-3}

echo "=============================================="
echo "Single GPU Training with Gradient Accumulation"
echo "=============================================="
echo "Max samples: $MAX_SAMPLES"
echo "Epochs: $EPOCHS"
echo "=============================================="

# Set environment variables
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo ""
echo "Starting training on single GPU..."
python scripts/train_multi_dataset.py \
    --max_samples $MAX_SAMPLES \
    --epochs $EPOCHS \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr 2e-4 \
    --gpus 0

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="
