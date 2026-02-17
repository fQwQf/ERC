#!/bin/bash
# Quick start guide for ERC project

echo "=============================================="
echo "ERC 细粒度情绪识别项目 - 快速开始"
echo "=============================================="
echo ""

# Check environment
echo "[1] 检查环境..."
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 Anaconda/Miniconda"
    exit 1
fi

# Activate environment
echo "[2] 激活 conda 环境..."
source $(conda info --base)/etc/profile.d/conda.sh
if ! conda activate erc 2>/dev/null; then
    echo "创建新的 conda 环境..."
    conda create -n erc python=3.10 -y
    conda activate erc
    pip install -r requirements.txt
fi

# Set mirrors
echo "[3] 配置国内镜像..."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

echo ""
echo "=============================================="
echo "可用命令"
echo "=============================================="
echo ""
echo "1. 评估 EmpatheticDialogues 数据集 (100样本):"
echo "   python scripts/evaluate_dataset.py --dataset empathetic --max_samples 100"
echo ""
echo "2. 评估 GoEmotions 数据集 (100样本):"
echo "   python scripts/evaluate_dataset.py --dataset goemotions --max_samples 100"
echo ""
echo "3. 整理项目结构:"
echo "   python scripts/organize_project.py"
echo ""
echo "4. 下载数据集:"
echo "   bash scripts/download_datasets.sh"
echo ""
echo "=============================================="
echo "数据集位置"
echo "=============================================="
echo "数据目录: data/"
echo "  - raw/:      原始数据"
echo "  - processed/: 预处理数据"
echo "  - json/:     JSON格式数据"
echo ""
echo "=============================================="
echo "模型检查点"
echo "=============================================="
echo "位置: outputs/checkpoints/run_20260211_112814/"
echo "  - final_model/: 最终模型"
echo "  - best_model/:  最佳模型 (F1=0.668)"
echo ""
echo "=============================================="
echo "评估结果"
echo "=============================================="
echo "位置: outputs/results/"
echo ""

# Show current status
echo "当前项目状态:"
python scripts/organize_project.py 2>/dev/null || echo "请确保已安装依赖"
