#!/bin/bash
# Setup script for emotion recognition experiments with Chinese mirror support

echo "Setting up Hugging Face Chinese mirrors..."

# 配置 Hugging Face 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 配置 pip 国内镜像
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "PIP_INDEX_URL=$PIP_INDEX_URL"

# 创建 conda 环境（如果不存在）
if ! conda info --envs | grep -q "erc"; then
    echo "Creating conda environment 'erc'..."
    conda create -n erc python=3.10 -y
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate erc

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
