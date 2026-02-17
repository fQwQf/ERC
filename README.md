# Sisyphus ERC: SOTA Fine-Grained Emotion Recognition

This repository contains the code and methodology for achieving State-of-the-Art (SOTA) results in Emotion Recognition in Conversation (ERC) using Large Language Models.

## Quick Results
- **EmpatheticDialogues (28 classes)**: **53.55% Weighted F1** (Full Test Set SOTA)
- **Coarse-Grained Upper Bound (8 classes)**: **73.38% Weighted F1**

## Project Structure
- `src/`: Core architecture (Data, Models, Retrieval).
- `scripts/`: Evaluation, Training, and TUI Demo scripts.
- `paper/`: LaTeX source for the research paper.

## Installation
```bash
conda create -n erc python=3.10 -y
conda activate erc
pip install -r requirements.txt
```

## 🧠 Model Zoo
Our State-of-the-Art model weights (LoRA adapters) are hosted on Hugging Face:
- **SOTA Model**: [fQwQf/erc-qwen2.5-7b-sota](https://huggingface.co/fQwQf/erc-qwen2.5-7b-sota)

### Run Inference using Remote Weights
Our scripts support loading LoRA weights directly from Hugging Face repo IDs:

```bash
# Start the Interactive TUI Demo with the SOTA model
python scripts/demo_tui.py --model_path fQwQf/erc-qwen2.5-7b-sota
```

## Research Paper
The complete research documentation is available in `paper/main.tex`. It details:
- **Retrieval-Augmented Input (RAI)**: Dynamically injecting semantic anchors using Sentence-BERT.
- **Multi-task Alignment**: Jointly learning Emotion, Speaker, and Context Impact.
- **Experimental Findings**: Analysis of sequence truncation and prompt sensitivity.

## Key Innovations
1. **Dynamic Retrieval Module**: Implemented using FAISS for real-time semantic demonstration retrieval.
2. **Role Alignment Fix**: Critical rectification of narrator/listener label flipping in the EmpatheticDialogues dataset.
3. **High-Granularity Calibration**: Optimized 768-token context window to prevent label truncation.

## 📊 Evaluation
To replicate the SOTA results on the full test set:
```bash
python scripts/eval_sota.py \
    --model_path fQwQf/erc-qwen2.5-7b-sota \
    --dataset empathetic \
    --num_samples 0
```

