# LoGo: Local-Global Protonet for Abdominal Organ Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official (refactored) implementation of the **LoGo (Local-Global) Encoder** for Few-Shot Medical Image Segmentation. It leverages a student-teacher distillation framework (DINOv2 + DeepLabv3) to learn robust feature representations without extensive annotations.

## 🚀 Key Features

*   **Dual-Pathway Architectures**: Combines global context (DINOv2-inspired) and local details (DeepLabv3-inspired).
*   **Knowledge Distillation**: Trains a lightweight student encoder using state-of-the-art foundation models.
*   **ProtoSegNet**: A prototypical network wrapper for few-shot segmentation tasks.
*   **Unified Dataset**: Supports CHAOS (MRI) and SABS (CT) datasets with on-the-fly normalization and pairing.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/LoGo.git
    cd LoGo
    ```

2.  **Install dependencies:**
    It is recommended to use a Conda environment (Python 3.8+).
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Data Setup

This project uses the **CHAOS** (MRI) and **SABS** (CT) datasets. We provide a helper script to download the preprocessed versions from Kaggle.

1.  **Configure Kaggle API:**
    Ensure you have your `kaggle.json` key in `~/.kaggle/` or `%USERPROFILE%\.kaggle\`.

2.  **Download Data:**
    ```bash
    python setup_data.py
    ```
    This command downloads and extracts:
    - `chaos_MR_T2_normalized/`
    - `sabs_CT_normalized/`

## 🛠️ Usage

### 1. Distillation (Pre-training)

Train the LoGo Encoder using knowledge distillation from DINOv2 and DeepLabv3 teachers. This step learns the feature representation.

```bash
python distill.py \
  --epochs 60 \
  --batch_size 1 \
  --device cuda \
  --output_dir checkpoints_distill
```

*   **Outputs:** `checkpoints_distill/distilled_logoencoder.pt` and loss curves.

### 2. Training (Few-Shot Segmentation)

Train the full `ProtoSegNet` capability using the pre-trained encoder.

```bash
python train.py \
  --distilled_weights checkpoints_distill/distilled_logoencoder.pt \
  --epochs 100 \
  --lr 1e-4 \
  --output_dir checkpoints_train
```

*   **Outputs:** `checkpoints_train/protosegnet_best.pt` (Best Validation Dice) and training plots.

### 3. Inference / Demo

Run inference on random validation samples to visualize performance.

```bash
python inference.py \
  --checkpoint checkpoints_train/protosegnet_best.pt \
  --n_samples 5 \
  --output_dir inference_results
```

*   **Outputs:** Visualization images in `inference_results/`.

## 📂 Project Structure

```
LoGo/
├── src/
│   ├── dataset.py       # UnifiedAbdominalDataset & Loader Factory
│   ├── model.py         # LoGoEncoder, ProtoSegNet, PrototypeRefiner
│   ├── loss.py          # SOTA losses (Dice, Focal, Boundary, Triplet)
│   ├── train_utils.py   # Training & Evaluation loops
│   └── utils.py         # Visualization & Stats
├── distill.py           # Stage 1: Encoder Distillation
├── train.py             # Stage 2: ProtoSegNet Training
├── inference.py         # Inference Demo
├── setup_data.py        # Data downloader
└── requirements.txt     # Dependencies
```

## 📝 Configuration

All scripts support arguments for paths and hyperparameters. Run with `--help` to see options:

```bash
python train.py --help
```
