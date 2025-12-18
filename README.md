# PlantCLEF - Plant Species Classification and Retrieval

This project implements deep learning models for plant species classification and image retrieval on the PlantCLEF dataset. It evaluates multiple state-of-the-art vision models including ConvNeXt, DINOv2, and BioCLIP.

## Overview

The project supports two main tasks:
- **Classification**: Multi-class plant species classification (7,806 species)
- **Retrieval**: Cross-species image retrieval where query and gallery contain species not seen during training

## Project Structure

```
plantclef/
├── dataset.py                          # Classification dataset class
├── dataset_retrieval.py                # Retrieval dataset class
├── dataloader.py                       # Classification data loading utilities
├── dataloader_retrieval.py             # Retrieval data loading utilities
├── retrieval_metrics.py                # Retrieval evaluation metrics (mAP, Recall@K)
├── split_for_class.py                  # Generate train/val/test splits for classification
├── split_for_retrieval.py              # Generate splits for retrieval task
├── train_convnext_cls.py               # Train ConvNeXt for classification
├── train_convnext_retri.py             # Train ConvNeXt for retrieval
├── train_dinov2_cls.py                 # Train DINOv2 for classification
├── train_dinov2_retri.py               # Train DINOv2 for retrieval
├── test_convnext_cls.py                # Test ConvNeXt classification
├── test_convnext_retri.py              # Test ConvNeXt retrieval
├── test_dinov2_cls.py                  # Test DINOv2 classification
├── test_dinov2_retri.py                # Test DINOv2 retrieval
├── test_bioclip_zeroshot_cls.py        # BioCLIP zero-shot classification
├── test_bioclip_zeroshot_retri.py      # BioCLIP zero-shot retrieval
├── test_dataloader.py                  # Dataloader testing utilities
├── demo.py                             # Demo script for inference on single images
├── checkpoints/                        # Pre-trained model checkpoints
│   └── README.md                       # Model download instructions
├── test_examples/                      # Sample test images for demo
│   └── README.md                       # Usage instructions
└── results/                            # Test results directory
```

## Models

### 1. ConvNeXt Tiny
- Modern ConvNet architecture with competitive performance
- Pre-trained on ImageNet-1K
- Fine-tuned with dual learning rates for backbone and head

### 2. DINOv2 ViT-B/14
- Vision Transformer trained with self-supervised learning
- Excellent transfer learning capabilities
- Base model with 14x14 patch size

### 3. BioCLIP
- Vision-language model specialized for biological images
- Zero-shot evaluation without fine-tuning
- Uses text embeddings of species names for classification

## Dataset

The PlantCLEF dataset contains plant images across 7,806 species.

**Expected data structure:**
```
/fs/scratch/PAS2099/plantclef/
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── [image files]
```

**CSV format:**
```
image_path,species_id
path/to/image1.jpg,species_001
path/to/image2.jpg,species_002
```

## Setup

### Requirements
- Python 3.8+
- PyTorch with CUDA support
- torchvision
- pandas
- PIL (Pillow)
- wandb (for experiment tracking)

### Installation

The installation depends on which models you want to use:

#### For DINOv2 and ConvNeXt

DINOv2 and ConvNeXt share compatible environments. Follow the official DINOv2 installation guide:

```bash
# Clone DINOv2 repository
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2

# Install dependencies according to the official guide
# See: https://github.com/facebookresearch/dinov2
```

The ConvNeXt models are compatible with the DINOv2 environment and use standard torchvision implementations.

#### For BioCLIP

BioCLIP requires only the OpenCLIP library:

```bash
pip install open_clip_torch
```

#### Additional Dependencies

Install common dependencies for data processing and experiment tracking:

```bash
pip install pandas pillow wandb
```

### Download Pre-trained Models

Pre-trained model checkpoints are available on Google Drive:

**Download Link**: https://drive.google.com/drive/folders/14c_88401aQ4go1w_g9b9W1LIK-qRnaGY?usp=drive_link

Download the checkpoint files and place them in the `checkpoints/` directory. See [checkpoints/README.md](checkpoints/README.md) for details.

## Usage

### Quick Demo

The `demo.py` script provides a simple command-line interface for testing the models on individual images.

**Features:**
- Support for both classification and retrieval tasks
- Works with DINOv2 and ConvNeXt models
- Automatic checkpoint loading and model initialization
- Outputs top-5 predictions for classification
- Outputs 512-dim normalized feature vectors for retrieval

**Usage:**

```bash
# Classification demo - Get top-5 species predictions
python demo.py --task classification --model dinov2 --image test_examples/sample1.jpg

# Example output:
# Rank   Species ID      Confidence
# 1      species_001     85.23%
# 2      species_042      7.89%
# ...

# Retrieval demo - Extract feature embeddings
python demo.py --task retrieval --model convnext --image test_examples/sample1.jpg

# Example output:
# Feature dimension: 512
# Feature norm: 1.0000 (should be ~1.0)
# First 10 feature values: [0.123, -0.456, ...]
```

**Arguments:**
- `--task`: Choose `classification` or `retrieval`
- `--model`: Choose `dinov2` or `convnext`
- `--image`: Path to input image (required)
- `--checkpoint`: Custom checkpoint path (optional)

**Note**: Make sure to download the model checkpoints first (see above).

### Data Preparation

Generate splits for classification:
```bash
python split_for_class.py
```

Generate splits for retrieval:
```bash
python split_for_retrieval.py
```

### Training

Train ConvNeXt for classification:
```bash
python train_convnext_cls.py
```

Train DINOv2 for classification:
```bash
python train_dinov2_cls.py
```

Train for retrieval task:
```bash
python train_convnext_retri.py
python train_dinov2_retri.py
```

### Evaluation

Test classification models:
```bash
python test_convnext_cls.py
python test_dinov2_cls.py
```

Test retrieval models:
```bash
python test_convnext_retri.py
python test_dinov2_retri.py
```

Zero-shot BioCLIP evaluation:
```bash
python test_bioclip_zeroshot_cls.py
python test_bioclip_zeroshot_retri.py
```

## Results

### Classification Task (7,806 species)

| Model | Test Top-1 Accuracy | Test Top-5 Accuracy |
|-------|---------------------|---------------------|
| DINOv2 ViT-B/14 | **69.99%** | **89.41%** |
| ConvNeXt Tiny | 64.75% | 86.21% |
| BioCLIP (zero-shot) | 26.50% | 45.71% |

### Retrieval Task (Cross-species, 3,903 test species)

| Model | mAP | R@1 | R@5 |
|-------|-----|-----|-----|
| DINOv2 ViT-B/14 | **37.90%** | **85.41%** | **94.63%** |
| BioCLIP (zero-shot) | 18.14% | 63.59% | 82.21% |
| ConvNeXt Tiny | 7.75% | 49.91% | 72.52% |

**Test Set Size:**
- DINOv2: 10K queries, 50K gallery samples (limited due to computational constraints)
- BioCLIP: 138K queries, 559K gallery samples (full test set)
- ConvNeXt: 10K queries, 50K gallery samples (limited)

**Metrics Explanation:**
- **mAP**: Mean Average Precision across all queries
- **R@K**: Recall@K - percentage of queries where correct species appears in top-K results

## Training Configuration

### Classification
- Batch size: 512
- Epochs: 10
- Optimizer: AdamW with weight decay
- Learning rate: Dual rates for backbone (5e-4) and head (5e-3)
- Warmup: 1 epoch with cosine annealing
- Freeze epochs: 1 epoch (linear probe before fine-tuning)
- Mixed precision: BF16
- Data augmentation: Random crop, flip, color jitter
- Label smoothing: 0.1

### Retrieval
- Loss: Triplet loss with hard negative mining
- Margin: 0.3
- Embedding dimension: 512
- Cross-species evaluation setup

## Features

- Mixed precision training (BF16) for faster training
- Gradient clipping for stable training
- Wandb integration for experiment tracking
- Checkpoint saving with best model selection
- Comprehensive evaluation metrics
- Hard negative mining for triplet loss
- Cross-species retrieval evaluation

## Notes

- The retrieval task is particularly challenging as it evaluates on species NOT seen during training
- BioCLIP shows strong zero-shot performance despite no fine-tuning
- DINOv2 consistently outperforms ConvNeXt across both tasks
- All models use ImageNet or self-supervised pre-training as initialization

## Citation

If you use this code, please cite the PlantCLEF challenge and respective model papers.
