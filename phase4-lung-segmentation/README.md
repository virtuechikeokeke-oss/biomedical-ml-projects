# Phase 4 — Chest X-Ray Lung Segmentation

**Architecture:** U-Net (built from scratch, PyTorch)  
**Dataset:** Montgomery County Chest X-Ray (138 images)  
**Result:** 0.93 val IoU  
**Live Demo:** [huggingface.co/spaces/vchike-okeke/lung-segmentation](https://huggingface.co/spaces/vchike-okeke/lung-segmentation)

---

## Overview

Built a U-Net segmentation model from scratch to identify and mask lung regions in chest X-rays. Unlike Phases 2 and 3 which used pretrained backbones (ResNet-18, DenseNet-121), this model was trained entirely from random weight initialization — no transfer learning. Implemented as modular Python scripts rather than a notebook to reflect a production-style codebase.

---

## Architecture

- **Encoder:** 4 downsampling blocks (DoubleConv → MaxPool), channels: 64 → 128 → 256 → 512
- **Bottleneck:** DoubleConv at 1024 channels
- **Decoder:** 4 upsampling blocks (ConvTranspose2d + skip connection concatenation + DoubleConv)
- **Output:** 1×1 conv → sigmoid → binary lung mask
- **Skip connections:** concatenate encoder feature maps to decoder at matching resolution — preserves spatial detail lost during downsampling

---

## Key Design Decisions

- **Grayscale input (1 channel):** Medical X-rays carry no color information — RGB would add noise, not signal
- **Loss function:** BCE + Dice combined (0.5/0.5) — BCE provides pixel-level gradient signal; Dice handles class imbalance and global shape accuracy
- **Normalization:** mean=0.5, std=0.5 (not ImageNet stats — no pretrained weights to match)
- **Mask combination:** Left and right lung masks merged via `np.maximum` before training

---

## Files

| File | Purpose |
|---|---|
| `dataset.py` | MontgomeryDataset — loads X-rays and combined lung masks, applies augmentation |
| `unet.py` | Full U-Net architecture from scratch (DoubleConv, EncoderBlock, DecoderBlock) |
| `utils.py` | DiceLoss, BCEDiceLoss, iou_score |
| `train.py` | Training loop, validation, best-model checkpointing, curve visualization |

---

## Results

| Metric | Value |
|---|---|
| Val IoU | 0.93 |
| Val Loss | 0.12 |
| Loss function | BCE + Dice |
| Epochs | 30 |
| Optimizer | Adam + ReduceLROnPlateau |

Training curves and sample predictions included in `training_curves.png` and `predictions.png`.

---

## Part of the Biomedical ML Series

| Phase | Task | Model | Key Result |
|---|---|---|---|
| 1 | ECG AFib Classification | Random Forest | 85% acc, 0.90 AUC |
| 2 | Brain Tumor MRI Classification | ResNet-18 | 95% acc, 0.9897 AUC |
| 3 | Chest X-Ray Classification | DenseNet-121 | 0.842 val AUC |
| **4** | **Lung Segmentation** | **U-Net (scratch)** | **0.93 val IoU** |
