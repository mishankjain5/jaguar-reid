# 🐆 Jaguar Re-Identification Challenge

**Course:** Applied Hands-On Computer Vision    
**Author:** Mishank Jain | Universität Potsdam  

[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)](https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/mishankjain)

---

## Overview

This repository contains all experiments, notebooks, and documentation for the Jaguar Re-Identification Kaggle competition. The goal is to identify individual jaguars from camera trap images collected in Brazil's Pantanal Park using deep metric learning.

**Best Result:** 0.783 mAP on public leaderboard (+5.7% over baseline)  
**Baseline:** MegaDescriptor + ArcFace = 0.741 mAP  
**Total Experiment:** 8 distinct experiments (including forked baseline)

---

## Competition Links

| Competition | URL | Best Score |
|---|---|---|
| Round 1 (with background) | [Jaguar Re-Identification Challenge](https://www.kaggle.com/competitions/jaguar-re-id/) | 0.783 mAP |
| Round 2 (background removed) | [Round-2 Jaguar Re-identification Challenge](https://www.kaggle.com/competitions/round-2-jaguar-reidentification-challenge/) | 0.041 mAP |

---

## Repository Structure

```
jaguar-reid/
│
├── README.md                          
├── EDA_EXPERIMENTS.md                 ← Exploratory data analysis experiments
├── LEADERBOARD_EXPERIMENTS.md         ← Leaderboard optimization experiments
├── report/
│   └── jaguar_reid_report.pdf         
│
├── notebooks/
├── jaguar-reid-exp01-baseline-diagnostic.ipynb
├── jaguar-reid-exp02-loss-comparison.ipynb
├── jaguar-reid-exp03-background-intervention.ipynb
├── jaguar-reid-exp04-reranking.ipynb
├── jaguar-reid-exp05-hyperparam-sweep.ipynb
├── jaguar-reid-exp06-embedding-analysis.ipynb
├── jaguar-reid-exp07-near-duplicate-analysis.ipynb
└── jaguar-reid-exp08-data-augmentation.ipynb
```

---

## Experiment Summary

### EDA Experiments → [EDA_EXPERIMENTS.md](./EDA_EXPERIMENTS.md)

| Exp | Description | Key Finding |
|---|---|---|
| 1 | Baseline diagnostic (R1 vs R2) | 17x performance drop reveals background reliance |
| 3 | Background intervention analysis | Removing backgrounds hurts (-0.048 val mAP) |
| 6 | Embedding space analysis | Fine-tuning improves separation from 0.157 → 0.930 |
| 7 | Near-duplicate detection | 317 near-duplicate pairs, 86 cross train/val split |

### Leaderboard Experiments → [LEADERBOARD_EXPERIMENTS.md](./LEADERBOARD_EXPERIMENTS.md)

| Exp | Description | Val mAP | R1 mAP | R2 mAP |
|---|---|---|---|---|
| 2 | Loss function comparison (6 losses) | 0.790 | 0.742 | 0.041 |
| 4 | K-reciprocal re-ranking | 0.689 | 0.778 | 0.041 |
| 5 | Bayesian hyperparameter sweep | 0.804 | 0.783 | 0.040 |
| 8 | Data augmentation comparison | 0.819 | N/A | N/A |

---

## Architecture

All experiments use a frozen MegaDescriptor-L-384 backbone with a trainable projection network:

```
MegaDescriptor-L-384 (frozen, ~307M params)
        ↓ 1536-dim embeddings (cached)
Linear(1536 → 512) + BatchNorm + ReLU + Dropout(0.5)
        ↓
Linear(512 → 256) + BatchNorm
        ↓
L2 Normalize
        ↓ 256-dim identity embeddings
```

**Best Loss:** Combined ArcFace (margin=0.6, scale=48) + Triplet Loss (margin=0.4), alpha=0.3  
**Best Optimizer:** AdamW, lr=7.67e-4, weight_decay=1e-4  
**Post-processing:** K-reciprocal re-ranking (k1=15, k2=6, lambda=0.5)

---

## Leaderboard Progression

| Stage | Round 1 mAP | Change |
|---|---|---|
| Baseline (MegaDescriptor + ArcFace) | 0.741 | — |
| Exp 2: Combined loss | 0.742 | +0.001 |
| Exp 4: + Re-ranking | 0.778 | +0.036 |
| Exp 5: + Sweep + Re-ranking | **0.783** | +0.005 |

---

## Setup and Reproduction

All experiments run on **Kaggle** using free GPU compute. To reproduce:

1. Fork the baseline notebook on Kaggle:  
   [Jaguar Re-Identification: MegaDescriptor + ArcFace Loss](https://www.kaggle.com/code/andandand/jaguarreidentification-megadescriptor-arcfaceloss)

2. Add the competition dataset:
   - [Jaguar Re-Identification Challenge](https://www.kaggle.com/competitions/jaguar-re-id/)
   - [Round-2 Jaguar Re-identification Challenge](https://www.kaggle.com/competitions/round-2-jaguar-reidentification-challenge/)

3. Add secrets in Kaggle notebook settings:
   - `wandb_api` — your W&B API key
   - `hf_api` — your HuggingFace API key (for MegaDescriptor)

4. Run notebooks in order (Exp01 → Exp07)

**Note:** MegaDescriptor embeddings are cached after the first run in each notebook. Subsequent runs load from cache, reducing training time to under 5 minutes per experiment.

---

## Key Results

### Loss Function Comparison (Exp 2)

| Loss | Val mAP |
|---|---|
| Combined ArcFace + Triplet | **0.790** |
| ArcFace | 0.785 |
| CosFace | 0.782 |
| Triplet (hard mining) | 0.684 |
| SubCenter ArcFace | 0.610 |
| Focal Loss | 0.418 |

### Hyperparameter Sweep Best Config (Exp 5)

| Parameter | Default | Best |
|---|---|---|
| Learning rate | 1e-4 | 7.67e-4 |
| ArcFace margin | 0.5 | 0.6 |
| Alpha (ArcFace weight) | 0.5 | 0.3 |
| Dropout | 0.3 | 0.5 |

### Near-Duplicate Analysis (Exp 7)

| Threshold | Pairs | Same Identity | Cross Split |
|---|---|---|---|
| 0.99 | 317 | 317 (100%) | 86 (27%) |
| 0.95 | 1,993 | 1,901 (95%) | 661 (33%) |

### Data Augmentation Comparison (Exp 8)
 
| Augmentation | Val mAP | vs Control |
|---|---|---|
| None (control) | 0.806 | baseline |
| Light (flip + mild jitter) | **0.819** | +0.013 |
| Heavy (rotation + strong jitter + erasing) | 0.766 | −0.040 |

---

## W&B Tracking

All experiments tracked at:  
**https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank**

| Run Name | Experiment |
|---|---|
| loss-arcface, loss-cosface, loss-triplet, loss-combined, loss-focal, loss-subcenter | Exp 2: Loss comparison |
| bg-with-background-r1, bg-removed-background-r2 | Exp 3: Background intervention |
| reranking-k1-15-lambda-0.5 | Exp 4: Re-ranking |
| sweep (15 trials) + sweep-best-final-training | Exp 5: Hyperparameter sweep |
| exp06-embedding-analysis | Exp 6: Embedding analysis |
| exp07-near-duplicate-analysis | Exp 7: Near-duplicate detection |
| aug-none-control, aug-light, aug-heavy | Exp 8: Data augmentation |

---

## References

1. Deng, J., et al. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019. https://arxiv.org/abs/1801.07698
2. Vidal, S., et al. (2024). MegaDescriptor: Universal Animal Re-Identification via Large-Scale Pretraining. arXiv. https://arxiv.org/abs/2311.00169
3. Rueda-Toicen, A. (2026). Jaguar Re-Identification: MegaDescriptor + ArcFace Loss. Baseline Notebook, Kaggle. https://www.kaggle.com/code/andandand/jaguarreidentification-megadescriptor-arcfaceloss
