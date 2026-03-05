## Experiment 3: Background Intervention Analysis

**Research Question**: Does removing background during training 
force the model to learn jaguar-specific identity features, 
improving re-identification performance?

**Intervention**: Trained two identical models using Combined 
ArcFace+Triplet loss — one on original images with backgrounds 
(Round 1 training data), one on SAM3 background-removed images 
(Round 2 training data). Everything else fixed: same backbone, 
same embedding dim, same optimizer, same seed.

**What is controlled**: MegaDescriptor backbone, embedding dim (256),
learning rate (1e-4), batch size (32), 25 epochs, seed=42, 
Combined ArcFace+Triplet loss.

**Results**:
| Training Data | Val mAP | Val Loss |
|---|---|---|
| With background (Round 1) | 0.7900 | 2.2577 |
| Background removed — SAM3 (Round 2) | 0.7421 | 2.7970 |

**mAP Delta**: -0.048 when background is removed from training

**Kaggle Submissions (Round 1 model)**:
| Competition | Public mAP | Date |
|---|---|---|
| Jaguar Re-Identification Challenge | 0.742 | March 3, 2026 |
| [Round-2] Jaguar Re-Identification Challenge | 0.041 | March 3, 2026 |

**Analysis**: Contrary to expectations, removing backgrounds during 
training hurt performance by 0.048 mAP. Two explanations are likely. 
First, the SAM3 segmentation used in Round 2 introduces artifacts 
around jaguar edges, adding noise that makes it harder for 
MegaDescriptor to extract clean features. Second, background context 
may actually carry useful identity information in this dataset — 
individual jaguars may consistently appear in the same habitat zones, 
making location a weak but real signal. The persistent low Round 2 
score (0.041) confirms that the model trained on background-present 
images cannot generalise to background-removed test images, regardless 
of which training set is used. This suggests the background reliance 
is deeply embedded in MegaDescriptor's pretrained features rather than 
something easily corrected by training data choice alone.

**W&B Runs**: (https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/o1nfegmu)

(https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/rpqh58m1)
              
**Notebook**: (https://www.kaggle.com/code/mishankjain/jaguar-reid-exp03-background-intervention)

## Experiment 6: Embedding Space Analysis & Nearest Neighbor Visualization

**Research Question**: How does fine-tuning transform the embedding 
space? Which identities are hardest to distinguish?

**Intervention**: Computed embedding statistics and nearest neighbor 
visualizations before and after fine-tuning using the best model 
(Combined ArcFace+Triplet, sweep best config). Analysed intra-class 
cohesion and inter-class separation across all 31 identities.

**Dataset Statistics**:
- Total images: 1895
- Unique identities: 31
- Most common: Marcela (183 images)
- Rarest: Ipepo, Bernard (13 images each)
- Mean images per identity: 61.1
- Significant class imbalance: 14x difference between most and least common

**Embedding Statistics**:
| Metric | Before Fine-Tuning | After Fine-Tuning | Improvement |
|---|---|---|---|
| Intra-class similarity | 0.310 | 0.883 | +0.573 |
| Inter-class similarity | 0.153 | -0.047 | -0.200 |
| Separation | 0.157 | 0.930 | **+0.774** |

**Nearest Neighbor Results (Top-5)**:
| Identity | Before (correct/5) | After (correct/5) |
|---|---|---|
| Marcela (common, 183 imgs) | 0/5 | 5/5 |
| Medrosa (common, 170 imgs) | 0/5 | 5/5 |
| Bernard (rare, 13 imgs) | 0/5 | 5/5 |
| Ipepo (rare, 13 imgs) | 0/5 | 5/5 |

**Hardest Identities (lowest intra-class similarity after fine-tuning)**:
| Identity | Images | Mean Intra-Sim |
|---|---|---|
| Ipepo | 13 | 0.659 |
| Patricia | 19 | 0.728 |
| Bernard | 13 | 0.731 |
| Bororo | 22 | 0.787 |
| Apeiara | 20 | 0.788 |

**Analysis**: Fine-tuning dramatically improves embedding quality. 
Before fine-tuning, raw MegaDescriptor embeddings fail completely — 
0/5 correct neighbors for all query jaguars, with inter-class 
similarity (0.153) dangerously close to intra-class similarity (0.310). 
After fine-tuning, the separation jumps from 0.157 to 0.930 (+0.774), 
meaning same-jaguar images are now very tightly clustered while 
different-jaguar images are pushed to negative similarity scores. 

The hardest identities (Ipepo, Patricia, Bernard) all have fewer than 
20 images, confirming that data scarcity is the primary challenge. 
The identity-balanced mAP metric is therefore well-chosen for this 
dataset — without it, rare jaguars would be ignored during optimization. 
The class imbalance (14x between most and least common) motivates 
future work on data augmentation for rare identities.

**W&B Run**: (https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/gi4zwzzk)
**Notebook**: (https://www.kaggle.com/code/mishankjain/jaguar-reid-exp06-deduplication)
