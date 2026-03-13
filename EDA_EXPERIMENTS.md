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

## Experiment 7: Near-Duplicate Detection Analysis

**Research Question**: Does the training set contain near-duplicate 
images, and do they leak across the train/val split, potentially 
inflating validation mAP estimates?

**Intervention**: Computed pairwise cosine similarity between all 
1,895 MegaDescriptor embeddings. Identified near-duplicate pairs 
at four similarity thresholds (0.95, 0.97, 0.98, 0.99). Analyzed 
per-identity duplicate rates and cross-split contamination.

**What is controlled**: Same train/val split as all experiments 
(seed=42, 80/20 stratified). Raw MegaDescriptor embeddings used 
(no fine-tuning) to detect visual similarity at the feature level.

**Results at threshold 0.99:**
| Metric | Value |
|---|---|
| Total near-duplicate pairs | 317 |
| Same identity pairs | 317 (100%) |
| Cross identity pairs | 0 (0%) |
| Cross-split pairs (train/val leak) | 86 (27%) |

**Threshold sensitivity:**
| Threshold | Total Pairs | Same Identity | Cross Identity | Cross Split |
|---|---|---|---|---|
| 0.99 | 317 | 317 | 0 | 86 |
| 0.98 | 755 | 754 | 1 | 223 |
| 0.97 | 1,225 | 1,188 | 37 | 397 |
| 0.95 | 1,993 | 1,901 | 92 | 661 |

**Most affected identities (threshold=0.99):**
| Identity | Total Images | Duplicate Pairs | Cross-Split Pairs | Duplicate Rate |
|---|---|---|---|---|
| Tomas | 63 | 163 | 32 | 2.59 |
| Kamaikua | 105 | 35 | 12 | 0.33 |
| Benita | 86 | 26 | 11 | 0.30 |
| Medrosa | 170 | 19 | 5 | 0.11 |
| Lua | 120 | 14 | 5 | 0.12 |

**Analysis**: The dataset contains 317 near-duplicate pairs at the 
0.99 similarity threshold. All pairs belong to the same identity — 
there are no cross-identity near-duplicates, confirming the labels 
are clean. The duplicates are primarily consecutive video frames 
from camera trap sequences, as evidenced by Tomas having 163 
duplicate pairs from 63 images (2.59x duplication rate) and the 
visual inspection showing nearly identical frames.

The critical finding is that 86 pairs (27% of all duplicate pairs) 
cross the train/val split boundary. This means some validation 
images are near-identical to training images, which likely inflates 
validation mAP estimates. Models that memorize specific frames 
will appear to generalize better than they actually do. This 
partially explains the consistent gap between validation mAP 
and public leaderboard scores observed across all experiments.

At the looser 0.95 threshold, 92 cross-identity pairs appear, 
suggesting some jaguars are genuinely visually similar at a 
coarse level — further motivating the use of fine-grained 
identity-specific features.

**Implication for future work**: Deduplication before splitting 
(grouping near-duplicate frames into clusters and splitting 
at cluster level) would produce a cleaner train/val split and 
more reliable validation mAP estimates.

**W&B Run**: (https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/2zy5g79m)
**Notebook**: (https://www.kaggle.com/code/mishankjain/jaguar-reid-exp07-near-duplicate-analysis)
