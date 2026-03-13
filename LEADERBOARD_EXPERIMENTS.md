# Leaderboard Experiments

## Experiment 1: Forked Baseline Diagnostic

**Research Question**: How much does background context contribute
to the baseline model's re-identification performance at inference time?

**Intervention**: Submitted identical baseline model (MegaDescriptor +
ArcFace, no modifications) to both Round 1 (background intact) and
Round 2 (background removed from test images). Zero change to model
weights or training — only the test set differs between rounds.

**What is controlled**: Model weights, training data, hyperparameters,
submission file — all identical. Only the test images differ.

**Results**:
| Competition | Public mAP | 
|---|---|
| Jaguar Re-Identification Challenge | 0.735 |
| [Round-2] Jaguar Re-Identification Challenge | 0.043 |

**mAP Delta**: 0.692 drop when background is removed

**Analysis**: Jaguar Re-Identification challenge and Round 2 differ in both training and test 
images — Round 2 has backgrounds removed from everything using SAM3 
segmentation. Submitting the Round 1 baseline model to Round 2 
(0.043) therefore measures how well features learned from 
background-heavy images transfer to segmented jaguar-only images. 
The severe drop suggests the baseline model learned background 
shortcuts rather than jaguar coat patterns, and that training on 
segmented images (Round 2) requires a dedicated model.

**W&B Run**: (https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/q23er6i8)

## Experiment 2: Loss Function Comparison (6 Losses)

**Research Question**: Which loss function produces the best 
identity-balanced mAP for jaguar re-identification under identical 
training conditions?

**Intervention**: Trained 6 different loss functions keeping everything 
else fixed — same MegaDescriptor backbone, same embedding dim (256), 
same learning rate (1e-4), same batch size (32), same 25 epochs, 
same val split (seed=42).

**What is controlled**: Backbone, embedding dimension, optimizer (AdamW), 
scheduler (ReduceLROnPlateau), batch size, val split, random seed.

**Results**:
| Loss Function | Val mAP | Val Loss |
|---|---|---|
| Combined ArcFace + Triplet | 0.7900 | 2.2577 |
| ArcFace | 0.7850 | 4.4457 |
| CosFace | 0.7823 | 3.2073 |
| Triplet (hard mining) | 0.6837 | 0.0756 |
| SubCenter ArcFace | 0.6101 | 4.4128 |
| Focal Loss | 0.4177 | 0.5060 |

**Kaggle Submissions (best model — Combined ArcFace + Triplet)**:
| Competition | Public mAP | Date |
|---|---|---|
| Jaguar Re-Identification Challenge | 0.742 | March 2, 2026 |
| [Round-2] Jaguar Re-Identification Challenge | 0.041 | March 2, 2026 |

**Analysis**: The Combined ArcFace + Triplet loss achieved the best 
validation mAP (0.790), outperforming plain ArcFace (0.785). This 
suggests that combining a classification-based margin loss with a 
distance-based triplet loss is beneficial — ArcFace pushes embeddings 
toward class centers while Triplet Loss directly optimises pairwise 
distances, and together they provide complementary learning signals. 
CosFace performed comparably to ArcFace, confirming that margin-based 
losses are well suited to this dataset. Triplet Loss alone was 
significantly weaker (0.684), likely because it receives less 
structured gradient signal without the classification head. Focal Loss 
performed worst (0.418) — it is designed for classification tasks and 
does not directly optimise the embedding space geometry needed for 
re-identification. SubCenter ArcFace underperformed unexpectedly (0.610), 
possibly because 25 epochs was insufficient for its K=3 sub-centers to 
converge. The Combined loss will be used as the foundation for all 
remaining experiments. The Round 2 score (0.041) remains low for both 
models, confirming that background reliance is a training-time problem 
that requires dedicated intervention.

**W&B Run**: (https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/g6xel2lf)

## Experiment 4: K-Reciprocal Re-ranking

**Research Question**: Can post-processing similarity scores using 
k-reciprocal re-ranking improve re-identification performance without 
any retraining?

**Intervention**: Applied k-reciprocal re-ranking to the output 
embeddings of the best model (Combined ArcFace+Triplet). No model 
weights changed. Searched over k1 ∈ {10, 15, 20, 25} and 
lambda ∈ {0.2, 0.3, 0.4, 0.5} on the validation set.

**What is controlled**: Model weights, training data, embeddings — 
all identical to Experiment 2. Only the similarity computation 
at inference time changes.

**Best Parameters**: k1=15, k2=6, lambda=0.5

**Validation Results**:
| Method | Val mAP |
|---|---|
| Baseline (cosine similarity) | 0.6828 |
| K-reciprocal re-ranking | 0.6892 |
| Improvement | +0.0064 |

**Kaggle Results**:
| Competition | Without Re-ranking | With Re-ranking | Delta |
|---|---|---|---|
| Jaguar Re-Identification Challenge | 0.742 | 0.778 | +0.036 |
| [Round-2] Jaguar Re-Identification Challenge | 0.041 | 0.041 | 0.000 |

**Analysis**: K-reciprocal re-ranking improved the Round 1 leaderboard 
score by 0.036, which is a substantial gain for a zero-cost 
post-processing step. The method works by replacing raw cosine 
similarity with a Jaccard distance based on shared k-reciprocal 
neighbors — two images are considered more similar if they mutually 
appear in each other's nearest neighbor lists. This is more robust 
than cosine similarity alone because it considers the local 
neighborhood structure of the embedding space. The Round 2 score 
was unaffected, confirming that the background reliance issue exists 
at the feature level and cannot be corrected by re-ranking alone. 
The validation improvement (+0.006) was smaller than the leaderboard 
improvement (+0.036), suggesting the test set benefits more from 
re-ranking than the validation set, possibly because the test set 
has more diverse query-gallery pairs.

**W&B Run**: (https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/nsxd6ham)
**Notebook**: (https://www.kaggle.com/code/mishankjain/jaguar-reid-exp04-reranking)

## Experiment 5: Hyperparameter Sweep (Bayesian Optimisation)

**Research Question**: Can Bayesian hyperparameter optimisation find 
a better configuration than the default settings used in Experiment 2?

**Intervention**: Used W&B Bayesian sweep to search over 7 
hyperparameters across 15 trials. Each trial trained for 20 epochs 
on the same cached embeddings with the Combined ArcFace+Triplet loss. 
Best config then retrained for 25 epochs.

**Search Space**:
| Parameter | Range |
|---|---|
| learning_rate | 1e-5 to 1e-3 (log uniform) |
| arcface_margin | 0.3, 0.4, 0.5, 0.6 |
| arcface_scale | 32, 48, 64 |
| triplet_margin | 0.2, 0.3, 0.4 |
| alpha | 0.3, 0.5, 0.7 |
| embedding_dim | 128, 256, 512 |
| dropout | 0.1, 0.3, 0.5 |

**Best Configuration Found**:
| Parameter | Default (Exp 2) | Sweep Best |
|---|---|---|
| learning_rate | 1e-4 | 7.67e-4 |
| arcface_margin | 0.5 | 0.6 |
| arcface_scale | 64 | 48 |
| triplet_margin | 0.3 | 0.4 |
| alpha | 0.5 | 0.3 |
| embedding_dim | 256 | 256 |
| dropout | 0.3 | 0.5 |

**Results**:
| Configuration | Val mAP |
|---|---|
| Default (Exp 2) | 0.7900 |
| Sweep best | 0.8039 |
| Improvement | +0.0139 |

**Kaggle Results (sweep best + re-ranking)**:
| Competition | Public mAP | Date |
|---|---|---|
| Jaguar Re-Identification Challenge | 0.783 | March 3, 2026 |
| [Round-2] Jaguar Re-Identification Challenge | 0.040 | March 3, 2026 |

**Analysis**: Bayesian optimisation found that a higher learning rate 
(7.67e-4 vs 1e-4) combined with stronger dropout (0.5 vs 0.3) and 
a lower alpha (0.3 vs 0.5) significantly improved performance. The 
lower alpha means 30% ArcFace + 70% Triplet Loss works better than 
50/50 — suggesting that direct distance optimisation is more important 
than classification margin for this dataset. The higher arcface margin 
(0.6) pushes class boundaries further apart, which helps with the 
fine-grained nature of jaguar identification. Combined with re-ranking 
from Experiment 4, this configuration achieves 0.783 on the public 
leaderboard — our best result overall.

**W&B Sweep**: (https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/lcvs2mt6)
**Notebook**: (https://www.kaggle.com/code/mishankjain/jaguar-reid-exp05-hyperparam-sweep)
