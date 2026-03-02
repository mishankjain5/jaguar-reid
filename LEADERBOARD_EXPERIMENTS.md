# Leaderboard Experiments

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

**W&B Run**: (https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/q23er6i8)
