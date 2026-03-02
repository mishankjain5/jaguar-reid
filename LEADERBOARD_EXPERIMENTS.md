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

**W&B Run**: (https://wandb.ai/jain5-university-of-potsdam/jaguar-reid-mishank/runs/q23er6i8)
