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
