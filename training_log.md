# Training Log - AI vs Real Image Classifier

## Project Overview
**Objective:** Build a binary image classifier to detect AI-generated images vs. Real images
**Dataset:** 10,000 high-quality images (5,000 Real, 5,000 AI-generated)
**Target Accuracy:** 80-95%
**Critical Constraint:** Minimize false positives (Real images classified as AI)

---

## Operational Rules
- Step-by-step execution with approval
- System safety & cleanliness
- Terminal-first approach for all training/evaluation
- Document all steps, commands, and metrics

---

## Training Runs

### Run #1
*Date:* [Pending]
*Configuration:* [Pending]
*Results:* [Pending]

---

## Model Comparisons

### Comparison #1
*Date:* [Pending]
*Models Compared:* [Pending]
*Results:* [Pending]

---

## Script Development Log

### 2026-04-25: train.py Created
- **Location:** `src/train.py`
- **Features:**
  - Full argparse configuration (epochs, lr, batch_size, model, augmentation, class weights)
  - Data augmentation: RandomHorizontalFlip, RandomRotation(15), ColorJitter, RandomAffine
  - Pretrained ResNet18 backbone with custom FC head + dropout
  - Class weighting to protect "Real" class (`--real_weight` default 1.5)
  - Configurable decision threshold (`--threshold` default 0.5)
  - Metrics tracked: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
  - Saves best model by F1-score and periodic checkpoints

### 2026-04-25: compare.py Created
- **Location:** `src/compare.py`
- **Features:**
  - Takes two model checkpoint paths as positional arguments
  - Side-by-side terminal comparison of all metrics
  - Highlights winning model per metric
  - Prints confusion matrices for both models

### Example Commands
```bash
# Quick test run (5 epochs)
python src/train.py --epochs 5 --batch_size 16 --lr 1e-4 --data_dir dataset --output_dir models

# Full training run (targeting 80-95% accuracy)
python src/train.py --epochs 25 --batch_size 32 --lr 1e-4 --model resnet18 --pretrained --real_weight 1.5 --threshold 0.5

# Compare two saved models
python src/compare.py models/resnet18_best.pt models/resnet18_epoch20.pt --data_dir dataset
```

---

### 2026-04-25: Corrupted Images Fixed
- **Issue:** 2 real images were corrupted (0 bytes): `real_01144.jpg` and `real_01215.jpg`
- **Fix:** Deleted corrupted files, re-downloaded from picsum.photos
- **Result:** Dataset now complete — 5,000 Real + 5,000 AI = 10,000 total valid images

---

## Notes & Observations
