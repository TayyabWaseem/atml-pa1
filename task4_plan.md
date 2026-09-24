# Task 4: Open-Set Recognition (OSR)

## Overview
In the real world, deployed models often encounter objects they have never been trained on. A standard closed-set classifier will confidently (and incorrectly) assign these inputs to one of its known classes. **Open-Set Recognition (OSR)** adds the ability to reject inputs that do not belong to the known classes.

- **Dataset:** CIFAR-10 (the 10 known classes) and selected CIFAR-100 classes (the unknowns).
  - *Near Unknowns:* Animals and vehicles visually similar to CIFAR-10 (e.g., wolf, bus).
  - *Far Unknowns:* Objects visually distinct from CIFAR-10 (e.g., bottle, keyboard).
- **Objective:** Compare different ways to calculate an "unknownness" score to reject novel inputs while maintaining high accuracy on known inputs.
- **Methods Evaluated:**
  1. **Vanilla Baseline:** A standard ResNet-18 adapted for 32x32 CIFAR images. Evaluated with 4 different post-hoc scoring functions:
     - **MSP:** Maximum Softmax Probability (1 - max probability).
     - **MLS:** Maximum Logit Score (negative max logit).
     - **Energy:** Negative log-sum-exp of logits.
     - **Mahalanobis:** Distance from known-class feature clusters.
  2. **GCSC (Strong Closed-Set Classifier):** Tests if simply adding strong data augmentation (`RandAugment`) improves open-set rejection alongside closed-set accuracy. Evaluated with MLS.
  3. **PROSER:** Changes the training process to prepare for unknowns by adding 5 "dummy" classifiers and creating synthetic "unknown" features by mixing known features halfway through the network (Manifold Mixup).

## Required Decisions / Choices
There is only one decision to make regarding the scope of Task 4:

1. **Optional Extension:** Would you like me to implement the optional **Reciprocal Point Learning (RPL)** extension? 
   - RPL learns what a class *is not* by pushing features away from a learned reciprocal point. It requires an additional training run and an open-space regularization objective. It can grant extra points/insight but is not strictly required by the base assignment.

---

## Execution Plan

We will follow the modular directory structure specified in the assignment.

### Phase 1: Data & Models
1. **`task4/data/`**: Implement scripts to split CIFAR-10 (90/10 train/val) and carefully extract only the specific Near and Far unknown classes from the CIFAR-100 test set. (No CIFAR-100 images are allowed during training or threshold tuning).
2. **`task4/models/resnet_cifar.py`**: Modify the standard ResNet-18 to replace the initial 7x7 stride-2 conv with a 3x3 stride-1 conv and remove the initial max-pool (standard practice for 32x32 images).

### Phase 2: Training Methods
1. **`task4/methods/vanilla.py`**: Standard SGD, cosine decay, 100 epochs.
2. **`task4/methods/gcsc.py`**: Adds `RandAugment(num_ops=2, magnitude=9)` to the pipeline.
3. **`task4/methods/proser.py` & `manifold_mixup.py`**: Initializes from the Vanilla checkpoint. Splits the batch in half: one half trains the 5 dummy classifiers (Classifier Placeholders), and the other half uses manifold mixup between `layer2` and `layer3` to create proxy unknowns (Data Placeholders).

### Phase 3: Post-Hoc Scores & Extraction
1. **`task4/extract_outputs.py`**: Runs the models over the datasets and caches the penultimate features $f(x)$ and logits $z(x)$ to disk.
2. **`task4/scores/`**: Implement MSP, MLS, Energy, and Mahalanobis scoring functions. (Mahalanobis requires computing the class means and shared covariance matrix from unaugmented training features).

### Phase 4: Evaluation & Diagnostics
1. **`task4/evaluation/thresholds.py`**: Calculates the rejection threshold $\tau$ defined as the 95th percentile of unknownness on the CIFAR-10 validation set.
2. **`task4/evaluate_osr.py`**: Calculates AUROC (Known vs. Near, Far, All) and FPR@95TPR (fraction of unknowns incorrectly accepted). Also identifies specific semantically plausible confusions vs surprising failures.
3. **Plotting**: Generates the required Score Distribution / ROC plots.
