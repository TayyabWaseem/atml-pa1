# ATML PA1: What Should a Representation Preserve?

This repository contains the codebase and experiments for four complementary tasks investigating model failure modes under distribution shift and open-set conditions.

## Repository Structure

- `task1/` - Inductive Biases (Shape vs Texture, Color, Translation, Patch Shuffle)
- `task2/` - Unsupervised Domain Adaptation (Source-only, DAN, DANN, CDAN)
- `task3/` - Domain Generalization (DAN-DG, SAM, Sharpness Proxy)
- `task4/` - Open-Set Recognition (Vanilla, GCSC, PROSER, Post-hoc Scoring)
- `shared/` - Shared utilities, dataloaders (PACS), and cross-task modules
- `report.pdf` - Comprehensive report of all findings

## Environment Setup

The code was developed and executed using Python 3.10 and PyTorch. 

To install the required dependencies locally:
```bash
pip install -r requirements.txt
```

---

## Reproducing the Experiments

### Task 1: Inductive Biases (STL-10)

Task 1 explores the inductive biases of ResNet-50, ViT-B/16, and CLIP ViT-B/32 on the STL-10 dataset using linear probes and zero-shot evaluation.

1. **Generate the AdaIN Cue-Conflict Dataset:**
   This script downloads the pretrained VGG/Decoder weights and uses AdaIN to generate 200 cue-conflict images (e.g., airplane shape with cat texture).
   ```bash
   python task1/data/make_cue_conflicts.py
   ```
2. **Run all Task 1 Experiments:**
   This script evaluates clean baselines, color interventions (grayscale, hue rotation), spatial interventions (translation, patch shuffling), and the generated cue-conflict dataset.
   ```bash
   python task1/run_task1.py
   ```
   *Results are saved in `task1/results/` as CSV files, along with UMAP visualisations (`.png`).*

### Task 2: Unsupervised Domain Adaptation (PACS)

Task 2 evaluates statistical (DAN) and adversarial (DANN, CDAN) feature alignment to close the domain gap from Photo, Art Painting, and Cartoon to the unlabeled Sketch domain.

1. **Train UDA Models:**
   ```bash
   python task2/train.py --method source_only
   python task2/train.py --method dan
   python task2/train.py --method dann --max_grl 1.0
   python task2/train.py --method dann --max_grl 0.5
   python task2/train.py --method dann --max_grl 0.25
   python task2/train.py --method cdan
   ```
2. **Evaluate and Plot:**
   This computes target Sketch accuracy, macro-F1, and domain separability using a logistic regression probe.
   ```bash
   python task2/evaluate_final.py
   python task2/plot_results.py
   ```
   *Results are saved to `task2/results/evaluation.json`.*

### Task 3: Domain Generalization (PACS)

Task 3 tackles generalization to the Sketch domain when it is completely hidden during training and model selection. It evaluates Sharpness-Aware Minimization (SAM) and multi-source Domain Generalization (DAN-DG).

1. **Train DG Models:**
   *Note: These methods initialize from the Task 2 `source_only` checkpoint.*
   ```bash
   python task3/train.py --method sam
   python task3/train.py --method dan_dg --lambda_dg 0.1
   python task3/train.py --method dan_dg --lambda_dg 1.0
   python task3/train.py --method dan_dg --lambda_dg 10.0
   ```
2. **Evaluate on Unseen Target (Sketch):**
   ```bash
   python task3/evaluate_sketch.py
   ```
   *Results and sharpness proxy metrics ($\Delta_{sharp}$) are saved to `task3/results/evaluation.json`.*

### Task 4: Open-Set Recognition (CIFAR)

Task 4 tests the ability of a CIFAR-10 classifier to confidently reject Near and Far unknown classes drawn from CIFAR-100.

1. **Train OSR Models:**
   ```bash
   python task4/train.py --method vanilla
   python task4/train.py --method gcsc
   python task4/train.py --method proser
   ```
2. **Evaluate Post-Hoc Scores and Extract Outputs:**
   ```bash
   python task4/evaluate_osr.py
   python task4/extract_outputs.py
   ```
3. **Plot Distributions and Generate Failure Analysis:**
   ```bash
   python task4/plot_osr.py
   python task4/failure_analysis.py
   ```
   *Plots are saved to `task4/results/figures/` and metrics (AUROC, FPR@95) to `task4/results/evaluation.json`.*

---

### Loss Curves (Tasks 2 & 3)

Keep track of logs during Task 2 aand Task 3 runs. To generate the aggregated loss curves from the raw training logs (`missing-data.txt`):
```bash
python plot_training_logs.py
```
*(This generates `task2/results/figures/loss_curves.png` and `task3/results/figures/loss_curves.png`)*