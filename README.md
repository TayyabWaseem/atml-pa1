# Transfer Learning - ATML

# Task 1 (rough working notes)

Status key: [x] done, [ ] todo

## Setup / Decisions
- Dataset: STL-10 (official train -> stratified 80/20 train/val; official test -> balanced 500 subset)
- Seed: 6304 everywhere
- Backbones (all frozen): ResNet-50 (IMAGENET1K_V2), ViT-B/16 (IMAGENET1K_V1), OpenCLIP ViT-B-32 (pretrained='openai')
- Features: ResNet = pooled 2048-d, ViT = final class token 768-d, CLIP = normalized embedding 512-d
- Images: STL-10 96x96 upsampled ONCE to 224x224 RGB; interventions built on that, normalization inside each model
- Head: linear, AdamW lr 1e-3, wd 1e-4, <=50 epochs, patience 5 on val acc, batch size 128 (my choice, spec silent)
- CLIP zero-shot prompt: "a photo of a {class}."
- CLIP activation check: printed `[CLIP] ... activation:` line (note result here: ______)
- Workflow: edit code locally -> push -> Kaggle clones, runs, pushes small results back
- Experimental choices still TO DECIDE (state hypothesis + metric BEFORE seeing results):
  - [x] dataset = STL-10
  - [ ] cue-conflict class pairs + style strength
  - [ ] extra color intervention (leaning: fixed hue rotation, angle = ___)
  - [ ] visualization method + settings (t-SNE or UMAP; perplexity/n_neighbors, seed)
  - [x] cue-conflict class pairs + style strength (airplane/cat, car/monkey, ship/horse, truck/bird, dog/airplane | alpha=1.0)
  - [x] extra color intervention (leaning: fixed hue rotation, angle = +180 degrees)
  - [x] visualization method + settings (UMAP; n_neighbors=15, min_dist=0.1, seed=6304)

## Pipeline / Files
- [x] `data/make_subset.py` -> `subset_ids.json`, `train_val_split.json`
- [x] `data/common.py` (config, seed, image loading)
- [x] `models/backbones.py` (3 wrappers + CLIP zero-shot)
- [x] `analysis/train_heads.py` (features cache, heads, clean baseline)
- [ ] `data/transforms.py` (grayscale, hue, translation, patch shuffle)
- [ ] `analysis/evaluate_bias.py` (accuracy, consistency, shape bias/coverage)
- [ ] `analysis/feature_similarity.py` (cosine stability)
- [ ] `analysis/representation.py` (t-SNE/UMAP)
- [ ] `data/make_cue_conflicts.py` (AdaIN; do last)
- [ ] `scripts/run_task1.py`
- [x] `data/transforms.py` (grayscale, hue, translation, patch shuffle)
- [x] `analysis/evaluate_bias.py` (accuracy, consistency, shape bias/coverage)
- [x] `analysis/feature_similarity.py` (cosine stability)
- [x] `analysis/representation.py` (UMAP)
- [x] `data/make_cue_conflicts.py` (AdaIN)
- [x] `run_task1.py`

## Step 1: Clean Baseline
- Metrics: top-1, macro-F1, mean max confidence (CLIP zero-shot: softmax over scaled similarities)
- Hypothesis: ______
- Results: see `results/clean_baseline.csv`
- Notes: are the models' starting quality similar? (compare interventions by absolute AND change vs own clean)

## Step 2: Color Bias
- Interventions: grayscale + one extra (hue rotation / palette transfer / class-swapped stats)
- Hypothesis: ______
- Metric: accuracy change vs clean, prediction consistency vs clean
- What extra transform changes / preserves: ______
- What extra transform changes / preserves: Fixed hue rotation perfectly preserves spatial structure, brightness, and contrast, but completely shifts chromaticity.
- Results: ______

## Step 3: Shape vs Texture (cue conflicts)
- Tool: AdaIN (weights: vgg_normalised.pth, decoder.pth -> get these onto Kaggle early)
- >=5 unordered class pairs, both directions, >=200 valid conflicts, balanced
- Class pairs: ______ | Style strength (alpha): ______
- Visual rejection rule (write BEFORE evaluating, never use model predictions): ______
- Class pairs: airplane/cat, car/monkey, ship/horse, truck/bird, dog/airplane | Style strength (alpha): 1.0
- Visual rejection rule (write BEFORE evaluating, never use model predictions): Reject any stylization where pixel variance < 100 (indicating a washed out / collapsed color block).
- Record accepted / rejected counts
- Predictions labelled: shape/content, texture/style, or other
- Metrics: Shape Bias = Ns/(Ns+Nt)*100, Coverage = (Ns+Nt)/Ntotal*100
- Hypothesis: ______
- Results: ______
- Save a few example agreements/disagreements/failures with predictions

## Step 4: Translation
- Shifts 0, 8, 16, 32 px, four directions, reflection pad + shifted crop, average over directions
- Plot accuracy and consistency vs displacement
- Hypothesis: ______
- Results: ______

## Step 5: Patch Structure
- 4x4 pixel-space grid, one non-identity permutation per image, seed 6304, same shuffled images for all models
- Metrics: accuracy drop, prediction consistency vs clean
- Hypothesis: ______
- Results: ______
- Watch: confident != sensible after shuffling; look at what evidence remains

## Step 6: Representation Analysis
- Cosine stability for: grayscale, cue conflict, translation, patch shuffle (clean vs transformed pair)
- Projection: ______ (settings: ______), one fit per backbone on clean+transformed combined
- Colour = class, marker = clean vs transformed; do NOT compare coordinates across backbones
- Hypothesis: ______
- Results: ______

## Required Evidence Checklist
- [ ] Table: clean, grayscale, extra color, patch shuffle
- [ ] Shape / texture / other counts + shape bias + coverage
- [ ] Translation curve
- [ ] Representation stability for all interventions + t-SNE/UMAP plots
- [ ] Small set of cue-conflict examples with predictions

## Research Questions (answers at the end)
1. Color + cue-conflict: reliance on shape / texture / color? Effect of coverage on conclusions?
2. Translation + patch shuffle: locality, global organization, positional sensitivity per model?
3. Prediction changes vs feature changes: one agreement/mismatch; CLIP zero-shot vs its linear head
4. Architecture vs pretraining data / supervision / augmentation / capacity: what can be attributed to what?

## Housekeeping
- `.gitignore`: .venv/, task1/data/raw/, task1/data/interventions/, task1/features/, task1/checkpoints/, *.pt, *.pth, *.npy
- Never paste the GitHub token into a cell; revoke and regenerate if leaked
- Kaggle sessions wipe /kaggle/working: re-clone at start, push results at end
- Tag final run: `git tag final-results`