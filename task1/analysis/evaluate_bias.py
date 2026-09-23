import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from task1.data.common import get_stl10, load_images, load_json, load_config, CKPT_DIR, RESULTS_DIR, FEATURE_DIR
from task1.data.transforms import apply_grayscale, apply_hue_rotation, apply_translation, apply_patch_shuffle
from task1.models.backbones import BACKBONE_NAMES, load_backbone
from task1.analysis.train_heads import compute_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_head(name, n_classes, device):
    if name == "clip":
        return None
    bb = load_backbone(name, device)
    head = torch.nn.Linear(bb.feature_dim, n_classes).to(device)
    head.load_state_dict(torch.load(CKPT_DIR / f"{name}_head.pt", map_location=device, weights_only=True))
    head.eval()
    del bb
    return head

def evaluate_transform(bb, head, name, class_names, images, yte, clean_preds):
    """
    bb: Backbone
    head: nn.Linear or None (for CLIP zero-shot)
    name: str backbone name
    images: transformed numpy images
    yte: ground truth labels
    clean_preds: predictions on clean images
    """
    feats = bb.extract_array(images, batch_size=64)
    with torch.no_grad():
        if name == "clip" and head is None:
            logits = bb.zero_shot_logits_from_features(torch.from_numpy(feats), class_names)
        else:
            logits = head(torch.from_numpy(feats).float().to(DEVICE))
            
    probs = logits.softmax(dim=1)
    preds = probs.argmax(1).cpu().numpy()
    
    acc = float((preds == yte).mean())
    consistency = float((preds == clean_preds).mean())
    return acc, consistency

def get_clean_preds(bb, head, name, class_names, clean_images):
    feats = bb.extract_array(clean_images, batch_size=64)
    with torch.no_grad():
        if name == "clip" and head is None:
            logits = bb.zero_shot_logits_from_features(torch.from_numpy(feats), class_names)
        else:
            logits = head(torch.from_numpy(feats).float().to(DEVICE))
    return logits.softmax(dim=1).argmax(1).cpu().numpy()

def main():
    cfg = load_config()
    subset = load_json("data/subset_ids.json")
    test_idx = subset["test_indices"]
    yte = np.asarray(subset["test_labels"])
    class_names = subset["class_names"]
    n_classes = len(class_names)
    
    ds = get_stl10("test", cfg)
    clean_images = load_images(ds, test_idx)
    
    print("Generating transformed images...")
    gray_imgs = apply_grayscale(clean_images)
    hue_imgs = apply_hue_rotation(clean_images, factor=0.5) # 180 degrees
    shuffle_imgs = apply_patch_shuffle(clean_images, seed=cfg["seed"])
    
    # Translations
    shifts = [0, 8, 16, 32]
    dirs = ['up', 'down', 'left', 'right']
    
    rows = []
    translation_results = []
    
    # Pre-calculate translation images
    trans_imgs = {s: {d: apply_translation(clean_images, s, d) for d in dirs} for s in shifts}
    
    for name in BACKBONE_NAMES:
        print(f"\nEvaluating {name}...")
        bb = load_backbone(name, DEVICE)
        
        # We need to evaluate both the linear head and CLIP zero-shot if it's clip
        evaluators = [(name, load_head(name, n_classes, DEVICE))]
        if name == "clip":
            evaluators.append(("clip_zero_shot", None))
            
        for eval_name, head in evaluators:
            clean_preds = get_clean_preds(bb, head, name, class_names, clean_images)
            clean_acc = float((clean_preds == yte).mean())
            
            # Basic transforms
            gray_acc, gray_cons = evaluate_transform(bb, head, name, class_names, gray_imgs, yte, clean_preds)
            hue_acc, hue_cons = evaluate_transform(bb, head, name, class_names, hue_imgs, yte, clean_preds)
            shuf_acc, shuf_cons = evaluate_transform(bb, head, name, class_names, shuffle_imgs, yte, clean_preds)
            
            rows.append({
                "model": eval_name,
                "clean_acc": clean_acc,
                "gray_acc": gray_acc, "gray_cons": gray_cons,
                "hue_acc": hue_acc, "hue_cons": hue_cons,
                "shuffle_acc": shuf_acc, "shuffle_cons": shuf_cons
            })
            
            # Translation curve
            for s in shifts:
                if s == 0:
                    translation_results.append({
                        "model": eval_name, "shift": s, "acc": clean_acc, "cons": 1.0
                    })
                    continue
                
                accs, conss = [], []
                for d in dirs:
                    a, c = evaluate_transform(bb, head, name, class_names, trans_imgs[s][d], yte, clean_preds)
                    accs.append(a); conss.append(c)
                
                translation_results.append({
                    "model": eval_name, "shift": s,
                    "acc": np.mean(accs), "cons": np.mean(conss)
                })
                
        del bb
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    df_basic = pd.DataFrame(rows)
    df_trans = pd.DataFrame(translation_results)
    
    df_basic.to_csv(RESULTS_DIR / "transform_bias.csv", index=False)
    df_trans.to_csv(RESULTS_DIR / "translation_curve.csv", index=False)
    
    print("\n--- Basic Transforms ---")
    print(df_basic.to_string(index=False))
    
    print("\n--- Translation Curve ---")
    print(df_trans.groupby(['model', 'shift'])[['acc', 'cons']].mean().to_string())
    
    # Now evaluate cue conflicts
    conflicts_path = REPO_ROOT / "task1" / "data" / "interventions" / "cue_conflicts" / "conflicts.json"
    if conflicts_path.exists():
        print("\nEvaluating Cue Conflicts...")
        with open(conflicts_path, "r") as f:
            conflicts_info = json.load(f)
            
        from PIL import Image
        conflict_imgs = []
        for info in conflicts_info:
            img_path = REPO_ROOT / "task1" / "data" / "interventions" / "cue_conflicts" / info["filename"]
            conflict_imgs.append(np.array(Image.open(img_path).convert("RGB")))
        conflict_imgs = np.array(conflict_imgs)
        
        conflict_rows = []
        
        for name in BACKBONE_NAMES:
            bb = load_backbone(name, DEVICE)
            evaluators = [(name, load_head(name, n_classes, DEVICE))]
            if name == "clip":
                evaluators.append(("clip_zero_shot", None))
                
            for eval_name, head in evaluators:
                feats = bb.extract_array(conflict_imgs, batch_size=64)
                with torch.no_grad():
                    if name == "clip" and head is None:
                        logits = bb.zero_shot_logits_from_features(torch.from_numpy(feats), class_names)
                    else:
                        logits = head(torch.from_numpy(feats).float().to(DEVICE))
                
                preds = logits.softmax(dim=1).argmax(1).cpu().numpy()
                
                n_shape, n_texture, n_other = 0, 0, 0
                for i, info in enumerate(conflicts_info):
                    pred_class = class_names[preds[i]]
                    if pred_class == info["content_class"]:
                        n_shape += 1
                    elif pred_class == info["style_class"]:
                        n_texture += 1
                    else:
                        n_other += 1
                        
                shape_bias = n_shape / (n_shape + n_texture + 1e-9) * 100
                coverage = (n_shape + n_texture) / len(conflicts_info) * 100
                
                conflict_rows.append({
                    "model": eval_name,
                    "n_shape": n_shape,
                    "n_texture": n_texture,
                    "n_other": n_other,
                    "shape_bias": shape_bias,
                    "coverage": coverage
                })
        
        df_conflicts = pd.DataFrame(conflict_rows)
        df_conflicts.to_csv(RESULTS_DIR / "cue_conflicts_bias.csv", index=False)
        print("\n--- Cue Conflicts Bias ---")
        print(df_conflicts.to_string(index=False))

if __name__ == "__main__":
    main()

