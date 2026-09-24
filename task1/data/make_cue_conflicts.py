import os
import urllib.request
import json
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm
import sys

# Repo root = PA1/
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from task1.data.common import get_stl10, load_images, load_config, load_json, set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = REPO_ROOT / "task1" / "data" / "weights"
OUT_DIR = REPO_ROOT / "task1" / "data" / "interventions" / "cue_conflicts"

VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
DEC_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"

# Extreme contrast pairs (class names)
CLASS_PAIRS = [
    ("airplane", "cat"),
    ("car", "monkey"),
    ("ship", "horse"),
    ("truck", "bird"),
    ("dog", "airplane")
]
STYLE_ALPHA = 1.0

def download_weights():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    vgg_path = WEIGHTS_DIR / "vgg_normalised.pth"
    dec_path = WEIGHTS_DIR / "decoder.pth"
    
    if not vgg_path.exists():
        print(f"Downloading VGG weights from {VGG_URL}...")
        urllib.request.urlretrieve(VGG_URL, vgg_path)
    if not dec_path.exists():
        print(f"Downloading Decoder weights from {DEC_URL}...")
        urllib.request.urlretrieve(DEC_URL, dec_path)
        
    return vgg_path, dec_path

# AdaIN definition
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1
)

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class AdaINModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encoder(style)
        content_feats = self.encoder(content)
        t = adain(content_feats, style_feats)
        t = alpha * t + (1 - alpha) * content_feats
        return self.decoder(t)

def preprocess(img_np):
    # Convert uint8 numpy array to [0,1] tensor
    return TF.to_tensor(img_np).unsqueeze(0).to(DEVICE)

def postprocess(tensor):
    # Convert [0,1] tensor back to uint8 numpy array
    tensor = tensor.clamp(0, 1).squeeze(0).cpu()
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def visual_rejection_proxy(stylized_img_np):
    """
    Reject stylizations that are 'broken'.
    Simple proxy: if the variance of the image is too low (washed out / solid color).
    """
    variance = np.var(stylized_img_np)
    return variance < 100.0 # Arbitrary threshold, tune if needed
def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    
    vgg_path, dec_path = download_weights()
    
    # 1. Define the FULL VGG (the weights file expects this)
    #    (copy the full definition from the official net.py – it goes up to relu5-4)
    full_vgg = nn.Sequential(   # ← put the COMPLETE architecture here
        # ... all layers up to and including the last ReLU of block 5
    )
    
    # 2. Load the full weights
    full_vgg.load_state_dict(torch.load(vgg_path, map_location="cpu"))
    
    # 3. Truncate to relu4-1 (first 31 layers)
    encoder = nn.Sequential(*list(full_vgg.children())[:31])
    
    # 4. Load decoder
    decoder.load_state_dict(torch.load(dec_path, map_location="cpu"))
    
    model = AdaINModel(encoder, decoder)
    model.eval()
    model.to(DEVICE)

    subset = load_json("data/subset_ids.json")
    test_idx = subset["test_indices"]
    test_labels = np.array(subset["test_labels"])
    class_names = subset["class_names"]
    
    ds = get_stl10("test", cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Organize indices by class
    class_to_indices = {c: [] for c in class_names}
    for idx, label in zip(test_idx, test_labels):
        c_name = class_names[label]
        class_to_indices[c_name].append(idx)
        
    conflicts_info = []
    
    for cA, cB in CLASS_PAIRS:
        print(f"Generating conflicts for {cA} (content) and {cB} (style)...")
        # We need ~20 valid conflicts per pair per direction to reach >200 total (10 directions * 20 = 200)
        
        for c_content, c_style in [(cA, cB), (cB, cA)]:
            content_indices = class_to_indices[c_content]
            style_indices = class_to_indices[c_style]
            
            valid_count = 0
            # Try up to 50 pairs to get 20 valid ones
            for i in range(min(50, len(content_indices))):
                if valid_count >= 20:
                    break
                    
                idx_c = content_indices[i]
                idx_s = style_indices[i % len(style_indices)] # just wrap around if fewer style images
                
                img_c = load_images(ds, [idx_c])[0]
                img_s = load_images(ds, [idx_s])[0]
                
                with torch.no_grad():
                    out_tensor = model(preprocess(img_c), preprocess(img_s), alpha=STYLE_ALPHA)
                
                out_img = postprocess(out_tensor)
                
                if not visual_rejection_proxy(out_img):
                    valid_count += 1
                    filename = f"content_{c_content}_style_{c_style}_{valid_count}.png"
                    Image.fromarray(out_img).save(OUT_DIR / filename)
                    
                    conflicts_info.append({
                        "filename": filename,
                        "content_class": c_content,
                        "style_class": c_style,
                        "content_idx": idx_c,
                        "style_idx": idx_s
                    })
                    
            print(f"  {c_content} (c) + {c_style} (s): {valid_count} valid conflicts")

    with open(OUT_DIR / "conflicts.json", "w") as f:
        json.dump(conflicts_info, f, indent=2)
        
    print(f"\nGenerated {len(conflicts_info)} valid conflicts in total.")
    print(f"Saved to {OUT_DIR}")

if __name__ == "__main__":
    main()

