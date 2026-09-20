"""
backbones.py

Frozen feature extractors with one shared interface.

Every backbone:
  - takes a float tensor in [0, 1], shape (B, 3, 224, 224)  (un-normalized RGB)
  - applies ITS OWN normalization internally
  - returns a (B, D) feature tensor

This is what lets us build interventions (grayscale, shuffle, ...) once on a
common 224x224 RGB image and feed the identical pixels to every model.

Features:
  resnet50 : global-average-pooled feature           (D = 2048)
  vit_b16  : final class token (after final LayerNorm) (D = 768)
  clip     : L2-normalized image embedding           (D = 512)

CLIP additionally exposes zero_shot_logits() for the zero-shot baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE_NAMES = ["resnet50", "vit_b16", "clip"]


def uint8_to_tensor(images: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 array -> (N, 3, H, W) float tensor in [0, 1]."""
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    return x


class Backbone(nn.Module):
    """Base class: handles device, freezing, normalization, batching."""

    name = "base"
    feature_dim = 0

    def __init__(self, mean, std, device):
        super().__init__()
        self.device = torch.device(device)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def _finalize(self, model):
        """Freeze, set eval mode, move to device."""
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.to(self.device)
        return model

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _forward_features(self, x_norm: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """x: float [0,1] (B,3,224,224) on any device -> (B, D) features."""
        x = x.to(self.device)
        return self._forward_features(self.normalize(x))

    @torch.inference_mode()
    def extract_array(self, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """(N, H, W, 3) uint8 array -> (N, D) numpy features, in batches."""
        out = []
        for i in range(0, len(images), batch_size):
            batch = uint8_to_tensor(images[i : i + batch_size])
            out.append(self.extract(batch).cpu().numpy())
        return np.concatenate(out, axis=0)


class ResNet50Backbone(Backbone):
    name = "resnet50"
    feature_dim = 2048

    def __init__(self, device="cpu"):
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V2
        tf = weights.transforms()
        super().__init__(tf.mean, tf.std, device)
        model = resnet50(weights=weights)
        model.fc = nn.Identity()  # output = global-average-pooled 2048-d feature
        self.model = self._finalize(model)

    def _forward_features(self, x_norm):
        return self.model(x_norm)


class ViTB16Backbone(Backbone):
    name = "vit_b16"
    feature_dim = 768

    def __init__(self, device="cpu"):
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        tf = weights.transforms()
        super().__init__(tf.mean, tf.std, device)
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()  # output = final class token (post encoder LayerNorm)
        self.model = self._finalize(model)

    def _forward_features(self, x_norm):
        return self.model(x_norm)


class CLIPBackbone(Backbone):
    name = "clip"
    feature_dim = 512

    def __init__(self, device="cpu"):
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        super().__init__(open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD, device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self._finalize(model)
        self._text_cache = {}

    def _forward_features(self, x_norm):
        feats = self.model.encode_image(x_norm)
        return F.normalize(feats, dim=-1)  # normalized CLIP image embedding

    @torch.inference_mode()
    def text_features(self, class_names, prompt="a photo of a {}."):
        key = (tuple(class_names), prompt)
        if key not in self._text_cache:
            tokens = self.tokenizer([prompt.format(c) for c in class_names]).to(self.device)
            self._text_cache[key] = F.normalize(self.model.encode_text(tokens), dim=-1)
        return self._text_cache[key]

    @torch.inference_mode()
    def zero_shot_logits_from_features(self, img_feats, class_names, prompt="a photo of a {}."):
        """img_feats: (B, 512) normalized image embeddings -> (B, C) scaled similarities."""
        txt = self.text_features(class_names, prompt)
        scale = self.model.logit_scale.exp()
        return scale * img_feats.to(self.device) @ txt.T

    @torch.inference_mode()
    def zero_shot_logits(self, x, class_names, prompt="a photo of a {}."):
        """x: float [0,1] image batch -> (B, C) scaled similarity logits."""
        return self.zero_shot_logits_from_features(self.extract(x), class_names, prompt)


def load_backbone(name: str, device=None) -> Backbone:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    factories = {
        "resnet50": ResNet50Backbone,
        "vit_b16": ViTB16Backbone,
        "clip": CLIPBackbone,
    }
    if name not in factories:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {BACKBONE_NAMES}")
    return factories[name](device=device)


if __name__ == "__main__":
    # Smoke test: random images through every backbone.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(4, 3, 224, 224)
    for n in BACKBONE_NAMES:
        bb = load_backbone(n, dev)
        f = bb.extract(x)
        print(f"{n:9s} features {tuple(f.shape)}  (expected D={bb.feature_dim})")
        if n == "clip":
            classes = ["airplane", "bird", "car", "cat", "deer",
                       "dog", "horse", "monkey", "ship", "truck"]
            print("          zero-shot logits", tuple(bb.zero_shot_logits(x, classes).shape))