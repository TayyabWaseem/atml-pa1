import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

def apply_grayscale(images: np.ndarray) -> np.ndarray:
    """
    Convert (N, H, W, 3) uint8 RGB images to grayscale, 
    replicated across 3 channels to maintain shape.
    """
    # Standard luminance weights
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    gray = np.dot(images[..., :3], weights).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)

def apply_hue_rotation(images: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """
    Apply fixed hue rotation to (N, H, W, 3) uint8 RGB images.
    factor: float in [-0.5, 0.5]. 0.5 corresponds to a 180-degree shift.
    """
    out = np.empty_like(images)
    for i in range(len(images)):
        img_pil = Image.fromarray(images[i])
        out_pil = TF.adjust_hue(img_pil, factor)
        out[i] = np.array(out_pil)
    return out

def apply_translation(images: np.ndarray, shift: int, direction: str) -> np.ndarray:
    """
    Translate (N, H, W, 3) uint8 RGB images by `shift` pixels in `direction`.
    Uses reflection padding followed by a shifted crop.
    Directions: 'up', 'down', 'left', 'right'.
    """
    if shift == 0:
        return images.copy()
        
    N, H, W, C = images.shape
    out = np.empty_like(images)
    
    # We can do this efficiently via numpy pad and slice
    for i in range(N):
        img = images[i]
        # Pad with reflection mode
        pad_width = ((shift, shift), (shift, shift), (0, 0))
        padded = np.pad(img, pad_width, mode='symmetric') # 'symmetric' behaves like reflection padding in pytorch commonly
        
        # Crop back to HxW depending on direction
        # If shifting content UP, the top `shift` pixels are removed, and bottom pulls from padding
        if direction == 'up':
            crop = padded[shift*2 : shift*2 + H, shift : shift + W]
        elif direction == 'down':
            crop = padded[0 : H, shift : shift + W]
        elif direction == 'left':
            crop = padded[shift : shift + H, shift*2 : shift*2 + W]
        elif direction == 'right':
            crop = padded[shift : shift + H, 0 : W]
        else:
            raise ValueError(f"Unknown direction {direction}")
            
        out[i] = crop
        
    return out

def apply_patch_shuffle(images: np.ndarray, seed: int = 6304) -> np.ndarray:
    """
    Divide 224x224 images into a 4x4 grid (56x56 patches).
    Shuffle the 16 patches using the given seed.
    Returns (N, H, W, 3) uint8 images.
    """
    N, H, W, C = images.shape
    assert H == 224 and W == 224, "Patch shuffle assumes 224x224 images"
    
    rng = np.random.RandomState(seed)
    
    # Generate one non-identity permutation
    perm = np.arange(16)
    while np.all(perm == np.arange(16)):
        rng.shuffle(perm)
        
    out = np.empty_like(images)
    patch_h, patch_w = H // 4, W // 4
    
    for i in range(N):
        img = images[i]
        # Extract patches
        patches = []
        for row in range(4):
            for col in range(4):
                patch = img[row*patch_h:(row+1)*patch_h, col*patch_w:(col+1)*patch_w]
                patches.append(patch)
                
        # Shuffle and reconstruct
        shuffled_img = np.empty_like(img)
        for idx, p_idx in enumerate(perm):
            row, col = idx // 4, idx % 4
            shuffled_img[row*patch_h:(row+1)*patch_h, col*patch_w:(col+1)*patch_w] = patches[p_idx]
            
        out[i] = shuffled_img
        
    return out

