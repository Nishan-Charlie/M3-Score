import os
import glob
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityd, Resized, ToTensord
)
from monai.data import CacheDataset, DataLoader, set_track_meta

# Disable metadata tracking to avoid "resize storage" errors with multiprocessing
set_track_meta(False)

def ensure_channel_first(d):
    img = d["image"]
    # If it's (H, W), make it (1, H, W)
    if len(img.shape) == 2:
        d["image"] = img[None, ...]
    # If it's (H, W, C), make it (C, H, W)
    elif len(img.shape) == 3 and img.shape[2] in [1, 3]:
        # Move channel from last to first
        if isinstance(img, np.ndarray):
            d["image"] = np.transpose(img, (2, 0, 1))
        else:
            d["image"] = img.permute(2, 0, 1)
    return d

def tensor_collate_fn(batch):
    # Batch is a list of dicts: [{"image": tensor}, ...]
    img_tensors = []
    for item in batch:
        t = item["image"]
        # Ensure 1 channel (Grayscale)
        if t.shape[0] == 3: # RGB
            t = t.mean(dim=0, keepdim=True)
        elif t.shape[0] > 3: # Fallback for odd shapes
            t = t[0:1, ...]
        img_tensors.append(t)
    return torch.stack(img_tensors)

def get_mri_2d_dataloader(data_dir, batch_size=16, spatial_size=(256, 256), num_workers=4):
    """Loads 2D MRI slices (JPG, PNG, etc.) from class-indexed directories."""
    # Find all images recursively
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    
    images = sorted(images)
    if not images:
        raise FileNotFoundError(f"No images found in {data_dir} with extensions {extensions}")
        
    print(f"Found {len(images)} images in {data_dir}")
    data_dicts = [{"image": img_path} for img_path in images]

    train_transforms = Compose([
        # PILReader preserves the standard (H, W) axis convention.
        # MONAI's default ITKReader loads PNG/JPG as (W, H) — transposed —
        # which causes generated images to appear rotated 90° vs real images.
        LoadImaged(keys=["image"], image_only=True, reader="PILReader"),
        ensure_channel_first,
        Resized(keys=["image"], spatial_size=spatial_size),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        ToTensord(keys=["image"]),
    ])

    ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0, progress=False)

    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=tensor_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
