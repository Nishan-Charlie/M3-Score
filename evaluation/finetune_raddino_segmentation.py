"""
finetune_radiodino_segmentation.py
==================================
Fine-tunes the RaDDINO (microsoft/rad-dino) backbone on BraTS 2D FLAIR slices
using a segmentation proxy task. The fine-tuned backbone is then usable as
a brain-MRI-aware feature extractor for the Rad-FID metric.

Key design decisions vs. the original script:
  1. Differential learning rates: backbone 1e-5, decoder 1e-4 (10x ratio)
     — prevents catastrophic forgetting of the pre-trained medical features.
  2. Dice + CrossEntropy combined loss: handles severe class imbalance
     (most pixels are background class 0 in BraTS masks).
  3. Class-weighted CrossEntropy: further down-weights the dominant background.
  4. Data augmentation: random horizontal/vertical flips + brightness jitter
     on the raw PIL images before the processor, increasing effective dataset size.
  5. Validation Dice score: per-class and mean Dice logged each epoch.
  6. More epochs (30): ViT fine-tuning with frozen layers needs more time.
  7. Backbone frozen for epoch 1..freeze_epochs, then unfrozen gradually
     (linear probing → fine-tuning strategy).
  8. Backbone CLS token saved separately for metric extraction.
  9. Cosine LR scheduler with linear warmup.
 10. Gradient clipping (max_norm=1.0) for stability.

BraTS label map:
  0 = background
  1 = necrotic tumour core (NCR)
  2 = peritumoral oedema (ED)
  4 = enhancing tumour (ET)  -->  remapped to 3
"""

import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from transformers import AutoModel, ViTImageProcessor, AutoImageProcessor
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def map_label(mask: np.ndarray) -> np.ndarray:
    """Remap BraTS label 4 → 3 for contiguous class indices [0,1,2,3]."""
    mask = mask.copy()
    mask[mask == 4] = 3
    return mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BraTSSegmentationDataset(Dataset):
    """
    Loads paired (FLAIR slice PNG, segmentation mask PNG) from two directories.
    Applies augmentation during training.
    """

    def __init__(self, images_dir: str, masks_dir: str, processor, augment: bool = False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor
        self.augment = augment

        img_files = set(f for f in os.listdir(images_dir) if f.endswith(".png"))
        mask_files = set(f for f in os.listdir(masks_dir) if f.endswith(".png"))
        self.image_files = sorted(img_files.intersection(mask_files))

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No matched PNG pairs found.\n  images: {images_dir}\n  masks:  {masks_dir}"
            )

    def __len__(self):
        return len(self.image_files)

    def _augment(self, image: Image.Image, mask: np.ndarray):
        """Apply consistent spatial augmentation to image+mask pair."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask)
        # Random vertical flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = np.flipud(mask)
        # Random brightness/contrast jitter (image only)
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image, np.ascontiguousarray(mask)

    def __getitem__(self, idx: int):
        fname = self.image_files[idx]
        image = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
        mask = np.array(Image.open(os.path.join(self.masks_dir, fname)).convert("L"))
        mask = map_label(mask)

        if self.augment:
            image, mask = self._augment(image, mask)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)   # (C, H, W)
        mask_tensor = torch.tensor(mask, dtype=torch.long)  # (H, W) values 0-3
        return pixel_values, mask_tensor


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft multi-class Dice loss.
    logits : (B, C, H, W)   raw logits
    targets: (B, H, W)      LongTensor class indices
    """
    probs = F.softmax(logits, dim=1)                          # (B, C, H, W)
    one_hot = F.one_hot(targets, num_classes)                 # (B, H, W, C)
    one_hot = one_hot.permute(0, 3, 1, 2).float()            # (B, C, H, W)

    intersection = (probs * one_hot).sum(dim=(2, 3))          # (B, C)
    union = (probs + one_hot).sum(dim=(2, 3))                 # (B, C)
    dice_per_class = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice_per_class.mean()


def combined_loss(logits, targets, ce_criterion, num_classes, alpha=0.5):
    """alpha * CE + (1-alpha) * Dice."""
    ce = ce_criterion(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    return alpha * ce + (1.0 - alpha) * dc


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _is_timm_model(model_name: str) -> bool:
    """RadioDino-s16 and similar are timm/HF-hub models, not transformers AutoModel."""
    return model_name.startswith("Snarcy/") or model_name.startswith("hf_hub:")


def _load_backbone(model_name: str):
    """Load backbone via timm (RadioDino-s16) or transformers (rad-dino)."""
    if _is_timm_model(model_name):
        import timm
        hf_id = f"hf_hub:{model_name}" if not model_name.startswith("hf_hub:") else model_name
        model = timm.create_model(hf_id, pretrained=True)
        model.hidden_size  = model.embed_dim          # expose as attribute
        model.is_timm      = True
        return model
    else:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(model_name)
        m.hidden_size = m.config.hidden_size
        m.is_timm     = False
        return m


def _backbone_forward(backbone, pixel_values: torch.Tensor):
    """Unified forward: returns (last_hidden_state, cls_token) regardless of backbone type."""
    if getattr(backbone, "is_timm", False):
        x = backbone.patch_embed(pixel_values)
        if hasattr(backbone, "_pos_embed"):
            x = backbone._pos_embed(x)
        else:
            cls = backbone.cls_token.expand(x.shape[0], -1, -1)
            x   = torch.cat([cls, x], dim=1) + backbone.pos_embed
        if hasattr(backbone, "patch_drop"): x = backbone.patch_drop(x)
        if hasattr(backbone, "norm_pre"):   x = backbone.norm_pre(x)
        for blk in backbone.blocks:
            x = blk(x)
        x = backbone.norm(x)
        return x, x[:, 0, :]   # (B, N+1, D), (B, D)
    else:
        out = backbone(pixel_values)
        hs  = out.last_hidden_state
        return hs, hs[:, 0, :]


class RadioDinoSegmentationModel(nn.Module):
    """
    RadioDino backbone (ViT-S/16 @ 224px or ViT-B/14 @ 518px) + ConvTranspose2d decoder.

    Input  : (B, 3, H, W) — 224×224 for RadioDino-s16, 518×518 for rad-dino
    Output : (B, num_classes, H, W) segmentation logits
    """

    def __init__(self, model_name: str = "Snarcy/RadioDino-s16", num_classes: int = 4):
        super().__init__()
        self.backbone = _load_backbone(model_name)
        hidden_size = self.backbone.hidden_size  # 384 for ViT-S, 768 for ViT-B

        # Lightweight 3-stage upsampling decoder (8× upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hs, _ = _backbone_forward(self.backbone, pixel_values)
        patch_embeddings = hs[:, 1:, :]   # drop CLS  (B, N, D)

        B, N, D = patch_embeddings.shape
        G = int(round(N ** 0.5))
        if G * G != N:
            raise ValueError(
                f"Patch count {N} is not a perfect square. "
                "The preprocessor resolution may not match the model's patch stride."
            )

        x = patch_embeddings.reshape(B, G, G, D).permute(0, 3, 1, 2).contiguous()
        logits = self.decoder(x)
        logits = F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def get_cls_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns the CLS token embedding (B, hidden_size) for metric extraction."""
        with torch.no_grad():
            _, cls = _backbone_forward(self.backbone, pixel_values)
        return cls


# ---------------------------------------------------------------------------
# Dice metric for validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_val_dice(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    class_tp = torch.zeros(num_classes)
    class_fp = torch.zeros(num_classes)
    class_fn = torch.zeros(num_classes)

    for pixel_values, masks in loader:
        pixel_values, masks = pixel_values.to(device), masks.to(device)
        logits = model(pixel_values)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1)       # (B, H, W)

        for c in range(num_classes):
            pred_c = (preds == c)
            true_c = (masks == c)
            class_tp[c] += (pred_c & true_c).sum().item()
            class_fp[c] += (pred_c & ~true_c).sum().item()
            class_fn[c] += (~pred_c & true_c).sum().item()

    dice_per_class = {}
    eps = 1e-6
    for c in range(num_classes):
        dice_per_class[c] = (2 * class_tp[c] + eps) / (2 * class_tp[c] + class_fp[c] + class_fn[c] + eps)
    mean_dice = float(np.mean(list(dice_per_class.values())))
    return {"mean_dice": mean_dice, "per_class": {c: float(v) for c, v in dice_per_class.items()}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune RaDDINO for BraTS segmentation.")
    parser.add_argument("--images_dir", default="/home/e21283/mediGAN/mri-diffuser/huggingface_models/data_mri/brats_axial")
    parser.add_argument("--masks_dir",  default="/home/e21283/mediGAN/mri-diffuser/huggingface_models/data_mri/brats_axial_masks")
    parser.add_argument("--output_dir", default="/home/e21283/mediGAN/mri-diffuser/huggingface_models/output/radiodino_segmentation")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--backbone_lr",type=float, default=1e-5,  help="LR for frozen→unfrozen backbone")
    parser.add_argument("--decoder_lr", type=float, default=1e-4,  help="LR for decoder head (always trained)")
    parser.add_argument("--freeze_epochs", type=int, default=5,    help="Epochs to keep backbone frozen (linear probing)")
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--num_classes",type=int,   default=4)
    parser.add_argument("--model_name", type=str,   default="Snarcy/RadioDino-s16",
                        help="Backbone model ID (timm: Snarcy/RadioDino-s16 or transformers: microsoft/rad-dino)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Processor / transforms --
    print(f"Loading backbone processor for {args.model_name} ...")
    if _is_timm_model(args.model_name):
        # timm models use standard ImageNet transforms at 224×224
        from torchvision import transforms as _tv
        img_size = 224
        processor = _tv.Compose([
            _tv.Resize((img_size, img_size)),
            _tv.ToTensor(),
            _tv.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        try:
            processor = ViTImageProcessor.from_pretrained(args.model_name)
        except Exception:
            processor = AutoImageProcessor.from_pretrained(args.model_name)

    # -- Dataset --
    print("Building dataset...")
    full_dataset_train = BraTSSegmentationDataset(args.images_dir, args.masks_dir, processor, augment=True)
    full_dataset_val   = BraTSSegmentationDataset(args.images_dir, args.masks_dir, processor, augment=False)

    n = len(full_dataset_train)
    train_n = int(0.8 * n)
    val_n   = n - train_n
    # Use same indices for both (augment=True for train, False for val)
    indices = list(range(n))
    random.shuffle(indices)
    train_indices, val_indices = indices[:train_n], indices[train_n:]

    from torch.utils.data import Subset
    train_ds = Subset(full_dataset_train, train_indices)
    val_ds   = Subset(full_dataset_val,   val_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # -- Model --
    print("Building RadioDino segmentation model...")
    model = RadioDinoSegmentationModel(model_name=args.model_name, num_classes=args.num_classes).to(device)

    # FIX: Differential LR — backbone starts at 10x lower than decoder
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.decoder.parameters(),  "lr": args.decoder_lr},
    ], weight_decay=1e-4)

    # Cosine LR scheduler with a short linear warmup
    warmup_steps = len(train_loader) * 2
    total_steps  = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # FIX: Class-weighted CE to handle background dominance
    # Weights: background down-weighted, tumour classes up-weighted
    class_weights = torch.tensor([0.25, 1.5, 1.0, 2.0], device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # FIX: Freeze backbone for the first few epochs (linear probing)
    def set_backbone_grad(requires_grad: bool):
        for p in model.backbone.parameters():
            p.requires_grad = requires_grad

    set_backbone_grad(False)   # Start frozen
    print(f"Backbone frozen for first {args.freeze_epochs} epochs.")

    best_dice = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):

        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            set_backbone_grad(True)
            print(f"\n>>> Epoch {epoch}: Backbone UNFROZEN — full fine-tuning begins.")

        # --- Training ---
        model.train()
        train_loss = 0.0
        for pixel_values, masks in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs} [train]", leave=False):
            pixel_values = pixel_values.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(pixel_values)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            # FIX: Combined Dice + Weighted CE loss
            loss = combined_loss(logits, masks, ce_criterion, args.num_classes, alpha=0.5)
            loss.backward()

            # FIX: Gradient clipping for ViT stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        val_metrics = compute_val_dice(model, val_loader, device, args.num_classes)
        mean_dice   = val_metrics["mean_dice"]

        label_names = {0: "BG", 1: "NCR", 2: "ED", 3: "ET"}
        per_class_str = "  ".join(
            f"{label_names[c]}={v:.3f}" for c, v in val_metrics["per_class"].items()
        )
        print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {avg_train_loss:.4f} | "
              f"Val Dice: {mean_dice:.4f} [{per_class_str}]")

        # Save best checkpoint
        if mean_dice > best_dice:
            best_dice = mean_dice
            # Save full model (backbone + decoder)
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            # FIX: Also save backbone weights separately for metric extraction (Rad-FID)
            backbone_path = os.path.join(args.output_dir, "backbone_only.pth")
            torch.save(model.backbone.state_dict(), backbone_path)
            print(f"  ✓ New best Dice={mean_dice:.4f}. Saved to {args.output_dir}")

    # -- Final save --
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    backbone_final = os.path.join(args.output_dir, "backbone_final.pth")
    torch.save(model.backbone.state_dict(), backbone_final)
    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")
    print(f"Backbone saved to: {backbone_final}  (use this for Rad-FID metric extraction)")


if __name__ == "__main__":
    main()
