"""
Downstream Classification Evaluator
=====================================
Trains a lightweight CNN on (a) only generated images or (b) real + generated
images, then evaluates on real held-out images.

Expected directory structure (subfolder = class label):
    real_dir/
        tumour/  *.png
        healthy/ *.png
    gen_dir/
        tumour/  *.png
        healthy/ *.png

Single-class fallback: if only one class dir exists, binary separation is
done automatically via the source (real vs generated).

Usage
-----
    from metrics.downstream_classifier import evaluate_downstream

    results = evaluate_downstream(
        real_dir   = "path/to/real",
        gen_dir    = "path/to/generated",
        output_dir = "path/to/results",
        mode       = "augment",   # or "only_gen"
        device     = "cuda:0",
        epochs     = 10,
        use_tqdm   = True,
    )
"""

from __future__ import annotations

import os
import glob
import random
from typing import Optional

import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

_TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class LabelledImageDataset(Dataset):
    """
    Multi-class dataset from a directory with class-named subdirectories.
    Falls back to binary (real=0, gen=1) if no subdirs found.
    """

    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _collect_class_paths(root: str) -> tuple[list[str], list[int], list[str]]:
    """
    Returns (paths, label_ids, class_names).
    Searches for class subdirectories; falls back to flat directory (label=0).
    """
    subdirs = [d for d in sorted(os.listdir(root))
               if os.path.isdir(os.path.join(root, d))]

    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")

    if subdirs:
        paths, labels, classes = [], [], subdirs
        for cls_id, cls_name in enumerate(classes):
            cls_dir = os.path.join(root, cls_name)
            for ext in exts:
                for p in glob.glob(os.path.join(cls_dir, ext)):
                    paths.append(p)
                    labels.append(cls_id)
        return paths, labels, classes
    else:
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(root, ext)))
        labels = [0] * len(paths)
        return sorted(paths), labels, ["images"]


# ---------------------------------------------------------------------------
# Lightweight CNN (fine-tuned ResNet-18)
# ---------------------------------------------------------------------------

def _build_classifier(num_classes: int, device: str) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final FC for the downstream task
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model.to(device)


def _train_one_epoch(model, loader, criterion, optimizer, device, use_tqdm):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    itr = tqdm(loader, desc="  Train", leave=False, disable=not use_tqdm)
    for imgs, lbls in itr:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += len(imgs)
    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(lbls.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


@torch.no_grad()
def _get_proba(model, loader, device) -> np.ndarray:
    """Return softmax probabilities for AUC calculation."""
    model.eval()
    all_proba = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        proba = torch.softmax(model(imgs), dim=1).cpu().numpy()
        all_proba.append(proba)
    return np.concatenate(all_proba)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_downstream(
    real_dir: str,
    gen_dir: str,
    output_dir: str = "./downstream_output",
    mode: str = "augment",        # "augment" | "only_gen"
    num_real: Optional[int] = None,
    num_gen: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cpu",
    use_tqdm: bool = True,
    seed: int = 42,
    val_split: float = 0.2,
) -> dict:
    """
    Train and evaluate a downstream classifier.

    Parameters
    ----------
    mode : "augment"  → train on real + generated, evaluate on real hold-out
           "only_gen" → train ONLY on generated, evaluate on all real images

    Returns
    -------
    dict : accuracy, auc (if multi-class: macro AUC), confusion_matrix
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Collect paths and labels
    real_paths, real_labels, real_classes = _collect_class_paths(real_dir)
    gen_paths, gen_labels, gen_classes = _collect_class_paths(gen_dir)

    # Use real class names as canonical (generated might be flat)
    classes = real_classes if len(real_classes) >= len(gen_classes) else gen_classes
    num_classes = len(classes)

    if num_real:
        real_paths, real_labels = real_paths[:num_real], real_labels[:num_real]
    if num_gen:
        gen_paths, gen_labels = gen_paths[:num_gen], gen_labels[:num_gen]

    print(f"[Downstream] Real images : {len(real_paths)} | Gen images: {len(gen_paths)}")
    print(f"[Downstream] Classes     : {classes}")
    print(f"[Downstream] Mode        : {mode}")

    # ---- Build train / val / test splits -----------------------------------
    # Hold out val_split of real images as the test set
    n_real = len(real_paths)
    indices = list(range(n_real))
    random.shuffle(indices)
    n_val = max(1, int(n_real * val_split))
    val_idx, train_real_idx = indices[:n_val], indices[n_val:]

    real_val_paths  = [real_paths[i] for i in val_idx]
    real_val_labels = [real_labels[i] for i in val_idx]

    if mode == "augment":
        train_paths  = [real_paths[i] for i in train_real_idx] + gen_paths
        train_labels = [real_labels[i] for i in train_real_idx] + gen_labels
    else:  # only_gen
        train_paths  = gen_paths
        train_labels = gen_labels

    # ---- Datasets & loaders ------------------------------------------------
    train_ds = LabelledImageDataset(train_paths, train_labels, _TRANSFORM_TRAIN)
    val_ds   = LabelledImageDataset(real_val_paths, real_val_labels, _TRANSFORM_EVAL)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ---- Model, loss, optimiser --------------------------------------------
    model     = _build_classifier(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- Training loop -----------------------------------------------------
    history = []
    print(f"[Downstream] Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device, use_tqdm)
        val_preds, val_lbls = _evaluate(model, val_loader, device)
        val_acc = float((val_preds == val_lbls).mean())
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_acc": val_acc})
        if use_tqdm:
            print(f"  Epoch {epoch:03d}/{epochs} | loss={tr_loss:.4f} | train_acc={tr_acc:.3f} | val_acc={val_acc:.3f}")

    # ---- Final evaluation --------------------------------------------------
    val_preds, val_lbls = _evaluate(model, val_loader, device)
    final_acc = float((val_preds == val_lbls).mean())

    # AUC (macro OvR for multi-class)
    val_proba = _get_proba(model, val_loader, device)
    try:
        if num_classes == 2:
            auc = roc_auc_score(val_lbls, val_proba[:, 1])
        else:
            auc = roc_auc_score(val_lbls, val_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(val_lbls, val_preds, labels=list(range(num_classes)))

    # ---- Confusion matrix plot ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes[:num_classes])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Downstream Classifier — {mode} mode\nAcc={final_acc:.3f} | AUC={auc:.3f}")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "downstream_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # ---- Learning curve ----------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=120)
    epochs_arr = [h["epoch"] for h in history]
    ax2.plot(epochs_arr, [h["train_loss"] for h in history], label="Train Loss", color="#4fc3f7")
    ax2.plot(epochs_arr, [h["val_acc"]    for h in history], label="Val Acc",    color="#ef9a9a")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Value"); ax2.legend()
    ax2.set_title("Downstream Classifier — Training Curve")
    plt.tight_layout()
    curve_path = os.path.join(output_dir, "downstream_training_curve.png")
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()

    results = {
        "accuracy": round(final_acc, 6),
        "auc": round(auc, 6) if not np.isnan(auc) else None,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "mode": mode,
        "num_train": len(train_paths),
        "num_val": len(real_val_paths),
    }

    results_path = os.path.join(output_dir, "downstream_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[Downstream] Accuracy={final_acc:.4f} | AUC={auc:.4f}")
    print(f"[Downstream] Saved → {results_path}, {cm_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Downstream classification evaluation for generated MRI images.")
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir",  required=True)
    parser.add_argument("--output_dir", default="./downstream_output")
    parser.add_argument("--mode", choices=["augment", "only_gen"], default="augment")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_downstream(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        use_tqdm=not args.no_tqdm,
        seed=args.seed,
    )
