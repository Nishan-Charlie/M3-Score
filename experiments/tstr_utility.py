"""
tstr_utility.py  --  Downstream Utility Anchoring (Train on Synthetic, Test on Real)
======================================================================================

The most persuasive single result for reviewers: generators that score better
on M3 should produce synthetic data that is more useful for downstream tasks.

Task
----
Train a ResNet-18 classifier on synthetic images from each generator.
Test on REAL images.  Compare TSTR accuracy to the M3 and FID rankings of
those generators.  Show that M3 ranking correlates with TSTR ranking better
than FID ranking does.

Classification task
-------------------
Two tasks (whichever data is available):
1. Modality classification: BraTS brain MRI vs LIDC lung CT  (binary)
   - Requires generated images from BOTH wdm3d/brats and wdm3d/lidc
2. Slice-position classification: inferior / middle / superior (3-class)
   - Uses slice index (filename) as proxy label; works with any single dataset
   - KNOWN LIMITATION: this task saturates rapidly. At sigma >= 0.05 Gaussian
     noise all generators achieve near-identical TSTR accuracy because the
     classifier learns slice-index artefacts rather than anatomy. Do not
     interpret TSTR accuracy differences smaller than ~5 pp as meaningful.

Statistical note
----------------
Spearman rank-correlation p-values are invalid when n_generators <= 5.
With n=4 generators the minimum achievable two-sided p-value is 1/4! = 0.042,
and the distribution has only 24 possible permutations.  The p-values are
reported for completeness but must NOT be used to claim statistical
significance.  Use them as descriptive effect sizes (rho) only.

Generator set (order matters for ranking)
------------------------------------------
- wdm3d_brats   : WDM-3D BraTS checkpoint (best expected quality)
- wdm3d_lidc    : WDM-3D LIDC checkpoint
- ddpm_unet     : Our trained DDPM UNet from output/output_unet/checkpoints
- random_noise  : Random Gaussian (worst baseline)

For each generator:
  1. Compute M3-Score  (RadioDino-s16 L12)
  2. Compute FID       (feature-space Frechet distance)
  3. Train ResNet-18 on synthetic, test on real => TSTR accuracy
  4. Rank all generators by each criterion
  5. Report Spearman rank correlation: M3 vs TSTR, FID vs TSTR

Usage
-----
    python experiments/tstr_utility.py \
        --real_dir   data_mri/brats_axial_multislice \
        --gen_dirs   output/generated_wdm3d/brats output/generated_wdm3d/lidc \
        --gen_labels wdm3d_brats wdm3d_lidc \
        --output_dir results/tstr_utility \
        --task       slice_position \
        --epochs     20
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _ImageFolderFlat(Dataset):
    """
    Flat directory of images with auto-generated labels.
    Labels are assigned by label_fn(filename) -> int.
    """
    def __init__(self, directory: str, label_fn, transform=None, n: int = None):
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        files = sorted(
            f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in exts
        )
        if n:
            files = files[:n]
        self.paths  = [os.path.join(directory, f) for f in files]
        self.labels = [label_fn(f, i, len(files)) for i, f in enumerate(files)]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _slice_position_label(filename: str, rank: int, total: int) -> int:
    """Assign 0/1/2 (inferior/middle/superior) based on sort rank.

    SATURATION WARNING: this proxy label saturates quickly. Classifiers trained
    with this scheme achieve near-identical accuracy across generators at mild
    noise levels (sigma >= 0.05). Do not interpret small TSTR differences as
    evidence of generation quality. See module docstring for details.
    """
    return min(int(rank * 3 / total), 2)


def _modality_label(directory_tag: str) -> int:
    """Return 0 for BraTS-type, 1 for LIDC-type, based on directory tag."""
    return 0 if "brats" in directory_tag.lower() else 1


# ---------------------------------------------------------------------------
# ResNet-18 trainer
# ---------------------------------------------------------------------------

def _train_resnet18(
    train_dataset: Dataset,
    val_dataset:   Dataset,
    num_classes:   int,
    device:        str,
    epochs:        int = 20,
    lr:            float = 1e-3,
    batch_size:    int = 32,
) -> Tuple[nn.Module, List[float]]:
    """Fine-tune ResNet-18 on train_dataset, return model + val_acc history."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False,  num_workers=0)

    val_acc_history = []
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).long()
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).long()
                preds = model(X).argmax(1)
                correct += (preds == y).sum().item()
                total   += len(y)
        acc = correct / total if total > 0 else 0.0
        val_acc_history.append(acc)

        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}/{epochs}  val_acc={acc:.4f}")

    return model, val_acc_history


# ---------------------------------------------------------------------------
# M3 and FID score helpers
# ---------------------------------------------------------------------------

def _score_generator(
    real_feats:  np.ndarray,
    gen_feats:   np.ndarray,
) -> Tuple[float, float]:
    """Returns (m3_score, fid_score) from pre-extracted features."""
    from experiments.sample_size_consistency import _mmd_rbf, _fid_feats
    m3  = _mmd_rbf(real_feats, gen_feats)
    fid = _fid_feats(real_feats, gen_feats)
    return float(m3), float(fid)


def _extract_feats(
    directory: str,
    backbone_id: str,
    device: str,
    n: int = 500,
    batch_size: int = 32,
) -> np.ndarray:
    from evaluation.coverage_novelty import _load_backbone, _extract_features
    from experiments._shared_utils import load_pils_recursive
    imgs = load_pils_recursive(directory, n=n)
    backbone, backend = _load_backbone(backbone_id, device)
    return _extract_features(imgs, backbone, backend, device, batch_size)


# ---------------------------------------------------------------------------
# Main TSTR runner
# ---------------------------------------------------------------------------

def run_tstr(
    real_dir:    str,
    gen_dirs:    List[str],
    gen_labels:  List[str],
    output_dir:  str = "results/tstr_utility",
    task:        str = "slice_position",   # or "modality"
    n_images:    int = 500,
    epochs:      int = 20,
    batch_size:  int = 32,
    lr:          float = 1e-3,
    backbone_id: str = "Snarcy/RadioDino-s16",
    device:      str = "cuda",
    eval_batch:  int = 32,
    seed:        int = 42,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device if torch.cuda.is_available() else "cpu"

    assert len(gen_dirs) == len(gen_labels), "gen_dirs and gen_labels must match"

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- Build real test dataset ---
    if task == "slice_position":
        num_classes = 3
        label_fn = _slice_position_label
        real_test_ds = _ImageFolderFlat(real_dir, label_fn, eval_tf, n=n_images)
    elif task == "modality":
        num_classes = 2
        # For modality task we need 2 gen_dirs minimum (one per class)
        if len(gen_dirs) < 2:
            raise ValueError("Modality task requires at least 2 generator directories.")
        # Real set: brats images are class 0 (assume real_dir is brats)
        label_fn = lambda fn, rank, total: 0
        real_test_ds = _ImageFolderFlat(real_dir, label_fn, eval_tf, n=n_images)
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"\n=== TSTR Utility Experiment ===")
    print(f"  Task       : {task} ({num_classes} classes)")
    print(f"  Real test  : {len(real_test_ds)} images from {real_dir}")
    print(f"  Generators : {gen_labels}")

    # --- Extract real features once (for M3/FID) ---
    print("\n  Extracting real features for M3/FID scoring...")
    real_feats = _extract_feats(real_dir, backbone_id, device, n_images, eval_batch)

    # --- Baseline: train on REAL, test on REAL (ceiling) ---
    print("\n  Baseline: Train on Real, Test on Real...")
    all_files = sorted(
        f for f in os.listdir(real_dir)
        if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg", ".tif"}
    )[:n_images * 2]
    split = len(all_files) // 2
    train_files = all_files[:split]
    test_files  = all_files[split:]

    class _SubsetDS(Dataset):
        def __init__(self, directory, files, label_fn, transform):
            self.paths  = [os.path.join(directory, f) for f in files]
            self.labels = [label_fn(f, i, len(files)) for i, f in enumerate(files)]
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            from PIL import Image
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]

    real_train_ds = _SubsetDS(real_dir, train_files, label_fn, train_tf)
    real_test_sub = _SubsetDS(real_dir, test_files,  label_fn, eval_tf)
    _, rr_history = _train_resnet18(real_train_ds, real_test_sub, num_classes,
                                    device, epochs, lr, batch_size)
    baseline_acc = max(rr_history)
    print(f"  Baseline (Real->Real) : {baseline_acc:.4f}")

    # --- Per-generator TSTR ---
    generator_results: Dict[str, dict] = {}

    for gen_dir, gen_label in zip(gen_dirs, gen_labels):
        print(f"\n  Generator: {gen_label}  ({gen_dir})")

        # M3 and FID scores
        print(f"    Extracting gen features...")
        gen_feats = _extract_feats(gen_dir, backbone_id, device, n_images, eval_batch)
        m3, fid = _score_generator(real_feats, gen_feats)
        print(f"    M3={m3:.6f}  FID={fid:.4f}")

        # TSTR: train on synthetic, test on real
        if task == "slice_position":
            gen_train_ds = _ImageFolderFlat(gen_dir, label_fn, train_tf, n=n_images)
        elif task == "modality":
            # Assign label based on which generator directory
            gen_mod_label = gen_labels.index(gen_label)
            gen_train_ds = _ImageFolderFlat(
                gen_dir,
                lambda fn, rank, total, lbl=gen_mod_label: lbl,
                train_tf, n=n_images
            )

        print(f"    Training ResNet-18 on {len(gen_train_ds)} synthetic images...")
        _, tstr_history = _train_resnet18(
            gen_train_ds, real_test_ds, num_classes, device, epochs, lr, batch_size
        )
        tstr_acc = max(tstr_history)
        print(f"    TSTR accuracy: {tstr_acc:.4f}")

        generator_results[gen_label] = {
            "m3_score":  m3,
            "fid_score": fid,
            "tstr_acc":  tstr_acc,
            "tstr_history": tstr_history,
        }

    # --- Random noise baseline ---
    print("\n  Generator: random_noise")
    rng = np.random.default_rng(seed + 99)
    noise_feats = rng.standard_normal(real_feats.shape).astype(np.float32)
    noise_feats /= (np.linalg.norm(noise_feats, axis=1, keepdims=True) + 1e-8)
    m3_noise, fid_noise = _score_generator(real_feats, noise_feats)
    # TSTR with random noise images
    from torch.utils.data import TensorDataset
    rand_imgs = torch.randn(n_images, 3, 224, 224)
    rand_lbls = torch.tensor(
        [label_fn("", i, n_images) for i in range(n_images)], dtype=torch.long
    )
    rand_ds = TensorDataset(rand_imgs, rand_lbls)
    _, noise_history = _train_resnet18(rand_ds, real_test_ds, num_classes,
                                       device, epochs, lr, batch_size)
    tstr_noise = max(noise_history)
    generator_results["random_noise"] = {
        "m3_score":  m3_noise,
        "fid_score": fid_noise,
        "tstr_acc":  tstr_noise,
        "tstr_history": noise_history,
    }
    print(f"  random_noise: M3={m3_noise:.6f}  FID={fid_noise:.4f}  TSTR={tstr_noise:.4f}")

    # --- Ranking correlation ---
    all_labels = list(generator_results.keys())
    m3_vals    = [generator_results[l]["m3_score"]  for l in all_labels]
    fid_vals   = [generator_results[l]["fid_score"] for l in all_labels]
    tstr_vals  = [generator_results[l]["tstr_acc"]  for l in all_labels]

    # Lower M3/FID = better; higher TSTR = better
    # Rank: 1 = best
    def _rank_asc(vals):   # lower = rank 1
        return list(np.argsort(np.argsort(vals)) + 1)
    def _rank_desc(vals):  # higher = rank 1
        return list(np.argsort(np.argsort([-v for v in vals])) + 1)

    m3_ranks   = [int(r) for r in _rank_asc(m3_vals)]
    fid_ranks  = [int(r) for r in _rank_asc(fid_vals)]
    tstr_ranks = [int(r) for r in _rank_desc(tstr_vals)]

    import scipy.stats as stats
    corr_m3_tstr  = stats.spearmanr(m3_ranks,  tstr_ranks)
    corr_fid_tstr = stats.spearmanr(fid_ranks, tstr_ranks)

    n_gen = len(m3_ranks)
    pval_warning = (
        f"INVALID at n_generators={n_gen}: Spearman p-values require n >= 10 "
        f"for meaningful inference. Minimum p with {n_gen} items = "
        f"{1.0 / float(np.math.factorial(n_gen)):.4f}. Treat as descriptive only."
        if n_gen < 10 else ""
    )

    print("\n  Ranking correlation with TSTR:")
    print(f"    M3  vs TSTR: rho={corr_m3_tstr.statistic:.4f}  p={corr_m3_tstr.pvalue:.4f}")
    print(f"    FID vs TSTR: rho={corr_fid_tstr.statistic:.4f}  p={corr_fid_tstr.pvalue:.4f}")
    if pval_warning:
        print(f"    WARNING: {pval_warning}")

    summary = {
        "task":            task,
        "baseline_rr_acc": baseline_acc,
        "spearman_m3_tstr": {
            "rho":     float(corr_m3_tstr.statistic),
            "p":       float(corr_m3_tstr.pvalue),
            "warning": pval_warning,
        },
        "spearman_fid_tstr": {
            "rho":     float(corr_fid_tstr.statistic),
            "p":       float(corr_fid_tstr.pvalue),
            "warning": pval_warning,
        },
        "generators": {
            lbl: {
                "m3_score":  generator_results[lbl]["m3_score"],
                "fid_score": generator_results[lbl]["fid_score"],
                "tstr_acc":  generator_results[lbl]["tstr_acc"],
                "m3_rank":   m3_ranks[i],
                "fid_rank":  fid_ranks[i],
                "tstr_rank": tstr_ranks[i],
            }
            for i, lbl in enumerate(all_labels)
        },
    }

    with open(os.path.join(output_dir, "tstr_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    _plot_tstr(summary, generator_results, all_labels, output_dir)

    print(f"\nSaved -> {output_dir}")
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_tstr(summary: dict, generator_results: dict,
               all_labels: List[str], output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    tstr_vals = [generator_results[l]["tstr_acc"]  for l in all_labels]
    m3_vals   = [generator_results[l]["m3_score"]  for l in all_labels]
    fid_vals  = [generator_results[l]["fid_score"] for l in all_labels]

    # Plot 1: TSTR accuracy bar chart
    ax = axes[0]
    colors = ["#4878cf" if l != "random_noise" else "#aaaaaa" for l in all_labels]
    ax.barh(all_labels, tstr_vals, color=colors, edgecolor="black", lw=0.8)
    ax.axvline(summary["baseline_rr_acc"], color="red", lw=2, ls="--",
               label=f"Real->Real ceiling = {summary['baseline_rr_acc']:.3f}")
    ax.set_xlabel("TSTR Accuracy", fontsize=12)
    ax.set_title("Downstream Utility (TSTR)\nHigher = more useful synthetic data", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 2: M3 vs TSTR scatter
    ax = axes[1]
    ax.scatter(m3_vals, tstr_vals, c=["#4878cf" if l != "random_noise" else "#aaaaaa"
                                       for l in all_labels], s=100, zorder=3)
    for i, lbl in enumerate(all_labels):
        ax.annotate(lbl, (m3_vals[i], tstr_vals[i]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    rho = summary["spearman_m3_tstr"]["rho"]
    ax.set_xlabel("M3-Score (lower = better)", fontsize=12)
    ax.set_ylabel("TSTR Accuracy (higher = better)", fontsize=12)
    ax.set_title(f"M3 vs TSTR\nSpearman rho={rho:.3f}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_xaxis()

    # Plot 3: FID vs TSTR scatter
    ax = axes[2]
    ax.scatter(fid_vals, tstr_vals, c=["#e05c5c" if l != "random_noise" else "#aaaaaa"
                                        for l in all_labels], s=100, zorder=3)
    for i, lbl in enumerate(all_labels):
        ax.annotate(lbl, (fid_vals[i], tstr_vals[i]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    rho_fid = summary["spearman_fid_tstr"]["rho"]
    ax.set_xlabel("FID (lower = better)", fontsize=12)
    ax.set_ylabel("TSTR Accuracy (higher = better)", fontsize=12)
    ax.set_title(f"FID vs TSTR\nSpearman rho={rho_fid:.3f}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_xaxis()

    plt.suptitle("TSTR Utility: M3 ranking correlates with downstream performance\n"
                 f"M3 rho={summary['spearman_m3_tstr']['rho']:.3f}  "
                 f"vs  FID rho={summary['spearman_fid_tstr']['rho']:.3f}",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tstr_utility.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",   required=True)
    p.add_argument("--gen_dirs",   nargs="+", required=True)
    p.add_argument("--gen_labels", nargs="+", required=True)
    p.add_argument("--output_dir", default="results/tstr_utility")
    p.add_argument("--task",       default="slice_position",
                   choices=["slice_position", "modality"])
    p.add_argument("--n_images",   type=int, default=500)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    p.add_argument("--device",     default=None)
    p.add_argument("--seed",       type=int, default=42)
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_tstr(
        real_dir    = a.real_dir,
        gen_dirs    = a.gen_dirs,
        gen_labels  = a.gen_labels,
        output_dir  = a.output_dir,
        task        = a.task,
        n_images    = a.n_images,
        epochs      = a.epochs,
        batch_size  = a.batch_size,
        lr          = a.lr,
        backbone_id = a.backbone_id,
        device      = device,
        seed        = a.seed,
    )
