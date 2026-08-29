"""
conditional_mmd.py  --  Per-stratum conditional MMD for M3-Score
=================================================================

Computes M3-Score independently within each clinical stratum and reports
the WORST-CASE stratum.  This makes mode-dropping of rare subtypes visible
where the global average would mask it.

Stratification schemes
----------------------
1. slice_position   : inferior / middle / superior thirds of the volume
   (proxy for anatomy; uses filename index or image rank)

2. intensity_quartile : dark / medium-dark / medium-bright / bright
   (proxy for tissue content / acquisition contrast)

3. kmeans_cluster   : K-Means in RadioDino feature space
   (unsupervised pseudo-strata; most general)

4. manual_labels    : user-supplied CSV with filename -> stratum

Each scheme returns a dict:
    {stratum_name: {"m3": float, "n_real": int, "n_gen": int, "fid": float}}

Summary statistics returned:
    mean_m3      -- mean M3 over strata
    worst_m3     -- max M3 over strata  (worst-case stratum)
    worst_stratum -- name of worst stratum
    std_m3       -- std across strata (heterogeneity)
    stratum_results -- per-stratum dict

Public API
----------
    result = run_conditional_mmd(
        real_dir, gen_dir,
        stratify_by="intensity_quartile",   # or "slice_position" / "kmeans" / "manual"
        n_strata=4,
        n_images=500,
        backbone_id="Snarcy/RadioDino-s16",
        device="cuda",
        output_dir="results/conditional_mmd",
    )

    ConditionalM3Metric   -- class wrapper
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Stratum builders
# ---------------------------------------------------------------------------

def _strata_slice_position(filenames: List[str], n_strata: int = 3) -> np.ndarray:
    """
    Assign strata based on sort order of filenames (proxy for slice index).
    Returns integer array of length N.
    """
    n = len(filenames)
    return (np.arange(n) * n_strata // n).astype(int)


def _strata_intensity(images, n_strata: int = 4) -> np.ndarray:
    """
    Assign strata based on mean pixel intensity of each image.
    """
    means = []
    for img in images:
        import numpy as _np
        arr = _np.array(img.convert("L"), dtype=np.float32) / 255.0
        means.append(arr.mean())
    means = np.array(means)
    # Quantile-based bins
    quantiles = np.linspace(0, 100, n_strata + 1)
    thresholds = [np.percentile(means, q) for q in quantiles]
    labels = np.digitize(means, thresholds[1:-1]).astype(int)
    return labels


def _strata_kmeans(feats: np.ndarray, n_strata: int = 4, seed: int = 42) -> np.ndarray:
    """
    K-Means clustering in feature space.  Returns cluster assignments.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_strata, random_state=seed, n_init=10)
    return km.fit_predict(feats).astype(int)


def _strata_manual(filenames: List[str], label_csv: str) -> np.ndarray:
    """
    Load stratum labels from CSV: columns = [filename, stratum].
    """
    import csv
    label_map = {}
    with open(label_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[os.path.basename(row["filename"])] = int(row["stratum"])
    return np.array([label_map.get(os.path.basename(fn), 0) for fn in filenames])


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def _extract_features_for_conditional(
    images: List,
    backbone_id: str,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract L2-normalised CLS features (N, D)."""
    from evaluation.coverage_novelty import _load_backbone, _extract_features
    backbone, backend = _load_backbone(backbone_id, device)
    return _extract_features(images, backbone, backend, device, batch_size)


# ---------------------------------------------------------------------------
# Per-stratum M3 computation
# ---------------------------------------------------------------------------

def _mmd_sq_rbf(X: np.ndarray, Y: np.ndarray,
                n_bandwidths: int = 3) -> float:
    """Fast unbiased RBF MMD^2 (reused from statistical_rigor logic)."""
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()

    def sq(A, B):
        return (A.unsqueeze(1) - B.unsqueeze(0)).pow(2).sum(-1)

    Dxx = sq(Xt, Xt)
    Dyy = sq(Yt, Yt)
    Dxy = sq(Xt, Yt)

    med_val = torch.cat([Dxy.flatten(),
                         Dxx[Dxx > 0].flatten(),
                         Dyy[Dyy > 0].flatten()]).median().item()
    if med_val < 1e-10:
        med_val = 1.0

    bws = [med_val * (2 ** k) for k in range(-(n_bandwidths // 2),
                                               n_bandwidths - n_bandwidths // 2)]
    n, m = len(X), len(Y)
    total = 0.0
    for bw in bws:
        g = 1.0 / (2.0 * bw)
        Kxx = torch.exp(-g * Dxx)
        Kyy = torch.exp(-g * Dyy)
        Kxy = torch.exp(-g * Dxy)
        Kxx.fill_diagonal_(0)
        Kyy.fill_diagonal_(0)
        total += (Kxx.sum() / (n * (n - 1)) +
                  Kyy.sum() / (m * (m - 1)) -
                  2 * Kxy.mean()).item()
    return total / len(bws)


def _compute_stratum_results(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
    real_strata: np.ndarray,
    gen_strata:  np.ndarray,
    stratum_names: Optional[Dict[int, str]] = None,
) -> Dict[str, dict]:
    """
    Compute MMD for each stratum independently.

    Parameters
    ----------
    real_feats   : (N_real, D)
    gen_feats    : (N_gen,  D)
    real_strata  : (N_real,) integer stratum assignments
    gen_strata   : (N_gen,)  integer stratum assignments
    stratum_names: optional {int: str} name map

    Returns
    -------
    {stratum_name: {"m3": float, "n_real": int, "n_gen": int}}
    """
    all_strata = sorted(set(real_strata.tolist()) | set(gen_strata.tolist()))
    results = {}

    for s in all_strata:
        name = (stratum_names or {}).get(s, f"stratum_{s}")
        r_idx = np.where(real_strata == s)[0]
        g_idx = np.where(gen_strata  == s)[0]

        if len(r_idx) < 5 or len(g_idx) < 5:
            results[name] = {
                "m3": float("nan"), "n_real": len(r_idx), "n_gen": len(g_idx),
                "note": "too few samples (<5)"
            }
            continue

        mmd = _mmd_sq_rbf(real_feats[r_idx], gen_feats[g_idx])
        results[name] = {
            "m3":    float(mmd),
            "n_real": int(len(r_idx)),
            "n_gen":  int(len(g_idx)),
        }

    return results


def _summary_stats(stratum_results: dict) -> dict:
    """Compute mean, worst-case, std across valid strata."""
    valid = {k: v for k, v in stratum_results.items()
             if not np.isnan(v.get("m3", float("nan")))}
    if not valid:
        return {"mean_m3": float("nan"), "worst_m3": float("nan"),
                "worst_stratum": None, "std_m3": float("nan")}

    values = np.array([v["m3"] for v in valid.values()])
    names  = list(valid.keys())
    worst_idx = int(np.argmax(values))

    return {
        "mean_m3":      float(np.mean(values)),
        "worst_m3":     float(np.max(values)),
        "best_m3":      float(np.min(values)),
        "std_m3":       float(np.std(values)),
        "worst_stratum": names[worst_idx],
        "best_stratum":  names[int(np.argmin(values))],
        "n_strata_valid": len(valid),
    }


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def run_conditional_mmd(
    real_dir:    str,
    gen_dir:     str,
    stratify_by: str = "intensity_quartile",
    n_strata:    int = 4,
    n_images:    int = 500,
    backbone_id: str = "Snarcy/RadioDino-s16",
    device:      str = "cuda",
    output_dir:  str = "results/conditional_mmd",
    batch_size:  int = 32,
    label_csv:   Optional[str] = None,
    seed:        int = 42,
    real_imgs:   Optional[List] = None,
    gen_imgs:    Optional[List] = None,
) -> dict:
    """
    Run conditional MMD with the chosen stratification scheme.

    Parameters
    ----------
    stratify_by : "intensity_quartile" | "slice_position" | "kmeans" | "manual"
    """
    from experiments._shared_utils import load_pils_recursive

    os.makedirs(output_dir, exist_ok=True)
    device = device if torch.cuda.is_available() else "cpu"

    # --- Load images ---
    if real_imgs is None:
        from PIL import Image as _PILImage
        real_files = sorted(
            f for f in os.listdir(real_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        )[:n_images]
        real_imgs = [_PILImage.open(
            os.path.join(real_dir, f)).convert("RGB") for f in real_files]
    else:
        real_files = [f"real_{i:05d}.png" for i in range(len(real_imgs))]

    if gen_imgs is None:
        from PIL import Image as _PILImage
        gen_files = sorted(
            f for f in os.listdir(gen_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        )[:n_images]
        gen_imgs = [_PILImage.open(
            os.path.join(gen_dir, f)).convert("RGB") for f in gen_files]
    else:
        gen_files = [f"gen_{i:05d}.png" for i in range(len(gen_imgs))]

    print(f"Real: {len(real_imgs)}  Gen: {len(gen_imgs)}")
    print(f"Stratify by: {stratify_by}  (n_strata={n_strata})")

    # --- Extract features (needed for kmeans; also reused for MMD) ---
    print("Extracting features...")
    real_feats = _extract_features_for_conditional(
        real_imgs, backbone_id, device, batch_size)
    gen_feats  = _extract_features_for_conditional(
        gen_imgs,  backbone_id, device, batch_size)

    # --- Assign strata ---
    stratum_names: Optional[Dict[int, str]] = None

    if stratify_by == "slice_position":
        snames = {0: "inferior", 1: "middle", 2: "superior"}
        n_strata = 3
        real_strata = _strata_slice_position(real_files, n_strata)
        gen_strata  = _strata_slice_position(gen_files,  n_strata)
        stratum_names = snames

    elif stratify_by == "intensity_quartile":
        real_strata = _strata_intensity(real_imgs, n_strata)
        gen_strata  = _strata_intensity(gen_imgs,  n_strata)
        snames = {i: f"intensity_Q{i+1}" for i in range(n_strata)}
        stratum_names = snames

    elif stratify_by == "kmeans":
        combined_feats = np.concatenate([real_feats, gen_feats], axis=0)
        all_labels = _strata_kmeans(combined_feats, n_strata, seed)
        real_strata = all_labels[:len(real_feats)]
        gen_strata  = all_labels[len(real_feats):]
        stratum_names = {i: f"cluster_{i}" for i in range(n_strata)}

    elif stratify_by == "manual":
        if label_csv is None:
            raise ValueError("label_csv required for stratify_by='manual'")
        real_strata = _strata_manual(real_files, label_csv)
        gen_strata  = _strata_manual(gen_files,  label_csv)

    else:
        raise ValueError(f"Unknown stratify_by: {stratify_by}")

    # --- Compute per-stratum M3 ---
    print("Computing per-stratum MMD...")
    stratum_results = _compute_stratum_results(
        real_feats, gen_feats,
        real_strata, gen_strata,
        stratum_names,
    )

    summary = _summary_stats(stratum_results)

    result = {
        "stratify_by":      stratify_by,
        "n_strata":         n_strata,
        "backbone_id":      backbone_id,
        **summary,
        "stratum_results":  stratum_results,
    }

    # --- Save report ---
    with open(os.path.join(output_dir, "conditional_mmd_report.json"), "w") as f:
        json.dump(result, f, indent=2)

    # --- Bar chart ---
    _plot_stratum_bars(stratum_results, summary, output_dir, stratify_by)

    print(f"\nMean M3     : {summary['mean_m3']:.6f}")
    print(f"Worst M3    : {summary['worst_m3']:.6f}  ({summary['worst_stratum']})")
    print(f"Std M3      : {summary['std_m3']:.6f}")
    print(f"Saved -> {output_dir}")

    return result


def _plot_stratum_bars(stratum_results: dict, summary: dict,
                       output_dir: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in stratum_results.items()
             if not np.isnan(v.get("m3", float("nan")))}
    if not valid:
        return

    names  = list(valid.keys())
    values = [valid[n]["m3"] for n in names]
    colors = ["#e05c5c" if n == summary["worst_stratum"] else "#4878cf"
              for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 5))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(summary["mean_m3"], color="gray", lw=1.5, ls="--",
               label=f"Mean M3 = {summary['mean_m3']:.4f}")
    ax.set_ylabel("M3-Score (MMD)")
    ax.set_title(f"Conditional M3 by {title}\n"
                 f"Worst: {summary['worst_stratum']} = {summary['worst_m3']:.4f}")
    ax.legend(fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conditional_mmd_bars.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Stateful class
# ---------------------------------------------------------------------------

class ConditionalM3Metric:
    """
    Stateful wrapper: cache features and evaluate multiple generators.

    Usage:
        cm3 = ConditionalM3Metric(stratify_by="intensity_quartile")
        cm3.set_real(real_imgs)
        result = cm3.evaluate(gen_imgs, output_dir="results/conditional")
    """

    def __init__(
        self,
        backbone_id: str = "Snarcy/RadioDino-s16",
        device:      str = "cuda",
        stratify_by: str = "intensity_quartile",
        n_strata:    int = 4,
        batch_size:  int = 32,
        seed:        int = 42,
    ):
        self.backbone_id = backbone_id
        self.device      = device if torch.cuda.is_available() else "cpu"
        self.stratify_by = stratify_by
        self.n_strata    = n_strata
        self.batch_size  = batch_size
        self.seed        = seed
        self._real_feats  = None
        self._real_imgs   = None
        self._real_files  = None

    def set_real(self, real_imgs, real_files=None):
        self._real_imgs  = real_imgs
        self._real_files = real_files or [f"real_{i}.png" for i in range(len(real_imgs))]
        self._real_feats = _extract_features_for_conditional(
            real_imgs, self.backbone_id, self.device, self.batch_size)
        return self._real_feats

    def evaluate(self, gen_imgs, output_dir: str = "results/conditional_mmd",
                 gen_files=None) -> dict:
        if self._real_feats is None:
            raise RuntimeError("Call set_real() first.")

        os.makedirs(output_dir, exist_ok=True)
        gen_files = gen_files or [f"gen_{i}.png" for i in range(len(gen_imgs))]
        gen_feats = _extract_features_for_conditional(
            gen_imgs, self.backbone_id, self.device, self.batch_size)

        # Build strata
        if self.stratify_by == "intensity_quartile":
            real_strata = _strata_intensity(self._real_imgs, self.n_strata)
            gen_strata  = _strata_intensity(gen_imgs,        self.n_strata)
            snames = {i: f"intensity_Q{i+1}" for i in range(self.n_strata)}
        elif self.stratify_by == "slice_position":
            real_strata = _strata_slice_position(self._real_files, 3)
            gen_strata  = _strata_slice_position(gen_files, 3)
            snames = {0: "inferior", 1: "middle", 2: "superior"}
        elif self.stratify_by == "kmeans":
            combined = np.concatenate([self._real_feats, gen_feats])
            all_lbl  = _strata_kmeans(combined, self.n_strata, self.seed)
            real_strata = all_lbl[:len(self._real_feats)]
            gen_strata  = all_lbl[len(self._real_feats):]
            snames = {i: f"cluster_{i}" for i in range(self.n_strata)}
        else:
            raise ValueError(f"Unknown stratify_by: {self.stratify_by}")

        stratum_results = _compute_stratum_results(
            self._real_feats, gen_feats,
            real_strata, gen_strata, snames)
        summary = _summary_stats(stratum_results)

        result = {
            "stratify_by": self.stratify_by,
            **summary,
            "stratum_results": stratum_results,
        }
        with open(os.path.join(output_dir, "conditional_mmd_report.json"), "w") as f:
            json.dump(result, f, indent=2)
        _plot_stratum_bars(stratum_results, summary, output_dir, self.stratify_by)
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",    required=True)
    p.add_argument("--gen_dir",     required=True)
    p.add_argument("--output_dir",  default="results/conditional_mmd")
    p.add_argument("--n_images",    type=int,   default=500)
    p.add_argument("--stratify_by", default="intensity_quartile",
                   choices=["intensity_quartile", "slice_position", "kmeans", "manual"])
    p.add_argument("--n_strata",    type=int,   default=4)
    p.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    p.add_argument("--device",      default=None)
    p.add_argument("--label_csv",   default=None)
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_conditional_mmd(
        real_dir    = a.real_dir,
        gen_dir     = a.gen_dir,
        stratify_by = a.stratify_by,
        n_strata    = a.n_strata,
        n_images    = a.n_images,
        backbone_id = a.backbone_id,
        device      = device,
        output_dir  = a.output_dir,
        label_csv   = a.label_csv,
    )
