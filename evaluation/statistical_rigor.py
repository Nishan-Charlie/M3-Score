"""
statistical_rigor.py  --  Statistical hypothesis-testing layer for M3-Score
=============================================================================

MMD is structurally a two-sample hypothesis test, not just a distance.
This module adds three things FID cannot provide:

1. Permutation p-value
   Randomly permute the real/gen labels N times; compute MMD for each
   permutation to build a null distribution.  Report the fraction of
   null-MMD values that exceed the observed MMD (one-sided p-value).
   Interpretation: p < 0.05 means "the two sets are distinguishable at
   5% significance level."

2. Bootstrap Confidence Intervals
   Resample real and gen with replacement B times; compute M3 each time.
   Report 95% (or configurable) CI over the bootstrap distribution.
   Per-layer bootstrap CIs expose WHICH scale drives the difference.

3. Effect Size
   Cohen's d analog for MMD:
       effect_size = (MMD_obs - mean(null)) / std(null)
   Interpretation: how many null-std deviations above the null mean is the
   observed score?  Equivalent to a z-score under the null.

Public API
----------
    result = statistical_m3(
        real_imgs, gen_imgs,
        backbone_id="Snarcy/RadioDino-s16",
        device="cuda",
        n_perm=500,
        n_boot=200,
        ci_level=0.95,
    )
    # result keys:
    #   m3_score     -- float
    #   p_value      -- float  (permutation test)
    #   effect_size  -- float  (z-score above null)
    #   ci_low       -- float  (ci_level/2 bootstrap percentile)
    #   ci_high      -- float  (1 - ci_level/2 bootstrap percentile)
    #   null_mmd     -- np.ndarray(n_perm,)
    #   boot_mmd     -- np.ndarray(n_boot,)
    #   per_layer_ci -- dict  {layer_idx: (ci_low, ci_high)}

    StatisticalM3   -- class wrapper (backbone loaded once)
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Thin feature extractor  (re-uses m3_score_v2 logic but standalone)
# ---------------------------------------------------------------------------

def _extract_all_layers(
    images: List,
    backbone_id: str,
    device: str,
    batch_size: int = 16,
) -> tuple:
    """
    Extract L2-normalised CLS features using the same backbone as coverage_novelty.
    Returns (extractor_fn, None) where extractor_fn(imgs) -> np.ndarray (N, D).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from evaluation.coverage_novelty import _load_backbone, _extract_features

    backbone, backend = _load_backbone(backbone_id, device)

    def _batch_feats(imgs):
        return _extract_features(imgs, backbone, backend, device, batch_size)

    return _batch_feats, None


# ---------------------------------------------------------------------------
# MMD kernel  (RBF, median bandwidth)
# ---------------------------------------------------------------------------

def _rbf_mmd_sq(X: np.ndarray, Y: np.ndarray,
                n_bandwidths: int = 5) -> float:
    """
    Unbiased RBF MMD^2 estimator.
    X: (n, d), Y: (m, d) -- L2-normalised.
    Bandwidth set: multiples of median heuristic.
    """
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    # pairwise squared L2 distances (equivalent to 2(1 - cosine) for unit vecs)
    def _sq_dists(A, B):
        return (A.unsqueeze(1) - B.unsqueeze(0)).pow(2).sum(-1)  # (n,m)

    D_xx = _sq_dists(X_t, X_t)
    D_yy = _sq_dists(Y_t, Y_t)
    D_xy = _sq_dists(X_t, Y_t)

    # Median heuristic on joint sample
    joint = torch.cat([D_xy.flatten(),
                       D_xx[D_xx > 0].flatten(),
                       D_yy[D_yy > 0].flatten()])
    med   = joint.median().item()
    if med < 1e-10:
        med = 1.0

    bws = [med * (2 ** k) for k in range(-(n_bandwidths // 2),
                                          n_bandwidths - n_bandwidths // 2)]

    n, m = len(X), len(Y)
    mmd2 = 0.0
    for bw in bws:
        gamma = 1.0 / (2.0 * bw)
        Kxx   = torch.exp(-gamma * D_xx)
        Kyy   = torch.exp(-gamma * D_yy)
        Kxy   = torch.exp(-gamma * D_xy)

        # Unbiased: zero-out diagonal of Kxx, Kyy
        Kxx.fill_diagonal_(0)
        Kyy.fill_diagonal_(0)

        mmd2 += (Kxx.sum() / (n * (n - 1)) +
                 Kyy.sum() / (m * (m - 1)) -
                 2 * Kxy.mean()).item()

    return mmd2 / len(bws)


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------

def permutation_pvalue(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
    n_perm:     int = 500,
    seed:       int = 42,
) -> Tuple[float, np.ndarray]:
    """
    Permutation test for MMD.

    Randomly permute the combined (real || gen) pool and compute MMD on
    each split.  Returns (p_value, null_distribution).
    """
    rng = np.random.default_rng(seed)
    n, m = len(real_feats), len(gen_feats)
    combined = np.concatenate([real_feats, gen_feats], axis=0)

    obs_mmd = _rbf_mmd_sq(real_feats, gen_feats)
    null_mmd = np.empty(n_perm, dtype=np.float64)

    for i in tqdm(range(n_perm), desc="Permutation test", leave=False):
        perm    = rng.permutation(n + m)
        X_perm  = combined[perm[:n]]
        Y_perm  = combined[perm[n:]]
        null_mmd[i] = _rbf_mmd_sq(X_perm, Y_perm)

    p_value = float(np.mean(null_mmd >= obs_mmd))
    return p_value, null_mmd


def bootstrap_ci(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
    n_boot:     int = 200,
    ci_level:   float = 0.95,
    seed:       int = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    Bootstrap confidence interval for MMD.

    Resample real and gen independently with replacement.
    Returns (ci_low, ci_high, boot_distribution).
    """
    rng = np.random.default_rng(seed)
    n, m = len(real_feats), len(gen_feats)
    boot_mmd = np.empty(n_boot, dtype=np.float64)

    for i in tqdm(range(n_boot), desc="Bootstrap CI", leave=False):
        idx_r = rng.integers(0, n, size=n)
        idx_g = rng.integers(0, m, size=m)
        boot_mmd[i] = _rbf_mmd_sq(real_feats[idx_r], gen_feats[idx_g])

    alpha   = (1.0 - ci_level) / 2.0
    ci_low  = float(np.percentile(boot_mmd, 100 * alpha))
    ci_high = float(np.percentile(boot_mmd, 100 * (1 - alpha)))
    return ci_low, ci_high, boot_mmd


def effect_size(obs_mmd: float, null_distribution: np.ndarray) -> float:
    """
    Cohen's d analog: (obs - null_mean) / null_std.
    Positive = generator distinguishable from real;
    near 0   = indistinguishable.
    """
    mu  = float(np.mean(null_distribution))
    std = float(np.std(null_distribution))
    if std < 1e-12:
        return 0.0
    return (obs_mmd - mu) / std


# ---------------------------------------------------------------------------
# Full statistical package
# ---------------------------------------------------------------------------

def statistical_m3(
    real_imgs,
    gen_imgs,
    backbone_id:  str   = "Snarcy/RadioDino-s16",
    device:       str   = "cuda",
    n_perm:       int   = 500,
    n_boot:       int   = 200,
    ci_level:     float = 0.95,
    batch_size:   int   = 16,
    single_layer: int   = 12,
    real_feats:   Optional[np.ndarray] = None,
    gen_feats:    Optional[np.ndarray] = None,
    seed:         int   = 42,
) -> dict:
    """
    Full statistical package: M3 score + p-value + effect size + CI.

    If real_feats/gen_feats are provided, skips feature extraction.
    """
    device = device if torch.cuda.is_available() else "cpu"

    # --- Feature extraction ---
    if real_feats is None or gen_feats is None:
        extractor_fn, _ = _extract_all_layers(
            real_imgs, backbone_id, device, batch_size
        )
        if real_feats is None:
            real_feats = extractor_fn(real_imgs)
        if gen_feats is None:
            gen_feats  = extractor_fn(gen_imgs)

    # --- Observed M3 score ---
    obs_mmd = _rbf_mmd_sq(real_feats, gen_feats)

    # --- Permutation p-value + null distribution ---
    p_val, null_dist = permutation_pvalue(
        real_feats, gen_feats, n_perm=n_perm, seed=seed
    )

    # --- Effect size ---
    eff = effect_size(obs_mmd, null_dist)

    # --- Bootstrap CI ---
    ci_lo, ci_hi, boot_dist = bootstrap_ci(
        real_feats, gen_feats, n_boot=n_boot, ci_level=ci_level, seed=seed
    )

    return {
        "m3_score":          obs_mmd,
        "p_value":           p_val,
        "effect_size":       eff,
        "ci_low":            ci_lo,
        "ci_high":           ci_hi,
        "ci_level":          ci_level,
        "null_mean":         float(np.mean(null_dist)),
        "null_std":          float(np.std(null_dist)),
        "null_mmd":          null_dist,
        "boot_mmd":          boot_dist,
        "n_real":            len(real_feats),
        "n_gen":             len(gen_feats),
        "n_perm":            n_perm,
        "n_boot":            n_boot,
        "backbone_id":       backbone_id,
        "single_layer":      single_layer,
    }


# ---------------------------------------------------------------------------
# Stateful class wrapper
# ---------------------------------------------------------------------------

class StatisticalM3:
    """
    Stateful wrapper: load backbone once, cache real features,
    evaluate multiple generators efficiently.

    Usage:
        sm3 = StatisticalM3(device="cuda")
        sm3.set_real(real_imgs)
        result_a = sm3.evaluate(gen_imgs_a, label="ddpm")
        result_b = sm3.evaluate(gen_imgs_b, label="wdm3d")
        sm3.compare_report([result_a, result_b])
    """

    def __init__(
        self,
        backbone_id:  str   = "Snarcy/RadioDino-s16",
        device:       str   = "cuda",
        n_perm:       int   = 500,
        n_boot:       int   = 200,
        ci_level:     float = 0.95,
        batch_size:   int   = 16,
        single_layer: int   = 12,
        seed:         int   = 42,
    ):
        self.backbone_id  = backbone_id
        self.device       = device if torch.cuda.is_available() else "cpu"
        self.n_perm       = n_perm
        self.n_boot       = n_boot
        self.ci_level     = ci_level
        self.batch_size   = batch_size
        self.single_layer = single_layer
        self.seed         = seed
        self._real_feats: Optional[np.ndarray] = None

        self._extractor_fn, _ = _extract_all_layers(
            [], backbone_id, self.device, batch_size, single_layer
        )

    def set_real(self, real_imgs) -> np.ndarray:
        self._real_feats = self._extractor_fn(real_imgs)
        return self._real_feats

    def evaluate(self, gen_imgs, label: str = "gen",
                 gen_feats: Optional[np.ndarray] = None) -> dict:
        if self._real_feats is None:
            raise RuntimeError("Call set_real() first.")
        if gen_feats is None:
            gen_feats = self._extractor_fn(gen_imgs)
        result = statistical_m3(
            None, None,
            backbone_id=self.backbone_id,
            device=self.device,
            n_perm=self.n_perm,
            n_boot=self.n_boot,
            ci_level=self.ci_level,
            batch_size=self.batch_size,
            single_layer=self.single_layer,
            real_feats=self._real_feats,
            gen_feats=gen_feats,
            seed=self.seed,
        )
        result["label"] = label
        return result

    @staticmethod
    def compare_report(results: list) -> None:
        """Print a side-by-side comparison table."""
        header = f"{'Generator':<20}  {'M3':>10}  {'p-val':>8}  {'ES':>6}  {'CI_95':>18}"
        print(header)
        print("-" * len(header))
        for r in results:
            ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            print(f"{r.get('label','?'):<20}  "
                  f"{r['m3_score']:>10.6f}  "
                  f"{r['p_value']:>8.4f}  "
                  f"{r['effect_size']:>6.2f}  "
                  f"{ci_str:>18}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, os
    from experiments._shared_utils import load_pils_recursive
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",    required=True)
    p.add_argument("--gen_dir",     required=True)
    p.add_argument("--output_dir",  default="results/statistical_rigor")
    p.add_argument("--n_images",    type=int, default=500)
    p.add_argument("--n_perm",      type=int, default=500)
    p.add_argument("--n_boot",      type=int, default=200)
    p.add_argument("--device",      default=None)
    p.add_argument("--backbone_id", default="Snarcy/RadioDino-s16")
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.output_dir, exist_ok=True)

    real_imgs = load_pils_recursive(a.real_dir, n=a.n_images)
    gen_imgs  = load_pils_recursive(a.gen_dir,  n=a.n_images)
    print(f"Real: {len(real_imgs)}  Gen: {len(gen_imgs)}")

    result = statistical_m3(
        real_imgs, gen_imgs,
        backbone_id=a.backbone_id,
        device=device,
        n_perm=a.n_perm,
        n_boot=a.n_boot,
    )

    report = {k: (float(v) if np.isscalar(v) else v)
              for k, v in result.items()
              if not isinstance(v, np.ndarray)}
    with open(os.path.join(a.output_dir, "statistical_rigor_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Plot null distribution with observed score
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(result["null_mmd"], bins=30, color="#4878cf", alpha=0.7,
                 label="Null distribution")
    axes[0].axvline(result["m3_score"], color="red", lw=2,
                    label=f"Observed  M3={result['m3_score']:.4f}")
    axes[0].set_xlabel("MMD value")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Permutation Test  (p={result['p_value']:.4f}, "
                      f"ES={result['effect_size']:.2f})")
    axes[0].legend()

    axes[1].hist(result["boot_mmd"], bins=30, color="#6acc65", alpha=0.7,
                 label="Bootstrap distribution")
    axes[1].axvline(result["ci_low"],  color="navy",  lw=1.5, ls="--",
                    label=f"95% CI [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    axes[1].axvline(result["ci_high"], color="navy",  lw=1.5, ls="--")
    axes[1].axvline(result["m3_score"], color="red",  lw=2, label="Observed")
    axes[1].set_xlabel("MMD value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Bootstrap Confidence Interval (95%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(a.output_dir, "statistical_rigor_plot.png"))
    plt.close()

    print(f"\nM3 Score     : {result['m3_score']:.6f}")
    print(f"p-value      : {result['p_value']:.4f}  ({'significant' if result['p_value'] < 0.05 else 'not sig.'})")
    print(f"Effect size  : {result['effect_size']:.2f}")
    print(f"95% CI       : [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    print(f"Saved -> {a.output_dir}")
