"""
experiments/debug_metric_ablation.py
====================================
Workstream A of the "Debug-then-Prove" plan: empirically settle the M3 metric design.

Answers three questions before any paper reframing:

  A1  Layer discriminability — for each ViT layer L, how decisively (permutation Z)
      and how stably (bootstrap CV) does MMD2(real, gen) separate, and how does each
      layer respond to near-OOD (WDM-3D brain) and far-OOD (retinal)?

  A2  Weighting ablation — does the entropy x stability x uniqueness weighting beat
      single-layer L12 and a uniform layer average, judged by Z (decisiveness) and
      CV (stability) under one shared permutation/bootstrap null?

  A3  OOD-vs-Spearman contradiction — reproduce and explain why M3 wins OOD AUC while
      InceptionV3 has higher per-sample Spearman with the real/OOD label (rank vs value).

All comparisons use raw [0,1] 224x224 tensors (the metric normalizes internally).
Runnable on the existing generated sets — no checkpoint or retraining required.

Usage:
    python experiments/debug_metric_ablation.py --n 300 --device cuda \
        --out results/debug_metric
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.m3_score_v2 import M3EntropyMetric


# ---------------------------------------------------------------------------
# Data loading (raw [0,1], 224x224 — NO ImageNet normalization; metric does it)
# ---------------------------------------------------------------------------

def _raw01(size: int = 224):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])


def load_raw01(directory: str, n: int, exclude_masks: bool = True) -> torch.Tensor:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths: list[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(directory, "**", e), recursive=True))
    paths = sorted(paths)
    if exclude_masks:
        paths = [p for p in paths if "_segmask_" not in os.path.basename(p)
                 and not os.path.basename(p).endswith("_mask.png")]
    paths = paths[:n]
    if not paths:
        raise FileNotFoundError(f"No images found under {directory}")
    t = _raw01()
    return torch.stack([t(Image.open(p).convert("RGB")) for p in paths])


# ---------------------------------------------------------------------------
# Weight computation — replicates M3EntropyMetric.forward() exactly, so the
# "weighted" config is the shipping metric's own scheme.
# ---------------------------------------------------------------------------

def compute_weights_full(metric, real_feats, gen_feats, entropy_vals, num_sub_batches=5):
    """Full replication of forward() weight scheme for the (real,gen) pair."""
    M = len(real_feats)
    nr = real_feats[0].shape[0]
    ng = gen_feats[0].shape[0]
    n_sub = max(1, min(num_sub_batches, min(nr, ng) // 4))
    sub_bs = max(2, min(nr, ng) // n_sub)
    sub_gen = torch.Generator().manual_seed(metric.seed)

    redundancy = []
    for i in range(M):
        cka_sum, npair = 0.0, 0
        for j in range(M):
            if i == j:
                continue
            cka_sum += metric._linear_cka(real_feats[i], real_feats[j])
            npair += 1
        redundancy.append(cka_sum / max(npair, 1))

    sem, stab, uniq = [], [], []
    for i in range(M):
        rf = real_feats[i].to(metric.device)
        gf = gen_feats[i].to(metric.device)
        sub_dists = []
        for _ in range(n_sub):
            ir = torch.randperm(rf.shape[0], generator=sub_gen)[:sub_bs].to(rf.device)
            ig = torch.randperm(gf.shape[0], generator=sub_gen)[:sub_bs].to(gf.device)
            sub_dists.append(metric._mmd2(rf[ir], gf[ig]).item())
        mean_sub = float(np.mean(sub_dists))
        std_sub = float(np.std(sub_dists))
        sem.append(float(np.exp(-entropy_vals[i] / max(metric.entropy_temperature, 1e-8))))
        stab.append(float(min(mean_sub / (std_sub + 1e-6), 1e4)))
        uniq.append(float(max(1.0 - redundancy[i], 1e-4)))

    raw = np.array(sem) * np.array(stab) * np.array(uniq)
    w = raw / (raw.sum() + 1e-8)
    return w, {"semanticity": sem, "stability": stab, "uniqueness": uniq,
               "redundancy_cka": redundancy}


# ---------------------------------------------------------------------------
# Combined test statistics from per-layer MMD2 values
# ---------------------------------------------------------------------------

def _stat(per_layer_mmd2: np.ndarray, config: str, weights: np.ndarray, best_idx: int):
    if config == "L12":
        return float(per_layer_mmd2[-1])
    if config == "best_single":
        return float(per_layer_mmd2[best_idx])
    if config == "uniform_all":
        return float(per_layer_mmd2.mean())
    if config == "weighted_all":
        return float((per_layer_mmd2 * weights).sum())
    raise ValueError(config)


def perm_test_configs(metric, real_feats, gen_feats, weights, best_idx,
                      configs, n_perm=200, seed=0):
    """Shared-null permutation test: one index shuffle applied across all layers."""
    dev = metric.device
    M = len(real_feats)
    nr = real_feats[0].shape[0]
    ng = gen_feats[0].shape[0]
    pooled = [torch.cat([real_feats[l].to(dev), gen_feats[l].to(dev)], 0) for l in range(M)]

    obs_layer = np.array([metric._mmd2(real_feats[l].to(dev), gen_feats[l].to(dev)).item()
                          for l in range(M)])
    obs = {c: _stat(obs_layer, c, weights, best_idx) for c in configs}

    rng = torch.Generator().manual_seed(seed)
    null = {c: [] for c in configs}
    for _ in range(n_perm):
        perm = torch.randperm(nr + ng, generator=rng)
        ri, gi = perm[:nr], perm[nr:]
        ml = np.array([metric._mmd2(pooled[l][ri], pooled[l][gi]).item() for l in range(M)])
        for c in configs:
            null[c].append(_stat(ml, c, weights, best_idx))

    res = {}
    for c in configs:
        arr = np.array(null[c])
        mu, sd = float(arr.mean()), float(arr.std())
        res[c] = {
            "observed": obs[c],
            "null_mean": mu,
            "null_std": sd,
            "z": float((obs[c] - mu) / max(sd, 1e-12)),
            "p_value": float((arr >= obs[c]).mean()),
        }
    return res, obs_layer.tolist()


def bootstrap_cv_configs(metric, real_feats, gen_feats, weights, best_idx,
                         configs, n_boot=200, seed=0):
    dev = metric.device
    M = len(real_feats)
    nr = real_feats[0].shape[0]
    ng = gen_feats[0].shape[0]
    rf = [real_feats[l].to(dev) for l in range(M)]
    gf = [gen_feats[l].to(dev) for l in range(M)]
    rng = torch.Generator().manual_seed(seed)
    boot = {c: [] for c in configs}
    for _ in range(n_boot):
        ri = torch.randint(0, nr, (nr,), generator=rng)
        gi = torch.randint(0, ng, (ng,), generator=rng)
        ml = np.array([metric._mmd2(rf[l][ri], gf[l][gi]).item() for l in range(M)])
        for c in configs:
            boot[c].append(_stat(ml, c, weights, best_idx))
    cv = {}
    for c in configs:
        arr = np.array(boot[c])
        cv[c] = {"mean": float(arr.mean()), "std": float(arr.std()),
                 "cv_pct": float(100.0 * arr.std() / (abs(arr.mean()) + 1e-12))}
    return cv


# ---------------------------------------------------------------------------
# A3 — OOD AUC vs per-sample Spearman for M3(L12) and InceptionV3(FID)
# ---------------------------------------------------------------------------

def centroid_scores(feats_real: torch.Tensor, feats_ood: torch.Tensor):
    """Per-sample distance (1 - cosine) to the real centroid. Returns (labels, scores)."""
    def l2n(x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-8)
    r = l2n(feats_real.float())
    o = l2n(feats_ood.float())
    centroid = l2n(r.mean(0, keepdim=True))
    s_real = (1.0 - (r @ centroid.t()).squeeze(1)).cpu().numpy()
    s_ood = (1.0 - (o @ centroid.t()).squeeze(1)).cpu().numpy()
    labels = np.concatenate([np.zeros(len(s_real)), np.ones(len(s_ood))])
    scores = np.concatenate([s_real, s_ood])
    return labels, scores


def inception_features(imgs01: torch.Tensor, device: str, batch: int = 32) -> torch.Tensor:
    """2048-d InceptionV3 pool features (ImageNet-normalized, 299x299)."""
    from torchvision import models
    net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                              aux_logits=True)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs01), batch):
            x = imgs01[i:i + batch].to(device)
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear",
                                                align_corners=False)
            x = (x - mean) / std
            feats.append(net(x).cpu())
    return torch.cat(feats, 0)


def auc_spearman(labels, scores):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    auc = float(roc_auc_score(labels, scores))
    rho, p = spearmanr(labels, scores)
    return auc, float(rho), float(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "debug_metric"))
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    sets = {
        "real": os.path.join(ROOT, "data_mri", "brats_axial_multislice"),
        "gen": os.path.join(ROOT, "output", "generated_500_standard"),
        "ood_near": os.path.join(ROOT, "output", "generated_wdm3d", "brats"),
        "ood_far": os.path.join(ROOT, "output", "generated_retinal"),
    }
    imgs = {}
    for k, d in sets.items():
        imgs[k] = load_raw01(d, args.n)
        print(f"[data] {k:9s} n={len(imgs[k])}  ({d})")

    metric = M3EntropyMetric(device=args.device, single_layer=None, cka_threshold=0.85,
                             seed=args.seed)
    L = metric.num_layers
    layers = set(range(1, L + 1))

    # Per-layer features + entropy for each set
    feats, ent = {}, {}
    for k in imgs:
        f, e = metric._extract_features_and_entropy(imgs[k].to(args.device),
                                                    layers_to_keep=layers)
        feats[k] = [f[i].cpu() for i in range(L)]
        ent[k] = [float(e[i]) for i in range(L)]
    print(f"[extract] done in {time.time()-t0:.1f}s  (L={L}, dim={feats['real'][-1].shape[1]})")

    report = {"config": {"n": args.n, "n_perm": args.n_perm, "n_boot": args.n_boot,
                         "seed": args.seed, "num_layers": L,
                         "backbone": metric.backbone_id}}

    # ---- A1: per-layer discriminability (real vs gen), + OOD per layer ----
    a1 = {"per_layer": []}
    for li in range(L):
        d_gen = metric._mmd2(feats["real"][li].to(args.device),
                             feats["gen"][li].to(args.device)).item()
        p, z, nm = metric._permutation_test_fidelity(feats["real"][li], feats["gen"][li],
                                                     n_permutations=args.n_perm, seed=args.seed)
        clo, chi = metric._bootstrap_ci_fidelity(feats["real"][li], feats["gen"][li],
                                                 n_bootstrap=args.n_boot, seed=args.seed)
        cv = 100.0 * (chi - clo) / (4.0 * abs(d_gen) + 1e-12)  # ~ (CI width / 4) / mean
        d_near = metric._mmd2(feats["real"][li].to(args.device),
                              feats["ood_near"][li].to(args.device)).item()
        d_far = metric._mmd2(feats["real"][li].to(args.device),
                             feats["ood_far"][li].to(args.device)).item()
        a1["per_layer"].append({
            "layer": li + 1, "mmd2_gen": d_gen, "perm_p": p, "z": z,
            "ci": [clo, chi], "cv_pct_approx": cv,
            "mmd2_ood_near": d_near, "mmd2_ood_far": d_far,
            "entropy_real": ent["real"][li],
        })
    best_idx = int(np.argmax([r["z"] for r in a1["per_layer"]]))
    a1["best_layer_by_z"] = best_idx + 1
    report["A1_layer_discriminability"] = a1
    print(f"[A1] best layer by Z = L{best_idx+1}  "
          f"(Z={a1['per_layer'][best_idx]['z']:.1f})")

    # ---- A2: weighting ablation (shared null) ----
    weights, comps = compute_weights_full(metric, feats["real"], feats["gen"], ent["real"])
    configs = ["L12", "best_single", "uniform_all", "weighted_all"]
    perm_res, obs_layer = perm_test_configs(metric, feats["real"], feats["gen"],
                                            weights, best_idx, configs,
                                            n_perm=args.n_perm, seed=args.seed)
    cv_res = bootstrap_cv_configs(metric, feats["real"], feats["gen"], weights, best_idx,
                                  configs, n_boot=args.n_boot, seed=args.seed)
    a2 = {"weights_per_layer": weights.tolist(), "weight_components": comps,
          "configs": {}}
    for c in configs:
        a2["configs"][c] = {**perm_res[c], **cv_res[c]}
    report["A2_weighting_ablation"] = a2
    print("[A2] config Z / CV%:")
    for c in configs:
        print(f"      {c:13s} Z={perm_res[c]['z']:8.1f}  p={perm_res[c]['p_value']:.3f}  "
              f"CV={cv_res[c]['cv_pct']:.2f}%")

    # ---- A3: OOD AUC vs Spearman (M3 L12 vs InceptionV3) ----
    a3 = {}
    inc_feats = {k: inception_features(imgs[k], args.device) for k in
                 ("real", "ood_near", "ood_far")}
    for ood_key in ("ood_near", "ood_far"):
        # M3 uses L12 features
        lab_m, sc_m = centroid_scores(feats["real"][-1], feats[ood_key][-1])
        auc_m, rho_m, pm = auc_spearman(lab_m, sc_m)
        # FID uses Inception pool features
        lab_i, sc_i = centroid_scores(inc_feats["real"], inc_feats[ood_key])
        auc_i, rho_i, pi = auc_spearman(lab_i, sc_i)
        a3[ood_key] = {
            "M3_L12": {"auc": auc_m, "spearman_rho": rho_m, "spearman_p": pm},
            "InceptionV3": {"auc": auc_i, "spearman_rho": rho_i, "spearman_p": pi},
        }
        print(f"[A3] {ood_key}: M3 AUC={auc_m:.3f} rho={rho_m:.3f} | "
              f"Inception AUC={auc_i:.3f} rho={rho_i:.3f}")
    report["A3_ood_vs_spearman"] = a3

    out_json = os.path.join(args.out, "ablation_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {time.time()-t0:.1f}s  ->  {out_json}")


if __name__ == "__main__":
    main()
