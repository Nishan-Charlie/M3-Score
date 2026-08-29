"""
Master Experiment Runner — RadioDino (microsoft/rad-dino)
==========================================================
Runs all key experiments and produces publication-ready plots with:
  - Black axis spines and ticks
  - 14 pt font size (axis labels), 13 pt (tick labels), 15 pt (titles)
  - Consistent font (DejaVu Sans / default matplotlib)
  - Same axis label naming convention across all plots

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --run_id 1 --seed 42
    python run_all_experiments.py --run_id 2 --seed 123
    python run_all_experiments.py --out_dir results/my_run --seed 0
"""

from __future__ import annotations

import argparse
import json, os, sys
import numpy as np
import torch

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="M3-Score full experiment suite")
_parser.add_argument("--run_id", type=int, default=None,
                     help="Numeric run ID; appended to output dir name (e.g. 1 → radiodino_run1)")
_parser.add_argument("--out_dir", type=str, default=None,
                     help="Override output directory path completely")
_parser.add_argument("--seed", type=int, default=42,
                     help="Global random seed (default 42)")
_parser.add_argument("--device", type=str, default=None,
                     help="Torch device string (default: auto-detect cuda/cpu)")
_parser.add_argument("--n_subset", type=int, default=500,
                     help="Subset size for sample_efficiency experiment (default 500)")
_parser.add_argument("--backbone", type=str, default="Snarcy/RadioDino-s16",
                     help="HuggingFace backbone ID (default: Snarcy/RadioDino-s16)")
_args, _ = _parser.parse_known_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Global plot style ─────────────────────────────────────────────────────────
FONT_TITLE  = 15
FONT_LABEL  = 14
FONT_TICK   = 13
FONT_LEGEND = 12
LW = 2.0
MS = 7

plt.rcParams.update({
    "font.size":         FONT_TICK,
    "axes.titlesize":    FONT_TITLE,
    "axes.labelsize":    FONT_LABEL,
    "xtick.labelsize":   FONT_TICK,
    "ytick.labelsize":   FONT_TICK,
    "legend.fontsize":   FONT_LEGEND,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "black",
    "axes.linewidth":    1.4,
    "xtick.color":       "black",
    "ytick.color":       "black",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.8,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

def pub_style(ax, grid=True):
    """Apply standard publication style to an axes object."""
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(1.4)
        ax.spines[spine].set_color("black")
    ax.tick_params(axis="both", which="major", colors="black",
                   labelsize=FONT_TICK, width=1.2, length=5)
    ax.set_facecolor("white")
    if grid:
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)


# ── Backbone & paths ─────────────────────────────────────────────────────────
BACKBONE_ID  = _args.backbone
# Short tag used in directory names: "microsoft/rad-dino" → "rad-dino"
BACKBONE_TAG = BACKBONE_ID.split("/")[-1].lower().replace("_", "-")

ROOT      = os.path.dirname(os.path.abspath(__file__))
REAL_DIR  = os.path.join(ROOT, "data_mri", "brats_axial_multislice")
GEN_DIR   = os.path.join(ROOT, "output",   "generated_500_best")
LAYERS_CACHE = os.path.join(ROOT, f"canonical_layers_{BACKBONE_TAG}.json")

# Output directory: --out_dir > --run_id > default
if _args.out_dir:
    OUT_DIR = _args.out_dir
elif _args.run_id is not None:
    OUT_DIR = os.path.join(ROOT, "results", f"{BACKBONE_TAG}_run{_args.run_id}")
else:
    OUT_DIR = os.path.join(ROOT, "results", f"{BACKBONE_TAG}_rerun")

DEVICE    = _args.device if _args.device else ("cuda" if torch.cuda.is_available() else "cpu")
N_IMAGES  = 500
N_NOISE   = 200
SEED      = _args.seed
N_PERM    = 50
N_SUBSET  = _args.n_subset   # subset size for sample_efficiency (paper uses 500)

# WDM-3D generated image directories (produced by tools/generate_wdm3d.py)
_WDM3D_BRATS_DIR = os.environ.get(
    "M3_WDM3D_BRATS_DIR",
    os.path.join(ROOT, "output", "generated_wdm3d", "brats")
)
_WDM3D_LIDC_DIR = os.environ.get(
    "M3_WDM3D_LIDC_DIR",
    os.path.join(ROOT, "output", "generated_wdm3d", "lidc")
)

print(f"\n{'='*60}")
print(f"  Run config")
print(f"  Output dir : {OUT_DIR}")
print(f"  Seed       : {SEED}")
print(f"  Device     : {DEVICE}")
print(f"  N_subset   : {N_SUBSET}")
print(f"{'='*60}\n")

os.makedirs(OUT_DIR, exist_ok=True)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.m3_score_v2 import M3V2Metric


# =============================================================================
# Helper: load image paths
# =============================================================================
import glob
from PIL import Image
from torchvision import transforms

def _load_paths(directory, n=None, recursive=True):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        if recursive:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        else:
            paths.extend(glob.glob(os.path.join(directory, ext)))
    paths = sorted(set(paths))
    if n:
        paths = paths[:n]
    if not paths:
        raise FileNotFoundError(f"No images in {directory}")
    return paths

def _load_m3(paths):
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    return torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])


# =============================================================================
# Initialise RadioDino M3 metric
# =============================================================================
print("\n" + "="*60)
print(f"  Initialising backbone: {BACKBONE_ID}")
print("="*60)
metric = M3V2Metric(device=DEVICE, backbone_id=BACKBONE_ID, cka_threshold=0.80, seed=SEED)

real_paths = _load_paths(REAL_DIR, N_IMAGES)
gen_paths  = _load_paths(GEN_DIR,  N_IMAGES, recursive=False)
cap = min(len(real_paths), len(gen_paths))
real_paths, gen_paths = real_paths[:cap], gen_paths[:cap]
print(f"Real: {len(real_paths)} | Gen: {len(gen_paths)}")

real_imgs = _load_m3(real_paths)
gen_imgs  = _load_m3(gen_paths)

# Use ≥200 images for stable CKA selection; seed ensures identical draws across re-runs.
# 200 images are enough to make the greedy backward CKA insensitive to the specific subset.
torch.manual_seed(SEED); np.random.seed(SEED)
cka_ref_n = min(200, len(real_imgs))
cka_ref_imgs = real_imgs[torch.randperm(len(real_imgs), generator=torch.Generator().manual_seed(SEED))[:cka_ref_n]]
metric.prune_layers_via_cka(cka_ref_imgs, cache_path=LAYERS_CACHE, seed=SEED)
print(f"Active layers: {metric.active_layers}")

# Load cached exp 1 & 2 results (already computed)
CACHE_FILE = os.path.join(OUT_DIR, "exp12_cache.json")
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        exp12 = json.load(f)
    m3_score   = exp12["m3_score"]
    layer_wt   = exp12["layer_weights"]
    Z_score    = exp12["Z_score"]
    mu_null    = exp12["null_mean"]
    sig_null   = exp12["null_std"]
    null_arr   = np.array(exp12["null_scores"])
    print(f"[CACHE] M3 Score={m3_score:.6f}  Z={Z_score:.1f}")
    master = {
        "active_layers": metric.active_layers,
        "device": DEVICE,
        "core_m3": exp12["core_m3"],
        "permutation_test": exp12["permutation_test"],
    }
else:
    # ==========================================================================
    # EXPERIMENT 1 — Core M3 Score + Layer Weights
    # ==========================================================================
    print("\n" + "="*60)
    print("  EXPERIMENT 1: Core M3 Score + Layer Weights")
    print("="*60)

    with torch.no_grad():
        result = metric(real_imgs, gen_imgs)

    m3_score   = result["m3_score"]
    layer_dist = result["layer_distances"]
    layer_wt   = result["layer_weights"]
    sem        = result["semanticity"]
    stab       = result["stability"]
    uniq       = result["uniqueness"]

    print(f"M3 Score: {m3_score:.6f}")
    for l in metric.active_layers:
        k = f"L{l}"
        print(f"  {k}: MMD²={layer_dist[k]:.4f}  w={layer_wt[k]:.4f}  "
              f"sem={sem[k]:.4f}  stab={stab[k]:.2f}  uniq={uniq[k]:.4f}")

    # ==========================================================================
    # EXPERIMENT 2 — Permutation Test
    # ==========================================================================
    print("\n" + "="*60)
    print("  EXPERIMENT 2: Permutation Test")
    print("="*60)

    torch.manual_seed(SEED); np.random.seed(SEED)
    null_scores = []
    n_half = cap // 2

    for i in range(N_PERM):
        idx = torch.randperm(cap)
        split_a = real_imgs[idx[:n_half]]
        split_b = real_imgs[idx[n_half:n_half*2]]
        with torch.no_grad():
            r = metric(split_a, split_b)
        null_scores.append(r["m3_score"])
        if (i+1) % 10 == 0:
            print(f"  Permutation {i+1}/{N_PERM}: score={r['m3_score']:.6f}")

    null_arr = np.array(null_scores)
    mu_null  = float(null_arr.mean())
    sig_null = float(null_arr.std())
    Z_score  = (m3_score - mu_null) / (sig_null + 1e-12)
    emp_p    = float((null_arr >= m3_score).mean())

    print(f"\nNull: mu={mu_null:.4e}  sigma={sig_null:.4e}")
    print(f"S_obs={m3_score:.6f}  Z={Z_score:.2f}  empirical p={emp_p:.4f} (<{1/N_PERM:.2f})")

    master = {
        "active_layers": metric.active_layers,
        "device": DEVICE,
        "core_m3": {
            "m3_score": m3_score,
            "layer_distances": layer_dist,
            "layer_weights":   layer_wt,
            "semanticity":     sem,
            "stability":       stab,
            "uniqueness":      uniq,
        },
        "permutation_test": {
            "null_mean":   mu_null,
            "null_std":    sig_null,
            "S_obs":       m3_score,
            "Z_score":     round(Z_score, 2),
            "empirical_p": emp_p,
            "n_perm":      N_PERM,
        },
    }
    # cache
    with open(CACHE_FILE, "w") as f:
        json.dump({**master,
                   "m3_score": m3_score,
                   "layer_weights": layer_wt,
                   "Z_score": round(Z_score, 2),
                   "null_mean": mu_null, "null_std": sig_null,
                   "null_scores": null_scores}, f, indent=2)


# =============================================================================
# Permutation test plot (always regenerate with good style)
# =============================================================================
S_obs = m3_score
fig, ax = plt.subplots(figsize=(7, 4.5))
pub_style(ax)
ax.hist(null_arr, bins=15, color="#4fc3f7", edgecolor="black", linewidth=0.8,
        alpha=0.85, label=f"Null distribution (n={N_PERM})")
ax.axvline(S_obs, color="#e53935", linewidth=2.2, linestyle="--",
           label=f"Observed M3-Score = {S_obs:.4f}")
ax.set_xlabel("M3-Score (real vs. real splits)", fontsize=FONT_LABEL, color="black")
ax.set_ylabel("Count", fontsize=FONT_LABEL, color="black")
ax.set_title(f"Permutation Test  (Z = {Z_score:.1f},  empirical p < {1/N_PERM:.2f})",
             fontsize=FONT_TITLE, color="black")
ax.legend(fontsize=FONT_LEGEND, framealpha=1, edgecolor="black")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "permutation_test.png"))
plt.close()
print("Plot saved: permutation_test.png")


# =============================================================================
# EXPERIMENT 3 — OOD / Binary Separation (M3 vs InceptionV3 vs CLIP-MMD)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 3: OOD Separation AUC (M3 vs InceptionV3 vs CLIP-MMD)")
print("="*60)

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy import stats as sp_stats
from torchvision import models
import torch.nn as nn

def _l2norm(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms < 1e-8, 1.0, norms)

# ── Centroid-fit / eval split ────────────────────────────────────────────────
# To avoid in-sample centroid bias: fit the centroid on the first half of
# real images, evaluate distances on the held-out second half + all generated.
# This gives an honest upper bound on separability.
rng_ood = np.random.default_rng(SEED)
real_idx_shuffled = rng_ood.permutation(cap)
n_fit  = cap // 2
fit_idx  = real_idx_shuffled[:n_fit]
eval_idx = real_idx_shuffled[n_fit:]
eval_cap = len(eval_idx)       # number of held-out reals = gen images evaluated
gen_idx  = np.arange(cap)      # all generated images

print(f"  OOD centroid-fit n={n_fit}, eval n={eval_cap} (real) + {cap} (gen)")

# ── M3 per-image distances ───────────────────────────────────────────────────
print("  Computing M3 per-image distances ...")
active_set = set(metric.active_layers)
with torch.no_grad():
    real_feats_list = metric._extract_raw_features(real_imgs, use_attention=True,
                                                    layers_to_keep=active_set)
    gen_feats_list  = metric._extract_raw_features(gen_imgs,  use_attention=True,
                                                    layers_to_keep=active_set)

m3_real_dist_all = np.zeros(cap)
m3_gen_dist_all  = np.zeros(cap)
for lk, wk in layer_wt.items():
    idx = int(lk[1:]) - 1
    rf_all = _l2norm(real_feats_list[idx].numpy())
    gf_all = _l2norm(gen_feats_list[idx].numpy())
    centroid = rf_all[fit_idx].mean(axis=0)          # fit centroid on held-out half
    m3_real_dist_all += wk * np.linalg.norm(rf_all - centroid, axis=1)
    m3_gen_dist_all  += wk * np.linalg.norm(gf_all - centroid, axis=1)

m3_real_dist = m3_real_dist_all[eval_idx]
m3_gen_dist  = m3_gen_dist_all

# ── InceptionV3 per-image distances ─────────────────────────────────────────
print("  Computing InceptionV3 per-image distances ...")
inc_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
inc_model.fc = nn.Identity(); inc_model.eval().to(DEVICE)
inc_tfm = transforms.Compose([
    transforms.Resize((299, 299)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def _inc_feats(paths):
    all_f = []
    BS = 32
    for s in range(0, len(paths), BS):
        batch = torch.stack([inc_tfm(Image.open(p).convert("RGB"))
                             for p in paths[s:s+BS]]).to(DEVICE)
        with torch.no_grad():
            out = inc_model(batch)
            if hasattr(out, "logits"): out = out.logits
        all_f.append(out.cpu().numpy())
    return np.concatenate(all_f, axis=0)

inc_real_f = _inc_feats(real_paths)
inc_gen_f  = _inc_feats(gen_paths)
inc_centroid = inc_real_f[fit_idx].mean(axis=0)      # same held-out split
inc_real_dist = np.linalg.norm(inc_real_f[eval_idx] - inc_centroid, axis=1)
inc_gen_dist  = np.linalg.norm(inc_gen_f             - inc_centroid, axis=1)
del inc_model

# ── CLIP-MMD (CMMD) per-image distances ─────────────────────────────────────
print("  Computing CLIP-MMD (CMMD) per-image distances ...")
clip_real_dist = np.full(eval_cap, np.nan)
clip_gen_dist  = np.full(cap,      np.nan)
try:
    from evaluation.cmmd_metric import CMMDMetric
    cmmd = CMMDMetric(device=DEVICE)
    clip_real_f_all = cmmd.extract_features(real_paths, desc="CLIP [real]")
    clip_gen_f      = cmmd.extract_features(gen_paths,  desc="CLIP [gen]")
    clip_centroid   = clip_real_f_all[fit_idx].mean(axis=0)
    clip_real_dist  = np.linalg.norm(clip_real_f_all[eval_idx] - clip_centroid, axis=1)
    clip_gen_dist   = np.linalg.norm(clip_gen_f                - clip_centroid, axis=1)
    del cmmd, clip_real_f_all
    print("  CLIP-MMD distances computed.")
except Exception as _clip_e:
    print(f"  [WARN] CLIP-MMD failed: {_clip_e}")

# ── AUC (eval reals vs all generated) ───────────────────────────────────────
labels_ood = np.concatenate([np.zeros(eval_cap), np.ones(cap)])

def _auc_pair(real_d, gen_d):
    all_d = np.concatenate([real_d, gen_d])
    if np.any(np.isnan(all_d)):
        return float("nan"), float("nan")
    return (round(float(roc_auc_score(labels_ood, all_d)), 4),
            round(float(average_precision_score(labels_ood, all_d)), 4))

m3_roc_auc,   m3_pr_auc   = _auc_pair(m3_real_dist,   m3_gen_dist)
inc_roc_auc,  inc_pr_auc  = _auc_pair(inc_real_dist,  inc_gen_dist)
clip_roc_auc, clip_pr_auc = _auc_pair(clip_real_dist, clip_gen_dist)

def _rel(a, b):
    return round((a - b) / max(abs(b), 1e-8) * 100, 2) if not (np.isnan(a) or np.isnan(b)) else float("nan")

rel_vs_inc_roc  = _rel(m3_roc_auc,  inc_roc_auc)
rel_vs_clip_roc = _rel(m3_roc_auc,  clip_roc_auc)

print(f"  M3 ({BACKBONE_TAG})  ROC-AUC={m3_roc_auc:.4f}  PR-AUC={m3_pr_auc:.4f}")
print(f"  InceptionV3 (FID)   ROC-AUC={inc_roc_auc:.4f}  PR-AUC={inc_pr_auc:.4f}")
print(f"  CLIP-MMD            ROC-AUC={clip_roc_auc:.4f}  PR-AUC={clip_pr_auc:.4f}")
print(f"  M3 vs FID:  {rel_vs_inc_roc:+.1f}%   M3 vs CMMD: {rel_vs_clip_roc:+.1f}%")

# ── Isolation Forest ─────────────────────────────────────────────────────────
print("  Fitting Isolation Forest ...")
rn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
rn_feat = nn.Sequential(*list(rn.children())[:-1]); rn_feat.eval().to(DEVICE)
rn_tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def _rn_feats(paths):
    all_f = []
    for s in range(0, len(paths), 64):
        b = torch.stack([rn_tfm(Image.open(p).convert("RGB"))
                         for p in paths[s:s+64]]).to(DEVICE)
        with torch.no_grad():
            all_f.append(rn_feat(b).squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(all_f, axis=0)

rn_real = _rn_feats(real_paths)
rn_gen  = _rn_feats(gen_paths)
del rn, rn_feat

iforest = IsolationForest(n_estimators=200, contamination=0.05,
                           random_state=SEED, n_jobs=-1)
iforest.fit(rn_real[fit_idx])            # fit on same held-out half
anomaly_scores      = -iforest.decision_function(rn_gen)
real_anomaly_scores = -iforest.decision_function(rn_real[eval_idx])
is_anomaly          =  iforest.predict(rn_gen) == -1
pct_ood = float(is_anomaly.mean()) * 100

ood_threshold = float(np.percentile(real_anomaly_scores, 95))

rho_m3,   p_m3   = sp_stats.spearmanr(anomaly_scores, m3_gen_dist)
rho_inc,  p_inc  = sp_stats.spearmanr(anomaly_scores, inc_gen_dist)
rho_clip, p_clip = sp_stats.spearmanr(anomaly_scores, clip_gen_dist) \
                   if not np.any(np.isnan(clip_gen_dist)) else (float("nan"), float("nan"))
print(f"  OOD rate: {pct_ood:.1f}%")
print(f"  Spearman M3:   rho={rho_m3:.4f} p={p_m3:.2e}")
print(f"  Spearman Inc:  rho={rho_inc:.4f} p={p_inc:.2e}")
print(f"  Spearman CLIP: rho={rho_clip:.4f} p={p_clip:.2e}")

master["ood"] = {
    "m3_backbone":                    BACKBONE_TAG,
    "m3_roc_auc":                     m3_roc_auc,
    "m3_pr_auc":                      m3_pr_auc,
    "inceptionv3_roc_auc":            inc_roc_auc,
    "inceptionv3_pr_auc":             inc_pr_auc,
    "clip_mmd_roc_auc":               clip_roc_auc,
    "clip_mmd_pr_auc":                clip_pr_auc,
    "m3_vs_fid_roc_pct":              rel_vs_inc_roc,
    "m3_vs_cmmd_roc_pct":             rel_vs_clip_roc,
    "pct_ood":                        round(pct_ood, 2),
    "spearman_m3_rho":                round(float(rho_m3),   4),
    "spearman_m3_p":                  float(p_m3),
    "spearman_inc_rho":               round(float(rho_inc),  4),
    "spearman_inc_p":                 float(p_inc),
    "spearman_clip_rho":              round(float(rho_clip), 4),
    "spearman_clip_p":                float(p_clip),
    "centroid_fit_n":                 int(n_fit),
    "centroid_eval_n":                int(eval_cap),
}

# ── ROC curves (3-way: M3 vs FID vs CLIP-MMD) ───────────────────────────────
m3_all_d   = np.concatenate([m3_real_dist,   m3_gen_dist])
inc_all_d  = np.concatenate([inc_real_dist,  inc_gen_dist])
clip_all_d = np.concatenate([clip_real_dist, clip_gen_dist])

fpr_m3,   tpr_m3,   _ = roc_curve(labels_ood, m3_all_d)
fpr_inc,  tpr_inc,  _ = roc_curve(labels_ood, inc_all_d)
fpr_clip, tpr_clip, _ = (roc_curve(labels_ood, clip_all_d)
                          if not np.any(np.isnan(clip_all_d))
                          else (np.array([0,1]), np.array([0,1]), None))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor("white")

ax = axes[0]
pub_style(ax)
ax.plot([0,1],[0,1], "--", color="#aaaaaa", linewidth=1.2, label="Random classifier")
ax.plot(fpr_inc,  tpr_inc,  color="#ff7f0e", linewidth=LW,
        label=f"InceptionV3 / FID  AUC = {inc_roc_auc:.3f}")
ax.plot(fpr_clip, tpr_clip, color="#2ca02c", linewidth=LW,
        label=f"CLIP-MMD           AUC = {clip_roc_auc:.3f}")
ax.plot(fpr_m3,   tpr_m3,   color="#1f77b4", linewidth=LW,
        label=f"M3 / {BACKBONE_TAG}  AUC = {m3_roc_auc:.3f}")
ax.set_xlabel("False Positive Rate", fontsize=FONT_LABEL, color="black")
ax.set_ylabel("True Positive Rate",  fontsize=FONT_LABEL, color="black")
ax.set_title("ROC: Real vs Generated (held-out centroid split)",
             fontsize=FONT_TITLE, color="black")
ax.legend(fontsize=FONT_LEGEND, framealpha=1, edgecolor="black")

ax = axes[1]
pub_style(ax)
ax.hist(real_anomaly_scores, bins=30, alpha=0.65,
        color="#1f77b4", density=True, label="Real images (eval half)")
ax.hist(anomaly_scores, bins=30, alpha=0.65,
        color="#d62728", density=True, label="Generated images")
ax.axvline(ood_threshold, color="black", linewidth=1.8,
           linestyle=":", label="OOD threshold (95th pct real)")
ax.set_xlabel("Isolation Forest Anomaly Score", fontsize=FONT_LABEL, color="black")
ax.set_ylabel("Density", fontsize=FONT_LABEL, color="black")
ax.set_title(f"Score Distributions  ({pct_ood:.1f}% generated OOD)",
             fontsize=FONT_TITLE, color="black")
ax.legend(fontsize=FONT_LEGEND, framealpha=1, edgecolor="black")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ood_roc_auc.png"))
plt.close()
print("Plot saved: ood_roc_auc.png")

# ── OOD scatter plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
pub_style(ax)
inlier_mask = ~is_anomaly
ax.scatter(anomaly_scores[inlier_mask], m3_gen_dist[inlier_mask],
           c="#1f77b4", s=20, alpha=0.55, label=f"In-distribution (n={inlier_mask.sum()})",
           edgecolors="none")
ax.scatter(anomaly_scores[is_anomaly],  m3_gen_dist[is_anomaly],
           c="#d62728", s=30, alpha=0.80, label=f"OOD-flagged (n={is_anomaly.sum()})",
           edgecolors="black", linewidths=0.4)
ax.set_xlabel("Isolation Forest Anomaly Score", fontsize=FONT_LABEL, color="black")
ax.set_ylabel(f"M3 Per-Image Distance ({BACKBONE_TAG})", fontsize=FONT_LABEL, color="black")
ax.set_title(f"Per-Image Scores: M3 vs Anomaly Detector\n"
             f"Spearman ρ = {rho_m3:.3f},  p = {p_m3:.1e}",
             fontsize=FONT_TITLE, color="black")
ax.legend(fontsize=FONT_LEGEND, framealpha=1, edgecolor="black")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ood_comparison_scatter.png"))
plt.close()
print("Plot saved: ood_comparison_scatter.png")


# =============================================================================
# EXPERIMENT 4 — Noise & Blur Robustness
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 4: Noise & Blur Robustness")
print("="*60)

from experiments.noise_robustness import run_noise_robustness

noise_out = os.path.join(OUT_DIR, "noise_robustness")
noise_res = run_noise_robustness(
    real_dir=REAL_DIR,
    output_dir=noise_out,
    num_images=N_NOISE,
    device=DEVICE,
    seed=SEED,
    backbone_id=BACKBONE_ID,
)
master["noise_robustness"] = noise_res

# ── Regenerate plots with standardised style ──────────────────────────────────
colors_map = {
    "m3":        "#1f77b4",
    "fid":       "#ff7f0e",
    "kid":       "#2ca02c",
    "ssim":      "#d62728",
    "psnr":      "#9467bd",
    "ms_ssim":   "#8c564b",
    "lpips":     "#e377c2",
    "precision": "#7f7f7f",
    "recall":    "#bcbd22",
}
labels_map = {
    "m3": "M3-Score (RadioDino)", "fid": "FID", "kid": "KID",
    "ssim": "SSIM", "psnr": "PSNR", "ms_ssim": "MS-SSIM",
    "lpips": "LPIPS", "precision": "α-Precision", "recall": "β-Recall",
}

for sweep_name, level_key, xlabel in [
    ("noise", "sigma",  "Gaussian Noise Level σ"),
    ("blur",  "radius", "Gaussian Blur Radius (pixels)"),
]:
    sweep = noise_res[sweep_name]["metric_values"]
    corr  = noise_res[sweep_name]["spearman"]
    levels = np.array([r[level_key] for r in sweep])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]; pub_style(ax)
    for m in ["m3", "fid", "kid"]:
        vals = [r.get(m, float("nan")) for r in sweep]
        ax.plot(levels, vals, "o-", color=colors_map[m], label=labels_map[m],
                linewidth=LW, markersize=MS)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, color="black")
    ax.set_ylabel("Metric Value", fontsize=FONT_LABEL, color="black")
    ax.set_title("Distributional Metrics", fontsize=FONT_TITLE, color="black")
    ax.legend(fontsize=FONT_LEGEND, framealpha=1, edgecolor="black")

    ax = axes[0, 1]; pub_style(ax)
    for m in ["ssim", "psnr", "ms_ssim", "lpips"]:
        vals = [r.get(m, float("nan")) for r in sweep]
        ax.plot(levels, vals, "o-", color=colors_map[m], label=labels_map[m],
                linewidth=LW, markersize=MS)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, color="black")
    ax.set_ylabel("Metric Value", fontsize=FONT_LABEL, color="black")
    ax.set_title("Perceptual Metrics", fontsize=FONT_TITLE, color="black")
    ax.legend(fontsize=FONT_LEGEND, framealpha=1, edgecolor="black")

    ax = axes[1, 0]; pub_style(ax)
    for m in list(colors_map.keys()):
        vals = np.array([r.get(m, float("nan")) for r in sweep])
        valid = ~np.isnan(vals)
        if valid.sum() >= 2:
            vmin, vmax = vals[valid].min(), vals[valid].max()
            normed = (vals - vmin) / (vmax - vmin + 1e-9)
            ax.plot(levels[valid], normed[valid], "o-", color=colors_map[m],
                    label=labels_map[m], linewidth=LW, markersize=MS - 1)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, color="black")
    ax.set_ylabel("Normalised Metric Value [0, 1]", fontsize=FONT_LABEL, color="black")
    ax.set_title("All Metrics Normalised", fontsize=FONT_TITLE, color="black")
    ax.legend(fontsize=FONT_LEGEND - 1, framealpha=1, edgecolor="black", ncol=2)

    ax = axes[1, 1]; pub_style(ax, grid=False)
    ms_list  = [m for m in colors_map if m in corr]
    rho_vals = [abs(corr[m]["spearman_r"]) for m in ms_list]
    bars = ax.barh([labels_map[m] for m in ms_list], rho_vals,
                   color=[colors_map[m] for m in ms_list],
                   edgecolor="black", linewidth=0.8, height=0.6)
    ax.bar_label(bars, fmt="%.3f", fontsize=FONT_TICK, color="black", padding=4)
    ax.set_xlabel("|Spearman ρ|", fontsize=FONT_LABEL, color="black")
    ax.set_title("Metric Sensitivity (|Spearman ρ|)", fontsize=FONT_TITLE, color="black")
    ax.set_xlim(0, 1.12)
    ax.tick_params(axis="y", labelsize=FONT_TICK)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"{sweep_name}_robustness.png")
    fig.savefig(path)
    plt.close()
    print(f"Plot saved: {sweep_name}_robustness.png")


# =============================================================================
# EXPERIMENT 5 — Sample Efficiency / CV
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 5: Sample Efficiency — CV at N=500")
print("="*60)

from experiments.sample_efficiency import run_sample_efficiency

eff_out = os.path.join(OUT_DIR, "sample_efficiency")
eff_res = run_sample_efficiency(
    real_dir=REAL_DIR,
    gen_dir=GEN_DIR,
    output_dir=eff_out,
    device=DEVICE,
    n_subset=N_SUBSET,
    backbone_id=BACKBONE_ID,
)
master["sample_efficiency"] = eff_res

try:
    summary = (eff_res.get("results", {}).get("summary") or
               eff_res.get("summary", {}))
    m3_cv_500  = None
    fid_cv_500 = None
    for mname, size_data in summary.items():
        data500 = size_data.get("500", {})
        cv = data500.get("cv", None)
        if cv is None and data500.get("std") and data500.get("mean"):
            cv = data500["std"] / (abs(data500["mean"]) + 1e-12) * 100
        if "m3" in mname.lower():
            m3_cv_500 = cv
        elif "fid" in mname.lower():
            fid_cv_500 = cv
    if m3_cv_500:
        print(f"  M3-Score CV at N=500:  {m3_cv_500:.2f}%")
    if fid_cv_500:
        print(f"  FID CV at N=500:       {fid_cv_500:.2f}%")
    master["cv_at_500"] = {
        "m3_cv_pct":  round(float(m3_cv_500), 2) if m3_cv_500 else None,
        "fid_cv_pct": round(float(fid_cv_500), 2) if fid_cv_500 else None,
    }
except Exception as e:
    print(f"  [WARN] CV extraction: {e}")


# =============================================================================
# EXPERIMENT 6 — CKA Threshold Ablation
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 6: CKA Threshold Ablation")
print("="*60)

from experiments.weight_ablation import run_weight_ablation

abl_out = os.path.join(OUT_DIR, "weight_ablation")
abl_res = run_weight_ablation(
    real_dir=REAL_DIR,
    gen_dir=GEN_DIR,
    output_dir=abl_out,
    n=N_IMAGES,
    device=DEVICE,
    backbone_id=BACKBONE_ID,
)
master["weight_ablation"] = abl_res


# =============================================================================
# EXPERIMENT 7 — Backbone Comparison (RAD-DINO vs RadioDINO-s16 vs DINOv2)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 7: Backbone Comparison")
print("="*60)

from experiments.backbone_comparison import run_backbone_comparison

bc_out = os.path.join(OUT_DIR, "backbone_comparison")
try:
    bc_res = run_backbone_comparison(
        real_dir=REAL_DIR,
        gen_dir=GEN_DIR,
        output_dir=bc_out,
        n_images=N_IMAGES,
        device=DEVICE,
        cka_tau=0.80,
    )
    master["backbone_comparison"] = bc_res
except Exception as _bc_e:
    print(f"[WARN] Backbone comparison failed: {_bc_e}")
    import traceback; traceback.print_exc()
    master["backbone_comparison"] = {"error": str(_bc_e)}


# =============================================================================
# CKA LAYER SIMILARITY — All Backbones
# =============================================================================
print("\n" + "="*60)
print("  CKA LAYER SIMILARITY (all backbones)")
print("="*60)

from experiments.cka_layer_similarity import run_cka_analysis

cka_out = os.path.join(OUT_DIR, "cka_analysis")
try:
    cka_res = run_cka_analysis(
        real_dir    = REAL_DIR,
        output_dir  = cka_out,
        n_images    = 256,
        device      = DEVICE,
        skip_cached = True,
    )
    master["cka_analysis"] = cka_res
    if "backbones" in cka_res:
        rs16 = cka_res["backbones"].get("radiodino-s16", {})
        print(f"  RadioDino-s16 L1↔L12: {rs16.get('l1_vs_lmax', '?')}")
except Exception as _cka_e:
    print(f"  CKA analysis FAILED: {_cka_e}")
    master["cka_analysis"] = {"error": str(_cka_e)}


# =============================================================================
# EXPERIMENT 8 — Weighting Justification
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 8: Weighting Justification")
print("="*60)

from experiments.weighting_justification import run_weighting_justification

wj_out = os.path.join(OUT_DIR, "weighting_justification")
try:
    wj_res = run_weighting_justification(
        real_dir=REAL_DIR,
        gen_dir=GEN_DIR,
        output_dir=wj_out,
        n_images=N_IMAGES,
        device=DEVICE,
        seed=SEED,
        backbone_id=BACKBONE_ID,
        layers_cache=LAYERS_CACHE,
    )
    master["weighting_justification"] = wj_res
except Exception as _wj_e:
    print(f"[WARN] Weighting justification failed: {_wj_e}")
    import traceback; traceback.print_exc()
    master["weighting_justification"] = {"error": str(_wj_e)}


# =============================================================================
# EXPERIMENT 9 — Calibration / Sanity Panel
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 9: Calibration / Sanity Panel")
print("="*60)

from experiments.calibration_sanity import run_calibration_sanity

cs_out = os.path.join(OUT_DIR, "calibration_sanity")
try:
    cs_res = run_calibration_sanity(
        real_dir=REAL_DIR,
        gen_dir=GEN_DIR,
        output_dir=cs_out,
        n_images=N_IMAGES,
        device=DEVICE,
        seed=SEED,
        backbone_id=BACKBONE_ID,
        layers_cache=LAYERS_CACHE,
    )
    master["calibration_sanity"] = cs_res
except Exception as _cs_e:
    print(f"[WARN] Calibration sanity failed: {_cs_e}")
    import traceback; traceback.print_exc()
    master["calibration_sanity"] = {"error": str(_cs_e)}


# =============================================================================
# EXPERIMENT 10: PATHOLOGY MASKING (clinical relevance)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 10: PATHOLOGY MASKING")
print("="*60)

from experiments.pathology_masking import run_pathology_masking

pm_out = os.path.join(OUT_DIR, "pathology_masking")
_MASK_DIR = os.environ.get("M3_MASK_DIR", None)   # optional: set in env before running
_GEN_DIR  = os.environ.get("M3_GEN_DIR",  os.path.join(os.path.dirname(OUT_DIR), "generated"))
try:
    pm_res = run_pathology_masking(
        real_dir    = REAL_DIR,
        gen_dir     = _GEN_DIR,
        output_dir  = pm_out,
        mask_dir    = _MASK_DIR,
        n_images    = min(200, cap),
        device      = DEVICE,
        seed        = SEED,
        backbone_id = _args.backbone,
    )
    master["pathology_masking"] = pm_res
    print(f"  [10] Pathology masking done. Sensitivity M3={pm_res['sensitivity_b_vs_a']['m3']:+.3f}")
except Exception as _pm_e:
    print(f"  [10] Pathology masking FAILED: {_pm_e}")
    master["pathology_masking"] = {"error": str(_pm_e)}


# =============================================================================
# EXPERIMENT 11: CHECKPOINT RANKING (generation metric validation)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 11: CHECKPOINT RANKING")
print("="*60)

from experiments.checkpoint_ranking import run_checkpoint_ranking

cr_out = os.path.join(OUT_DIR, "checkpoint_ranking")
_CP_GEN_DIR = os.environ.get("M3_CHECKPOINT_GEN_DIR", None)
try:
    cr_res = run_checkpoint_ranking(
        real_dir           = REAL_DIR,
        gen_dir            = _GEN_DIR,
        output_dir         = cr_out,
        checkpoint_gen_dir = _CP_GEN_DIR,
        n_images           = min(200, cap),
        device             = DEVICE,
        seed               = SEED,
        backbone_id        = _args.backbone,
    )
    master["checkpoint_ranking"] = cr_res
    print(f"  [11] Checkpoint ranking done (mode={cr_res['mode']}, "
          f"FID-paradox={cr_res['fid_paradox_detected']})")
except Exception as _cr_e:
    print(f"  [11] Checkpoint ranking FAILED: {_cr_e}")
    master["checkpoint_ranking"] = {"error": str(_cr_e)}


# =============================================================================
# EXPERIMENT 12: N-SCALING (sample efficiency)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 12: N-SCALING / SAMPLE EFFICIENCY")
print("="*60)

from experiments.n_scaling import run_n_scaling

ns_out = os.path.join(OUT_DIR, "n_scaling")
try:
    ns_res = run_n_scaling(
        real_dir    = REAL_DIR,
        gen_dir     = _GEN_DIR,
        output_dir  = ns_out,
        n_list      = [25, 50, 100, 200, 500],
        n_repeats   = 10,
        device      = DEVICE,
        seed        = SEED,
        backbone_id = _args.backbone,
    )
    master["n_scaling"] = ns_res
    # Report CV at N=50 for paper
    cv_m3_50  = ns_res["summary"].get("50", {}).get("m3",  {}).get("cv", float("nan"))
    cv_fid_50 = ns_res["summary"].get("50", {}).get("fid", {}).get("cv", float("nan"))
    print(f"  [12] N-scaling done. CV@N=50: M3={cv_m3_50:.1f}%  FID={cv_fid_50:.1f}%")
except Exception as _ns_e:
    print(f"  [12] N-scaling FAILED: {_ns_e}")
    master["n_scaling"] = {"error": str(_ns_e)}


# Exps 13-19 always use the primary generated dir (GEN_DIR), not the mask-experiment dir
_GEN_DIR = GEN_DIR

# =============================================================================
# EXPERIMENT 13: COVERAGE & NOVELTY (Axes 2 & 3)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 13: COVERAGE & NOVELTY")
print("="*60)

from evaluation.coverage_novelty import compute_coverage_novelty

cn_out = os.path.join(OUT_DIR, "coverage_novelty")
os.makedirs(cn_out, exist_ok=True)
try:
    from experiments._shared_utils import load_pils_recursive
    _real_imgs_cn = load_pils_recursive(REAL_DIR, n=N_IMAGES)
    _gen_imgs_cn  = load_pils_recursive(_GEN_DIR,  n=N_IMAGES)
    cn_res = compute_coverage_novelty(
        _real_imgs_cn, _gen_imgs_cn,
        backbone_id=BACKBONE_ID,
        device=DEVICE,
        k_neighbors=5,
        pct_memorize=5.0,
        batch_size=32,
    )
    # Also evaluate WDM-3D brats if available
    cn_wdm = {}
    if os.path.isdir(_WDM3D_BRATS_DIR):
        _wdm_imgs = load_pils_recursive(_WDM3D_BRATS_DIR, n=N_IMAGES)
        cn_wdm["wdm3d_brats"] = compute_coverage_novelty(
            _real_imgs_cn, _wdm_imgs,
            backbone_id=BACKBONE_ID, device=DEVICE, batch_size=32,
        )
    import json as _json2
    _cn_report = {k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else v)
                  for k, v in cn_res.items() if not hasattr(v, "__len__") or isinstance(v, str)}
    with open(os.path.join(cn_out, "coverage_novelty_report.json"), "w") as f:
        _json2.dump({**_cn_report, "wdm3d": cn_wdm}, f, indent=2, default=str)
    master["coverage_novelty"] = {
        "coverage":          cn_res["coverage"],
        "novelty":           cn_res["novelty"],
        "memorization_rate": cn_res["memorization_rate"],
        "wdm3d_brats":       cn_wdm.get("wdm3d_brats", {}),
    }
    print(f"  [13] Coverage={cn_res['coverage']:.4f}  "
          f"Novelty={cn_res['novelty']:.4f}  "
          f"Memorization={cn_res['memorization_rate']:.4f}")
except Exception as _cn_e:
    print(f"  [13] Coverage/Novelty FAILED: {_cn_e}")
    master["coverage_novelty"] = {"error": str(_cn_e)}


# =============================================================================
# EXPERIMENT 14: STATISTICAL RIGOR (p-value, effect size, CI)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 14: STATISTICAL RIGOR")
print("="*60)

from evaluation.statistical_rigor import statistical_m3

sr_out = os.path.join(OUT_DIR, "statistical_rigor")
os.makedirs(sr_out, exist_ok=True)
try:
    from experiments._shared_utils import load_pils_recursive
    _real_sr = load_pils_recursive(REAL_DIR, n=N_IMAGES)
    _gen_sr  = load_pils_recursive(_GEN_DIR,  n=N_IMAGES)
    sr_res = statistical_m3(
        real_imgs=_real_sr, gen_imgs=_gen_sr,
        backbone_id=BACKBONE_ID, device=DEVICE,
        n_perm=500, n_boot=200, ci_level=0.95,
        batch_size=16, single_layer=12, seed=SEED,
    )
    _sr_report = {k: (float(v) if isinstance(v, (int, float)) else v)
                  for k, v in sr_res.items() if not hasattr(v, "__len__") or isinstance(v, str)}
    import json as _json3
    with open(os.path.join(sr_out, "statistical_rigor_report.json"), "w") as f:
        _json3.dump(_sr_report, f, indent=2, default=str)
    master["statistical_rigor"] = {
        "m3_score":    sr_res["m3_score"],
        "p_value":     sr_res["p_value"],
        "effect_size": sr_res["effect_size"],
        "ci_low":      sr_res["ci_low"],
        "ci_high":     sr_res["ci_high"],
    }
    print(f"  [14] p={sr_res['p_value']:.4f}  ES={sr_res['effect_size']:.2f}  "
          f"CI=[{sr_res['ci_low']:.4f}, {sr_res['ci_high']:.4f}]")
except Exception as _sr_e:
    print(f"  [14] Statistical Rigor FAILED: {_sr_e}")
    master["statistical_rigor"] = {"error": str(_sr_e)}


# =============================================================================
# EXPERIMENT 15: CONDITIONAL MMD (per-stratum worst-case)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 15: CONDITIONAL MMD")
print("="*60)

from evaluation.conditional_mmd import run_conditional_mmd

cond_out = os.path.join(OUT_DIR, "conditional_mmd")
try:
    cond_res = run_conditional_mmd(
        real_dir    = REAL_DIR,
        gen_dir     = _GEN_DIR,
        stratify_by = "intensity_quartile",
        n_strata    = 4,
        n_images    = N_IMAGES,
        backbone_id = BACKBONE_ID,
        device      = DEVICE,
        output_dir  = cond_out,
        seed        = SEED,
    )
    master["conditional_mmd"] = {
        "mean_m3":       cond_res["mean_m3"],
        "worst_m3":      cond_res["worst_m3"],
        "worst_stratum": cond_res["worst_stratum"],
        "std_m3":        cond_res["std_m3"],
    }
    print(f"  [15] Mean={cond_res['mean_m3']:.6f}  "
          f"Worst={cond_res['worst_m3']:.6f} ({cond_res['worst_stratum']})")
except Exception as _cond_e:
    print(f"  [15] Conditional MMD FAILED: {_cond_e}")
    master["conditional_mmd"] = {"error": str(_cond_e)}


# =============================================================================
# EXPERIMENT 16: NORMALITY VIOLATION (FID blind-spot)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 16: NORMALITY VIOLATION (FID blind-spot)")
print("="*60)

from experiments.normality_violation import run_normality_violation

nv_out = os.path.join(OUT_DIR, "normality_violation")
try:
    nv_res = run_normality_violation(
        real_dir    = REAL_DIR,
        gen_dir     = _GEN_DIR,
        output_dir  = nv_out,
        n_images    = N_IMAGES,
        backbone_id = BACKBONE_ID,
        device      = DEVICE,
        seed        = SEED,
    )
    master["normality_violation"] = {
        k: {
            "levels": v["levels"],
            "fid_at_max_departure": v["fid_vals"][-1],
            "m3_at_max_departure":  v["m3_vals"][-1],
        }
        for k, v in nv_res.items()
    }
    print(f"  [16] Normality violation done. "
          f"Bimodal: FID={nv_res['bimodal_shift']['fid_vals'][-1]:.4f}  "
          f"M3={nv_res['bimodal_shift']['m3_vals'][-1]:.6f}")
except Exception as _nv_e:
    print(f"  [16] Normality Violation FAILED: {_nv_e}")
    master["normality_violation"] = {"error": str(_nv_e)}


# =============================================================================
# EXPERIMENT 17: DISTORTION MONOTONICITY PER SCALE
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 17: DISTORTION MONOTONICITY PER SCALE")
print("="*60)

from experiments.distortion_monotonicity_per_scale import run_distortion_monotonicity

dm_out = os.path.join(OUT_DIR, "distortion_monotonicity_per_scale")
try:
    dm_res = run_distortion_monotonicity(
        real_dir      = REAL_DIR,
        gen_dir       = _GEN_DIR,
        output_dir    = dm_out,
        n_images      = min(N_IMAGES, 300),
        backbone_id   = BACKBONE_ID,
        device        = DEVICE,
        seed          = SEED,
        target_layers = [1, 6, 12],
    )
    # Summarise Kendall tau for paper
    import json as _json4
    _mono_path = os.path.join(dm_out, "monotonicity_check.json")
    _mono = {}
    if os.path.exists(_mono_path):
        with open(_mono_path) as f:
            _mono = _json4.load(f)
    master["distortion_monotonicity"] = {
        "corruptions_tested": list(dm_res.keys()),
        "monotonicity_check": _mono,
    }
    print(f"  [17] Distortion monotonicity done for: {list(dm_res.keys())}")
except Exception as _dm_e:
    print(f"  [17] Distortion Monotonicity FAILED: {_dm_e}")
    master["distortion_monotonicity"] = {"error": str(_dm_e)}


# =============================================================================
# EXPERIMENT 18: SAMPLE SIZE CONSISTENCY
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 18: SAMPLE SIZE CONSISTENCY")
print("="*60)

from experiments.sample_size_consistency import run_sample_size_consistency

ssc_out = os.path.join(OUT_DIR, "sample_size_consistency")
try:
    ssc_res = run_sample_size_consistency(
        real_dir    = REAL_DIR,
        gen_dir     = _GEN_DIR,
        output_dir  = ssc_out,
        n_list      = [25, 50, 100, 200, 500],
        n_repeats   = 20,
        backbone_id = BACKBONE_ID,
        device      = DEVICE,
        seed        = SEED,
    )
    # Extract CVs at N=50 for paper
    _ssc_50 = ssc_res.get("50", {})
    master["sample_size_consistency"] = {
        str(N): {m: {"mean": v["mean"], "cv": v["cv"]}
                 for m, v in vals.items()}
        for N, vals in ssc_res.items()
    }
    if _ssc_50:
        print(f"  [18] CV@N=50:  M3={_ssc_50.get('m3_mmd', {}).get('cv', '?'):.4f}  "
              f"FID={_ssc_50.get('fid', {}).get('cv', '?'):.4f}")
    else:
        print(f"  [18] Sample-size consistency done.")
except Exception as _ssc_e:
    print(f"  [18] Sample Size Consistency FAILED: {_ssc_e}")
    master["sample_size_consistency"] = {"error": str(_ssc_e)}


# =============================================================================
# EXPERIMENT 19: TSTR UTILITY (Train on Synthetic, Test on Real)
# =============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 19: TSTR UTILITY")
print("="*60)

from experiments.tstr_utility import run_tstr

tstr_out = os.path.join(OUT_DIR, "tstr_utility")

# Build generator list from available directories
_tstr_gen_dirs   = []
_tstr_gen_labels = []
if os.path.isdir(_GEN_DIR):
    _tstr_gen_dirs.append(_GEN_DIR);  _tstr_gen_labels.append("ddpm_unet")
if os.path.isdir(_WDM3D_BRATS_DIR):
    _tstr_gen_dirs.append(_WDM3D_BRATS_DIR); _tstr_gen_labels.append("wdm3d_brats")
if os.path.isdir(_WDM3D_LIDC_DIR):
    _tstr_gen_dirs.append(_WDM3D_LIDC_DIR);  _tstr_gen_labels.append("wdm3d_lidc")

try:
    if len(_tstr_gen_dirs) == 0:
        raise ValueError("No generator directories found for TSTR.")
    tstr_res = run_tstr(
        real_dir    = REAL_DIR,
        gen_dirs    = _tstr_gen_dirs,
        gen_labels  = _tstr_gen_labels,
        output_dir  = tstr_out,
        task        = "slice_position",
        n_images    = min(N_IMAGES, 500),
        epochs      = 20,
        batch_size  = 32,
        backbone_id = BACKBONE_ID,
        device      = DEVICE,
        seed        = SEED,
    )
    master["tstr_utility"] = {
        "task":              tstr_res["task"],
        "baseline_rr_acc":  tstr_res["baseline_rr_acc"],
        "spearman_m3_tstr":  tstr_res["spearman_m3_tstr"],
        "spearman_fid_tstr": tstr_res["spearman_fid_tstr"],
        "generator_summary": {
            lbl: {"m3_rank": v["m3_rank"], "fid_rank": v["fid_rank"],
                  "tstr_rank": v["tstr_rank"], "tstr_acc": v["tstr_acc"]}
            for lbl, v in tstr_res["generators"].items()
        },
    }
    print(f"  [19] TSTR done. M3 rho={tstr_res['spearman_m3_tstr']['rho']:.3f}  "
          f"FID rho={tstr_res['spearman_fid_tstr']['rho']:.3f}")
except Exception as _tstr_e:
    print(f"  [19] TSTR FAILED: {_tstr_e}")
    master["tstr_utility"] = {"error": str(_tstr_e)}


# =============================================================================
# SAVE MASTER RESULTS JSON
# =============================================================================
master_path = os.path.join(OUT_DIR, "master_results.json")
with open(master_path, "w") as f:
    json.dump(master, f, indent=4, default=str)
print(f"\n[OK] Master results saved: {master_path}")

# =============================================================================
# PAPER-READY NUMBERS SUMMARY
# =============================================================================
print("\n" + "="*60)
print("  PAPER-READY NUMBERS SUMMARY")
print("="*60)
wts = master["core_m3"]["layer_weights"]
print(f"  Active layers:             {master['active_layers']}")
print(f"  M3-Score (N={cap}):           {master['core_m3']['m3_score']:.6f}")
print(f"  Layer weights:             {wts}")
print(f"  Z-score (permutation):     {master['permutation_test']['Z_score']:.1f}")
print(f"  Empirical p (50 perm):     < {1/N_PERM:.2f}")
print(f"  M3 ROC-AUC:                {master['ood']['m3_roc_auc']:.3f}")
print(f"  InceptionV3 ROC-AUC:       {master['ood']['inceptionv3_roc_auc']:.3f}")
print(f"  CLIP-MMD ROC-AUC:          {master['ood']['clip_mmd_roc_auc']:.3f}")
print(f"  M3 vs FID improvement:     {master['ood']['m3_vs_fid_roc_pct']:+.1f}%")
print(f"  M3 vs CMMD improvement:    {master['ood']['m3_vs_cmmd_roc_pct']:+.1f}%")
print(f"  OOD rate (generated):      {master['ood']['pct_ood']:.1f}%")
print(f"  Spearman M3 vs IF:         rho={master['ood']['spearman_m3_rho']:.3f}  p={master['ood']['spearman_m3_p']:.1e}")
print(f"  Spearman Inc vs IF:        rho={master['ood']['spearman_inc_rho']:.3f}  p={master['ood']['spearman_inc_p']:.1e}")
if "pathology_masking" in master and "sensitivity_b_vs_a" in master.get("pathology_masking", {}):
    _pm = master["pathology_masking"]
    print(f"  Path-mask sensitivity M3:  {_pm['sensitivity_b_vs_a']['m3']:+.3f}")
    print(f"  Path-mask sensitivity FID: {_pm['sensitivity_b_vs_a']['fid']:+.3f}")
    if _pm.get("heatmap_iou_mean") is not None:
        print(f"  Heatmap IoU (mean):        {_pm['heatmap_iou_mean']:.3f}")
if "checkpoint_ranking" in master and "fid_paradox_detected" in master.get("checkpoint_ranking", {}):
    _cr = master["checkpoint_ranking"]
    print(f"  FID-paradox detected:      {_cr['fid_paradox_detected']}")
    print(f"  Checkpoint mode:           {_cr.get('mode', '?')}")
if "n_scaling" in master and "summary" in master.get("n_scaling", {}):
    _ns = master["n_scaling"]["summary"]
    for _n in ["50", "100", "200"]:
        if _n in _ns:
            print(f"  CV@N={_n:>4s}  M3={_ns[_n]['m3']['cv']:.1f}%  "
                  f"FID={_ns[_n]['fid']['cv']:.1f}%  "
                  f"CMMD={_ns[_n]['cmmd']['cv']:.1f}%")

# --- New tri-axial metrics ---
if "coverage_novelty" in master and "coverage" in master.get("coverage_novelty", {}):
    _cn = master["coverage_novelty"]
    print(f"  Manifold Coverage:         {_cn['coverage']:.4f}")
    print(f"  Calibrated Novelty:        {_cn['novelty']:.4f}")
    print(f"  Memorization Rate:         {_cn['memorization_rate']:.4f}")

if "statistical_rigor" in master and "p_value" in master.get("statistical_rigor", {}):
    _sr = master["statistical_rigor"]
    print(f"  Permutation p-value:       {_sr['p_value']:.4f}")
    print(f"  Effect size (Cohen d):     {_sr['effect_size']:.2f}")
    print(f"  Bootstrap 95% CI:          [{_sr['ci_low']:.4f}, {_sr['ci_high']:.4f}]")

if "conditional_mmd" in master and "worst_m3" in master.get("conditional_mmd", {}):
    _cm = master["conditional_mmd"]
    print(f"  Conditional MMD (mean):    {_cm['mean_m3']:.6f}")
    print(f"  Conditional MMD (worst):   {_cm['worst_m3']:.6f} ({_cm['worst_stratum']})")

if "tstr_utility" in master and "spearman_m3_tstr" in master.get("tstr_utility", {}):
    _tstr = master["tstr_utility"]
    print(f"  TSTR Spearman M3:          rho={_tstr['spearman_m3_tstr']['rho']:.3f}")
    print(f"  TSTR Spearman FID:         rho={_tstr['spearman_fid_tstr']['rho']:.3f}")
    print(f"  TSTR baseline (real->real): {_tstr['baseline_rr_acc']:.4f}")

print("="*60)
