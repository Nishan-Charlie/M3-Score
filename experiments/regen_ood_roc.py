"""
experiments/regen_ood_roc.py
Regenerate the stale OOD ROC figure (figures/OOD/ood_roc_auc.png). The on-disk figure showed
AUC 0.885/0.785 and no CLIP curve, contradicting the paper's stated 0.974/0.778/0.645.
Reproduces the centroid-L2 (real=0 / gen=1) ROC-AUC for the three backbones and re-plots.
"""
import os, sys, glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

N = int(os.environ.get("OOD_N", "500"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def paths(d, n):
    ps = []
    for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        ps.extend(glob.glob(os.path.join(d, "**", e), recursive=True))
    ps = [p for p in sorted(ps) if "_segmask_" not in os.path.basename(p)]
    return ps[:n]

real_p = paths(os.path.join(ROOT, "data_mri", "brats_axial_multislice"), N)
gen_p  = paths(os.path.join(ROOT, "output", "generated_500_standard"), N)
print(f"real={len(real_p)} gen={len(gen_p)}")

def centroid_scores(fr, fg, normalize=True):
    """Distance to real centroid (higher = more generated-like). Returns labels, scores.
    normalize=True: 1 - cosine on L2-normalized features (== centroid-L2 on unit sphere).
    normalize=False: raw Euclidean distance to raw real centroid."""
    fr, fg = fr.astype(np.float64), fg.astype(np.float64)
    if normalize:
        def l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        r, g = l2n(fr), l2n(fg); c = r.mean(0); c = c / (np.linalg.norm(c) + 1e-8)
        sr = 1 - r @ c; sg = 1 - g @ c
    else:
        c = fr.mean(0)
        sr = np.linalg.norm(fr - c, axis=1); sg = np.linalg.norm(fg - c, axis=1)
    return np.r_[np.zeros(len(sr)), np.ones(len(sg))], np.r_[sr, sg]

from sklearn.metrics import roc_auc_score, roc_curve
results = {}

def record(name, fr, fg):
    ln, sn = centroid_scores(fr, fg, True)
    lr, sr = centroid_scores(fr, fg, False)
    an, ar = roc_auc_score(ln, sn), roc_auc_score(lr, sr)
    print(f"  {name:26s} AUC norm={an:.3f}  raw={ar:.3f}")
    results[name] = (ar, *roc_curve(lr, sr)[:2])  # store RAW centroid-L2 for the plot

# ---- RadioDINO-s16 L12 (M3) ----
from evaluation.m3_score_v2 import M3EntropyMetric
t01 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
load = lambda ps: torch.stack([t01(Image.open(p).convert("RGB")) for p in ps])
m = M3EntropyMetric(device=DEV, single_layer=12)
L = m.num_layers
fr = m._extract_raw_features(load(real_p).to(DEV), layers_to_keep={L})[L-1].cpu().numpy()
fg = m._extract_raw_features(load(gen_p).to(DEV), layers_to_keep={L})[L-1].cpu().numpy()
record("RadioDINO-s16 (M3)", fr, fg)
del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ---- InceptionV3 (FID) ----
from torchvision import models
def inception_feats(ps, batch=32):
    net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    net.fc = torch.nn.Identity(); net.eval().to(DEV)
    mean = torch.tensor([0.485,0.456,0.406], device=DEV).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225], device=DEV).view(1,3,1,1)
    out = []
    with torch.no_grad():
        for i in range(0, len(ps), batch):
            x = load(ps[i:i+batch]).to(DEV)
            x = torch.nn.functional.interpolate(x, size=(299,299), mode="bilinear", align_corners=False)
            out.append(net((x-mean)/std).cpu().numpy())
    return np.concatenate(out, 0)
fr = inception_feats(real_p); fg = inception_feats(gen_p)
record("InceptionV3 (FID)", fr, fg)
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ---- CLIP ViT-L/14 (CMMD) ----
from evaluation.cmmd_metric import CMMDMetric
cm = CMMDMetric(device=DEV)
fr = cm.extract_features(real_p, "real"); fg = cm.extract_features(gen_p, "gen")
record("CLIP ViT-L/14 (CMMD)", fr, fg)

print("\n=== centroid-L2 ROC-AUC (real vs generated, N=%d) ===" % N)
for k, v in results.items():
    print(f"  {k:26s} AUC = {v[0]:.3f}")

# ---- plot ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
colors = {"RadioDINO-s16 (M3)": "#1565C0", "InceptionV3 (FID)": "#e65100", "CLIP ViT-L/14 (CMMD)": "#6a1a9a"}
ax.plot([0,1],[0,1], "--", color="#999999", label="Random classifier")
for k in ["CLIP ViT-L/14 (CMMD)", "InceptionV3 (FID)", "RadioDINO-s16 (M3)"]:
    auc, fpr, tpr = results[k]
    ax.plot(fpr, tpr, color=colors[k], lw=2.4, label=f"{k}   AUC = {auc:.3f}")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC: Real vs Generated Separation\n(centroid-$L_2$ distance, $N=%d$)" % N, fontsize=13)
ax.legend(loc="lower right", fontsize=10, frameon=True)
ax.grid(True, alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
fig.tight_layout()
out = os.path.join(ROOT, "figures", "OOD", "ood_roc_auc.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
