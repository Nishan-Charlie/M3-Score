"""
Builder that generates the MRI-Diffuser notebook suite with nbformat.
Run once:  python notebooks/_build_notebooks.py
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))


def md(s):
    return new_markdown_cell(s.strip("\n"))


def code(s):
    return new_code_cell(s.strip("\n"))


def save(name, cells):
    nb = new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


# ===========================================================================
# 00 — INDEX
# ===========================================================================
save("00_index.ipynb", [
    md(r"""
# MRI-Diffuser · Notebook Suite

Interactive notebooks for running experiments and exploring results for the
**M3-Score (Multi-Scale Medical-MMD)** project.

| Notebook | What it does |
|---|---|
| **00_index** (this) | Overview, environment check, shared config |
| **01_explore_data** | View real vs. generated MRI images, build collages |
| **02_compute_metrics** | Compute M3-Score / FID / KID / SSIM on a real↔gen pair |
| **03_run_experiments** | Launch any experiment script with a friendly UI |
| **04_results_dashboard** | Load + visualize every `*_report.json` already on disk |

All notebooks share **`nb_config.py`** for paths, image loading, and plotting.
Edit the default `REAL_DIR` / `GEN_DIR` there once and every notebook follows.
"""),
    md("## 1 · Environment check\nRun this to confirm the project is found and see image counts + device."),
    code(r"""
import nb_config as C
C.setup()
"""),
    md("## 2 · Dependency sanity check\nVerifies the key packages are importable before you run anything heavy."),
    code(r"""
import importlib
pkgs = ["torch", "torchmetrics", "transformers", "PIL", "matplotlib", "numpy", "sklearn"]
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f"  ok   {p:<14} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  MISS {p:<14} -> {e}")
"""),
    md("## 3 · What's on disk\nQuick inventory of generated-image folders and existing result reports."),
    code(r"""
import os
print("Generated-image folders under output/:")
out = os.path.join(C.PROJECT_ROOT, "output")
for d in sorted(os.listdir(out)) if os.path.isdir(out) else []:
    full = os.path.join(out, d)
    if os.path.isdir(full):
        print(f"  {d:<28} {C.count_images(full):>6} imgs")

print("\nExisting result reports:")
for r in C.find_reports():
    print("  ", os.path.relpath(r, C.PROJECT_ROOT))
"""),
    md("---\nNext: open **01_explore_data.ipynb** to look at the images, or jump to **04_results_dashboard.ipynb** if you already have results."),
])


# ===========================================================================
# 01 — EXPLORE DATA
# ===========================================================================
save("01_explore_data.ipynb", [
    md(r"""
# 01 · Explore Data — Real vs. Generated MRI

Visually compare the real training distribution against generated samples.
Use this to sanity-check a generator before spending time on metrics.
"""),
    code(r"""
import nb_config as C
C.setup()
"""),
    md("## Real images"),
    code(r"""
C.show_collage(C.REAL_DIR, n=16, ncols=4, title="Real MRI (BraTS axial)")
"""),
    md("## Generated images"),
    code(r"""
C.show_collage(C.GEN_DIR, n=16, ncols=4, title="Generated")
"""),
    md("## Compare any two folders side by side\nChange the paths below to compare, e.g. two different checkpoints."),
    code(r"""
import os
folder_a = C.REAL_DIR
folder_b = C.GEN_DIR   # try: os.path.join(C.PROJECT_ROOT, "output", "generated_retinal")

C.show_collage(folder_a, n=8, ncols=4, title=f"A: {os.path.basename(folder_a)}")
C.show_collage(folder_b, n=8, ncols=4, title=f"B: {os.path.basename(folder_b)}")
"""),
    md("## Pixel-intensity histograms\nDistribution overlap is a cheap first signal of how realistic samples are."),
    code(r"""
import numpy as np, matplotlib.pyplot as plt
from PIL import Image

def intensities(directory, n=64):
    vals = []
    for p in C.list_images(directory, n):
        vals.append(np.asarray(Image.open(p).convert("L"), dtype=np.float32).ravel())
    return np.concatenate(vals) if vals else np.array([])

real = intensities(C.REAL_DIR)
gen  = intensities(C.GEN_DIR)
plt.figure(figsize=(8, 4))
plt.hist(real, bins=80, alpha=0.5, density=True, label="real")
plt.hist(gen,  bins=80, alpha=0.5, density=True, label="generated")
plt.xlabel("pixel intensity (0-255)"); plt.ylabel("density")
plt.legend(); plt.title("Pixel-intensity distribution"); plt.tight_layout(); plt.show()
"""),
])


# ===========================================================================
# 02 — COMPUTE METRICS
# ===========================================================================
save("02_compute_metrics.ipynb", [
    md(r"""
# 02 · Compute Metrics — M3-Score, FID, KID, SSIM

Compute evaluation metrics directly in-notebook for a real↔generated pair.
This mirrors `evaluation/eval_pipeline.py` but lets you inspect intermediates
(per-layer MMD, weights, entropy) — the parts that make M3 interpretable.

> **Tip:** Start with a small `N` (e.g. 64) on CPU to smoke-test, then bump up.
"""),
    code(r"""
import nb_config as C
C.setup()
import torch
DEVICE = C.detect_device()
N = 64          # images per split — raise to 500 for paper-grade numbers
print("device:", DEVICE, "| N:", N)
"""),
    md("## Load images as `[N,3,224,224]` in `[0,1]`\nM3 / RadioDino expect raw `[0,1]` tensors and normalize internally."),
    code(r"""
import glob, os
from PIL import Image
from torchvision import transforms

_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def load_m3(directory, n):
    paths = C.list_images(directory, n)
    return torch.stack([_tf(Image.open(p).convert("RGB")) for p in paths])

real = load_m3(C.REAL_DIR, N).to(DEVICE)
gen  = load_m3(C.GEN_DIR,  N).to(DEVICE)
print("real:", tuple(real.shape), "| gen:", tuple(gen.shape))
"""),
    md("## M3-Score (the core metric)\nPrunes layers via CKA on a small real subset, then scores. Returns the final\nscore plus per-layer distances / weights so you can see *why* it moved."),
    code(r"""
from evaluation.m3_score_v2 import M3V2Metric

m3 = M3V2Metric(device=DEVICE)
m3.prune_layers_via_cka(real[:20])
res = m3(real, gen)

print(f"\nM3-Score = {res['m3_score']:.5f}")
print("active layers:", res.get("active_layers"))
import pandas as pd
def _row(d): return {f"L{k}": round(v, 4) for k, v in (d or {}).items()}
pd.DataFrame({
    "distance":   _row(res.get("layer_distances")),
    "weight":     _row(res.get("layer_weights")),
    "semanticity":_row(res.get("semanticity")),
    "stability":  _row(res.get("stability")),
    "uniqueness": _row(res.get("uniqueness")),
}).T
"""),
    md("## FID & KID (Inception baselines)\nFor comparison against M3. Uses torchmetrics; needs uint8 `[0,255]` inputs."),
    code(r"""
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

def load_uint8(directory, n, size=299):
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.PILToTensor()])
    paths = C.list_images(directory, n)
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])

real_u = load_uint8(C.REAL_DIR, N)
gen_u  = load_uint8(C.GEN_DIR,  N)

fid = FrechetInceptionDistance(normalize=False)
fid.update(real_u, real=True); fid.update(gen_u, real=False)
print(f"FID = {float(fid.compute()):.3f}")

subset = min(50, N)
kid = KernelInceptionDistance(subset_size=subset, normalize=False)
kid.update(real_u, real=True); kid.update(gen_u, real=False)
km, ks = kid.compute()
print(f"KID = {float(km):.5f} ± {float(ks):.5f}")
"""),
    md("## One-call full pipeline (optional)\nRuns the project's official multi-metric CLI and writes a report + radar plot\nto `results/notebook_eval/`. Slower but authoritative."),
    code(r"""
# Uncomment to run the official pipeline end-to-end:
# C.run_script("evaluation/eval_pipeline.py", [
#     "--real_dir", C.REAL_DIR,
#     "--gen_dir",  C.GEN_DIR,
#     "--output_dir", "results/notebook_eval",
#     "--metrics", "fid", "kid", "m3", "ssim",
#     "--num_images", str(N),
#     "--device", DEVICE,
# ])
"""),
])


# ===========================================================================
# 03 — RUN EXPERIMENTS
# ===========================================================================
save("03_run_experiments.ipynb", [
    md(r"""
# 03 · Run Experiments

Launch any of the experiment scripts in `experiments/` (or the master
orchestrator) with a clean interface. Output streams live into the notebook
and each script writes its own `*_report.json` + PNGs under `results/`.

Pick a row from the catalog, set the args, run the cell.
"""),
    code(r"""
import nb_config as C
C.setup()
DEVICE = C.detect_device()
N = 200    # images per split for the experiments below
"""),
    md("## Experiment catalog\nThe most useful, paper-mapped scripts. (`experiments/` has ~30 total — list them with the next cell.)"),
    code(r"""
import os
catalog = {
    "noise_robustness":       "Metric monotonicity under Gaussian noise / blur (Sec 3.5)",
    "ood_detection":          "OOD / anomaly detection AUC: M3 vs FID (Sec 3.6)",
    "hallucination_detection":"Pathology-masking sensitivity, M3 vs FID (Sec 3.11)",
    "interpretability":       "Radiomic + RadioDino feature decomposition (Sec 3.10)",
    "feature_orthogonality":  "Non-redundancy of the 3 scales (Sec 3.10)",
    "weight_ablation":        "Optimal layer weights ablation (Sec 3.8)",
    "permutation_test":       "Statistical significance via null distribution (Sec 3.7)",
    "n_scaling":              "Sample-size stability / coefficient of variation",
    "noise_quality_ladder":   "M3/FID/KID/CMMD vs increasing noise sigma",
}
for k, v in catalog.items():
    exists = os.path.exists(os.path.join(C.EXP_DIR, k + ".py"))
    print(f"  [{'x' if exists else ' '}] {k:<26} {v}")

print("\nAll scripts in experiments/:")
print("  " + ", ".join(sorted(f[:-3] for f in os.listdir(C.EXP_DIR)
                              if f.endswith('.py') and not f.startswith('_'))))
"""),
    md("## Run a single experiment\nMost scripts accept `--real_dir --gen_dir --output_dir --num_images --device`.\nNoise/robustness scripts only need `--real_dir` (they degrade the reals themselves)."),
    code(r"""
EXPERIMENT = "noise_robustness"          # <- change me
OUT = f"results/notebook/{EXPERIMENT}"

args = [
    "--real_dir",   C.REAL_DIR,
    "--output_dir", OUT,
    "--num_images", str(N),
    "--device",     DEVICE,
]
# Scripts that also need generated images:
if EXPERIMENT in {"ood_detection", "hallucination_detection",
                  "interpretability", "feature_orthogonality"}:
    args[2:2] = ["--gen_dir", C.GEN_DIR]

C.run_script(f"experiments/{EXPERIMENT}.py", args)
"""),
    md("## Show what it produced\nLoads the JSON report and renders any PNGs the script saved."),
    code(r"""
import os, glob
out_abs = os.path.join(C.PROJECT_ROOT, OUT)
reports = glob.glob(os.path.join(out_abs, "**", "*.json"), recursive=True)
for r in reports:
    print("==", os.path.relpath(r, C.PROJECT_ROOT))
    try:
        d = C.load_json(r)
        keys = list(d.keys()) if isinstance(d, dict) else f"list[{len(d)}]"
        print("   keys:", keys)
    except Exception as e:
        print("   (could not parse)", e)

pngs = glob.glob(os.path.join(out_abs, "**", "*.png"), recursive=True)
C.show_pngs(pngs, ncols=2, max_imgs=6)
"""),
    md("## Master orchestrator (everything at once)\nRuns the whole suite. This is **slow** — uncomment only when you mean it."),
    code(r"""
# C.run_script("evaluation/run_experiments.py", [
#     "--real_dir",   C.REAL_DIR,
#     "--gen_dir",    C.GEN_DIR,
#     "--output_dir", "results/notebook_full",
#     "--experiments", "all",
#     "--num_images", str(N),
#     "--device",     DEVICE,
# ])
"""),
])


# ===========================================================================
# 04 — RESULTS DASHBOARD
# ===========================================================================
save("04_results_dashboard.ipynb", [
    md(r"""
# 04 · Results Dashboard

Aggregate and visualize results that already exist under `results/` —
no recomputation. Good for reading off the numbers that go into the paper.
"""),
    code(r"""
import nb_config as C
C.setup()
import json, os
import pandas as pd
import matplotlib.pyplot as plt
"""),
    md("## All reports found"),
    code(r"""
reports = C.find_reports()
for r in reports:
    print("  ", os.path.relpath(r, C.PROJECT_ROOT))
print(f"\n{len(reports)} report(s).")
"""),
    md("## Sample-size stability (coefficient of variation)\nThe headline robustness claim: M3 should have far lower CV than FID at small N."),
    code(r"""
p = os.path.join(C.RESULTS_DIR, "n_scaling", "n_scaling_report.json")
if os.path.exists(p):
    d = C.load_json(p)
    rows = []
    for n, metrics in d["summary"].items():
        row = {"N": int(n)}
        for m, stat in metrics.items():
            row[f"{m}_cv"] = stat.get("cv")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("N").set_index("N")
    display(df)
    df.plot(marker="o", figsize=(8, 4), title="Coefficient of variation vs N (lower = more stable)")
    plt.ylabel("CV (%)"); plt.tight_layout(); plt.show()
else:
    print("Run experiments/n_scaling.py first (see notebook 03).")
"""),
    md("## Noise-quality ladder\nEach metric vs. injected Gaussian sigma. Monotonic, well-spread curves are good.\nNormalized so all metrics share one axis."),
    code(r"""
p = os.path.join(C.RESULTS_DIR, "noise_quality_ladder", "results.json")
if os.path.exists(p):
    rows = C.load_json(p)
    df = pd.DataFrame(rows)
    display(df)
    plt.figure(figsize=(8, 4))
    for col in ["m3", "fid", "kid", "cmmd"]:
        if col in df:
            v = df[col].astype(float)
            norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
            plt.plot(df["sigma"], norm, marker="o", label=col)
    plt.xlabel("noise sigma"); plt.ylabel("normalized metric")
    plt.legend(); plt.title("Metric response to noise (normalized)")
    plt.tight_layout(); plt.show()
else:
    print("Run experiments/noise_quality_ladder.py first.")
"""),
    md("## OOD / anomaly detection AUC\nKey claim: M3 separates anomalies better than FID."),
    code(r"""
import glob
cands = glob.glob(os.path.join(C.RESULTS_DIR, "**", "*ood*", "**", "*.json"), recursive=True)
cands += glob.glob(os.path.join(C.RESULTS_DIR, "**", "*ood*.json"), recursive=True)
for c in sorted(set(cands)):
    try:
        d = C.load_json(c)
        flat = {k: v for k, v in (d.items() if isinstance(d, dict) else [])
                if isinstance(v, (int, float))}
        print(os.path.relpath(c, C.PROJECT_ROOT))
        for k, v in flat.items():
            if "auc" in k.lower():
                print(f"    {k}: {v}")
    except Exception:
        pass
"""),
    md("## Browse any report\nPaste a path from the list above to dump its full contents + PNGs."),
    code(r"""
target = reports[0] if reports else None   # <- or set a path string
if target:
    print(os.path.relpath(target, C.PROJECT_ROOT), "\n")
    print(json.dumps(C.load_json(target), indent=2)[:2000])
    subdir = os.path.relpath(os.path.dirname(target), C.RESULTS_DIR)
    C.show_pngs(C.find_pngs(subdir), ncols=2, max_imgs=6)
"""),
    md("## Browse result figures by folder\nList of result subfolders, then render the PNGs in whichever you pick."),
    code(r"""
subdirs = sorted(d for d in os.listdir(C.RESULTS_DIR)
                 if os.path.isdir(os.path.join(C.RESULTS_DIR, d)))
print("Result folders:", subdirs)

PICK = subdirs[0] if subdirs else ""     # <- change to any folder above
C.show_pngs(C.find_pngs(PICK), ncols=2, max_imgs=8)
"""),
])

print("\nAll notebooks built.")
