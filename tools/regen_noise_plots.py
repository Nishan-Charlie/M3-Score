"""Regenerate noise_robustness plots from existing master JSON report with white background."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

report_path = "results/experiments_output_v5/noise_robustness/robustness_report.json"
cmmd_report_path = "results/experiments_output_v5/cmmd_sweep/cmmd_robustness_report.json"
output_dir  = "results/experiments_output_v5/noise_robustness"
os.makedirs(output_dir, exist_ok=True)

with open(report_path) as f:
    report = json.load(f)

# Load CMMD sweep results and build lookup by level value
cmmd_noise_lookup = {}
cmmd_blur_lookup  = {}
if os.path.exists(cmmd_report_path):
    with open(cmmd_report_path) as f:
        cmmd_report = json.load(f)
    for row in cmmd_report["noise"]["metric_values"]:
        cmmd_noise_lookup[float(row["sigma"])] = row["cmmd"]
    for row in cmmd_report["blur"]["metric_values"]:
        cmmd_blur_lookup[float(row["radius"])] = row["cmmd"]
    cmmd_noise_spearman = cmmd_report["noise"]["spearman"]
    cmmd_blur_spearman  = cmmd_report["blur"]["spearman"]
else:
    cmmd_noise_spearman = cmmd_blur_spearman = None

colors = {
    "m3":      "#1565C0",
    "fid":     "#e65100",
    "kid":     "#6a1a9a",
    "cmmd":    "#00695c",
    "ssim":    "#2e7d32",
    "psnr":    "#00838f",
    "ms_ssim": "#f57f17",
    "lpips":   "#b71c1c",
    "precision": "#ad1457",
    "recall":    "#26c6da",
}

metric_names = ["m3", "fid", "kid", "cmmd", "ssim", "psnr", "ms_ssim", "lpips", "precision", "recall"]


def _white_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="#333333")
    ax.grid(True, alpha=0.4)


def make_plot(sweep, level_key, corr, prefix, xlabel, cmmd_lookup=None, cmmd_spearman=None):
    levels = np.array([r[level_key] for r in sweep])

    # Inject CMMD values into each sweep row (or NaN if not available)
    augmented = []
    for r in sweep:
        row = dict(r)
        lvl = float(r[level_key])
        row["cmmd"] = cmmd_lookup.get(lvl, float("nan")) if cmmd_lookup else float("nan")
        augmented.append(row)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]; _white_ax(ax)
    for m in ["m3", "fid", "kid", "cmmd"]:
        vals = [r.get(m, float("nan")) for r in augmented]
        label = "CMMD" if m == "cmmd" else m.upper()
        ax.plot(levels, vals, "o-", color=colors[m], label=label, linewidth=2, markersize=6)
    ax.set_xlabel(xlabel, color="#222222", fontsize=11)
    ax.set_ylabel("Metric value", color="#222222", fontsize=11)
    ax.set_title("Distributional metrics", color="#222222", fontsize=13)
    ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc")

    ax = axes[0, 1]; _white_ax(ax)
    for m in ["ssim", "psnr", "ms_ssim", "lpips"]:
        vals = [r.get(m, float("nan")) for r in augmented]
        ax.plot(levels, vals, "o-", color=colors[m], label=m.upper(), linewidth=2, markersize=6)
    ax.set_xlabel(xlabel, color="#222222", fontsize=11)
    ax.set_ylabel("Metric value", color="#222222", fontsize=11)
    ax.set_title("Perceptual metrics", color="#222222", fontsize=13)
    ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc")

    ax = axes[1, 0]; _white_ax(ax)
    for m in metric_names:
        vals = np.array([r.get(m, float("nan")) for r in augmented])
        valid = ~np.isnan(vals)
        if valid.sum() >= 2:
            vmin, vmax = vals[valid].min(), vals[valid].max()
            normed = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
            label = "CMMD" if m == "cmmd" else m.upper()
            ax.plot(levels[valid], normed[valid], "o-", color=colors[m],
                    label=label, linewidth=2, markersize=5)
    ax.set_xlabel(xlabel, color="#222222", fontsize=11)
    ax.set_ylabel("Normalised metric (0 to 1)", color="#222222", fontsize=11)
    ax.set_title("All metrics normalised", color="#222222", fontsize=13)
    ax.legend(fontsize=8, facecolor="white", edgecolor="#cccccc", ncol=2)

    ax = axes[1, 1]
    ax.set_facecolor("white")
    ax.tick_params(colors="#333333")
    # Build extended corr dict including CMMD
    extended_corr = dict(corr)
    if cmmd_spearman is not None:
        extended_corr["cmmd"] = cmmd_spearman
    corr_ms = list(extended_corr.keys())
    corr_vs = [abs(extended_corr[m]["spearman_r"]) for m in corr_ms]
    bar_labels = ["CMMD" if m == "cmmd" else m.upper() for m in corr_ms]
    bars = ax.barh(bar_labels, corr_vs,
                   color=[colors.get(m, "#888") for m in corr_ms],
                   edgecolor="#cccccc", height=0.6)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, color="#222222", padding=4)
    ax.set_xlabel("|Spearman rho|", color="#222222", fontsize=11)
    ax.set_title("Metric sensitivity", color="#222222", fontsize=13)
    ax.set_xlim(0, 1.1)

    plt.suptitle(f"M3-Score Robustness: {prefix.capitalize()} Sweep",
                 fontsize=14, fontweight="bold", color="#222222")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_robustness.png")
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


noise_data = report["noise"]
blur_data  = report["blur"]

# Determine level key name
noise_row0 = noise_data["metric_values"][0] if noise_data["metric_values"] else {}
level_key_noise = "sigma" if "sigma" in noise_row0 else list(noise_row0.keys())[0]

blur_row0 = blur_data["metric_values"][0] if blur_data["metric_values"] else {}
level_key_blur = "radius" if "radius" in blur_row0 else list(blur_row0.keys())[0]

make_plot(noise_data["metric_values"], level_key_noise, noise_data["spearman"],
          "noise", "Noise level sigma",
          cmmd_lookup=cmmd_noise_lookup, cmmd_spearman=cmmd_noise_spearman)
make_plot(blur_data["metric_values"],  level_key_blur,  blur_data["spearman"],
          "blur",  "Blur radius (pixels)",
          cmmd_lookup=cmmd_blur_lookup, cmmd_spearman=cmmd_blur_spearman)

print("Done.")
