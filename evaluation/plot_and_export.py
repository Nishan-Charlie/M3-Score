#!/usr/bin/env python3
"""
plot_and_export.py — Comprehensive Metric Visualisation & CSV Export
=====================================================================
Reads the evaluation_report.json, produces publication-quality plots, and
exports all results as a clean CSV file for easy inclusion in papers/reports.

Usage:
    python plot_and_export.py \
        --report_path /path/to/evaluation_report.json \
        --output_dir  /path/to/plots_and_csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.size": 11,
})

# Colour palette
PAL = {
    "primary":   "#4fc3f7",
    "secondary": "#ef9a9a",
    "accent1":   "#81c784",
    "accent2":   "#ffb74d",
    "accent3":   "#ce93d8",
    "accent4":   "#4db6ac",
    "bg_bar":    "#1a1a24",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into {dotted.key: value}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def _add_value_labels(ax, bars, fmt=".4f"):
    """Put value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:{fmt}}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, color="#ccc",
        )


# ---------------------------------------------------------------------------
# 1. Bar chart — Base metrics (FID, KID, SSIM, PSNR)
# ---------------------------------------------------------------------------

def plot_base_metrics(base: dict, out: str):
    labels = ["FID", "KID (mean)", "SSIM", "PSNR"]
    values = [
        base.get("fid", 0),
        base.get("kid_mean", 0),
        base.get("ssim", 0),
        base.get("psnr", 0),
    ]
    colours = [PAL["primary"], PAL["secondary"], PAL["accent1"], PAL["accent2"]]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
    fig.suptitle("Base Image Quality Metrics", fontsize=15, color="white", y=1.02,
                 fontweight="bold")

    for ax, label, val, col in zip(axes, labels, values, colours):
        bar = ax.bar([label], [val], color=col, width=0.5, edgecolor="none",
                     alpha=0.85, zorder=3)
        _add_value_labels(ax, bar)
        ax.set_title(label, fontsize=12, color=col, fontweight="bold")
        ax.set_ylim(0, val * 1.4 if val > 0 else 1)
        ax.grid(axis="y", zorder=0)
        ax.set_xticks([])

    plt.tight_layout()
    path = os.path.join(out, "base_metrics_bars.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Base metrics → {path}")


# ---------------------------------------------------------------------------
# 2. M3 Score Decomposition
# ---------------------------------------------------------------------------

def plot_m3_decomposition(m3: dict, out: str):
    final = m3.get("m3_v2_final_score", 0)
    layer_dists = m3.get("layer_distances", {})
    active_layers = m3.get("active_layers", [])
    
    if not layer_dists or not active_layers:
        print("  [Plot] Missing layer distances in M3-Score output, skipping decomposition.")
        return

    # Use keys in order of layers
    keys = sorted(layer_dists.keys(), key=int)
    names = [f"L{k}" for k in keys]
    vals = [layer_dists[k] for k in keys]
    
    # Layers are equally weighted (mean aggregation)
    weight = 1.0 / len(vals)
    weighted = [v * weight for v in vals]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    fig.suptitle(f"M3-Score Decomposition  (Final = {final:.6f})",
                 fontsize=15, color="white", y=1.02, fontweight="bold")

    colors = [PAL["primary"], PAL["secondary"], PAL["accent1"], PAL["accent2"], PAL["accent3"], PAL["accent4"]] * 3

    # (a) Raw MMD² per layer
    ax = axes[0]
    bars = ax.bar(names, vals, color=colors[:len(vals)], edgecolor="none", alpha=0.85, zorder=3)
    _add_value_labels(ax, bars, ".6f")
    ax.set_title("Raw MMD² per Layer", color=PAL["secondary"], fontweight="bold")
    ax.grid(axis="y", zorder=0)

    # (b) Weighted contribution
    ax = axes[1]
    bars = ax.bar(names, weighted, color=colors[:len(vals)], edgecolor="none", alpha=0.85, zorder=3)
    _add_value_labels(ax, bars, ".6f")
    ax.set_title("Weighted Contribution", color=PAL["accent2"], fontweight="bold")
    ax.grid(axis="y", zorder=0)

    # (c) Pie chart of contribution
    ax = axes[2]
    ax.set_facecolor("white")
    wedges, texts, autotexts = ax.pie(
        weighted, labels=names,
        colors=colors[:len(vals)], autopct="%.1f%%",
        startangle=140, textprops={"color": "#222222", "fontsize": 10},
        wedgeprops={"edgecolor": "#0f0f14", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("white")
    ax.set_title("Contribution Share", color=PAL["accent4"], fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out, "m3_decomposition.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] M3 decomposition → {path}")


# ---------------------------------------------------------------------------
# 3. Manifold / Precision-Recall
# ---------------------------------------------------------------------------

def plot_manifold(manifold: dict, out: str):
    labels = [
        "α-Precision", "β-Recall", "Authenticity",
        "Imp. Precision", "Imp. Recall",
    ]
    keys = [
        "alpha_precision", "beta_recall", "authenticity",
        "improved_precision", "improved_recall",
    ]
    values = [manifold.get(k, 0) for k in keys]
    colours = [PAL["primary"], PAL["secondary"], PAL["accent1"],
               PAL["accent2"], PAL["accent3"]]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.barh(labels, values, color=colours, edgecolor="none", height=0.55,
                   alpha=0.85, zorder=3)
    ax.set_xlim(0, max(max(values) * 1.5, 0.1))
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, color="#ccc")
    ax.set_title("Manifold Metrics (α-Precision / β-Recall)",
                 fontsize=14, color="white", fontweight="bold")
    ax.grid(axis="x", zorder=0)

    plt.tight_layout()
    path = os.path.join(out, "manifold_metrics.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Manifold metrics → {path}")


# ---------------------------------------------------------------------------
# 4. Extended Metrics
# ---------------------------------------------------------------------------

def plot_extended(ext: dict, out: str):
    # Group into two panels: quality scores and distributional scores
    quality_keys = ["ms_ssim", "lpips"]
    quality_names = ["MS-SSIM", "LPIPS"]
    quality_vals = [ext.get(k, 0) for k in quality_keys]

    dist_keys = ["is_mean", "coverage", "density"]
    dist_names = ["IS (mean)", "Coverage", "Density"]
    dist_vals = [ext.get(k, 0) for k in dist_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle("Extended Evaluation Metrics", fontsize=15, color="white",
                 y=1.02, fontweight="bold")

    # Quality
    bars1 = ax1.bar(quality_names, quality_vals,
                    color=[PAL["primary"], PAL["secondary"]],
                    edgecolor="none", alpha=0.85, width=0.45, zorder=3)
    _add_value_labels(ax1, bars1, ".4f")
    ax1.set_title("Image Quality", color=PAL["accent1"], fontweight="bold")
    ax1.grid(axis="y", zorder=0)
    ax1.set_ylim(0, max(max(quality_vals) * 1.3, 0.5))

    # Distribution
    bars2 = ax2.bar(dist_names, dist_vals,
                    color=[PAL["accent2"], PAL["accent3"], PAL["accent4"]],
                    edgecolor="none", alpha=0.85, width=0.45, zorder=3)
    _add_value_labels(ax2, bars2, ".4f")
    ax2.set_title("Distribution Metrics", color=PAL["accent2"], fontweight="bold")
    ax2.grid(axis="y", zorder=0)
    ax2.set_ylim(0, max(max(dist_vals) * 1.3, 0.5))

    plt.tight_layout()
    path = os.path.join(out, "extended_metrics.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Extended metrics → {path}")


# ---------------------------------------------------------------------------
# 5. Comprehensive Radar Chart
# ---------------------------------------------------------------------------

def plot_comprehensive_radar(report: dict, out: str):
    flat = _flatten(report)

    # Define candidates: (display_name, key, max_val, invert?)
    candidates = [
        ("SSIM",          "base_metrics.ssim",               1.0,  False),
        ("MS-SSIM",       "extended_metrics.ms_ssim",        1.0,  False),
        ("1-LPIPS",       "extended_metrics.lpips",           1.0,  True),
        ("α-Precision",   "manifold_metrics.alpha_precision", 1.0,  False),
        ("β-Recall",      "manifold_metrics.beta_recall",     1.0,  False),
        ("Authenticity",  "manifold_metrics.authenticity",    1.0,  False),
        ("Coverage",      "extended_metrics.coverage",        1.0,  False),
        ("Density",       "extended_metrics.density",         1.0,  False),
        ("1-M3",          "m3_score.m3_v2_final_score",       1.0,  True),
    ]

    labels, values = [], []
    for display, key, scale, invert in candidates:
        val = flat.get(key)
        if val is not None:
            nv = val / scale
            if invert:
                nv = 1.0 - nv
            nv = float(np.clip(nv, 0.0, 1.0))
            labels.append(display)
            values.append(nv)

    if len(labels) < 3:
        return

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals = values + [values[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True}, dpi=150)
    ax.fill(angles, vals, color=PAL["primary"], alpha=0.2)
    ax.plot(angles, vals, color=PAL["primary"], linewidth=2.5)
    ax.scatter(angles[:-1], vals[:-1], color=PAL["secondary"], s=70, zorder=5,
               edgecolors="white", linewidths=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color="#e0e0e0")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#555", fontsize=8)
    ax.spines["polar"].set_color("#333")
    ax.set_title("Comprehensive Evaluation Radar\n(Higher = Better, Inverted where needed)",
                 color="white", fontsize=13, pad=25, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out, "comprehensive_radar.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Comprehensive radar → {path}")


# ---------------------------------------------------------------------------
# 6. Summary Dashboard (single image with all key metrics)
# ---------------------------------------------------------------------------

def plot_summary_dashboard(report: dict, out: str):
    flat = _flatten(report)

    # Collect all key metrics
    metrics_display = [
        ("FID ↓",            "base_metrics.fid",                   ".4f"),
        ("KID (mean) ↓",     "base_metrics.kid_mean",              ".6f"),
        ("SSIM ↑",           "base_metrics.ssim",                  ".4f"),
        ("PSNR ↑",           "base_metrics.psnr",                  ".2f"),
        ("MS-SSIM ↑",        "extended_metrics.ms_ssim",           ".4f"),
        ("LPIPS ↓",          "extended_metrics.lpips",              ".4f"),
        ("IS (mean) ↑",      "extended_metrics.is_mean",           ".4f"),
        ("α-Precision ↑",    "manifold_metrics.alpha_precision",   ".4f"),
        ("β-Recall ↑",       "manifold_metrics.beta_recall",       ".4f"),
        ("Authenticity ↑",   "manifold_metrics.authenticity",      ".4f"),
        ("Coverage ↑",       "extended_metrics.coverage",          ".4f"),
        ("Density ↑",        "extended_metrics.density",           ".4f"),
        ("M3 Score ↓",       "m3_score.m3_v2_final_score",         ".6f"),
    ]

    available = [(name, flat.get(key), fmt) for name, key, fmt in metrics_display if key in flat]

    fig, ax = plt.subplots(figsize=(12, max(len(available) * 0.45, 4)), dpi=150)
    ax.axis("off")

    y_positions = np.linspace(0.95, 0.05, len(available))
    for i, (name, val, fmt) in enumerate(available):
        y = y_positions[i]
        # Name
        ax.text(0.02, y, name, transform=ax.transAxes, fontsize=12,
                color=PAL["primary"], fontweight="bold", va="center")
        # Value
        val_str = f"{val:{fmt}}"
        ax.text(0.55, y, val_str, transform=ax.transAxes, fontsize=13,
                color="white", fontweight="bold", va="center",
                family="monospace")
        # Horizontal separator line
        ax.plot([0.01, 0.99], [y - 0.015, y - 0.015], color="#222",
                linewidth=0.5, transform=ax.transAxes, clip_on=False)

    ax.set_title("📊  Evaluation Summary Dashboard  —  500 Generated Images",
                 fontsize=16, color="white", fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(out, "summary_dashboard.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Summary dashboard → {path}")


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_csv(report: dict, out: str):
    flat = _flatten(report)

    # Define ordered rows for clean CSV
    rows = [
        ("Category", "Metric", "Value"),
        # Base
        ("Base", "FID", flat.get("base_metrics.fid", "")),
        ("Base", "KID (mean)", flat.get("base_metrics.kid_mean", "")),
        ("Base", "KID (std)", flat.get("base_metrics.kid_std", "")),
        ("Base", "SSIM", flat.get("base_metrics.ssim", "")),
        ("Base", "PSNR", flat.get("base_metrics.psnr", "")),
        # Manifold
        ("Manifold", "Alpha Precision", flat.get("manifold_metrics.alpha_precision", "")),
        ("Manifold", "Beta Recall", flat.get("manifold_metrics.beta_recall", "")),
        ("Manifold", "Authenticity", flat.get("manifold_metrics.authenticity", "")),
        ("Manifold", "Improved Precision", flat.get("manifold_metrics.improved_precision", "")),
        ("Manifold", "Improved Recall", flat.get("manifold_metrics.improved_recall", "")),
        # Extended
        ("Extended", "MS-SSIM", flat.get("extended_metrics.ms_ssim", "")),
        ("Extended", "LPIPS", flat.get("extended_metrics.lpips", "")),
        ("Extended", "IS (mean)", flat.get("extended_metrics.is_mean", "")),
        ("Extended", "IS (std)", flat.get("extended_metrics.is_std", "")),
        ("Extended", "Coverage", flat.get("extended_metrics.coverage", "")),
        ("Extended", "Density", flat.get("extended_metrics.density", "")),
        # M3
        ("M3-Score", "M3 Final Score", flat.get("m3_score.m3_v2_final_score", "")),
        ("M3-Score", "Active Layers", str(flat.get("m3_score.active_layers", ""))),
        # Meta
        ("Meta", "Total Evaluation Time (s)", flat.get("elapsed_seconds", "")),
        ("Meta", "Number of Images", flat.get("config.num_images", "")),
    ]

    csv_path = os.path.join(out, "evaluation_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  [CSV]  All metrics → {csv_path}")

    # Also export a flat single-row CSV (useful for comparison tables)
    flat_csv_path = os.path.join(out, "evaluation_metrics_flat.csv")
    with open(flat_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        vals = []
        for row in rows[1:]:  # skip header
            header.append(f"{row[0]}_{row[1]}")
            vals.append(row[2])
        writer.writerow(header)
        writer.writerow(vals)
    print(f"  [CSV]  Flat metrics → {flat_csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot and export evaluation results.")
    parser.add_argument("--report_path", required=True,
                        help="Path to evaluation_report.json")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for plots/CSV (defaults to report's parent dir)")
    args = parser.parse_args()

    if not os.path.exists(args.report_path):
        print(f"Error: {args.report_path} not found")
        sys.exit(1)

    with open(args.report_path) as f:
        report = json.load(f)

    out = args.output_dir or os.path.dirname(args.report_path)
    os.makedirs(out, exist_ok=True)

    print("=" * 60)
    print("  Generating plots and CSV exports")
    print("=" * 60)

    # Generate all plots
    if "base_metrics" in report:
        plot_base_metrics(report["base_metrics"], out)

    if "m3_score" in report and "error" not in report["m3_score"]:
        plot_m3_decomposition(report["m3_score"], out)

    if "manifold_metrics" in report and "error" not in report["manifold_metrics"]:
        plot_manifold(report["manifold_metrics"], out)

    if "extended_metrics" in report and "error" not in report["extended_metrics"]:
        plot_extended(report["extended_metrics"], out)

    plot_comprehensive_radar(report, out)
    plot_summary_dashboard(report, out)
    export_csv(report, out)

    print("\n" + "=" * 60)
    print("  All plots and CSV files generated!")
    print(f"  Output directory: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
