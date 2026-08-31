"""Regenerate the CKA-threshold ablation figure from its saved report JSON.

No GPU and no metric evaluation: the plot is rebuilt from
weight_ablation_report.json, so cosmetic fixes do not require re-running
experiments/weight_ablation.py. Mirrors tools/regen_noise_plots.py.

The legend is placed below each panel rather than inside it, so it never
overlaps the curves.

Usage
-----
    python tools/regen_weight_ablation_plot.py
    python tools/regen_weight_ablation_plot.py \\
        --report results/radiodino-s16_run5/weight_ablation/weight_ablation_report.json \\
        --out    paper/Images/weight_abalation/weight_ablation_discriminability.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_REPORT = "results/radiodino-s16_run5/weight_ablation/weight_ablation_report.json"
DEFAULT_OUT = "paper/Images/weight_abalation/weight_ablation_discriminability.png"

COLOR_DISC = "#1565C0"
COLOR_LAYERS = "#c62828"
COLOR_RG = "#e65100"
COLOR_RR = "#2e7d32"
COLOR_BEST = "#6a1a9a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    rows = sorted(report["rows"], key=lambda r: r["threshold"])
    best_tau = report["best_threshold"]

    thresholds = [r["threshold"] for r in rows]
    discs = [r["discriminability"] for r in rows]
    n_layers = [r["n_active_layers"] for r in rows]
    scores_rg = [r["score_rg"] for r in rows]
    scores_rr = [r["score_rr"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    fig.patch.set_facecolor("white")

    # ---- Panel 1: discriminability + active layer count ----
    ax1 = axes[0]
    ax1.set_facecolor("white")
    ax2 = ax1.twinx()
    ax1.plot(thresholds, discs, "o-", color=COLOR_DISC, lw=2.5, markersize=8,
             label="Discriminability", zorder=3)
    ax1.axvline(best_tau, color=COLOR_BEST, lw=1.8, linestyle="--",
                label=f"Best $\\tau$ = {best_tau}", zorder=2)
    ax1.set_xlabel("CKA threshold $\\tau$", fontsize=11, color="#222222")
    ax1.set_ylabel("Discriminability  $(S_\\mathrm{RG} - S_\\mathrm{RR})\\ /\\ S_\\mathrm{RR}$",
                   color=COLOR_DISC, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=COLOR_DISC)
    ax1.tick_params(axis="x", colors="#333333")
    ax1.set_title("Discriminability vs.\\ CKA threshold $\\tau$",
                  fontsize=11, color="#222222")
    ax1.grid(alpha=0.35, color="#dddddd")
    ax2.bar(thresholds, n_layers, width=0.015, color=COLOR_LAYERS,
            alpha=0.30, label="Active layers $M$", zorder=1)
    ax2.set_ylabel("Active layers retained ($M$)", color=COLOR_LAYERS, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=COLOR_LAYERS)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
               facecolor="white", edgecolor="#cccccc",
               loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=3, borderaxespad=0.0)

    # ---- Panel 2: raw RG / RR scores ----
    ax = axes[1]
    ax.set_facecolor("white")
    ax.plot(thresholds, scores_rg, "o-", color=COLOR_RG, lw=2.5,
            markersize=8, label="$S_{\\mathrm{RG}}$ (real vs generated)")
    ax.plot(thresholds, scores_rr, "s--", color=COLOR_RR, lw=2.5,
            markersize=8, label="$S_{\\mathrm{RR}}$ (real vs real)")
    ax.axvline(best_tau, color=COLOR_BEST, lw=1.8, linestyle="--",
               label=f"Best $\\tau$ = {best_tau}")
    ax.set_xlabel("CKA threshold $\\tau$", fontsize=11, color="#222222")
    ax.set_ylabel("M3 score (equal-weight)", fontsize=11, color="#222222")
    ax.set_title("Raw $S_\\mathrm{RG}$ and $S_\\mathrm{RR}$ vs.\\ CKA threshold $\\tau$",
                 fontsize=11, color="#222222")
    ax.tick_params(colors="#333333")
    ax.grid(alpha=0.35, color="#dddddd")
    ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc",
              loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, borderaxespad=0.0)

    plt.suptitle("M3-Score: CKA Threshold Ablation",
                 fontsize=13, fontweight="bold", color="#222222")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
