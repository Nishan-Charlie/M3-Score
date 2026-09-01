"""
Regenerate every manuscript figure from saved JSON, in the paper style.
=======================================================================

Reads the result files the experiments already wrote and redraws the figures
with the typography in ``experiments/_plot_style.py``: Times New Roman, bold,
sentence-cased titles and axis names, and tick labels large enough to stay
legible after the figure is reduced to journal column width.

No GPU and no recomputation -- this only reads JSON, so it is safe to run while
other jobs hold the device.

Usage
-----
    python tools/regen_paper_figures.py
    python tools/regen_paper_figures.py --out_dir paper_cmig/Images
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments._plot_style import apply_paper_style, finalize, SIZES

LAYERS = list(range(1, 13))
APRIORI = {"fidelity": 12, "memorization": 9, "coverage": 4}


def _load(path: str):
    if not os.path.exists(path):
        print(f"  [skip] missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure: depth x task grid
# ---------------------------------------------------------------------------

def fig_depth_grid(res: dict, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
    ap_col = "#c0392b"

    # (a) fidelity
    ax = axes[0, 0]
    z = [res["fidelity"]["per_layer"][str(L)]["perm_z"] for L in LAYERS]
    ax.plot(LAYERS, z, "o-", color="#2c3e50", label="Permutation $Z$")
    best = int(np.argmax(z)) + 1
    ax.plot([best], [z[best - 1]], "o", ms=15, mfc="none",
            mec="#27ae60", mew=3, label=f"Best block (L{best})")
    ax.axvline(APRIORI["fidelity"], color=ap_col, ls="--", lw=2.2,
               label=f"A priori (L{APRIORI['fidelity']})")
    for L in (best, APRIORI["fidelity"]):
        ax.annotate(f"{z[L - 1]:.0f}", (L, z[L - 1]),
                    textcoords="offset points", xytext=(0, -26),
                    ha="center", fontsize=SIZES["tick"], fontweight="bold")
    ax.set_xticks(LAYERS)
    finalize(ax, title="(a) Fidelity: discriminability peaks before the final block",
             xlabel="Encoder block", ylabel="Permutation $Z$ (real vs. DDPM)",
             legend=True, loc="lower right")

    # (b) memorization
    ax = axes[0, 1]
    bys = res["memorization"]["by_sigma"]
    keys = list(bys)
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(keys):
        mae = [bys[s]["per_layer"][str(L)]["mean_abs_error"] for L in LAYERS]
        ax.plot(LAYERS, mae, "o-", ms=6, lw=2.0,
                color=cmap(i / max(len(keys) - 1, 1)),
                label=f"$\\sigma$ = {bys[s]['jitter_sigma']:g}")
    ax.axvline(APRIORI["memorization"], color=ap_col, ls="--", lw=2.2,
               label=f"A priori (L{APRIORI['memorization']})")
    ax.set_xticks(LAYERS)
    finalize(ax, title="(b) Memorization: error falls with depth",
             xlabel="Encoder block", ylabel="Mean absolute recovery error",
             legend=True, ncol=2, loc="upper right")

    # (c) coverage
    ax = axes[1, 0]
    rho = [res["coverage"]["per_layer"][str(L)]["spearman_rho_drop_vs_recall"]
           for L in LAYERS]
    bars = ax.bar(LAYERS, rho, color="#95a5a6", edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="k", lw=1.6)
    ax.axvline(APRIORI["coverage"], color=ap_col, ls="--", lw=2.2,
               label=f"A priori (L{APRIORI['coverage']})")
    for b, v in zip(bars, rho):
        ax.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=12, fontweight="bold")
    # Headroom for the value labels, the warning text and the legend.
    ax.set_ylim(-0.12, max(rho) * 1.55)
    ax.set_xticks(LAYERS)
    finalize(ax, title="(c) Coverage: wrong sign at every depth",
             xlabel="Encoder block",
             ylabel="Spearman $\\rho$ (mode drop, recall)",
             legend=True, loc="lower right")
    ax.text(0.5, 0.965,
            "Required: $\\rho < 0$    Observed: $\\rho > 0$ at all 12 blocks",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="#c0392b", va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fdf2f0",
                      ec="#c0392b", lw=1.4))

    # (d) per-image scoring rules
    ax = axes[1, 1]
    tasks = res["ood_scoring"]["tasks"]
    task = "same_modality" if "same_modality" in tasks else list(tasks)[0]
    pretty = {"centroid": "Centroid $L_2$", "knn1": "1-NN",
              "knn5": "5-NN", "mahalanobis": "Mahalanobis"}
    styles = {"centroid": ("#e74c3c", "o"), "knn1": ("#2980b9", "s"),
              "knn5": ("#16a085", "^"), "mahalanobis": ("#8e44ad", "D")}
    for rule in res["ood_scoring"]["rules"]:
        auc = [tasks[task]["per_layer"][str(L)][rule] for L in LAYERS]
        c, m = styles.get(rule, ("#333", "o"))
        ax.plot(LAYERS, auc, marker=m, ls="-", ms=7, color=c,
                label=pretty.get(rule, rule))
    ax.axhline(0.5, color="k", lw=1.4, ls=":")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0.35, 1.05)
    finalize(ax, title="(d) Per-image scoring: rule matters more than depth",
             xlabel="Encoder block",
             ylabel="ROC-AUC (real vs. DDPM, per image)",
             legend=True, loc="lower right")

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure: reference set construction
# ---------------------------------------------------------------------------

def fig_reference_sweep(res: dict, out_path: str) -> None:
    agg = res["aggregate"]
    x = sorted(int(k) for k in agg)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))

    ax = axes[0]
    for L, c, m in ((8, "#2c3e50", "o"), (12, "#c0392b", "s")):
        mu = [agg[str(n)][f"mmd_gen_L{L}_mean"] for n in x]
        sd = [agg[str(n)][f"mmd_gen_L{L}_std"] for n in x]
        ax.errorbar(x, mu, yerr=sd, fmt=f"{m}-", color=c, capsize=5,
                    capthick=2, label=f"L{L}: real vs. DDPM")
        nu = [agg[str(n)][f"mmd_null_L{L}_mean"] for n in x]
        ax.plot(x, nu, marker=m, ls="--", ms=6, color=c, alpha=0.55,
                label=f"L{L}: real vs. real (null)")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.minorticks_off()
    finalize(ax, title="(a) Medical backbone (RadioDINO-s16)",
             xlabel="Distinct subjects in the real reference ($N$ = 500 slices)",
             ylabel="MMD$^2$", legend=True, loc="center right")

    ax = axes[1]
    mu = [agg[str(n)]["fid_gen_mean"] for n in x]
    sd = [agg[str(n)]["fid_gen_std"] for n in x]
    ax.errorbar(x, mu, yerr=sd, fmt="o-", color="#16a085", capsize=5,
                capthick=2, label="FID (InceptionV3)")
    for xi, yi in zip(x, mu):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 12), ha="center",
                    fontsize=SIZES["tick"], fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.minorticks_off()
    ax.set_ylim(min(mu) - 6, max(mu) + 8)
    finalize(ax, title="(b) Natural-image backbone",
             xlabel="Distinct subjects in the real reference ($N$ = 500 slices)",
             ylabel="FID", legend=True, loc="upper right")

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure: cohort heterogeneity null
# ---------------------------------------------------------------------------

def fig_cohort_null(res: dict, out_path: str, depth: int = 12) -> None:
    sizes = sorted(int(k) for k in res["by_cohort_size"])
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.2 * len(sizes), 5.2),
                             squeeze=False)
    colors = {"ddpm": "#2980b9", "wdm3d": "#e67e22",
              "lidc_ct": "#c0392b", "retinal": "#8e44ad"}
    pretty = {"ddpm": "DDPM", "wdm3d": "WDM-3D (MRI)",
              "lidc_ct": "LIDC CT", "retinal": "Retinal"}

    for ax, n_sub in zip(axes[0], sizes):
        e = res["by_cohort_size"][str(n_sub)]
        vals = np.array(e["null"][f"L{depth}"]["values"])
        gens = {n: g[f"L{depth}"]["mmd"] for n, g in e["generators"].items()}

        # The null sits two orders of magnitude below the generators, so a
        # linear axis collapses it onto the spine and hides its shape. Log
        # spacing shows both the null distribution and the separation.
        lo = max(vals.min() * 0.6, 1e-4)
        hi = max(gens.values()) * 1.35
        bins = np.logspace(np.log10(lo), np.log10(vals.max() * 1.05), 22)

        ax.hist(vals, bins=bins, color="#95a5a6", edgecolor="#2c3e50",
                linewidth=0.7,
                label=f"Real vs. real null\n({e['n_null_pairs']} cohort pairs)")
        for name, v in gens.items():
            ax.axvline(v, color=colors.get(name, "#333"), lw=3.0,
                       label=f"{pretty.get(name, name)} = {v:.3f}")
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        # Headroom so the legend never overlaps the histogram.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.75)
        finalize(ax, title=f"{n_sub} subjects per cohort",
                 xlabel=f"MMD$^2$ at block L{depth} (log scale)",
                 ylabel="Number of cohort pairs",
                 legend=True, loc="upper left")

    fig.suptitle("Generator distances against the real cohort-heterogeneity null",
                 fontweight="bold", fontsize=SIZES["suptitle"], y=0.99)
    fig.tight_layout(pad=2.0, rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure: LGG lesion specificity
# ---------------------------------------------------------------------------

def fig_lesion_sweep(res: dict, out_path: str) -> None:
    """Lesion-selectivity ratio vs perturbation strength.

    Points whose bootstrap interval is degenerate are dropped rather than
    drawn. Below the measurable range the control-region response is close to
    zero, so the ratio explodes (values above 10^9 appear in the raw file) and
    plotting it would imply a precision the experiment does not have. The
    omitted range is shaded and labelled instead.
    """
    sweep = res["sweep"]
    series = {
        "M3_mirror":   ("#c0392b", "o", "-",  "RadioDINO MMD (mirrored control)"),
        "M3_texture":  ("#e67e22", "s", "--", "RadioDINO MMD (texture-matched)"),
        "KID_mirror":  ("#2c3e50", "^", "-",  "Inception MMD (mirrored control)"),
        "KID_texture": ("#7f8c8d", "D", "--", "Inception MMD (texture-matched)"),
    }

    def usable(e, key):
        v = e.get(key)
        if v is None or not v.get("measurable", True):
            return False
        r, lo, hi = v["ratio"], v["lo"], v["hi"]
        return all(np.isfinite([r, lo, hi])) and 0 < r < 10 and hi < 10

    fig, ax = plt.subplots(figsize=(11, 7))

    measurable_sigmas = [e["sigma"] for e in sweep
                         if any(usable(e, k) for k in ("M3_mirror", "M3_texture"))]
    if measurable_sigmas:
        lo_edge = min(measurable_sigmas)
        all_s = [e["sigma"] for e in sweep]
        if lo_edge > min(all_s):
            ax.axvspan(min(all_s) - 4, lo_edge, color="#ecf0f1", zorder=0)
            ax.text((min(all_s) + lo_edge) / 2, 2.24,
                    "RadioDINO ratio not measurable here\n"
                    "(control-region response $\\approx 0$)",
                    ha="center", va="top", fontsize=13, fontweight="bold",
                    color="#7f8c8d")
            ax.set_xlim(min(all_s) - 4, max(all_s) + 4)

    for key, (c, m, ls, lab) in series.items():
        xs, ys, los, his = [], [], [], []
        for e in sweep:
            if usable(e, key):
                xs.append(e["sigma"])
                ys.append(e[key]["ratio"])
                los.append(e[key]["lo"])
                his.append(e[key]["hi"])
        if not xs:
            continue
        ax.plot(xs, ys, marker=m, ls=ls, color=c, ms=9, label=lab)
        ax.fill_between(xs, los, his, color=c, alpha=0.15, linewidth=0)

    ax.axhline(1.0, color="k", lw=1.8, ls=":", zorder=1)
    ax.text(0.985, 1.0, "No selectivity", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylim(0.8, 2.3)
    finalize(ax, title="Lesion selectivity on LGG with expert tumour masks ($N$ = 200)",
             xlabel="Noise $\\sigma$ (region-local additive Gaussian)",
             ylabel="Lesion / healthy response ratio",
             legend=True, loc="upper right")

    fig.tight_layout(pad=1.5)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="paper_cmig/Images")
    ap.add_argument("--also_results", action="store_true", default=True,
                    help="also refresh the copy under results/")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    apply_paper_style(bold=True)
    print("[regen_paper_figures] style: Times New Roman, bold, enlarged")

    jobs = [
        ("results/depth_task_grid/depth_task_grid.json",
         "depth_task_grid.png", fig_depth_grid),
        ("results/reference_set_construction/reference_set_construction.json",
         "reference_set_construction.png", fig_reference_sweep),
        ("results/cohort_heterogeneity_null/cohort_heterogeneity_null.json",
         "cohort_heterogeneity_null.png", fig_cohort_null),
        ("results/lgg_noise_sweep/sweep.json",
         "lgg_noise_sweep.png", fig_lesion_sweep),
    ]

    for src, name, fn in jobs:
        data = _load(src)
        if data is None:
            continue
        fn(data, os.path.join(args.out_dir, name))
        if args.also_results:
            fn(data, os.path.join(os.path.dirname(src), name))


if __name__ == "__main__":
    main()
