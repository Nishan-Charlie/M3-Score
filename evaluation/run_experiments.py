#!/usr/bin/env python3
"""
Experimental Evaluation Runner
================================
Master orchestrator for all metric evaluation experiments.

Usage:
    python run_experiments.py \\
        --real_dir data_mri/brats_axial \\
        --gen_dir output/generated_1000_clean \\
        --output_dir results/experiments_output \\
        --experiments all \\
        --device cuda:0

Experiments:
    interpretability          -- Radiomic + RadioDino feature decomposition
    ood                       -- OOD / anomaly detection on generated images
    noise                     -- Noise robustness stress test
    efficiency                -- Sample efficiency & stability
    fid_infinity              -- FID-infinity extrapolation
    weight_ablation           -- M3 weight scheme ablation (Section 3.8)
    permutation_test          -- Statistical significance Z-score test (Section 3.7)
    hallucination             -- Hallucination injection detection (Section 3.11)
    orthogonality             -- Feature scale orthogonality / non-redundancy (Section 3.10)
    frd_comparison            -- FRD vs FID vs KID vs M3-Score (Section 3.12)
    metric_interpretability   -- Metric interpretability validation (Section 3.13)
    comparative_sweep         -- Comparative M3/FID/KID/SSIM/PSNR sensitivity across 4 conditions
    all                       -- Run everything
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _section(title: str):
    width = 68
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _get_module_info(module_path: str) -> dict:
    if not os.path.exists(module_path):
        return {"description": "File not found", "code_content": ""}
    with open(module_path, "r") as f:
        content = f.read()
    import ast
    try:
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree) or "No description provided."
    except Exception:
        docstring = "Failed to parse docstring."
    return {
        "description": docstring,
        "code_path":   os.path.abspath(module_path),
        "code_content": content,
    }


def _safe_run(fn, name: str, module_path: str | None = None):
    metadata = {}
    if module_path:
        metadata = _get_module_info(module_path)
    try:
        results = fn()
        return {**metadata, "results": results if isinstance(results, dict) else results}
    except Exception as exc:
        import traceback
        print(f"\n  [ERROR] {name} failed:")
        traceback.print_exc()
        return {**metadata, "error": str(exc)}


def _n(args_num_images: int | None, fallback: int) -> int:
    return args_num_images if args_num_images is not None else fallback


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental Evaluation Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir",  required=True)
    parser.add_argument("--output_dir", default="./results/experiments_output")
    parser.add_argument(
        "--experiments", nargs="+", default=["all"],
        choices=[
            "all", "interpretability", "ood", "noise", "efficiency",
            "fid_infinity", "weight_ablation", "permutation_test",
            "hallucination", "orthogonality",
            "frd_comparison", "metric_interpretability",
            "comparative_sweep",
        ],
    )
    parser.add_argument("--num_images",  type=int, default=None)
    parser.add_argument("--device",      default="cuda:1")
    parser.add_argument("--no_tqdm",     action="store_true")
    parser.add_argument("--K",           type=int, default=10)
    parser.add_argument("--noise_num_images", type=int, default=None)
    parser.add_argument("--rad_fid_checkpoint",
                        default="output/radiodino_segmentation/backbone_final.pth")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n_perm",      type=int, default=50)
    parser.add_argument("--blob_size",   type=int, default=32)
    parser.add_argument("--cka_threshold", type=float, default=0.85)
    # Required by metric_interpretability experiment only
    parser.add_argument(
        "--interp_report", default=None,
        help=(
            "Path to interpretability_report.json produced by the "
            "interpretability experiment. Required when "
            "--experiments includes metric_interpretability. "
            "If not supplied and metric_interpretability is requested, "
            "the orchestrator will attempt to locate the file at "
            "<output_dir>/interpretability/interpretability_report.json."
        ),
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    use_tqdm = not args.no_tqdm
    os.makedirs(args.output_dir, exist_ok=True)

    noise_n = (args.noise_num_images if args.noise_num_images is not None
               else _n(args.num_images, 200))

    all_experiments = {
        "interpretability", "ood", "noise", "efficiency", "fid_infinity",
        "weight_ablation", "permutation_test",
        "hallucination", "orthogonality", "frd_comparison",
        "metric_interpretability", "comparative_sweep",
    }
    requested = all_experiments if "all" in args.experiments else set(args.experiments)

    master_report = {
        "config": {
            "real_dir":    args.real_dir,
            "gen_dir":     args.gen_dir,
            "experiments": sorted(requested),
            "num_images":  args.num_images,
            "device":      args.device,
        }
    }

    t0 = time.time()

    # ------------------------------------------------------------------
    # Experiment 1: Interpretability Analysis
    # ------------------------------------------------------------------
    if "interpretability" in requested:
        _section("Experiment 1: Interpretability Analysis")
        from experiments.interpretability import run_interpretability
        master_report["interpretability"] = _safe_run(
            lambda: run_interpretability(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "interpretability"),
                num_images=args.num_images,
                device=args.device,
                use_tqdm=use_tqdm,
                cka_threshold=args.cka_threshold,
            ),
            "Interpretability",
            "experiments/interpretability.py",
        )

    # ------------------------------------------------------------------
    # Experiment 2: OOD / Anomaly Detection
    # ------------------------------------------------------------------
    if "ood" in requested:
        _section("Experiment 2: OOD / Anomaly Detection")
        from experiments.ood_detection import run_ood_detection
        master_report["ood_detection"] = _safe_run(
            lambda: run_ood_detection(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "ood"),
                num_images=args.num_images,
                device=args.device,
                use_tqdm=use_tqdm,
                seed=args.seed,
            ),
            "OOD Detection",
            "experiments/ood_detection.py",
        )

    # ------------------------------------------------------------------
    # Experiment 3: Noise Robustness
    # ------------------------------------------------------------------
    if "noise" in requested:
        _section("Experiment 3: Noise Robustness")
        from experiments.noise_robustness import run_noise_robustness
        master_report["noise_robustness"] = _safe_run(
            lambda: run_noise_robustness(
                real_dir=args.real_dir,
                output_dir=os.path.join(args.output_dir, "noise_robustness"),
                num_images=noise_n,
                device=args.device,
                use_tqdm=use_tqdm,
                seed=args.seed,
            ),
            "Noise Robustness",
            "experiments/noise_robustness.py",
        )

    # ------------------------------------------------------------------
    # Experiment 4: Sample Efficiency & Stability
    # ------------------------------------------------------------------
    if "efficiency" in requested:
        _section("Experiment 4: Sample Efficiency & Stability")
        from experiments.sample_efficiency import run_sample_efficiency
        master_report["sample_efficiency"] = _safe_run(
            lambda: run_sample_efficiency(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "sample_efficiency"),
                device=args.device,
            ),
            "Sample Efficiency",
            "experiments/sample_efficiency.py",
        )

    # ------------------------------------------------------------------
    # Experiment 5: FID-infinity
    # ------------------------------------------------------------------
    if "fid_infinity" in requested:
        _section("Experiment 5: FID-infinity")
        from experiments.fid_infinity import run_fid_infinity
        master_report["fid_infinity"] = _safe_run(
            lambda: run_fid_infinity(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "fid_infinity"),
                device=args.device,
                use_tqdm=use_tqdm,
                seed=args.seed,
            ),
            "FID-infinity",
            "experiments/fid_infinity.py",
        )

    # ------------------------------------------------------------------
    # Experiment 6: Weight Ablation (Section 3.8)
    # ------------------------------------------------------------------
    if "weight_ablation" in requested:
        _section("Experiment 6: M3-Score Weight Ablation (Section 3.8)")
        from experiments.weight_ablation import run_weight_ablation
        master_report["weight_ablation"] = _safe_run(
            lambda: run_weight_ablation(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "weight_ablation"),
                n=_n(args.num_images, 500),
                device=args.device,
            ),
            "Weight Ablation",
            "experiments/weight_ablation.py",
        )

    # ------------------------------------------------------------------
    # Experiment 7: Permutation Test (Section 3.7)
    # ------------------------------------------------------------------
    if "permutation_test" in requested:
        _section("Experiment 7: Permutation Test (Section 3.7)")
        from experiments.permutation_test import run_permutation_test
        master_report["permutation_test"] = _safe_run(
            lambda: run_permutation_test(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "permutation_test"),
                n_perm=args.n_perm,
                n_images=_n(args.num_images, 500),
                device=args.device,
                seed=args.seed,
            ),
            "Permutation Test",
            "experiments/permutation_test.py",
        )

    # ------------------------------------------------------------------
    # Experiment 8: Hallucination Detection (Section 3.11)
    # ------------------------------------------------------------------
    if "hallucination" in requested:
        _section("Experiment 9: Hallucination Detection (Section 3.11)")
        from experiments.hallucination_detection import run_hallucination_detection
        master_report["hallucination"] = _safe_run(
            lambda: run_hallucination_detection(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "hallucination"),
                n=_n(args.num_images, 300),
                blob_size=args.blob_size,
                device=args.device,
                seed=args.seed,
            ),
            "Hallucination Detection",
            "experiments/hallucination_detection.py",
        )

    # ------------------------------------------------------------------
    # Experiment 10: Feature Orthogonality (Section 3.10)
    # ------------------------------------------------------------------
    if "orthogonality" in requested:
        _section("Experiment 10: Feature Orthogonality (Section 3.10)")
        from experiments.feature_orthogonality import run_feature_orthogonality
        master_report["orthogonality"] = _safe_run(
            lambda: run_feature_orthogonality(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "orthogonality"),
                n=_n(args.num_images, 300),
                device=args.device,
                seed=args.seed,
            ),
            "Feature Orthogonality",
            "experiments/feature_orthogonality.py",
        )

    # ------------------------------------------------------------------
    # Experiment 11: FRD vs FID vs KID vs M3-Score
    # ------------------------------------------------------------------
    if "frd_comparison" in requested:
        _section("Experiment 11: FRD vs FID vs KID vs M3-Score (Section 3.12)")
        from experiments.frd_comparison import run_frd_comparison
        master_report["frd_comparison"] = _safe_run(
            lambda: run_frd_comparison(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "frd_comparison"),
                num_images=args.num_images,
                device=args.device,
                use_tqdm=use_tqdm,
                cka_threshold=args.cka_threshold,
                seed=args.seed,
            ),
            "FRD Comparison",
            "experiments/frd_comparison.py",
        )

    # ------------------------------------------------------------------
    # Experiment 12: Metric Interpretability Validation (Section 3.13)
    # ------------------------------------------------------------------
    if "metric_interpretability" in requested:
        _section("Experiment 12: Metric Interpretability Validation (Section 3.13)")
        from experiments.metrics_interpretability import (
            run_metric_interpretability_validation,
        )

        # Resolve the interpretability report path.
        # Priority: explicit --interp_report flag > auto-locate in output_dir.
        interp_report_path = args.interp_report
        if interp_report_path is None:
            # Auto-locate: the interpretability experiment writes its report here
            interp_report_path = os.path.join(
                args.output_dir,
                "interpretability",
                "interpretability_report.json",
            )

        if not os.path.exists(interp_report_path):
            print(
                f"\n  [SKIP] metric_interpretability requires an "
                f"interpretability report at:\n    {interp_report_path}\n"
                f"  Run the interpretability experiment first, or supply "
                f"--interp_report <path>."
            )
            master_report["metric_interpretability"] = {
                "error": (
                    f"interpretability_report.json not found at "
                    f"{interp_report_path}. Run experiment 1 first or "
                    f"pass --interp_report."
                )
            }
        else:
            master_report["metric_interpretability"] = _safe_run(
                lambda: run_metric_interpretability_validation(
                    real_dir=args.real_dir,
                    gen_dir=args.gen_dir,
                    interp_report=interp_report_path,
                    output_dir=os.path.join(
                        args.output_dir, "metric_interpretability"
                    ),
                    num_images=args.num_images,
                    device=args.device,
                    seed=args.seed,
                ),
                "Metric Interpretability Validation",
                "experiments/metrics_interpretability.py",
            )

    # ------------------------------------------------------------------
    # Experiment 13: Comparative Sensitivity Sweep
    # ------------------------------------------------------------------
    if "comparative_sweep" in requested:
        _section("Experiment 13: Comparative Sensitivity Sweep (M3/FID/KID/SSIM/PSNR)")
        from experiments.comparative_metrics_sweep import run_comparative_sweep
        master_report["comparative_sweep"] = _safe_run(
            lambda: run_comparative_sweep(
                real_dir=args.real_dir,
                gen_dir=args.gen_dir,
                output_dir=os.path.join(args.output_dir, "comparative_sweep"),
                n=_n(args.num_images, 300),
                device=args.device,
                seed=args.seed,
            ),
            "Comparative Sweep",
            "experiments/comparative_metrics_sweep.py",
        )


    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _section("Summary")

    master_report["total_time_seconds"] = round(time.time() - t0, 2)
    report_path = os.path.join(args.output_dir, "master_report.json")
    with open(report_path, "w") as f:
        json.dump(master_report, f, indent=4, default=str)

    if ("sample_efficiency" in master_report and
            isinstance(master_report.get("sample_efficiency"), dict) and
            "summary" in master_report["sample_efficiency"].get("results", {})):
        _generate_latex_table(
            master_report["sample_efficiency"]["results"]["summary"],
            args.output_dir,
        )

    print(f"\n  Total time: {master_report['total_time_seconds']:.1f}s")
    print(f"  Report:     {report_path}")
    print(f"  Output:     {args.output_dir}")


def _generate_latex_table(summary: dict, output_dir: str):
    metrics = list(summary.keys())
    sizes   = sorted(set(int(s) for m in summary.values() for s in m.keys()))
    lines   = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sample Efficiency: Metric Stability (mean $\pm$ std across $K=10$ subsets)}",
        r"\label{tab:sample_efficiency}",
        r"\begin{tabular}{l" + "c" * len(sizes) + "}",
        r"\toprule",
        "Metric & " + " & ".join([f"$N={s}$" for s in sizes]) + r" \\",
        r"\midrule",
    ]
    for metric in metrics:
        row = [metric.upper()]
        for s in sizes:
            data = summary[metric].get(str(s), {})
            mean = data.get("mean", float("nan"))
            std  = data.get("std",  float("nan"))
            row.append("--" if (isinstance(mean, float) and mean != mean)
                       else f"${mean:.3f} \\pm {std:.3f}$")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(output_dir, "sample_efficiency_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX table -> {path}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()