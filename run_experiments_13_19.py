"""
Run only experiments 13-19 (new tri-axial evaluation framework).
Merges results into the existing master_results.json for a given run_id.

Usage:
    python run_experiments_13_19.py --run_id 5 --seed 42 --device cuda
"""
from __future__ import annotations

import argparse, json, os, sys
import torch

_p = argparse.ArgumentParser()
_p.add_argument("--run_id",  type=int, default=5)
_p.add_argument("--seed",    type=int, default=42)
_p.add_argument("--device",  default=None)
_p.add_argument("--backbone", default="Snarcy/RadioDino-s16")
_args = _p.parse_args()

ROOT     = os.path.dirname(os.path.abspath(__file__))
BACKBONE_TAG = _args.backbone.split("/")[-1].lower().replace("_", "-")
OUT_DIR  = os.path.join(ROOT, "results", f"{BACKBONE_TAG}_run{_args.run_id}")
REAL_DIR = os.path.join(ROOT, "data_mri", "brats_axial_multislice")
GEN_DIR  = os.path.join(ROOT, "output", "generated_500_best")
WDM_BRATS = os.path.join(ROOT, "output", "generated_wdm3d", "brats")
WDM_LIDC  = os.path.join(ROOT, "output", "generated_wdm3d", "lidc")
DEVICE   = _args.device or ("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE_ID = _args.backbone
SEED     = _args.seed
N_IMAGES = 500

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")

os.makedirs(OUT_DIR, exist_ok=True)

# Load existing master results
master_path = os.path.join(OUT_DIR, "master_results.json")
if os.path.exists(master_path):
    with open(master_path) as f:
        master = json.load(f)
else:
    master = {}

print(f"\n{'='*60}")
print(f"  Running Experiments 13-19 only")
print(f"  Output dir : {OUT_DIR}")
print(f"  GEN_DIR    : {GEN_DIR}")
print(f"  Device     : {DEVICE}")
print(f"{'='*60}\n")


# ============================================================================
# EXPERIMENT 13: COVERAGE & NOVELTY
# ============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 13: COVERAGE & NOVELTY")
print("="*60)

from evaluation.coverage_novelty import compute_coverage_novelty
from experiments._shared_utils import load_pils_recursive

cn_out = os.path.join(OUT_DIR, "coverage_novelty")
os.makedirs(cn_out, exist_ok=True)
try:
    _real_cn = load_pils_recursive(REAL_DIR, n=N_IMAGES)
    _gen_cn  = load_pils_recursive(GEN_DIR,  n=N_IMAGES)
    cn_res = compute_coverage_novelty(
        _real_cn, _gen_cn,
        backbone_id=BACKBONE_ID, device=DEVICE,
        k_neighbors=5, pct_memorize=5.0, batch_size=32,
    )
    cn_wdm = {}
    if os.path.isdir(WDM_BRATS):
        _wdm_imgs = load_pils_recursive(WDM_BRATS, n=N_IMAGES)
        cn_wdm["wdm3d_brats"] = compute_coverage_novelty(
            _real_cn, _wdm_imgs,
            backbone_id=BACKBONE_ID, device=DEVICE, batch_size=32,
        )
    _cn_rep = {k: float(v) for k, v in cn_res.items()
               if isinstance(v, (int, float))}
    with open(os.path.join(cn_out, "coverage_novelty_report.json"), "w") as f:
        json.dump({**_cn_rep, "wdm3d": {
            k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
            for k, v in cn_wdm.items()
        }}, f, indent=2, default=str)
    master["coverage_novelty"] = {
        "coverage":          float(cn_res["coverage"]),
        "novelty":           float(cn_res["novelty"]),
        "memorization_rate": float(cn_res["memorization_rate"]),
        "wdm3d_brats": {k: float(v) for k, v in cn_wdm.get("wdm3d_brats", {}).items()
                        if isinstance(v, (int, float))},
    }
    print(f"  [13] Coverage={cn_res['coverage']:.4f}  "
          f"Novelty={cn_res['novelty']:.4f}  "
          f"Memorization={cn_res['memorization_rate']:.4f}")
except Exception as e:
    print(f"  [13] FAILED: {e}")
    master["coverage_novelty"] = {"error": str(e)}


# ============================================================================
# EXPERIMENT 14: STATISTICAL RIGOR
# ============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 14: STATISTICAL RIGOR")
print("="*60)

from evaluation.statistical_rigor import statistical_m3

sr_out = os.path.join(OUT_DIR, "statistical_rigor")
os.makedirs(sr_out, exist_ok=True)
try:
    _real_sr = load_pils_recursive(REAL_DIR, n=N_IMAGES)
    _gen_sr  = load_pils_recursive(GEN_DIR,  n=N_IMAGES)
    sr_res = statistical_m3(
        real_imgs=_real_sr, gen_imgs=_gen_sr,
        backbone_id=BACKBONE_ID, device=DEVICE,
        n_perm=500, n_boot=200, ci_level=0.95,
        batch_size=16, single_layer=12, seed=SEED,
    )
    _sr_rep = {k: float(v) for k, v in sr_res.items()
               if isinstance(v, (int, float))}
    with open(os.path.join(sr_out, "statistical_rigor_report.json"), "w") as f:
        json.dump(_sr_rep, f, indent=2, default=str)
    master["statistical_rigor"] = {
        "m3_score":    float(sr_res["m3_score"]),
        "p_value":     float(sr_res["p_value"]),
        "effect_size": float(sr_res["effect_size"]),
        "ci_low":      float(sr_res["ci_low"]),
        "ci_high":     float(sr_res["ci_high"]),
    }
    print(f"  [14] p={sr_res['p_value']:.4f}  ES={sr_res['effect_size']:.2f}  "
          f"CI=[{sr_res['ci_low']:.6f}, {sr_res['ci_high']:.6f}]")
except Exception as e:
    print(f"  [14] FAILED: {e}")
    import traceback; traceback.print_exc()
    master["statistical_rigor"] = {"error": str(e)}


# ============================================================================
# EXPERIMENT 15: CONDITIONAL MMD
# ============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 15: CONDITIONAL MMD")
print("="*60)

from evaluation.conditional_mmd import run_conditional_mmd

cond_out = os.path.join(OUT_DIR, "conditional_mmd")
try:
    cond_res = run_conditional_mmd(
        real_dir=REAL_DIR, gen_dir=GEN_DIR,
        stratify_by="intensity_quartile", n_strata=4,
        n_images=N_IMAGES, backbone_id=BACKBONE_ID,
        device=DEVICE, output_dir=cond_out, seed=SEED,
    )
    master["conditional_mmd"] = {
        "mean_m3":       float(cond_res["mean_m3"]),
        "worst_m3":      float(cond_res["worst_m3"]),
        "worst_stratum": cond_res["worst_stratum"],
        "std_m3":        float(cond_res["std_m3"]),
    }
    print(f"  [15] Mean={cond_res['mean_m3']:.6f}  "
          f"Worst={cond_res['worst_m3']:.6f} ({cond_res['worst_stratum']})")
except Exception as e:
    print(f"  [15] FAILED: {e}")
    import traceback; traceback.print_exc()
    master["conditional_mmd"] = {"error": str(e)}


# ============================================================================
# EXPERIMENT 16: NORMALITY VIOLATION
# ============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 16: NORMALITY VIOLATION (FID blind-spot)")
print("="*60)

from experiments.normality_violation import run_normality_violation

nv_out = os.path.join(OUT_DIR, "normality_violation")
try:
    nv_res = run_normality_violation(
        real_dir=REAL_DIR, gen_dir=GEN_DIR,
        output_dir=nv_out, n_images=N_IMAGES,
        backbone_id=BACKBONE_ID, device=DEVICE, seed=SEED,
    )
    master["normality_violation"] = {
        k: {
            "levels":                v["levels"],
            "fid_at_max_departure":  float(v["fid_vals"][-1]),
            "m3_at_max_departure":   float(v["m3_vals"][-1]),
        }
        for k, v in nv_res.items()
    }
    print(f"  [16] Bimodal: FID={nv_res['bimodal_shift']['fid_vals'][-1]:.4f}  "
          f"M3={nv_res['bimodal_shift']['m3_vals'][-1]:.6f}")
except Exception as e:
    print(f"  [16] FAILED: {e}")
    import traceback; traceback.print_exc()
    master["normality_violation"] = {"error": str(e)}


# ============================================================================
# EXPERIMENT 17: DISTORTION MONOTONICITY PER SCALE
# ============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 17: DISTORTION MONOTONICITY PER SCALE")
print("="*60)

from experiments.distortion_monotonicity_per_scale import run_distortion_monotonicity

dm_out = os.path.join(OUT_DIR, "distortion_monotonicity_per_scale")
try:
    dm_res = run_distortion_monotonicity(
        real_dir=REAL_DIR, gen_dir=GEN_DIR,
        output_dir=dm_out, n_images=min(N_IMAGES, 300),
        backbone_id=BACKBONE_ID, device=DEVICE,
        seed=SEED, target_layers=[1, 6, 12],
    )
    _mono_path = os.path.join(dm_out, "monotonicity_check.json")
    _mono = {}
    if os.path.exists(_mono_path):
        with open(_mono_path) as f:
            _mono = json.load(f)
    master["distortion_monotonicity"] = {
        "corruptions_tested": list(dm_res.keys()),
        "monotonicity_check": _mono,
    }
    print(f"  [17] Done for: {list(dm_res.keys())}")
except Exception as e:
    print(f"  [17] FAILED: {e}")
    import traceback; traceback.print_exc()
    master["distortion_monotonicity"] = {"error": str(e)}


# ============================================================================
# EXPERIMENT 18: SAMPLE SIZE CONSISTENCY
# ============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 18: SAMPLE SIZE CONSISTENCY")
print("="*60)

from experiments.sample_size_consistency import run_sample_size_consistency

ssc_out = os.path.join(OUT_DIR, "sample_size_consistency")
try:
    ssc_res = run_sample_size_consistency(
        real_dir=REAL_DIR, gen_dir=GEN_DIR,
        output_dir=ssc_out,
        n_list=[25, 50, 100, 200, 500], n_repeats=20,
        backbone_id=BACKBONE_ID, device=DEVICE, seed=SEED,
    )
    _ssc_50 = ssc_res.get("50", {})
    master["sample_size_consistency"] = {
        str(N): {m: {"mean": float(v["mean"]), "cv": float(v["cv"])}
                 for m, v in vals.items()}
        for N, vals in ssc_res.items()
    }
    if _ssc_50:
        print(f"  [18] CV@N=50  M3={_ssc_50.get('m3_mmd',{}).get('cv',float('nan')):.4f}  "
              f"FID={_ssc_50.get('fid',{}).get('cv',float('nan')):.4f}  "
              f"CMMD={_ssc_50.get('cmmd',{}).get('cv',float('nan')):.4f}")
    else:
        print("  [18] Sample-size consistency done (N=50 not in n_list or empty).")
except Exception as e:
    print(f"  [18] FAILED: {e}")
    import traceback; traceback.print_exc()
    master["sample_size_consistency"] = {"error": str(e)}


# ============================================================================
# EXPERIMENT 19: TSTR UTILITY
# ============================================================================
print("\n" + "="*60)
print("  EXPERIMENT 19: TSTR UTILITY")
print("="*60)

from experiments.tstr_utility import run_tstr

tstr_out = os.path.join(OUT_DIR, "tstr_utility")
_gen_dirs = []; _gen_labels = []
if os.path.isdir(GEN_DIR):
    _gen_dirs.append(GEN_DIR);   _gen_labels.append("ddpm_unet")
if os.path.isdir(WDM_BRATS):
    _gen_dirs.append(WDM_BRATS); _gen_labels.append("wdm3d_brats")
if os.path.isdir(WDM_LIDC):
    _gen_dirs.append(WDM_LIDC);  _gen_labels.append("wdm3d_lidc")

try:
    if not _gen_dirs:
        raise ValueError("No generator directories found.")
    tstr_res = run_tstr(
        real_dir=REAL_DIR, gen_dirs=_gen_dirs, gen_labels=_gen_labels,
        output_dir=tstr_out, task="slice_position",
        n_images=min(N_IMAGES, 500), epochs=20, batch_size=32,
        backbone_id=BACKBONE_ID, device=DEVICE, seed=SEED,
    )
    master["tstr_utility"] = {
        "task":              tstr_res["task"],
        "baseline_rr_acc":  float(tstr_res["baseline_rr_acc"]),
        "spearman_m3_tstr":  tstr_res["spearman_m3_tstr"],
        "spearman_fid_tstr": tstr_res["spearman_fid_tstr"],
        "generator_summary": {
            lbl: {"m3_rank": v["m3_rank"], "fid_rank": v["fid_rank"],
                  "tstr_rank": v["tstr_rank"], "tstr_acc": float(v["tstr_acc"])}
            for lbl, v in tstr_res["generators"].items()
        },
    }
    print(f"  [19] M3 rho={tstr_res['spearman_m3_tstr']['rho']:.3f}  "
          f"FID rho={tstr_res['spearman_fid_tstr']['rho']:.3f}")
except Exception as e:
    print(f"  [19] FAILED: {e}")
    import traceback; traceback.print_exc()
    master["tstr_utility"] = {"error": str(e)}


# ============================================================================
# SAVE MERGED MASTER RESULTS
# ============================================================================
with open(master_path, "w") as f:
    json.dump(master, f, indent=4, default=str)
print(f"\n[OK] Merged master results saved: {master_path}")

# ============================================================================
# NEW METRICS SUMMARY
# ============================================================================
print("\n" + "="*60)
print("  NEW METRICS SUMMARY (Experiments 13-19)")
print("="*60)

if "coverage_novelty" in master and "coverage" in master.get("coverage_novelty", {}):
    _cn = master["coverage_novelty"]
    print(f"  Manifold Coverage    : {_cn['coverage']:.4f}")
    print(f"  Calibrated Novelty   : {_cn['novelty']:.4f}")
    print(f"  Memorization Rate    : {_cn['memorization_rate']:.4f}")
    if "wdm3d_brats" in _cn and "coverage" in _cn["wdm3d_brats"]:
        _wc = _cn["wdm3d_brats"]
        print(f"  WDM-3D BraTS Coverage: {_wc['coverage']:.4f}  "
              f"Novelty={_wc['novelty']:.4f}")

if "statistical_rigor" in master and "p_value" in master.get("statistical_rigor", {}):
    _sr = master["statistical_rigor"]
    print(f"  Permutation p-value  : {_sr['p_value']:.4f}")
    print(f"  Effect size (z)      : {_sr['effect_size']:.2f}")
    print(f"  Bootstrap 95%% CI   : [{_sr['ci_low']:.6f}, {_sr['ci_high']:.6f}]")

if "conditional_mmd" in master and "mean_m3" in master.get("conditional_mmd", {}):
    _cm = master["conditional_mmd"]
    print(f"  Cond-MMD mean        : {_cm['mean_m3']:.6f}")
    print(f"  Cond-MMD worst       : {_cm['worst_m3']:.6f}  ({_cm['worst_stratum']})")
    print(f"  Cond-MMD std         : {_cm['std_m3']:.6f}")

if "normality_violation" in master and "bimodal_shift" in master.get("normality_violation", {}):
    _nv = master["normality_violation"]["bimodal_shift"]
    print(f"  Normality bimodal    : FID={_nv['fid_at_max_departure']:.4f}  "
          f"M3={_nv['m3_at_max_departure']:.6f}")

if "distortion_monotonicity" in master and "monotonicity_check" in master.get("distortion_monotonicity", {}):
    _dm = master["distortion_monotonicity"]
    print(f"  Corruptions tested   : {_dm['corruptions_tested']}")
    for ct, cv in _dm.get("monotonicity_check", {}).items():
        print(f"    {ct}: tau={cv.get('kendall_tau',{})}")

if "sample_size_consistency" in master and "50" in master.get("sample_size_consistency", {}):
    _s = master["sample_size_consistency"]["50"]
    print(f"  Sample CV@N=50: M3={_s.get('m3_mmd',{}).get('cv',float('nan')):.4f}  "
          f"FID={_s.get('fid',{}).get('cv',float('nan')):.4f}  "
          f"CMMD={_s.get('cmmd',{}).get('cv',float('nan')):.4f}")

if "tstr_utility" in master and "spearman_m3_tstr" in master.get("tstr_utility", {}):
    _t = master["tstr_utility"]
    print(f"  TSTR baseline acc    : {_t['baseline_rr_acc']:.4f}")
    print(f"  M3-TSTR Spearman rho : {_t['spearman_m3_tstr']['rho']:.3f}  "
          f"p={_t['spearman_m3_tstr']['p']:.4f}")
    print(f"  FID-TSTR Spearman rho: {_t['spearman_fid_tstr']['rho']:.3f}  "
          f"p={_t['spearman_fid_tstr']['p']:.4f}")
