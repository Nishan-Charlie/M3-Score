"""Print the complete multi-metric comparison table + key findings."""
import json, sys, math
sys.path.insert(0, r'C:\Users\nisha\OneDrive\Desktop\Generativemodels\MRI-ImageGenration')

results_path = r'C:\Users\nisha\OneDrive\Desktop\Generativemodels\MRI-ImageGenration\results\multi_metric_comparison\results.json'

with open(results_path) as f:
    rows = json.load(f)

def na(v, fmt=".5f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "   NaN  "
    return format(v, fmt)

print("\n" + "=" * 80)
print("MULTI-METRIC COMPARISON: M3 vs FID vs KID*100 vs CMMD")
print("=" * 80)
print(f"{'Comparison':<25} {'M3':>10} {'FID':>8} {'KID*100':>9} {'CMMD':>12}  {'p(M3)':>7}")
print("-" * 80)

for r in rows:
    if not isinstance(r.get("name"), str):
        continue
    perm = r.get("m3_permutation", {})
    m3   = r.get("m3", float("nan"))
    fid  = r.get("fid", float("nan"))
    kid  = r.get("kid_mean", float("nan"))
    cmmd = r.get("cmmd", float("nan"))
    p    = perm.get("p_value", float("nan"))
    z    = perm.get("z_score", float("nan"))
    print(
        f"{r['name']:<25} "
        f"{na(m3, '.5f'):>10} "
        f"{na(fid, '.2f'):>8} "
        f"{na(kid, '.4f'):>9} "
        f"{na(cmmd, '.6f'):>12}  "
        f"{na(p, '.4f'):>7}"
    )
print("=" * 80)

print("\nKEY FINDING — CMMD cannot discriminate cross-modality OOD:")
print("  BraTS_WDM3D (same modality, different model) : CMMD=0.674")
print("  BraTS_vs_LIDC_OOD (CT lung vs MRI brain)     : CMMD=0.663")
print("  -> CMMD difference = 0.011  (indistinguishable)")
print("  M3: BraTS_WDM3D=0.200 vs LIDC_CT=0.380 -> difference = 0.180 (clear separation)")

print("\nNOISE LADDER SUMMARY (Gaussian corruption of BraTS real images):")
print("  All metrics: Spearman rho = +1.000 vs sigma (all monotone)")
print("  TSTR saturates at sigma=0.05 (binary task trivially easy)")
print("  M3 advantage: domain-sensitive RadioDino features, not unique monotonicity")
