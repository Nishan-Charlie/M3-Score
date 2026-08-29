"""Print noise ladder summary + Spearman rho."""
import json, sys
sys.path.insert(0, r'C:\Users\nisha\OneDrive\Desktop\Generativemodels\MRI-ImageGenration')
from scipy.stats import spearmanr
import numpy as np

with open(r'C:\Users\nisha\OneDrive\Desktop\Generativemodels\MRI-ImageGenration\results\noise_quality_ladder\results.json') as f:
    rows = json.load(f)

print('NOISE QUALITY LADDER - COMPLETE RESULTS')
print('=' * 72)
print(f"{'sigma':>6} {'M3':>10} {'FID':>8} {'KID*100':>9} {'CMMD':>10} {'TSTR':>7}")
print('-' * 72)
for r in rows:
    sigma = r['sigma']
    m3    = r.get('m3',   float('nan'))
    fid   = r.get('fid',  float('nan'))
    kid   = r.get('kid',  float('nan'))
    cmmd  = r.get('cmmd', float('nan'))
    tstr  = r.get('tstr_acc', float('nan'))
    print(f"{sigma:>6.2f} {m3:>10.5f} {fid:>8.2f} {kid:>9.4f} {cmmd:>10.5f} {tstr:>7.4f}")
print('=' * 72)

sigmas = [r['sigma'] for r in rows]
print('\nSpearman rho vs sigma (higher = metric tracks degradation monotonically):')
for key, label in [('m3','M3'), ('fid','FID'), ('kid','KID'), ('cmmd','CMMD')]:
    vals = [r.get(key, float('nan')) for r in rows]
    valid = [(s, v) for s, v in zip(sigmas, vals) if not np.isnan(v)]
    if len(valid) >= 4:
        rho, p = spearmanr([x[0] for x in valid], [x[1] for x in valid])
        print(f"  {label:<5}: rho={rho:+.4f}  p={p:.6f}")

print('\nNote: TSTR saturates at 1.0 for sigma>0 (binary discrimination task too easy).')
print('      All metrics achieve rho=+1.0 on the Gaussian degradation ladder.')
print('      M3 advantage is domain specificity (RadioDino), not monotonicity.')
