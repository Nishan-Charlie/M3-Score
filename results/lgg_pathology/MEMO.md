# Workstream B — LGG lesion-specificity: results & honest conclusion

Date: 2026-07-26 | Script: `experiments/lgg_lesion_specificity.py`
Data: Kaggle/Buda LGG (real expert FLAIR-abnormality masks), 600 image+mask pairs extracted
from the (re-downloaded, full 748 MB) dataset. Real MRI only — no LGG generator exists, so this
is a **metric-sensitivity** test with controlled perturbations, not a generator-quality test.

## Design

Within-image, **area-matched** contrast. For each image, perturb (i) the annotated tumour
region [LESION] vs (ii) an equal-area region on the contralateral healthy hemisphere obtained by
mirroring the tumour mask [CONTROL]. Images where the mirror lands off-brain or on tumour are
skipped, so lesion area == control area **by construction** (~1700 px each). Measure the
distributional shift each perturbation induces (real vs perturbed) with M3 (single-layer L12, the
Workstream-A-validated config) and with **KID** (unbiased Inception-MMD — small-N-valid, and the
same MMD estimator family as M3, so the *only* difference is the backbone).

Specificity ratio = Δ_lesion / Δ_control. >1 ⇒ metric reacts more to tumour damage than to
equal-area healthy damage.

## Results

| Perturbation | M3 ratio | KID ratio | M3 lesion-selective? |
|---|---:|---:|---|
| erase-to-black (N=200,s42) | 1.13 | 1.33 | No — KID slightly higher |
| blur r=6 (N=200,s42) | degenerate | degenerate | Uninformative (Δ≈0 both) |
| **noise σ=40 (N=200,s42)** | **2.02** | 1.05 | **Yes** |
| **noise σ=40 (N=250,s7)** | **1.73** | 1.02 | **Yes (replicated)** |

## Conclusion (honest, nuanced)

1. **The original "M3 is 4.23× more sensitive to pathology masking than FID" claim is NOT
   supported and should be dropped.** Under gross erasure — the closest analogue to masking — M3
   is *not* more lesion-specific than the Inception baseline (1.13 vs 1.33). The low-level "hole"
   signal dominates and swamps any semantic difference. This corroborates the reversed finding
   already logged in EXPERIMENT_FINDINGS.

2. **A narrower, defensible claim survives and is new:** under *subtle* noise degradation, M3 is
   **lesion-selective (1.7–2.0×) while the Inception-MMD baseline is not (≈1.0×)**, replicated
   across two seeds/sizes. Because both metrics score the identical perturbed images, the
   difference is **attributable to the radiology backbone**, not to pixel/texture magnitude:
   RadioDINO weights subtle degradation of a diagnostically-relevant region more heavily than
   Inception does. This is the "specific to semantically-meaningful shift, not to arbitrary
   artifacts" framing — now with real expert masks and a clean contralateral control.

3. **Scope it precisely.** The selectivity is perturbation-dependent (present for subtle noise,
   absent for gross erasure), so the paper must state it as *"M3 preferentially localizes subtle
   diagnostic-region degradation"*, NOT *"M3 is more sensitive to pathology."* Over-generalizing
   is exactly the error this experiment caught.

## Update (2026-07-26) — noise-σ sweep + texture-matched control

Script: `experiments/lgg_noise_sweep.py` → `results/lgg_noise_sweep/{sweep.json,sweep.png}`.
Added (a) a σ-sweep and (b) a second, **texture-matched** control (tumour mask translated to the
healthy-brain location with the closest local gradient-energy), alongside the mirror control.

Measurable regime (σ where M3 MMD² is above the underflow floor; σ≤30 underflows → excluded):

| σ | M3 (mirror) | M3 (texture) | KID (mirror) | KID (texture) |
|---:|---:|---:|---:|---:|
| 45 | 1.71 | 1.81 | 1.08 | 1.03 |
| 60 | 1.57 | 1.59 | 1.08 | 1.05 |
| 90 | 1.39 | 1.40 | 1.06 | 1.06 |

- **The effect is robust to the control:** M3's two curves (mirror vs texture-matched) agree to
  within ~0.1 at every σ, and both sit far above KID's flat ~1.05.
- **Not a texture artifact:** at N=200 the mirror control's gradient energy matches the lesion's
  to within 4% (tumour 49.7k vs mirror 47.7k), and the texture control is 7% higher — M3 is
  selective against both, KID against neither ⇒ backbone-attributable.
- **Selectivity decays with σ** (1.8→1.4 as 45→90), extrapolating toward the ~1.1 of gross erasure:
  consistent with "M3 localizes *subtle* diagnostic-region degradation; gross damage is a
  low-level signal both backbones catch equally."
- **Measurement limit:** below σ≈45 the L12 MMD² underflows to ~0 at N=200 (0/0 ratios); those
  points are flagged `measurable=false` and excluded, not plotted as spurious spikes.

Publication figure: `results/lgg_noise_sweep/sweep.png`.

## Caveats / follow-ups

- N=200–250; noise σ=40 fixed. Worth a σ-sweep and larger N for a publication figure.
- `blur` was too weak to register (Δ≈0); either raise blur strength or drop it — do not report it.
- Possible residual confound: lesion regions carry more high-frequency structure than healthy
  mirror regions. The KID control largely addresses this (same images, texture-sensitive backbone,
  no effect), but a texture-matched control (e.g. equal-gradient-energy region) would fully close it.

Artifacts: `results/lgg_pathology/lesion_specificity.json`, `results/lgg_pathology_s7/lesion_specificity.json`,
example triptychs `results/lgg_pathology/example_{erase,noise}.png`.
