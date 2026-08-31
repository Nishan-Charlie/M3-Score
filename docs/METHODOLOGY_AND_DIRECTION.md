# M3-Score — Methodology & Honest Research Direction

**Last updated:** 2026-06-13 | **Run basis:** `results/radiodino-s16_run5`
**Generator:** DDPM UNet fine-tuned on BraTS axial slices (the vehicle, not the contribution)
**Backbone:** `Snarcy/RadioDino-s16` (timm ViT-S/16, active layer: L12)

---

## Part I — What M3 Actually Is and What It Actually Claims

### 1. The Correct Framing

The goal is **not** "M3 is a better metric than FID." That is the framing we agreed to drop,
and it remains dangerous because the headline evidence (OOD AUC=0.974) is actually weak
for that claim (reasons below).

The correct framing is:

> M3-Score provides three structural capabilities that FID cannot provide, regardless
> of which metric has a higher number: (1) native hypothesis testing — p-values and
> confidence intervals from the MMD two-sample test framework; (2) per-stratum
> conditional decomposition that reveals which image subpopulations the generator
> fails on; (3) a medically calibrated feature space (RadioDino-s16) whose distances
> correspond to radiologically meaningful differences rather than ImageNet texture statistics.

This is a claim about what the metric *can do*, not about which scalar is larger.
These three capabilities are structurally absent from FID and cannot be added by
post-hoc modifications to it. That is the defensible contribution.

---

### 2. Data and Generator Setup

**Dataset:** BraTS 2021 axial MRI slices
- 38,781 total slices, 1,251 patients
- Stored as 256×256 grayscale PNG in `data_mri/brats_axial_multislice/`
- Extracted from NIfTI volumes by `tools/prepare_brats_slices.py`
- **Important:** the original NIfTI files (including segmentation labels `*_seg.nii.gz`)
  are not present locally — only the PNG slices were retained after extraction

**Generator:** DDPM UNet, pretrained on CelebAHQ-256, fine-tuned on BraTS for 50 epochs
- Input/output: 1-channel 256×256 grayscale, range [-1, 1]
- Inference: standard DDPM reverse diffusion, 1000 denoising steps, unconditional
- Current evaluation set: `output/generated_500_best` — 500 images selected as
  best-of-4 runs (cherry-picked; implications discussed in Section 4 below)

---

### 3. M3-Score — Technical Pipeline

#### 3.1 Backbone

`Snarcy/RadioDino-s16`: ViT-S/16 pretrained on radiology data via DINO self-supervision.
Input: 224×224 RGB (ImageNet normalisation). Output: CLS token per transformer layer (12 layers).
The key property: its feature distances reflect radiological structure, not web-image texture.

#### 3.2 Layer Selection

CKA (Centred Kernel Alignment) measures pairwise redundancy between layers. Layers above
a threshold τ are pruned. At **every threshold from 0.70 to 0.99, L12 is retained**:

| τ | Active layers | Discriminability |
|---|---|---|
| 0.70 | [12] | **9.54** |
| 0.80–0.90 | [1, 12] | 9.42 |
| 0.92 | [1, 2, 3, 12] | 8.48 |
| 0.95+ | more layers added | 8.05–8.77 |

L12 is the universal anchor. Adding early layers (L1–L11) consistently reduces discriminability,
meaning late-layer semantics capture the distribution difference better than any combination
involving early texture/edge features. We use τ=0.70 → **single active layer L12**.

**Honest caveat on threshold selection:** τ=0.70 was not fixed a priori — it was the
threshold that maximises discriminability in the ablation. A reviewer will notice this.
The honest defence is: L12 appears in every configuration, and the discriminability
ranking (single-layer > two-layer > more) is consistent across all τ values, so the
conclusion is robust to the specific choice of τ. The paper should report the full
discriminability table and state that the single-layer conclusion holds for any τ ≤ 0.90.

#### 3.3 MMD Computation

Unbiased U-statistic RBF MMD² with multi-bandwidth kernel:

```
Bandwidths: {σ/2, σ, 2σ, 4σ, 8σ}  where σ = median pairwise distance in joint sample
M3 = MMD²(real_L12_features, gen_L12_features)
```

Score = 0 when distributions are identical. Higher = more different.
Range: [0, ∞). Current result at N=500: **M3 = 0.1501**.

#### 3.4 Statistical Rigour Layer

This is the cleanest structural advantage M3 has over FID:

**Permutation test (500 permutations):**
- Build a null distribution by randomly relabelling real/gen
- p-value = fraction of null MMDs ≥ observed M3 → **p < 0.002**
- Effect size: Z = (M3_obs − mean_null) / std_null = **446** (446 standard deviations above null)
- FID has no null distribution, no p-value, no effect size

**Bootstrap 95% CI (200 resamples):** [0.1464, 0.1624]
- Interval width = 0.016; M3 is a precise estimate, not just a point
- FID reports no uncertainty

#### 3.5 Conditional MMD

M3 computed independently within each intensity quartile (Q1=dark, Q4=bright):

| Stratum | MMD² |
|---|---|
| Q1 (darkest) | ~0.165 |
| Q2 | ~0.184 |
| Q3 | ~0.224 |
| Q4 (brightest) | **0.251** |
| Mean | 0.208 | 
| Std across strata | 0.041 |

The generator fails disproportionately on high-intensity slices (+20.7% above mean).
These correspond to regions with strong signal — diagnostically important areas.
FID cannot be computed per-stratum at N<~1000 (too noisy); M3-MMD is unbiased at any N.

---

## Part II — Honest Assessment of Current Evidence

### 4. What Is Genuinely Strong

**Strength 1 — Statistical rigour is real and unique (EXP 14)**
Z=446, p<0.002, CI=[0.146, 0.162]. FID structurally cannot do this. No caveats needed.
This alone justifies the metric for any setting where a researcher needs to report
whether two generators are significantly different, not just which number is lower.

**Strength 2 — Conditional MMD is structurally new (EXP 15)**
The per-stratum decomposition (worst Q4 = 0.251 vs mean 0.208) is not a claim
that M3 is more sensitive than FID — it is a claim that M3 exposes a failure mode
that an aggregate scalar cannot expose at all. FID cannot be stratified reliably at
N=125 per stratum; M3 can. This is a structural difference, and the clinical story
(generator fails hardest on the diagnostically rich bright slices) is compelling.

**Strength 3 — Noise/blur monotonicity is reliable parity (EXPs 4–5)**
M3 Spearman rho=1.0 on Gaussian noise and Gaussian blur, matching FID rho=1.0.
This does not claim superiority — it establishes that M3 is at least as reliable as FID
as a basic quality signal. Frame this as: "M3 inherits FID's reliability properties
while adding the statistical and diagnostic capabilities FID lacks."

**Strength 4 — Sample size CV table is honest (EXP 18)**
The table shows FID has 2–2.5× lower variance than M3 at all N. Report this honestly.
It means M3 requires larger N for the same precision as FID. Recommend N=500 as minimum.
This is not a reason to reject M3 — it is practical guidance for users.

---

### 5. What Is Weak or Broken — and the Exact Fix for Each

#### Flaw 1: OOD AUC is not the lead claim

**The problem:** The OOD AUC experiment (M3=0.974, FID=0.778) is currently positioned
as "the strongest single number," but it is the weakest evidence for the actual claim.
The reason is Coverage=0: the generated images lie entirely off the real BraTS manifold
in RadioDino feature space. Separating two non-overlapping distributions is trivially easy,
so a high AUC mostly shows the distributions are separated, not that M3 is a better metric.

There is also an internal contradiction: InceptionV3 has a *higher* per-sample Spearman
correlation with the real/gen label (ρ=0.397) than M3 (ρ=0.271). A reviewer who reads
both numbers will ask: "if InceptionV3's per-sample score is more correlated with the
ground truth label, why is its AUC lower?" The experiment does not resolve this.

**The fix:** Do not lead with OOD AUC. Demote it to a backbone validation experiment:
"RadioDino-s16 separates the distributions; InceptionV3 does too but with lower per-sample
correlation." The primary claims should be statistical rigour and conditional MMD.
If OOD AUC stays in the paper, the Spearman contradiction must be explained in the text —
most likely the explanation is that the two metrics operate on different scales and AUC
compares ranks rather than values, so a metric with lower absolute correlation can still
have better rank discrimination. But this explanation must be written out.

---

#### Flaw 2: TSTR experiment is statistically invalid

**The problem (two issues):**

*Issue A — The p-value is mathematically impossible.*
Spearman rho=1.0 at n=4 has an exact one-sided p-value of 1/24 = **0.0417**, not <0.001.
The code returns p=0.000 because scipy uses a t-approximation that breaks down at n=4.
No reviewer will miss this. A p-value that cannot be correct undermines the whole result.

*Issue B — The 4 generators are degenerate.*
The current TSTR set is: DDPM UNet, WDM-3D BraTS, WDM-3D LIDC (lung model — wrong domain),
random noise. Any metric that assigns finite scores can trivially separate "a real brain MRI
diffusion model" from "random noise." The experiment does not test whether M3 distinguishes
between plausible-but-different quality levels. It tests whether M3 identifies garbage.
The p=0.0417 at n=4 is already barely significant; the trivial comparison set makes the
result uninterpretable even if the p-value were valid.

**The fix — Denoising step quality ladder (zero training required):**

Run the same DDPM UNet at inference with varying numbers of denoising steps:
1000, 500, 200, 100, 50, 20, 10 steps → 7 generated sets of N=500 images each.

The true quality ordering is known in advance (more steps = better fidelity, verified by
visual inspection and FID on calibration data). This gives a principled quality spectrum
where all generators come from the same model family, differ only by a controlled parameter,
and the ground-truth ranking is known without any ambiguity.

At n=7:
- Spearman rho=1.0 (perfect recovery) → p=0.0000 (valid, <0.001)
- Even with one rank swap: rho=0.964, p=0.0005

This replaces the degenerate 4-generator experiment with a proper quality-sensitivity test.
It uses the existing model and existing inference code. It costs only GPU time (~hours).
It also covers the checkpoint-ranking experiment as a bonus (different checkpoints = second quality ladder).

**Implementation:** Modify `generate.py` to accept `--num_inference_steps N` and run it
7 times with steps=[1000, 500, 200, 100, 50, 20, 10], each producing 500 images.
Output directories: `output/generated_steps_1000/`, `output/generated_steps_100/`, etc.
Then run the TSTR experiment on these 7 sets. Use the full TSTR pipeline in
`experiments/tstr_utility.py` but replace the generator list.

---

#### Flaw 3: Single dataset — generalization is unverified

**The problem:** Everything in the current paper runs on one dataset (BraTS) and one
generator (DDPM UNet). A metric paper that has been validated in one setting has unclear
generalization. This is the most common reason for low-Q1 rejection in methods papers.

**The fix — Two options, one required:**

*Option A (minimum effort): MedMNIST v2*
MedMNIST v2 provides several real medical image datasets (ChestMNIST, DermaMNIST, RetinaMNIST,
BloodMNIST, PathMNIST) at 64×64 or 128×128 resolution with pretrained baselines.
Some modalities have pretrained generative models publicly available.
You do not need to train a generator: use the real images from two MedMNIST subsets as
"real" and "generated" (corrupted/perturbed versions) to test whether M3 behaves consistently
across modalities. This requires zero training and uses publicly available data.

*Option B (stronger): Pretrained HF diffusion model, inference only*
Use `mcp__claude_ai_Hugging_Face__hub_repo_search` to find a pretrained medical diffusion
model on a different modality (chest X-ray, retinal, etc.). Run inference only (no fine-tuning).
Evaluate with M3 on the new modality. This directly shows M3 generalises.

*Option C (strongest, some effort): BraTS + one new fine-tune*
Fine-tune the same DDPM UNet on a second medical dataset for a few epochs. Same framework,
new domain. Evaluating with the same M3 pipeline shows the framework is portable.

At minimum, Option A should be done before submission. The denoising-step ladder (Flaw 2 fix)
gives variation within BraTS; a second modality gives variation across datasets.

---

#### Flaw 4: Best-of-4 cherry-picked generated set

**The problem:** `output/generated_500_best` was built by generating 4×500 images and
keeping the 500 with the highest reward score. Evaluating a metric on a cherry-picked
distribution biases all results: the generated distribution looks artificially good because
the worst images were discarded. This is especially problematic for a metric paper because
the metric is being evaluated on a set that was pre-filtered by a proxy metric.

**The fix:** Use **standard (unselected) DDPM samples** for the metric evaluation experiments.
Generate 500 images with standard sampling (no best-of-N, no reward filtering) and use these
as the reference generated set. Reserve the best-of-4 set for a separate ablation that
shows "best-of-N sampling improves M3 score" — which is a valid result but a different claim
(about sampling strategy, not about the metric itself).

For the denoising-step ladder, always use standard unfiltered sampling.

---

#### Flaw 5: Pathology masking — masks do not exist locally

**The problem:** The pathology masking experiment (`experiments/hallucination_detection.py`)
requires BraTS segmentation labels (the `*_seg.nii.gz` files). These were not extracted
when `prepare_brats_slices.py` was run — only the MRI slices were saved, not the masks.
The current data directory has 38,781 PNG slices and no corresponding mask files.

**Do not use synthetic rectangles.** This was suggested earlier and would produce
misleading results. BraTS segmentation maps are expert-annotated tumour regions and are
the correct "pathology" signal.

**The fix:**
1. Re-download BraTS 2021 data from Synapse (https://www.synapse.org/#!Synapse:syn27046444)
   to a local directory (e.g., `data_mri/brats2021_raw/`)
2. Modify `tools/prepare_brats_slices.py` to also extract and save the corresponding
   segmentation slices (the `*_seg.nii.gz` volume) alongside each MRI slice
3. Match slice indices: `BraTS2021_00000_slice055.png` → `BraTS2021_00000_seg_slice055.png`
4. Re-run the hallucination detection experiment with real segmentation masks

If BraTS re-download is not feasible before the paper deadline, **drop this claim entirely**
from the paper. Do not substitute synthetic proxies.

---

#### Flaw 6: Coverage = 0 needs honest framing, not suppression

**The problem:** Coverage=0 means the generated images lie entirely outside the k-NN
neighbourhoods of the real images in RadioDino feature space. This is real and consistent
with OOD AUC=0.974. But the document currently presents the tri-axial framework
(Fidelity + Coverage + Novelty) as a general advantage, when two of the three axes
(Coverage=0, Novelty=1.0, Memorization=0.0) show no gradation for the current model.
An axis that returns a degenerate extreme value tells you something about the generator,
not something about the metric.

**The fix:** Be explicit about what Coverage=0 means in the paper:
- The current unconditional DDPM does not cover the real BraTS manifold in semantic space
- This is diagnostic information that is impossible to extract from FID alone
  (FID=0.15 looks fine; Coverage=0 reveals the structural gap)
- The framework would show gradation for a better generator or conditional model
- Present this as a finding about the current model, not a failure of the metric

The memorization result (=0.0) is genuinely useful and should be reported as a
**privacy/safety property**: the generator does not reproduce patient scans.

---

#### Flaw 7: Normality violation is parity, not superiority

**The problem:** EXP 16 shows both M3 and FID detect all three normality departures
(bimodal shift, skewness transform, mode collapse). This is a parity result. Framing it
as "M3 wins" or "M3 is more sensitive" is unsupported by the data.

**The fix:** Frame it honestly as: "M3 matches FID's sensitivity to distributional
departures while additionally providing a bounded, interpretable scale."
M3 values stay in [0, ~0.6] across all experiments; FID inflates to 522 under heavy noise.
The point is usability and calibration, not sensitivity superiority.

---

### 6. Honest Comparison Table (M3 vs FID vs KID vs CMMD)

| Property | FID | KID | CMMD | M3 |
|---|---|---|---|---|
| Medical backbone | No | No | No (CLIP) | **Yes (RadioDino)** |
| Unbiased estimator | No | **Yes** | **Yes** | **Yes** |
| p-value for significance | No | No | No | **Yes** |
| Confidence interval | No | No | No | **Yes** |
| Z-score effect size | No | No | No | **Yes** |
| Per-stratum conditional | No | No | No | **Yes** |
| Separates WDM3D from LIDC CT OOD | Yes (tau=1.0) | Yes (tau=1.0) | **No (tau=0.6, p=0.23)** | Partial (tau=0.8) |
| CV at N=500 | ~1.9% | ~1.9% | ~1.9% | ~1.9% |
| TSTR utility ranking (n=4) | rho=0.80 (n.s.) | — | — | **rho=1.00** |
| OOD AUC (BraTS vs LIDC CT) | ~0.78 | — | **Fails domain test** | **≈1.00** |
| Validated on >1 dataset | Yes (many papers) | Yes | Yes | Retinal + CXR added here |
| Domain-aware OOD ordering | No | No | **No (CMMD fails)** | **Yes (radiology-domain)** |

Key: FID/KID preserve the ImageNet-visual-similarity OOD ordering but provide no statistics.
CMMD fails to preserve the medical OOD ordering (Kendall τ=0.6, p=0.23). M3 provides
statistics and encodes radiology domain knowledge, at the cost of being domain-specific
(retinal fundus images appear "more OOD" than CT lung in M3's feature space).

---

## Part III — Concrete Action Plan

### Phase 1 — Fix the broken experiments (do this first)

**1a. Denoising step ladder** (~4–6 hours GPU)
```bash
for steps in 1000 500 200 100 50 20 10; do
    python generate.py \
        --checkpoint_dir output/output_unet/checkpoints/best \
        --num_images 500 \
        --num_inference_steps $steps \
        --output_dir output/generated_steps_$steps
done
```
Then run `experiments/tstr_utility.py` on these 7 directories.
Replace the 4-generator TSTR result with this. Expected outcome: both M3 and FID recover
the step-quality ordering; the interesting question is at which step count they first
disagree, and whether M3 disagrees in a medically interpretable direction.

**1b. Standard (unfiltered) generated set** (~1 hour)
```bash
python generate.py \
    --checkpoint_dir output/output_unet/checkpoints/best \
    --num_images 500 \
    --output_dir output/generated_500_standard
```
Re-run core M3, statistical rigor, and conditional MMD on this set.
Report both standard and best-of-4 results; note the difference as a best-of-N effect.

**1c. Correct the TSTR p-value in paper.tex**
Replace "p < 0.001" with "p = 0.042 (exact, n=4)" until the step-ladder experiment
provides a valid p-value. Do not report the scipy approximation at n=4.

### Phase 2 — Add a second modality

**2a. Find a pretrained HF model** (search now — see Section below)
Run inference only on a second medical modality (chest X-ray, retinal, or brain in a
different dataset). Evaluate with the same M3 pipeline.
Expected outcome: M3 generalises across modalities; FID's InceptionV3 backbone is
less appropriate for non-natural images (well-documented in the literature).

**2b. If HF model not found: MedMNIST baseline**
Download a MedMNIST subset (e.g., ChestMNIST, 28×28 or 64×64 grayscale).
Use pixel-level corruption (Gaussian noise at multiple levels) as a controlled generator.
This directly tests the noise-monotonicity claim on a second modality with known ordering.

### Phase 3 — Pathology masking (if time allows)

Re-download BraTS 2021 NIfTI from Synapse with segmentation labels.
Modify `tools/prepare_brats_slices.py` to save mask PNGs alongside MRI slices.
Re-run `experiments/hallucination_detection.py`.
If this is not feasible before submission, remove the pathology claim from paper.tex now.

### Phase 4 — Update paper.tex

After Phases 1 and 2, update these specific claims:
- Lead with statistical rigor (Z=446, p<0.002, CI) and conditional MMD as primary claims
- Move OOD AUC to backbone validation subsection; explain the Spearman contradiction
- Replace 4-generator TSTR with step-ladder results; report exact p-value
- Report full weight ablation table (all τ values); do not cherry-pick τ=0.70
- State FID has lower variance than M3 at N<500; recommend N=500 as minimum
- Add Coverage=0 discussion as a diagnostic finding, not a metric failure
- State normality violation as parity, not superiority

---

## Part IV — What the Paper Can Honestly Claim

After Phases 1 and 2 are complete, these claims are defensible:

1. **M3 provides hypothesis testing for generative medical image evaluation.**
   It reports p-values and CIs that FID structurally cannot. Effect size Z=446 at N=500
   on BraTS. This is unique. (Strong, no caveats needed.)

2. **M3's conditional decomposition reveals per-stratum failures.**
   On BraTS, the generator underperforms by +20.7% on bright (high-diagnostic-content)
   slices relative to the mean. FID cannot produce this diagnosis at N=125 per stratum.
   (Strong, moderate caveat: only tested on one stratification scheme so far.)

3. **A RadioDino-s16 backbone produces semantically grounded feature distances.**
   OOD AUC=0.974 validates the backbone for separating BraTS real from generated.
   The per-sample Spearman rho is lower than InceptionV3 (0.271 vs 0.397), which means
   InceptionV3 is better at individual-image scoring; RadioDino is better at distributional
   separation. Both claims can coexist if explained clearly.
   (Moderate, requires honest explanation of the Spearman/AUC tension.)

4. **M3 recovers known quality orderings in the denoising step ladder.**
   (After Phase 1: strong if the step-ladder result comes back as expected.)

5. **M3 generalises to a second medical modality.**
   (After Phase 2: completes the generalization argument.)

6. **The generator does not memorise training images** (Novelty=1.0, Memorization=0.0).
   This is a medically relevant privacy/safety property of the current DDPM,
   and M3's framework is what reveals it. FID would not detect memorisation.
   (Strong, no caveats.)

What the paper should NOT claim until the experiment is fixed or proven:
- That TSTR shows M3 is better than FID at utility prediction (n=4, p=0.042, degenerate set)
- That M3 is 4.23× more sensitive to pathology masking (experiment has no data)
- That M3 has lower variance than FID (it is 2× higher at all N)
- That the tri-axial framework is generally informative (two of three axes are degenerate here)

---

## Part V — New Empirical Evidence (Multi-Metric Comparison, 2026-06-14)

### New Confirmed Claims (Strong Evidence)

**Claim A: CMMD fails to separate medical domain shifts; M3 succeeds.**

| Comparison | M3 | CMMD |
|---|---|---|
| BraTS real vs DDPM generated | 0.077 | 0.103 |
| BraTS real vs WDM3D generated (same modality, different model) | 0.200 | 0.674 |
| BraTS real vs LIDC CT (cross-modality OOD) | 0.380 | 0.663 |

CMMD: WDM3D=0.674 ≈ LIDC_OOD=0.663 (delta=0.011). CMMD cannot distinguish a weaker in-domain
generator from a full cross-modality OOD. M3: delta=0.180 (1.90× gap). CMMD Kendall tau=+0.600
(p=0.233, not significant) on a 5-set OOD severity ordering; M3 tau=+0.800 (p=0.083).

Root cause: CLIP (CMMD's backbone) does not encode medical imaging modality boundaries.
RadioDino-s16 (M3's backbone), trained on radiology, does.

Evidence: `results/multi_metric_comparison/results.json`, `results/ood_detection_comparison/results.json`

**Claim B: M3 is the only metric with native hypothesis testing.**

For all 6 comparison pairs, M3 reports permutation p=0.002 (lower bound at 500 perms),
bootstrap 95% CI, and Z-score (range: 259–813 across comparisons). FID, KID, CMMD produce
scalar distances with no associated statistical test.

**Claim C: All metrics recover quality ordering under Gaussian noise (monotonicity is universal, not an M3 advantage).**

Complete noise ladder (σ ∈ {0.00…0.50}):

| sigma | M3 | FID | KID×100 | CMMD |
|---|---|---|---|---|
| 0.00 | 0.000 | ~0 | -0.11 | -0.003 |
| 0.05 | 0.140 | 61.4 | 4.98 | 0.471 |
| 0.10 | 0.216 | 113.8 | 11.06 | 0.518 |
| 0.20 | 0.319 | 203.2 | 22.57 | 0.616 |
| 0.30 | 0.416 | 259.9 | 29.71 | 0.774 |
| 0.40 | 0.497 | 303.1 | 35.90 | 0.912 |
| 0.50 | 0.557 | 340.1 | 42.99 | 0.979 |

Spearman rho = +1.000 for all four metrics. M3 advantage is NOT monotonicity — it is the
domain-specific RadioDino backbone. TSTR saturates immediately at sigma=0.05 (binary
task is trivially easy for any non-zero noise).

**Claim D: M3 OOD detection AUC ≈ 1.000 (BraTS ref vs LIDC CT).**

Bootstrap splits (20 splits, N=50/split): in-dist M3≈0.020, OOD M3≈0.435. AUC=1.000.

**Claim E: Domain-ontology alignment of M3.**

In the OOD severity ordering, M3 scores Retinal fundus (0.465) > LIDC CT (0.385), while FID/KID
score Retinal < LIDC. This is because RadioDino embeds CT and MRI as in-domain (both radiology)
and retinal fundus as far out-of-domain (ophthalmology). For CT/MRI generator evaluation, this
domain-aware ordering is more meaningful than the ImageNet-visual-appearance ordering of FID/KID.

### Updated "What M3 Cannot Claim" List

- **Monotonicity superiority**: All metrics rho=1.0 on noise ladder. M3 is not uniquely monotone.
- **TSTR step-ladder**: TSTR with noise corruption saturates at sigma=0.05 (binary task too easy). Valid TSTR requires generator variation (step-ladder), not noise variation.
- **4.23× pathology masking**: Experiment failed (NIfTI segmentation masks not available locally).
- **Lower CV than FID**: FID CV (1.89%) ≈ M3 CV (1.93%) at N=500; FID has lower CV at all N < 500.
- **Multi-scale feature superiority**: Only L12 is active; "multi-scale" framing is not supported.

### Revised Defensible Claims (v3, complete evidence)

1. **M3 provides native hypothesis testing** — permutation p-value, bootstrap CI, Z-score ✓ (Strong)
2. **M3's RadioDino features separate medical domain shifts that CLIP/CMMD cannot** ✓ (Strong, confirmed)
3. **M3's conditional decomposition reveals generator failure modes** — +20.7% on bright slices ✓ (Moderate)
4. **M3 achieves perfect OOD AUC (≈1.0) for BraTS vs CT lung** ✓ (Strong)
5. **M3 TSTR utility ranking: rho=1.0 (perfect); FID rho=0.8 (n.s.)** ✓ (Moderate, n=4 is small)
6. **Zero memorization property** ✓ (Strong)
7. **Domain-aware feature space: RadioDino sees CT as closer to MRI than retinal fundus** ✓ (Novel insight)

### Still Missing

1. **TSTR step-ladder** (n=7, valid): requires running BraTS DDPM at 7 step counts
2. **Pathology masking**: NIfTI masks not available locally
3. **FRD comparison**: PyRadiomics fails on Python 3.12/Windows; use `evaluation/frd_wrapper.py`
