# Reference-cohort confound — the finding that supersedes most earlier numbers

**Date:** 2026-09-01
**Scripts:** `experiments/depth_task_grid.py` (EXP 34),
`experiments/reference_set_construction.py` (EXP 35),
`experiments/cohort_heterogeneity_null.py` (EXP 36),
`experiments/unified_metric_table.py` (EXP 37)
**Results:** `results/depth_task_grid/`, `results/reference_set_construction/`,
`results/cohort_heterogeneity_null/`, `results/unified_metric_table/`

> **Read this before quoting any number from `EXPERIMENT_FINDINGS.md`.** Most of
> the historical numbers were computed against a real reference set built by
> taking the first 500 filenames in sorted order. That set spans **17 subjects**
> from one contributing institution and inflates every distance. It is not a
> valid reference.

---

## 1. What was wrong

`_glob_images(real_dir, 500)` returns `sorted(paths)[:500]`. BraTS filenames are
ordered by subject ID, so the first 500 slices come from subjects
`BraTS2021_00000`–`00016` — 17 subjects, all from the same contributing site.

| Real reference (N=500) | Distinct subjects | M3 (L12) vs DDPM |
|---|---|---|
| First 500 sorted (`_glob_images` default) | **17** | **0.1501** |
| Random 500 from the cohort | 413 | **0.0768** |

This is the origin of the long-standing "0.077 vs 0.150" discrepancy between
`results/multi_metric_comparison/results.json` and
`results/frd_comparison/frd_comparison.json`, and of the FID discrepancy
(68.77 vs 101.45). The two scripts built the real set differently:
`multi_metric_comparison.py:72` samples randomly, `frd_comparison.py` takes the
head of the sorted list. Both captions said "N=500 real BraTS slices".

**The paper's headline number (M3 = 0.153, CI [0.146, 0.162]) is on the
17-subject reference.** The corrected value on a subject-diverse reference is
≈ 0.073–0.080.

## 2. It is a site effect, not a sample-size effect

Holding N=500 and the draw procedure fixed, only the *first* block is anomalous:

| 17-subject block | M3 (L12) vs DDPM |
|---|---|
| First 17 subjects (what the default selects) | **0.152** |
| Middle 17 subjects | 0.084 |
| Last 17 subjects | 0.082 |
| Random 17 subjects (two seeds) | 0.081, 0.082 |

## 3. Real-vs-real can exceed real-vs-generated

| Comparison | Content | M3 (L12) |
|---|---|---|
| Real cohort A vs real cohort B | real vs real | 0.017 |
| Real (diverse) vs DDPM | real vs generated | 0.080 |
| **First-17 block vs rest of cohort** | **real vs real** | **0.096** |
| First-17 block vs DDPM | real vs generated | 0.152 |

Two sets of *real* BraTS images are further apart than real is from generated,
when the two real sets straddle the site boundary the default protocol selects.

## 4. Cohort diversity moves the null, not the signal

N=500 fixed, sweeping distinct subjects (3 seeds, `reference_set_construction`):

| Subjects | M3 L8 | M3 L12 | null L12 | FID |
|---|---|---|---|---|
| 17 | 0.1021 | 0.0801 | 0.0072 | 85.3 |
| 50 | 0.0975 | 0.0751 | 0.0024 | 71.7 |
| 100 | 0.0958 | 0.0739 | 0.0010 | 72.5 |
| 400 | 0.0957 | 0.0736 | 0.0000 | 67.9 |

Generated distance moves ×1.09; the **null** moves from 0.0000 to 0.0072; FID
moves ×1.26. Low diversity mainly destroys the ability to separate real from
generated.

## 5. The permutation test is the wrong instrument

Cohort-heterogeneity null (distances between disjoint real subject blocks) vs
the conventional permutation null, for the DDPM at L12:

| Reference cohort | cohort z | permutation z |
|---|---|---|
| 17 subjects | 15.7 | 423.3 |
| 50 subjects | 49.5 | 381.6 |
| 100 subjects | 89.0 | 347.2 |

**They move in opposite directions.** Improving the reference makes the
permutation test report *less* significance, because a more heterogeneous real
set inflates the shuffled null. The permutation test also assigns z > 340 to
every generator including retinal fundus — it never discriminates.

Use `z_cohort` and the ratio to the **mean** real-vs-real cohort distance
(DDPM: 5.5× at 17 subjects, 14.5× at 50, 30.3× at 100). Do NOT use the ratio
to the maximum: it is an order statistic whose expectation grows with the
number of cohort pairs, so 17-subject (276 pairs) and 100-subject (66 pairs)
conditions are not comparable on it. The 17 vs 50 comparison holds K=24 and
276 pairs fixed and already shows the trend.

## 6. Depth results, recomputed on a patient-disjoint reference

Full grid in `results/depth_task_grid/depth_task_grid.json`.

| Task | Best block | A-priori | Verdict |
|---|---|---|---|
| Fidelity (permutation z) | **L7–L9 plateau** (max at L8; the three are not distinguishable) | L12 | a priori is 22% below the plateau |
| Severity ordering | all 12 (ρ=+1.000) | L12 | depth-robust |
| Degradation monotonicity | L3–L12 | L12 | L1, L2 non-monotone |
| Memorization (MAE) | L10–L11 (0.004) | L9 (0.007) | **a priori near-optimal** |
| Coverage (k-NN recall) | **none** | L4 | invalid at all 12 depths |
| Per-image AUC | L9–L10 + Mahalanobis (0.972) | L12 + centroid (0.810) | rule matters more than depth |

### Two earlier conclusions are reversed

- **"Shallow layers beat L9 for memorization" is dead.** That was an artefact of
  the 17-subject reference. On a patient-disjoint reference, MAE falls
  monotonically with depth: L1 = 0.512, L6 = 0.027, L9 = 0.007, L11 = 0.004.
  The a-priori L9 is vindicated.
- **Coverage failure is real and reference-independent.** All 12 depths show
  *positive* ρ(mode drop, recall) — recall rises as diversity is removed.
  Mechanism: k-NN radius inflation. Dropping 80% of generated samples raises
  recall at L9 from 0.008 to 0.114.

## 7. Also fixed / noted

- `output/generated_500_best` and `output/generated_500_standard` are
  **byte-identical** (verified by MD5 over all 500 files). The "best-of-4
  cherry-picking" concern in `METHODOLOGY_AND_DIRECTION.md` Flaw 4 is moot.
- Cross-modality OOD (real vs LIDC CT) is **saturated**: AUC = 1.000 at every
  depth for every scoring rule. It cannot discriminate depths or rules and
  should not be a headline experiment.
- `balanced_draw` must shuffle slices within subject. Taking them in filename
  order draws only low-index slices, which are volume-edge positions — a second
  composition confound. Fixed in `cohort_heterogeneity_null.py`.

## 8. Protocol going forward

1. Report distinct subjects, not just N.
2. Build references by subject: sample subjects, then draw slices balanced
   across them and shuffled within them.
3. Keep real-vs-real comparisons subject-disjoint.
4. Contextualise against the inter-cohort distribution; report `z_cohort` and
   the ratio to the **mean** cohort distance at a stated cohort size.
5. Audit read-out depth for your own encoder — do not transplant ours. Here the
   fidelity optimum is a plateau (L7–L9, mutually indistinguishable), not L8
   specifically. Do not report k-NN coverage without running the mode-drop
   validation first.
6. One implementation per baseline metric (torchmetrics for FID/KID).

---

## 9. Train/evaluation contamination — CONFIRMED, unresolved (2026-09-01)

`train.py --data_dir data_mri/brats_axial_multislice` fine-tuned the DDPM on
**all 38,781 slices** — the same pool every evaluation reference is drawn from.
Every reference image was a training image.

**Why it matters:** contamination pulls `D(G, R)` *down*, which makes the
headline claim (real-real 0.096 > real-gen 0.080) *easier* to obtain. The
confound biases toward our own conclusion.

**What it does NOT touch** (all generator-free, so they stand):
- inter-cohort heterogeneity (0.017 typical, 0.096 for the head block)
- the null shrinking with cohort size (0.0072 → 0.0000)
- the first-17-subject anomaly
- the `real_probe` / `real_headblk` rows of the unified table

**The fix, set up but not yet run:** `tools/make_subject_split.py` builds a
subject-disjoint split (876 train / 375 held-out, hard-linked, zero extra disk)
with a manifest at `data_mri/brats_split/split_manifest.json`. Train on
`data_mri/brats_split/train`, then compare `D(G, R_train_subjects)` against
`D(G, R_heldout_subjects)`. The difference is a direct estimate of the
contamination effect and needs only ONE training run, not two.

## 10. Unified metric table (corrected protocol, balanced(100), N=500)

| Set | Content | M3 L8 | M3 L12 | FID | KID |
|---|---|---|---|---|---|
| real_probe | real vs real | 0.0016 | 0.0015 | 36.0 | 0.0009 |
| **real_headblk** | **real vs real** | **0.0628** | **0.0674** | **64.6** | **0.0299** |
| ddpm | real vs gen | 0.1028 | 0.0792 | 70.8 | 0.0380 |
| wdm3d_mri | real vs gen | 0.2155 | 0.2204 | 174.8 | 0.1574 |
| lidc_ct | real vs gen | 0.3886 | 0.4034 | 322.2 | 0.3725 |
| retinal | real vs gen | 0.5026 | 0.4632 | 248.4 | 0.2851 |

**A real-vs-real comparison scores FID 64.6; a real generator scores 70.8.**
Six points apart, against a 29-point spread between the two real rows.

Also reproduced on the corrected protocol: RadioDINO puts retinal further from
brain MRI than CT (0.463 vs 0.403); FID/KID put CT further (322 vs 248). The
domain-ontology difference is real and survives the reference fix.

## 11. Environment gotchas (cost real time on 2026-09-01)

- **Do not `pip install monai` unpinned.** MONAI ≥1.5 requires torch ≥2.8 and
  will upgrade torch, breaking `torchvision 0.21.0+cu124` with
  `RuntimeError: operator torchvision::nms does not exist`. Use
  `pip install "monai==1.4.0" --no-deps`. Recovery:
  `pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124`.
- **`train.py` has no `--output_dir`** (CLAUDE.md is stale). It is
  `--checkpoint_dir` plus `--log_file`.
- **MONAI `CacheDataset` at `cache_rate=1.0` needs ~7 GB for a 27k-slice split**
  and will exhaust a 32 GB machine alongside another job. `train.py` now takes
  `--cache_rate`; use 0.05–0.2 when sharing the machine.
- **Training cannot share the GPU with another job on this 8 GB card.** Running
  both drove free VRAM to 53 MiB and free RAM to 0.4 GB. Schedule a dedicated
  window.
