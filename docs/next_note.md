# next_note.md — working orientation for future sessions

Companion to `CLAUDE.md`. CLAUDE.md = *how the repo is built*; this file = *what state the
research is in and what to do next*, so a fresh session can act without re-deriving context.

Last synced from repo + `EXPERIMENT_FINDINGS.md` + `METHODOLOGY_AND_DIRECTION.md`.

## 2026-08 debug-then-prove pass — DONE (A, B, C)

Spec: `docs/specs/2026-07-25-m3-debug-then-prove-design.md`. All three workstreams complete.

- **A — metric debug** (`experiments/debug_metric_ablation.py`, EXP 29): single-layer L12 ≥ the
  entropy×stability×uniqueness weighting on Z and CV; the entropy/"semanticity" axis is inert for
  the timm RadioDino-s16 backbone (constant `_DUMMY_ENTROPY`). Default `single_layer=12` kept;
  warning added in `m3_score_v2.py.__init__`.
- **B — lesion-specificity with REAL LGG expert masks** (`experiments/lgg_lesion_specificity.py`
  + `lgg_noise_sweep.py`, EXP 30/31): the old "4.23× pathology sensitivity" is dead. New defensible,
  replicated result: under *subtle* noise M3 is lesion-selective **1.4–1.8×** vs Inception-MMD's
  ~1.05×, robust to two area-matched controls (mirror + texture-matched) and a σ-sweep,
  backbone-attributable. Publication figure: `figures/lesion_specificity/lgg_noise_sweep.png`.
  (Full 748 MB LGG dataset re-downloaded — the local copy had been truncated at 6 MB.)
- **C — paper reframe**: `paper.tex` abstract/contributions/§Lesion-Specificity/discussion/conclusion
  rewritten to the real-mask result; `refs.bib` +buda2019association +liu2008isolation; compiles clean
  (exit 0, 0 undefined refs, 19 pp, Fig. 5 renders). Backup at `paper.tex.bak`.

**Open next:** (i) evaluate a lesion-conditioned generator against these real masks (turns the
metric-sensitivity test into a generator-quality claim); (ii) σ-sweep at larger N to tighten the
CI bands; (iii) put the coverage axis on firmer footing at N≥500. The paper is NOT git-tracked —
`paper.tex.bak` is the only undo.

## 2026-08 reviewer-response pass — DONE

Addressed a TMI-style major-revision review. Key outcomes:
- **Method figure**: `figures/M3Score-Calculation.{drawio,pdf}` (draw.io source + TikZ-rendered PDF,
  source `figures/M3Score-Calculation_src.tex`). Shows canonical CLS vs alternative attention pooling.
- **2.1 CLS-vs-attention (real bug)**: code truth = RadioDINO-s16 (timm) uses **CLS**; attention
  pooling only for `transformers` backbones. §III-D rewritten to match §III-B/Fig 1/code.
- **2.2 blur figure (real bug)**: was stale (M3 bar 0.700, non-monotone, empty perceptual panel).
  Regenerated from JSON (`tools/regen_noise_plots.py`) + computed & injected SSIM/PSNR/MS-SSIM
  (`scratchpad/fill_perceptual.py`); now M3 ρ=1.000 monotone, perceptual panel populated.
- **3.3 CMMD central claim (major)**: bootstrap CIs (`experiments/cmmd_structural_ci.py`, EXP 32)
  showed the paper's "Δ=0.011 within noise" does NOT reproduce. Reframed (user-approved) to a
  statistically-significant **rank inversion**: CMMD ranks LIDC-CT closer to BraTS-MRI (0.699) than
  a weak MRI generator (0.738), Δ=−0.039 [−0.059,−0.021]; M3 and FID order correctly. Updated
  Table V + abstract + contributions + conclusion. Also fixed a pre-existing internal inconsistency
  (Table V DDPM M3 0.077 → 0.150, now matching the abstract's 0.153).
- **2.4** layer-depth: honest limitation added (only L12 CKA-corroborated; L9/L4 depth-ablation = future work).
- Verified as **PDF-extraction artifacts, not source bugs**: garbled algorithms (2.6), table typos /
  "0.07'" (3.2), Table II direction (3.1); Z-score caveat (2.5) already present in §III-B.
- Paper compiles clean: `latexmk`/pdflatex exit 0, 0 undefined refs, 19 pp. Diagram + all figures render.

---

---

## 1. What this project actually is

A **metric paper**, not a generation paper. The product is **M3-Score**
(`evaluation/m3_score_v2.py`, class `M3EntropyMetric`). The DDPM generator and WDM-3D baseline
exist only to produce samples for validating the metric. Most work = running experiments,
reconciling results, and updating `paper.tex`.

**Source of truth for numbers:** `EXPERIMENT_FINDINGS.md` (dated experiment log) →
`METHODOLOGY_AND_DIRECTION.md` (honest assessment + plan) → then `paper.tex`.
`paper.tex` and `README.md` still carry the **original, now-superseded** numbers.

---

## 2. Current state — validated vs contested

**Solid / defensible (lead with these):**
- Native hypothesis testing: permutation p-value + bootstrap CI. FID structurally cannot.
  Effect size Z≈446 at N=500 on BraTS.
- Conditional MMD (intensity-quartile stratification) exposes per-stratum generator failures
  (generator worst on high-intensity Q4 slices).
- TSTR utility correlation ρ=1.000 for M3 vs 0.800 for FID (but n=4 → weak p-value; see 3a).
- Zero training-image memorization in the DDPM.
- Backbone choice: RadioDino-s16 wins OOD-AUC/discriminability across 6 backbones.

**Contested / must be reframed before publishing (see CLAUDE.md → Contested Claims):**
- OOD AUC: **0.9743**, not 0.863.
- Layers: **single layer [12]**, not multi-scale [1,4,12]. Update the "multi-scale" framing.
- CV: M3 **1.93%** and FID **1.89%** at N=500 — FID is *not* unstable at N=500; M3's edge is
  the statistical test, not lower variance. FID has *lower* variance at N<500.
- **Pathology-masking claim is reversed** — FID/CMMD are *more* artifact-sensitive; M3 is more
  *specific* to semantic shift. Do not claim "4.23× more sensitive." Reframe or drop.

---

## 3. Action plan (from METHODOLOGY Part III — do in order)

**Phase 1 — fix broken experiments (highest priority):**
- **3a. Denoising step ladder** (~4–6h GPU): generate at steps ∈ {1000,500,200,100,50,20,10}
  into `output/generated_steps_$steps`, then `experiments/tstr_utility.py` across them.
  Replaces the weak 4-generator TSTR; gives a valid p-value and a monotone-quality axis.
- **3b. Unfiltered standard set:** regenerate `output/generated_500_standard` (no best-of-N);
  re-run core M3 + statistical_rigor + conditional_mmd. Report standard vs best-of-4 as a
  best-of-N effect.
- **3c.** In `paper.tex`, replace TSTR "p<0.001" with the exact n=4 p-value until 3a lands.

**Phase 2 — second modality:** run inference-only with a pretrained HF model on CXR/retinal,
or fall back to MedMNIST + Gaussian-corruption as a controlled generator (already partly done —
`run_multidataset.py` covers RetinaMNIST + PneumoniaMNIST).

**Phase 3 — pathology masking (if time):** re-download BraTS NIfTI with seg labels, extend
`tools/prepare_brats_slices.py` to emit mask PNGs, re-run `hallucination_detection.py`.
If infeasible, remove the pathology claim from `paper.tex`.

**Phase 4 — update `paper.tex`:** lead with statistical rigor + conditional MMD; move OOD to a
backbone-validation subsection; report the full weight-ablation table (don't cherry-pick τ=0.70);
state FID's lower variance at N<500 and recommend N=500 minimum.

---

## 4. How to move efficiently

**Reproduce the main result set:**
```bash
python run_all_experiments.py --run_id 5 --seed 42 --device cuda:0     # EXP 1–12
python run_experiments_13_19.py --run_id 5 --seed 42 --device cuda     # EXP 13–19
# outputs → results/radiodino-s16_run5/  (+ master_results.json)
```
**Compare backbones:** `python run_all_backbones.py --device cuda:0` (all 6) or `--backbones rad-dino dinov2`.
**One-off metric check:** `evaluation/eval_pipeline.py --metrics fid kid ssim m3 alpha`.
**Single experiment:** run `experiments/<name>.py` directly (each has `run_<name>()` + argparse).

**Where things land:** `results/experiments_output_v5/` is the latest curated run;
purpose-named dirs (`ood_detection_comparison/`, `pathology_masking/`, `cka_analysis/`,
`multidataset_run/`) hold `*_report.json` + PNGs. Generated image sets are under `output/`.

---

## 5. Traps that will waste your time

- **Don't quote paper/README numbers.** Cross-check `EXPERIMENT_FINDINGS.md` first.
- **`M3EntropyMetric` defaults to `single_layer=12`** → CKA multi-scale is OFF and
  `prune_layers_via_cka()` is a no-op. Pass `single_layer=None` for the multi-scale path.
  `canonical_layers.json` is empty and unused right now.
- **`config/global.json` = medigan registry** (3rd-party), **`results/master.json` = experiment OUTPUT.**
  Neither is app configuration. Reading them fully is expensive and pointless.
- **Rotation fix:** `generate.py` rotates 90° CCW by default; disable with `--no_rotation_fix`
  only for models retrained after the `PILReader` dataloader fix.
- **Backbone loads need `timm` / `open_clip`** (not pinned in requirements).
- **OOM in feature extraction** → set `batch_size=1` in the metric.
- **No git history** — you can't `git log` your way to context here; use the `.md` logs.

---

## 6. Open questions / unfinished

- Reconcile the OOD Spearman-vs-AUC contradiction noted in `METHODOLOGY_AND_DIRECTION.md`.
- Decide final framing: metric is *more specific*, not *more sensitive* — audit every
  "sensitivity" sentence in `paper.tex`.
- `pretrained/wdm3d_new/` is empty — confirm whether a newer WDM-3D checkpoint is expected.
- Full weight-ablation table (all τ) still needs to go into the paper.

_When you finish a phase, update §2/§3 here and the corresponding row in `EXPERIMENT_FINDINGS.md`._
