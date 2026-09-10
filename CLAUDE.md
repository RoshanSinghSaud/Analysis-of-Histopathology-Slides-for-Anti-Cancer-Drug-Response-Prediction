# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Last reconciled with the code: 29 August 2026.** If what you read here
disagrees with `docs/analysis_plan_preregistration.md`, the pre-registration
wins and this file is stale — say so rather than working around it.

## Project Overview

Does the **molecular structure of the administered drug** add predictive signal
to a histology-only model of treatment response? The endpoint is **ΔAUROC,
with-SMILES minus without-SMILES**, on RECIST binary response. That contrast
*is* the paper; a well-controlled null is a publishable result and §8 of the
pre-registration says so.

**Scope is pan-cancer, not breast-only.** Older prose in `docs/` and in
archived scripts that says "TCGA-BRCA" is describing Arm B.

- **Arm A — the primary endpoint.** Top 10 TCGA projects. 1,031 patients /
  127 drug strings / 1,211 slides. **Analysable: 981 patients / 94 drug names /
  82 distinct molecules** — patients whose only drugs are biologics have no
  SMILES and are excluded by a declared inclusion criterion.
- **Arm B — TCGA-BRCA, a descriptive showcase.** 199 patients, a strict
  *subset* of Arm A. 22 non-responders in 199; it was never powered and it came
  out at chance three independent ways. Do not treat an Arm B number as a result.

## The pipeline that actually produced the study

Numbered scripts, each reading the previous step's manifest JSON. **Anything
not listed here is either a frozen reference or lives in `scripts/archive/`.**

| script | what it does |
|---|---|
| `check_drug_outcome_labels.py` | the go/no-go audit. **Generates** `cohort/cohort_*.csv` and the GDC download manifest. The cohort is never assembled by clicking GDC portal facets. |
| `normalize_drugs_to_smiles.py` | free-text TCGA drug names → verified structures → `drugs/drug_smiles_map.json`. Refuses to guess. |
| `cohort_labels.py` | **the single source of truth for a patient's label.** Every script that needs one imports it from here. |
| `00_fetch_embed_batches.py` | the batched fetch→md5→embed→delete driver used on the rented GPU |
| `01_preprocess_and_extract.py` | OpenSlide → Otsu tissue mask → 256 px tiles (cap 4,000/slide) → **1024-d MahmoodLab/UNI (ViT-L/16)** features. Writes `output/features/<slide_id>_features.pt` + `feature_manifest.json`. Checkpoint/resume. |
| `10_fetch_receptor_status.py` | ER/PR/HER2 from the BCR clinical XMLs → `cohort/receptor_status_<tag>.csv`, joined into the model's context vector |
| `11_build_drug_embeddings.py` | **the frozen pretrained molecular encoder.** ChemBERTa-zinc, pinned revision, run once offline → `drugs/drug_embeddings_chemberta_zinc.npz` + provenance JSON. Refuses to emit if the tokenizer is lossy on the drug set. |
| `08_evaluate_ablation.py` | **THE EXPERIMENT.** 5-fold CV grouped by patient and stratified by (response × cancer type), four arms, pooled out-of-fold predictions, permutation null, controls, sensitivity analyses, run log. |
| `merge_feature_manifests.py` | merges feature manifests from separate embedding runs (self-tested against 6 failure modes) |
| `06_attention_heatmap.py` | explainability figure. **Currently broken** — assumes per-slide attention, but `08_` pools all of a patient's slides into one bag. Needs a rewrite, not deletion. |

**Frozen references — do not run, do not edit:** `08_prebatch_reference.py`,
`08_evaluate_ablation.py.bak-*` (regression baselines for `--selftest`), and
`09_tune_baseline.py` (the single hyperparameter-tuning pass §6 permits has
been spent; re-running it is a protocol violation).

`bench_embed.py` and `setup_pod.sh` are the GPU-rental toolkit. All GPU
spending finished 29 Aug 2026; they are archival.

`scripts/archive/` holds the superseded ancestry (steps 02–05, 07, and two
`check_*` scripts). See `scripts/archive/README.md`. Nothing imports it and no
feature in the study came from it.

## The four arms and the contrasts

`without_smiles` (baseline) · `with_smiles` · `shuffled_drug` (capacity
control) · `onehot_drug` (identity control). Primary contrast is with vs
without; the **capacity-matched** secondary is `with_smiles` vs `shuffled_drug`,
which holds parameter count fixed and changes only whether the molecule is the
right one.

**The eval unit is the PATIENT.** Training samples are (patient, drug) pairs,
but a patient's predicted probabilities are averaged so every patient is scored
exactly once. A fold-level assert raises on any patient appearing in both train
and test.

## Rules that are load-bearing — violating these invalidates the study

- **Hyperparameters are frozen** (`dropout 0.25 / weight_decay 1.0 /
  max_epochs 40 / patience 8`). The one permitted tuning pass is spent. Never
  re-tune after seeing a ΔAUROC. Any change is a §10 deviation.
- **The drug branch gets no tuning budget.** `drug_feat_dim=32`,
  `drug_proj_dim=16`. Both encoders reduce to the same 32 dims on purpose: a
  wider drug branch produces a negative ΔAUROC out of parameter count alone
  (measured at −0.066 on data with no drug effect).
- **The primary p-value is `--permutation-null` (N ≥ 200), not DeLong.** DeLong
  assumes fixed models; every fold retrains both arms, so it is
  anti-conservative here. Never tune against the permutation null.
- **Every run against real labels is logged** to `output/eval/run_log.jsonl`
  with a config hash covering source, arms, seeds, folds, hparams,
  `code_sha256` and `drug_encoder`.
- **Anything that changes the drug branch must re-run `--selftest`** and clear
  the null-bias check at production width.

## Environment Setup

```bash
conda activate histopath
huggingface-cli login          # one-time; UNI requires an accepted licence on HF
pip install -r requirements.txt
```

Device auto-detection: CUDA → Apple MPS → CPU. `NUM_WORKERS=0` on MPS.
`transformers` is needed only to *build* the drug embeddings; running the
ablation reads the `.npz` with numpy.

## Running it

```bash
python scripts/check_drug_outcome_labels.py --arm-a        # rebuild the cohort
python scripts/normalize_drugs_to_smiles.py                # rebuild the drug map
python scripts/01_preprocess_and_extract.py                # tile + embed
python scripts/11_build_drug_embeddings.py --tag arm_a     # frozen molecular encoder
python scripts/08_evaluate_ablation.py --selftest          # never touches a real label
python scripts/08_evaluate_ablation.py --tag arm_a --dry-run
python scripts/08_evaluate_ablation.py --tag arm_a --permutation-null 200
```

⚠️ `01_preprocess_and_extract.py` has a module-level `COHORT_TAG`. With it set
to `None` while both cohort CSVs exist, the embedder cannot resolve which
cohort it is working on. Check it before any long run.

## Key Architecture Decisions

- **All paths anchor to the repo root** — every script sets
  `BASE_DIR = Path(__file__).resolve().parents[1]`. The repo can be moved or
  cloned anywhere and the launch directory does not matter.
- **Manifest-driven** — each step writes a JSON manifest the next step reads,
  tracking per-slide status (`done`, `failed`, `skipped`, `no_tissue`) and
  carrying the clinical label forward.
- **Labels** — responder (`Complete Response`, `Partial Response`) vs
  non-responder (`Stable Disease`, `Clinical Progressive Disease`). Note the
  model's positive class is **non-responder**. Patients whose treatment records
  give both outcomes are excluded, not counted twice.
- **Checkpoint/resume** — atomic writes (`os.replace` via temp file), manifest
  updated after every slide.
- **Both arms receive tissue context** (stage, cancer type, ER/PR/HER2). This
  is what makes ΔAUROC measure chemistry rather than confounding by indication,
  and it is load-bearing for the paper's central defence.

## Repository Layout

```
scripts/          ← live pipeline (table above) + frozen references
scripts/archive/  ← superseded ancestry; see its README
docs/             ← pre-registration (§10 deviations log), runbooks, checklists
cohort/           ← generated by check_drug_outcome_labels.py: cohort_*.csv,
                    cohort_slides_*.csv, cohort_*_summary.json,
                    gdc_manifest_*.txt, receptor_status_*.csv,
                    drug_outcome_audit.{json,md}
drugs/            ← drug_smiles_map.json, pubchem_cache.json,
                    drug_embeddings_chemberta_zinc.{npz,json},
                    drug_resolution_report.md, drug_review_for_pharmacist.xlsx
tools/            ← gdc-client binary (gitignored)
```

`cohort/` and `drugs/` are tracked — they are small and they are the paper's
provenance trail. The `.npz` belongs in git for the same reason. `data/` and
`output/` are gitignored.

## Data Layout

```
data/
  wsi/         ← .svs whole-slide images (deleted after embedding; 1.25 TB streamed)
  clinical/    ← GDC clinical XML per patient
output/
  features/    ← 2,422 per-slide .pt tensors + coords + feature_manifest.json (19.3 GB)
  eval/        ← run_log.jsonl — every configuration ever trained against labels
```
