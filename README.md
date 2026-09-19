# Does molecular structure add predictive signal to histology-only drug-response prediction?

Code and provenance for a study asking one question: **does the molecular
structure of the administered drug add predictive signal to a histology-only
model of treatment response?**

The endpoint is **ΔAUROC, with-SMILES minus without-SMILES**, on RECIST binary
response. That contrast is the study. A well-controlled null is a reportable
result and the analysis plan says so in advance.

## Cohort

- **Arm A — the primary endpoint.** Top 10 TCGA projects. 1,031 patients /
  127 drug strings / 1,211 slides. **Analysable: 981 patients / 94 drug names /
  82 distinct molecules** — patients whose only agents are biologics have no
  SMILES and are excluded by a declared inclusion criterion.
- **Arm B — TCGA-BRCA, descriptive only.** 199 patients, a strict subset of
  Arm A. 22 non-responders in 199; never powered, and it came out at chance
  three independent ways. No Arm B number is a result.

The cohort is **generated** by `scripts/check_drug_outcome_labels.py`, never by
clicking GDC portal facets. The generated tables in `cohort/` are tracked for
exactly that reason.

## Pipeline

| script | what it does |
|---|---|
| `check_drug_outcome_labels.py` | the go/no-go audit; generates `cohort/cohort_*.csv` and the GDC download manifest |
| `normalize_drugs_to_smiles.py` | free-text TCGA drug names → verified structures → `drugs/drug_smiles_map.json`. Refuses to guess. |
| `cohort_labels.py` | the single source of truth for a patient's label |
| `00_fetch_embed_batches.py` | batched fetch → md5 → embed → delete driver used on the rented GPU |
| `01_preprocess_and_extract.py` | OpenSlide → Otsu tissue mask → 256 px tiles (cap 4,000/slide) → 1024-d MahmoodLab/UNI (ViT-L/16) features |
| `10_fetch_receptor_status.py` | ER/PR/HER2 from the BCR clinical XMLs, joined into the model's context vector |
| `11_build_drug_embeddings.py` | the frozen pretrained molecular encoder (ChemBERTa-zinc, pinned revision, run once offline). Refuses to emit if the tokenizer is lossy on the drug set. |
| `08_evaluate_ablation.py` | **the experiment.** 5-fold CV grouped by patient and stratified by (response × cancer type), four arms, pooled out-of-fold predictions, permutation null, controls, sensitivity analyses |
| `merge_feature_manifests.py` | merges feature manifests from separate embedding runs |
| `06_attention_heatmap.py` | explainability figure — currently being rewritten against the per-patient model |

Frozen references, kept deliberately and **not** to be run:
`08_prebatch_reference.py`, `08_evaluate_ablation.py.bak-*` (regression
baselines for `--selftest`), and `09_tune_baseline.py` (the single permitted
hyperparameter-tuning pass has been spent).

`bench_embed.py` and `setup_pod.sh` are the GPU-rental toolkit that produced the
feature store.

## The four arms

`without_smiles` (baseline) · `with_smiles` · `shuffled_drug` (capacity control)
· `onehot_drug` (identity control).

The primary contrast is with vs without. The **capacity-matched** secondary is
`with_smiles` vs `shuffled_drug`, which holds parameter count fixed and changes
only whether the molecule is the right one.

**The evaluation unit is the patient.** Training samples are (patient, drug)
pairs, but a patient's predicted probabilities are averaged so every patient is
scored exactly once. A fold-level assertion raises if any patient appears in
both train and test.

Both arms receive tissue context (stage, cancer type, ER/PR/HER2). That is what
makes ΔAUROC measure chemistry rather than confounding by indication.

## Running it

```bash
conda activate histopath
huggingface-cli login          # UNI requires an accepted licence on HF
pip install -r requirements.txt

python scripts/check_drug_outcome_labels.py --arm-a     # rebuild the cohort
python scripts/normalize_drugs_to_smiles.py             # rebuild the drug map
python scripts/01_preprocess_and_extract.py             # tile + embed
python scripts/11_build_drug_embeddings.py --tag arm_a  # frozen molecular encoder
python scripts/08_evaluate_ablation.py --selftest       # never touches a real label
python scripts/08_evaluate_ablation.py --tag arm_a --dry-run
python scripts/08_evaluate_ablation.py --tag arm_a --permutation-null 200
```

Device auto-detection: CUDA → Apple MPS → CPU. `NUM_WORKERS=0` on MPS.
`transformers` is needed only to *build* the drug embeddings; running the
ablation reads the `.npz` with numpy.

## Layout

```
scripts/   live pipeline (table above) + frozen references
cohort/    generated cohort tables, GDC manifests, receptor status, audit
drugs/     verified structures, PubChem cache, frozen ChemBERTa embeddings
docs/      run records: gate4_runbook.md, arm_b_run_findings_*.md
output/    gitignored except the integrity record (see below)
data/      gitignored — .svs whole-slide images, 1.25 TB streamed from GDC
```

All paths anchor to the repo root (`BASE_DIR = Path(__file__).resolve().parents[1]`),
so the repo can be cloned anywhere and the launch directory does not matter.
Each step writes a JSON manifest the next step reads, carrying per-slide status
and the clinical label forward.

## What is tracked and why

`cohort/` and `drugs/` are small and they are the study's provenance trail —
they are how a reader checks 1,031 → 981 analysable, and
`drug_embeddings_chemberta_zinc.npz` is the frozen encoder the reproducibility
claim rests on.

`output/eval/run_log.jsonl` records **every configuration ever trained against a
real label**, with a config hash over source, arms, seeds, folds,
hyperparameters, `code_sha256` and `drug_encoder`. It cannot be reconstructed —
re-running writes a new log that proves nothing. `output/features/feature_manifest.json`
is the source of any "N slides excluded" statement.

The 18 GB feature store, the raw slides, and the 199 raw GDC clinical XMLs are
not tracked. The XMLs are re-fetchable from `cohort/gdc_manifest_*.txt`; the
derived `cohort/receptor_status_brca.csv` is tracked and is what the model reads.

## Data availability

All imaging and clinical data are from The Cancer Genome Atlas, obtained through
the NCI Genomic Data Commons. The exact cohort is reproducible from the tracked
manifests in `cohort/`. Drug structures come from PubChem; the resolution report
and cache are in `drugs/`.

## Analysis plan

The pre-registered analysis plan, including its deviations log, is maintained in
a separate private repository and will be deposited with the manuscript.
