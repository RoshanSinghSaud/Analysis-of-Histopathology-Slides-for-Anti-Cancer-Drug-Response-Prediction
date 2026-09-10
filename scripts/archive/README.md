# archive/ — superseded code, kept for provenance only

Moved here 29 August 2026. **Nothing in this folder is on the current path.**
No live script imports any of it, and nothing in `output/features/` was
produced by it. It is kept because it is the ancestry of the current pipeline
and because a reviewer asking "what did you try first?" deserves an answer.

**Do not run these. Do not import them. Do not repair them.**
If one of them looks like it does something you need, that thing now lives in
`08_evaluate_ablation.py`.

| file | what it was | what replaced it |
|---|---|---|
| `02_feature_extraction_local.py` | Phikon (768-d) tile features on the Mac | `01_preprocess_and_extract.py`, UNI (1024-d) on a rented GPU. **Every feature in the study came from 01_.** |
| `03_abmil_aggregation.py` | standalone gated-attention MIL, one embedding per slide | the `ResponseModel` class inside `08_evaluate_ablation.py`, which pools every diagnostic slide of a patient into ONE bag |
| `04_cross_modal_fusion.py` | histology + Morgan fingerprint fusion, three modes | the drug branch of `08_`, plus `11_build_drug_embeddings.py` for the frozen encoder |
| `05_predict_drug_response.py` | BRCA-only response head | `08_`, which is cohort-agnostic via `--tag` |
| `07_pcr_prediction.py` | pathologic-complete-response endpoint | nothing — **pCR was dropped from scope.** The endpoint is RECIST binary response. |
| `check_pcr_labels.py` | pCR label feasibility scan | dropped with pCR. Its BCR-XML parsing logic was the seed for `10_fetch_receptor_status.py`. |
| `check_svs_xml_match.py` | one-off slide↔XML pairing check, hardcoded paths | folded into `check_drug_outcome_labels.py` |

## What is still live, for contrast

`cohort_labels.py` (label source of truth) · `check_drug_outcome_labels.py`
(builds the cohort) · `normalize_drugs_to_smiles.py` (builds the drug map) ·
`00_` + `01_` (fetch, tile, embed) · `10_` (receptors) · `11_` (molecular
encoder) · `08_evaluate_ablation.py` (**the experiment**).

Frozen references that must not be run or edited: `08_prebatch_reference.py`,
`08_evaluate_ablation.py.bak-*`, `09_tune_baseline.py` (the single permitted
tuning pass is spent).

`06_attention_heatmap.py` stays in `scripts/` and is deliberately NOT archived:
it is broken against the current model (it assumes per-slide attention, while
`08_` pools slides per patient) but attention heatmaps are the explainability
figure the paper needs. It is a rewrite, not a relic.
