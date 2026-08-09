# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-modal drug response prediction pipeline for breast cancer (TCGA-BRCA). Predicts chemotherapy response from whole-slide histopathology images (WSI) fused with drug molecular fingerprints. Data is sourced from the GDC Data Portal using `gdc-client`.

The pipeline runs as a sequence of numbered Python scripts, each reading the previous step's manifest JSON:

1. **01_preprocess_and_extract.py** — Opens `.svs` slides via OpenSlide, builds a tissue mask (Otsu on HSV saturation), tiles at 256px, and extracts 1024-dim features with MahmoodLab/UNI (ViT-L/16). This merged script supersedes the older separate `02_feature_extraction_local.py` (Phikon, 768-dim). Outputs `output/features/<slide_id>_features.pt` and `feature_manifest.json`. Supports checkpoint/resume — completed slides are skipped on restart.
2. **02_feature_extraction_local.py** — Legacy standalone feature extractor using Phikon (768-dim). Kept for reference; the merged script 01 is the current path.
3. **03_abmil_aggregation.py** — Attention-Based MIL (Ilse et al. 2018) with gated attention. Aggregates tile features → slide embedding `[1, 256]` + per-tile attention scores. Training activates at ≥10 labelled slides; below that it runs inference-only with random weights. Outputs `output/abmil/abmil_manifest.json`.
4. **04_cross_modal_fusion.py** — Fuses histology slide embedding with drug Morgan fingerprint (ECFP4, 1024-bit via RDKit). Drug SMILES are resolved from PubChem API with a local cache + manual fallback dict. Three fusion modes: `concat` (default, no training), `mlp`, `cross_attn`. Outputs `output/fusion/fusion_manifest.json`.

## Environment Setup

```bash
conda activate histopath
huggingface-cli login          # one-time; UNI model requires accepted license on HF
pip install timm huggingface_hub openslide-python scikit-image tqdm torch torchvision transformers
conda install -c conda-forge rdkit   # required for Step 4 drug fingerprints
```

Device auto-detection: CUDA → Apple MPS → CPU. Scripts default to `NUM_WORKERS=0` (required on MPS).

## Running the Pipeline

```bash
python 01_preprocess_and_extract.py            # Steps 1+2 merged
python 01_preprocess_and_extract.py --no-resume # force re-extract all slides
python 03_abmil_aggregation.py                  # Step 3
python 04_cross_modal_fusion.py                 # Step 4
```

## Key Architecture Decisions

- **All paths are hardcoded to `~/Desktop/Research/`** — each script sets `BASE_DIR`, `SVS_DIR`, etc. as module-level constants using `Path.home()`. There are no CLI flags for paths; edit the CONFIG section at the top of each script.
- **Manifest-driven pipeline** — each step writes a JSON manifest that the next step reads. The manifests track per-slide status (`done`, `failed`, `skipped`, `no_tissue`) and carry forward the clinical label.
- **Clinical labels from GDC XML** — patient barcode is matched between `.svs` filename and XML. Drug response is mapped to binary classes: responder (`Complete Response`, `Partial Response` → 1) vs. non-responder (`Stable Disease`, `Clinical Progressive Disease` → 0).
- **Checkpoint/resume** — Step 1 uses atomic file writes (`os.replace` via temp file) and updates the manifest after every slide. Set `SKIP_EXISTING = True` (default) to resume.
- **INPUT_DIM coupling** — Step 3's `INPUT_DIM` must match the feature extractor's embed dim (1024 for UNI in step 01, 768 for Phikon in step 02).

## Data Layout

```
data/
  wsi/         ← .svs whole-slide images (TCGA-BRCA)
  clinical/    ← GDC clinical XML per patient
  raw/         ← raw GDC downloads
output/
  features/    ← per-slide .pt feature tensors + coords + manifest
  abmil/       ← slide embeddings + attention scores + manifest
  fusion/      ← fused vectors + drug SMILES cache + manifest
```
