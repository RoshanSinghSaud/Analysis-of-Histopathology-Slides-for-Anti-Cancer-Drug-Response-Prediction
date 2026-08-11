"""
=============================================================================
STEP 1+2 (MERGED): Preprocess + UNI Feature Extraction  — one streaming pass
=============================================================================

Replaces the old separate `preprocess.py` (tile -> save PNGs -> stain-norm ->
save more PNGs) and `02_feature_extraction_local.py` (read PNGs -> ViT/Phikon).

WHY MERGED
----------
The old flow wrote 2 copies of every tile (raw + normalized) to disk, then a
second script read them all back. For 200+ slides x up to ~30k tiles that is
hundreds of GB of disk I/O and the slowest part of the project. Here each slide
is processed fully IN MEMORY:

    open WSI -> tissue mask -> tile -> (optional stain norm) -> UNI -> features

Only two small files per slide are written:
    output/features/<slide_id>_features.pt   tensor [N, 1024]   (for ABMIL)
    output/features/<slide_id>_coords.pt     dict w/ [N,2] coords (for heatmaps)

No PNG tiles are persisted. Tile (x, y) level-0 coordinates ARE persisted, so
Step 6 can still paint attention heatmaps back onto the slide.

FEATURE MODEL : MahmoodLab/UNI  (ViT-L/16, 1024-dim)  -- gated HF model.
                You must (1) accept the license on the model page and
                (2) be logged in:  `huggingface-cli login`  (or set HF_TOKEN).

SAFEGUARDS for 200+ cases
-------------------------
1. CHECKPOINT / RESUME : the run manifest is written after EVERY slide and
   completed slides are skipped on restart. Crash at slide 150 -> rerun resumes
   at 151. Outputs are written atomically (temp file -> os.replace).
2. TILE CAP : MAX_TILES_PER_SLIDE caps tiles/slide (seeded random sample) so a
   huge slide cannot blow up memory or runtime.
3. COHORT CHECK : every slide is matched to a patient in the cohort CSV by
   TCGA barcode. Labels come from that CSV and are never re-derived here --
   see scripts/cohort_labels.py for why. Slides outside the cohort are skipped
   rather than embedded, and a summary is printed at the end.

Run:
    conda activate histopath
    huggingface-cli login        # one time, needs UNI access granted
    pip install timm huggingface_hub openslide-python scikit-image tqdm
    python scripts/01_preprocess_and_extract.py
=============================================================================
"""

import os
import json
import time
import random
import logging
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import openslide
from tqdm import tqdm

import torch

# =============================================================================
# ── CONFIG ───────────────────────────────────────────────────────────────────
# =============================================================================

BASE_DIR   = Path(__file__).resolve().parents[1]
SVS_DIR    = BASE_DIR / "data" / "wsi"
OUTPUT_DIR = BASE_DIR / "output"
FEAT_DIR   = OUTPUT_DIR / "features"

# --- Labels ---
# Which cohort arm to label against. None = auto-detect a single cohort_*.csv
# in cohort/, and refuse to guess when there are several.
COHORT_TAG = None

# --- Tiling ---
TILE_SIZE   = 256     # tile size in pixels at the processing level
MAG_LEVEL   = 0       # 0 = highest resolution
TISSUE_THRESH = 0.50  # min fraction of a tile that must be tissue to keep it
MASK_DOWNSAMPLE = 32  # build the fast tissue mask at ~this downsample factor

# --- 200+ case safeguards ---
MAX_TILES_PER_SLIDE = 1000   # hard cap; seeded random sample if exceeded
SEED                = 42     # makes the tile sampling reproducible
SKIP_EXISTING       = True   # resume: skip slides already extracted
SKIP_NOT_IN_COHORT  = True   # don't spend GPU time on slides outside the cohort

# --- Stain normalization (OFF by default: UNI is fairly stain-robust and this
#     is the biggest per-tile cost). Flip to True if a reviewer wants it. ---
APPLY_STAIN_NORM = False
STAIN_REF_SAMPLE = 100   # tiles sampled to compute the LAB reference stats

# --- UNI feature extractor ---
UNI_REPO   = "hf-hub:MahmoodLab/UNI"
EMBED_DIM  = 1024
BATCH_SIZE = 32          # raise on a CUDA GPU; 16-32 is safe on Mac MPS
NUM_WORKERS = 0          # keep 0 on MPS

# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess+uni")

random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# 1. COHORT LABELS  (read, never re-derived -- see scripts/cohort_labels.py)
# =============================================================================
#
# This script used to walk each patient's clinical XML and keep the LAST
# drug_name and the LAST measure_of_response tag it happened to encounter, with
# no pairing between the two. That silently produced a different label than the
# study is pre-registered on. Labels now come from the audit script's cohort
# CSV, which carries the strict drug<->outcome pairing, the mixed-outcome
# exclusions and the RECIST-only rules.

from cohort_labels import (BARCODE_RE, CohortError, compact, find_cohort_csv,  # noqa: E402
                           label_for_slide, load_cohort, summarise)


# =============================================================================
# 2. TISSUE MASK + TILING  (fast: mask first, only read tissue tiles)
# =============================================================================

def build_tissue_mask(slide: openslide.OpenSlide):
    """
    Build a low-res boolean tissue mask via Otsu on HSV saturation.
    Returns (mask[bool HxW], mask_downsample).
    """
    from skimage.color import rgb2hsv
    from skimage.filters import threshold_otsu

    w0, h0 = slide.level_dimensions[0]
    thumb_w = max(1, int(w0 / MASK_DOWNSAMPLE))
    thumb_h = max(1, int(h0 / MASK_DOWNSAMPLE))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    thumb_np = np.asarray(thumb)
    real_ds = w0 / thumb_np.shape[1]   # actual downsample after PIL rounding

    sat = rgb2hsv(thumb_np)[:, :, 1]
    try:
        thr = threshold_otsu(sat)
    except ValueError:
        thr = 0.0
    mask = sat > max(thr, 0.05)        # tissue = saturated; background = white
    return mask, real_ds


def tile_fraction(mask: np.ndarray, mask_ds: float,
                  x0: int, y0: int, span0: int) -> float:
    """Fraction of a level-0 tile bbox that is tissue, read from the low-res mask."""
    mx0 = int(x0 / mask_ds); my0 = int(y0 / mask_ds)
    mx1 = int((x0 + span0) / mask_ds); my1 = int((y0 + span0) / mask_ds)
    mx1 = max(mx1, mx0 + 1); my1 = max(my1, my0 + 1)
    region = mask[my0:my1, mx0:mx1]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def list_tissue_tiles(slide: openslide.OpenSlide) -> list[tuple[int, int]]:
    """
    Return level-0 (x, y) top-left coords of all tissue tiles for the slide,
    capped/sampled to MAX_TILES_PER_SLIDE.
    """
    mask, mask_ds = build_tissue_mask(slide)

    level = MAG_LEVEL
    ds = slide.level_downsamples[level]
    w, h = slide.level_dimensions[level]
    span0 = int(TILE_SIZE * ds)                  # tile footprint in level-0 px

    n_x = (w - TILE_SIZE) // TILE_SIZE + 1
    n_y = (h - TILE_SIZE) // TILE_SIZE + 1

    coords = []
    for row in range(n_y):
        for col in range(n_x):
            x0 = int(col * TILE_SIZE * ds)
            y0 = int(row * TILE_SIZE * ds)
            if tile_fraction(mask, mask_ds, x0, y0, span0) >= TISSUE_THRESH:
                coords.append((x0, y0))

    total = len(coords)
    if total > MAX_TILES_PER_SLIDE:
        coords = random.sample(coords, MAX_TILES_PER_SLIDE)   # seeded above
    coords.sort()
    return coords, total, level, ds


# =============================================================================
# 3. STAIN NORMALIZATION (optional, LAB/Reinhard — self-contained, no extra deps)
# =============================================================================

def compute_lab_reference(slide, coords, level, n_sample=STAIN_REF_SAMPLE):
    from skimage.color import rgb2lab
    sample = random.sample(coords, min(n_sample, len(coords)))
    means, stds = [], []
    for (x0, y0) in sample:
        t = slide.read_region((x0, y0), level, (TILE_SIZE, TILE_SIZE)).convert("RGB")
        lab = rgb2lab(np.asarray(t).astype(np.float32) / 255.0).reshape(-1, 3)
        means.append(lab.mean(0)); stds.append(lab.std(0))
    return np.mean(means, 0), np.mean(stds, 0)


def stain_normalize(tile: Image.Image, ref_mean, ref_std) -> Image.Image:
    from skimage.color import rgb2lab, lab2rgb
    lab = rgb2lab(np.asarray(tile).astype(np.float32) / 255.0)
    for c in range(3):
        ch = lab[:, :, c]
        lab[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-6) * ref_std[c] + ref_mean[c]
    return Image.fromarray((lab2rgb(lab) * 255).clip(0, 255).astype(np.uint8))


# =============================================================================
# 4. UNI MODEL
# =============================================================================

def load_uni(device: torch.device):
    """Load MahmoodLab/UNI (ViT-L/16, 1024-dim) + its preprocessing transform."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    # --- Hugging Face auth (UNI is a GATED model) ---
    # You only need ONE of the following:
    #   (a) run `huggingface-cli login` once in your terminal (recommended), OR
    #   (b) set an env var:  export HF_TOKEN=hf_xxxxxxxx   (handled below), OR
    #   (c) uncomment the two lines below and paste your token (do NOT commit it):
    # from huggingface_hub import login
    # login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxx")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        log.info("Authenticated to Hugging Face via HF_TOKEN env var.")

    log.info("Loading UNI (MahmoodLab/UNI). Requires accepted license + HF login.")
    model = timm.create_model(
        UNI_REPO, pretrained=True, init_values=1e-5, dynamic_img_size=True
    )
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval().to(device)
    log.info(f"✅ UNI loaded — embed_dim={EMBED_DIM}")
    return model, transform


@torch.no_grad()
def extract_features(slide, coords, level, model, transform, device,
                     ref_mean=None, ref_std=None) -> torch.Tensor:
    """Read tissue tiles, (optionally) stain-normalize, run UNI in batches."""
    feats, batch = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        out = model(x)                       # [B, 1024]
        feats.append(out.float().cpu())
        batch.clear()

    for (x0, y0) in tqdm(coords, desc="  tiles", unit="tile", leave=False):
        tile = slide.read_region((x0, y0), level, (TILE_SIZE, TILE_SIZE)).convert("RGB")
        if APPLY_STAIN_NORM and ref_mean is not None:
            tile = stain_normalize(tile, ref_mean, ref_std)
        batch.append(transform(tile))
        if len(batch) >= BATCH_SIZE:
            flush()
    flush()

    if not feats:
        return torch.empty(0, EMBED_DIM)
    return torch.cat(feats, dim=0)           # [N, 1024]


# =============================================================================
# 5. ATOMIC SAVE + MANIFEST  (checkpoint/resume)
# =============================================================================

def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# =============================================================================
# MAIN
# =============================================================================

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        log.info("Using Apple Silicon MPS")
        return torch.device("mps")
    log.warning("No GPU detected — running on CPU (slow).")
    return torch.device("cpu")


def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = FEAT_DIR / "feature_manifest.json"

    log.info("=" * 64)
    log.info("  MERGED PREPROCESS + UNI FEATURE EXTRACTION")
    log.info("=" * 64)

    # ── Resume: load existing manifest ───────────────────────────────────
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        done = sum(1 for v in manifest.values() if v.get("status") == "done")
        log.info(f"[RESUME] Existing manifest found — {done} slides already done")

    # ── Labels: read from the cohort CSV, never re-derived ────────────────
    cohort_csv = find_cohort_csv(BASE_DIR, COHORT_TAG)
    cohort = load_cohort(cohort_csv)
    log.info(f"[COHORT] {cohort_csv.name} — {summarise(cohort)}")

    svs_files = sorted(SVS_DIR.glob("*.svs"))
    log.info(f"[SVS] Found {len(svs_files)} slides")

    device = pick_device()
    model, transform = load_uni(device)

    unmatched = []

    for i, svs in enumerate(svs_files, 1):
        slide_id = svs.stem
        out_feat = FEAT_DIR / f"{slide_id}_features.pt"

        # ---- Resume check (checkpoint) ----
        if SKIP_EXISTING and out_feat.exists() and \
           manifest.get(slide_id, {}).get("status") == "done":
            log.info(f"[{i}/{len(svs_files)}] {slide_id} — already done, skipping")
            continue

        # ---- Cohort membership + label ----
        record, match_status = label_for_slide(slide_id, cohort)
        label = compact(record)
        if match_status != "matched":
            unmatched.append(f"{slide_id}  [{match_status}]")
            log.warning(f"[{i}/{len(svs_files)}] {slide_id} — {match_status}: "
                        f"not present in {cohort_csv.name}")
            if SKIP_NOT_IN_COHORT:
                manifest[slide_id] = {"status": f"skipped_{match_status}",
                                      "label": label,
                                      "match_status": match_status}
                atomic_json_save(manifest, manifest_path)
                continue

        log.info(f"[{i}/{len(svs_files)}] {slide_id} — processing "
                 f"(label match: {match_status})")

        try:
            slide = openslide.OpenSlide(str(svs))
            coords, n_total, level, ds = list_tissue_tiles(slide)
            log.info(f"    tissue tiles: {len(coords)} kept "
                     f"(of {n_total} found; cap={MAX_TILES_PER_SLIDE})")

            if len(coords) == 0:
                manifest[slide_id] = {"status": "no_tissue", "label": label,
                                      "match_status": match_status}
                atomic_json_save(manifest, manifest_path)
                slide.close()
                continue

            ref_mean = ref_std = None
            if APPLY_STAIN_NORM:
                ref_mean, ref_std = compute_lab_reference(slide, coords, level)

            t0 = time.time()
            features = extract_features(slide, coords, level, model, transform,
                                        device, ref_mean, ref_std)
            dt = time.time() - t0
            slide.close()

            assert features.shape == (len(coords), EMBED_DIM), \
                f"shape mismatch {tuple(features.shape)}"

            # ---- Atomic saves ----
            atomic_torch_save(features, out_feat)
            out_coords = FEAT_DIR / f"{slide_id}_coords.pt"
            atomic_torch_save(
                {"coords_level0": torch.tensor(coords, dtype=torch.long),
                 "level": level, "downsample": float(ds),
                 "tile_size": TILE_SIZE}, out_coords)

            manifest[slide_id] = {
                "feature_path": str(out_feat),
                "coords_path":  str(out_coords),
                "shape": list(features.shape),
                "embed_dim": EMBED_DIM,
                "n_tiles": int(features.shape[0]),
                "n_tiles_found": n_total,
                "stain_norm": APPLY_STAIN_NORM,
                "match_status": match_status,
                "label": label,
                "seconds": round(dt, 1),
                "status": "done",
            }
            atomic_json_save(manifest, manifest_path)   # checkpoint EVERY slide
            log.info(f"    ✅ saved {list(features.shape)} in {dt:.1f}s")

        except Exception as e:
            log.error(f"    ❌ {slide_id} failed: {e}")
            manifest[slide_id] = {"status": "failed", "error": str(e),
                                  "label": label, "match_status": match_status}
            atomic_json_save(manifest, manifest_path)

    # ── Summary ───────────────────────────────────────────────────────────
    done    = sum(1 for v in manifest.values() if v.get("status") == "done")
    failed  = sum(1 for v in manifest.values() if v.get("status") == "failed")
    notiss  = sum(1 for v in manifest.values() if v.get("status") == "no_tissue")

    log.info("=" * 64)
    log.info(f"  Slides done       : {done}")
    log.info(f"  Slides failed     : {failed}")
    log.info(f"  Slides no-tissue  : {notiss}")
    skipped  = sum(1 for v in manifest.values()
                   if str(v.get("status", "")).startswith("skipped_"))
    log.info(f"  Slides skipped    : {skipped} (not in cohort)")
    log.info(f"  Slides w/o label  : {len(unmatched)}")
    if unmatched:
        log.info(f"  Not in {cohort_csv.name} (extra downloads? wrong arm?):")
        for s in unmatched:
            log.info(f"      - {s}")
    log.info(f"  Manifest → {manifest_path}")
    log.info("=" * 64)
    log.info("Done. Next → scripts/03_abmil_aggregation.py (set INPUT_DIM=1024).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore existing outputs and re-extract everything")
    args = parser.parse_args()
    if args.no_resume:
        SKIP_EXISTING = False
    try:
        main()
    except CohortError as e:
        raise SystemExit(f"\nCohort problem: {e}\n")
