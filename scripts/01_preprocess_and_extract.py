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
2. TILE CAP : MAX_TILES_PER_SLIDE caps tiles/slide (per-slide seeded random
   sample) so a huge slide cannot blow up memory or runtime. Currently 4000
   -> 16.4 MB of features per slide.
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

from __future__ import annotations

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
COHORT_TAG = "arm_a"   # set on the pod for the Gate 4 Arm A run, 28 Aug 2026

# --- Tiling ---
TILE_SIZE   = 256     # tile size in pixels at the processing level
MAG_LEVEL   = 0       # 0 = highest resolution
TISSUE_THRESH = 0.50  # min fraction of a tile that must be tissue to keep it
MASK_DOWNSAMPLE = 32  # build the fast tissue mask at ~this downsample factor

# --- 200+ case safeguards ---
MAX_TILES_PER_SLIDE = 4000   # hard cap; seeded per-slide random sample if exceeded
#   Raised 1000 -> 4000 on 2026-08-25. 1000 was ~3-10% of the tissue on the
#   two test slides (10,381 and 32,632 tiles found), which is hard to defend
#   in a methods section. 4000 costs 16.4 MB/slide instead of 4.1 MB.
#   DO NOT change this mid-cohort: every slide in a feature store must share
#   one cap. Changing it means re-embedding everything (--no-resume).
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

# Throughput knobs. None = auto-tune for the detected device in configure_run().
# Tile decoding (openslide read_region + the timm transform) is CPU work and it
# is what actually limits this script, so NUM_WORKERS matters more than the GPU
# tier. Set any of these to a number to override, or pass the CLI flags.
BATCH_SIZE  = None       # auto: CUDA 128, MPS 32, CPU 16
NUM_WORKERS = None       # auto: CUDA min(8, cpu_count-1), MPS/CPU 0 (MPS needs 0)
USE_AMP     = None       # auto: fp16 autocast on CUDA only; never on MPS/CPU

# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess+uni")

np.random.seed(SEED)

# NOTE: there is deliberately no module-level random.seed() here. Tile sampling
# used to draw from the global RNG, which made a slide's tile set depend on how
# many slides had been drawn before it in the same process — so a resumed run,
# or a run launched per-batch by 00_fetch_embed_batches.py, silently sampled
# DIFFERENT tiles for the same slide than a fresh run did. The pre-registration
# claims a seeded sample, so the seed is now derived per slide instead; see
# slide_rng().


def slide_rng(slide_id: str) -> random.Random:
    """
    A private, reproducible RNG for one slide.

    Seeded from (SEED, slide_id) rather than from a shared global stream, so the
    tiles chosen for a slide depend only on its ID and the tissue mask — never
    on processing order, resume state, batch boundaries or which machine ran it.
    """
    return random.Random(f"{SEED}:{slide_id}")


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


def list_tissue_tiles(slide: openslide.OpenSlide, rng: random.Random):
    """
    Return (coords, n_total, level, downsample) where coords are the level-0
    (x, y) top-left corners of the tissue tiles, capped/sampled to
    MAX_TILES_PER_SLIDE using the slide's own RNG (see slide_rng).
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
        # rng is per-slide, so this sample is reproducible from the slide ID
        # alone — independent of processing order or resume state.
        coords = rng.sample(coords, MAX_TILES_PER_SLIDE)
    coords.sort()
    return coords, total, level, ds


# =============================================================================
# 3. STAIN NORMALIZATION (optional, LAB/Reinhard — self-contained, no extra deps)
# =============================================================================

def compute_lab_reference(slide, coords, level, rng, n_sample=STAIN_REF_SAMPLE):
    from skimage.color import rgb2lab
    # per-slide rng, same reproducibility argument as list_tissue_tiles
    sample = rng.sample(coords, min(n_sample, len(coords)))
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
    # login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
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


class TileDataset(torch.utils.data.Dataset):
    """
    Serves decoded, transformed tiles so DataLoader workers can decode in
    parallel while the GPU is busy with the previous batch.

    Why this exists: the old implementation was one serial loop —
    read_region -> transform -> accumulate 32 -> GPU. Every JPEG decode left the
    GPU idle, which made the script CPU-bound and meant a faster GPU bought
    nothing. Decoding across N workers is what turns a rented GPU into value.

    OpenSlide handles are not safe to share across processes, so each worker
    opens its own lazily on first access and keeps it for the worker's life.
    Note `_slide` must stay None at fork/spawn time — never touch it in the
    parent process.
    """

    def __init__(self, svs_path, coords, level, transform,
                 ref_mean=None, ref_std=None):
        self.svs_path  = str(svs_path)
        self.coords    = coords
        self.level     = level
        self.transform = transform
        self.ref_mean  = ref_mean
        self.ref_std   = ref_std
        self._slide    = None

    def __len__(self):
        return len(self.coords)

    def _handle(self):
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.svs_path)
        return self._slide

    def __getitem__(self, i):
        x0, y0 = self.coords[i]
        tile = self._handle().read_region(
            (x0, y0), self.level, (TILE_SIZE, TILE_SIZE)).convert("RGB")
        if self.ref_mean is not None:
            tile = stain_normalize(tile, self.ref_mean, self.ref_std)
        return self.transform(tile)


@torch.no_grad()
def extract_features(svs_path, coords, level, model, transform, device,
                     ref_mean=None, ref_std=None) -> torch.Tensor:
    """
    Decode tissue tiles in parallel workers and run UNI in batches.

    Row i of the returned [N, EMBED_DIM] tensor corresponds to coords[i]:
    shuffle stays off so step 06 can map attention back onto the slide.
    """
    ds = TileDataset(svs_path, coords, level, transform,
                     ref_mean if APPLY_STAIN_NORM else None, ref_std)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,                       # row order must match coords
        pin_memory=(device.type == "cuda"),
    )
    if NUM_WORKERS > 0:
        # keep the queue ahead of the GPU without ballooning RAM
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = False
    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    feats = []
    n_batches = (len(coords) + BATCH_SIZE - 1) // BATCH_SIZE
    for x in tqdm(loader, desc="  tiles", unit="batch",
                  total=n_batches, leave=False):
        x = x.to(device, non_blocking=(device.type == "cuda"))
        if USE_AMP:
            # inference-only, so fp16 needs no loss scaling; features are cast
            # back to fp32 below and stored fp32, as they were before.
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
        else:
            out = model(x)
        feats.append(out.float().cpu())

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


def configure_run(device: torch.device) -> None:
    """
    Fill in whichever of BATCH_SIZE / NUM_WORKERS / USE_AMP were left as None.

    The defaults differ per device for real reasons, not taste:
      - MPS requires num_workers=0 and has no usable fp16 autocast path here.
      - On CUDA the limit is tile decoding, so workers scale with vCPU count.
        A pod is worth renting only to the extent these workers are busy.
    """
    global BATCH_SIZE, NUM_WORKERS, USE_AMP

    n_cpu = os.cpu_count() or 4
    if device.type == "cuda":
        auto_bs, auto_nw, auto_amp = 128, max(1, min(8, n_cpu - 1)), True
    elif device.type == "mps":
        auto_bs, auto_nw, auto_amp = 32, 0, False
    else:
        auto_bs, auto_nw, auto_amp = 16, 0, False

    BATCH_SIZE  = auto_bs  if BATCH_SIZE  is None else BATCH_SIZE
    NUM_WORKERS = auto_nw  if NUM_WORKERS is None else NUM_WORKERS
    USE_AMP     = auto_amp if USE_AMP     is None else USE_AMP

    if USE_AMP and device.type != "cuda":
        log.warning("fp16 autocast requested off CUDA — disabling it.")
        USE_AMP = False
    if NUM_WORKERS > 0 and device.type == "mps":
        log.warning("MPS cannot use DataLoader workers — forcing NUM_WORKERS=0.")
        NUM_WORKERS = 0

    log.info(f"[RUN] device={device.type}  batch_size={BATCH_SIZE}  "
             f"num_workers={NUM_WORKERS}  fp16={USE_AMP}  (vCPU={n_cpu})")
    if device.type == "cuda" and NUM_WORKERS < 4:
        log.warning(f"Only {NUM_WORKERS} decode workers on a GPU box — tile "
                    f"decoding will bottleneck the GPU. Rent more vCPUs.")


def rel_to_repo(p: Path) -> str:
    """
    Store manifest paths relative to the repo root.

    Absolute paths used to be written here, so a manifest produced on a rented
    Linux pod said /workspace/... and broke the moment it came back to the Mac.
    Relative paths travel; see resolve_repo_path() for the read side.
    """
    try:
        return str(Path(p).resolve().relative_to(BASE_DIR))
    except ValueError:                       # outside the repo — keep absolute
        return str(Path(p).resolve())


def resolve_repo_path(p: str | None, base: Path = BASE_DIR) -> Path | None:
    """Read side of rel_to_repo(). Absolute paths from older manifests still work."""
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else (base / q)


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
    configure_run(device)
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
            rng = slide_rng(slide_id)        # reproducible from the ID alone
            slide = openslide.OpenSlide(str(svs))
            coords, n_total, level, ds = list_tissue_tiles(slide, rng)
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
                ref_mean, ref_std = compute_lab_reference(slide, coords, level, rng)

            # Close the parent handle before the workers open their own: an
            # OpenSlide handle inherited across fork is not safe to use.
            slide.close()

            t0 = time.time()
            features = extract_features(svs, coords, level, model, transform,
                                        device, ref_mean, ref_std)
            dt = time.time() - t0

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
                # repo-relative so the manifest survives the trip between the
                # Mac and a rented pod — see rel_to_repo()
                "feature_path": rel_to_repo(out_feat),
                "coords_path":  rel_to_repo(out_coords),
                "shape": list(features.shape),
                "embed_dim": EMBED_DIM,
                "n_tiles": int(features.shape[0]),
                "n_tiles_found": n_total,
                "stain_norm": APPLY_STAIN_NORM,
                "match_status": match_status,
                "label": label,
                "seconds": round(dt, 1),
                # provenance: what these vectors were actually produced with,
                # so a mixed Mac/pod feature store is auditable rather than a
                # guess when the methods section gets written
                "seed": SEED,
                "max_tiles": MAX_TILES_PER_SLIDE,
                "device": device.type,
                "fp16": bool(USE_AMP),
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
    log.info("Done. Next → scripts/08_evaluate_ablation.py --tag <tag>.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore existing outputs and re-extract everything")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="tiles per GPU batch (default: auto per device)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="parallel tile-decode workers (default: auto; "
                             "must be 0 on Apple MPS)")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help=f"override MAX_TILES_PER_SLIDE (default: "
                             f"{MAX_TILES_PER_SLIDE}). Changing this mid-cohort "
                             f"makes the feature store inconsistent.")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable fp16 autocast on CUDA (slower, exactly "
                             "reproduces fp32 features)")
    args = parser.parse_args()
    if args.no_resume:
        SKIP_EXISTING = False
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        NUM_WORKERS = args.num_workers
    if args.max_tiles is not None:
        MAX_TILES_PER_SLIDE = args.max_tiles
    if args.no_amp:
        USE_AMP = False
    try:
        main()
    except CohortError as e:
        raise SystemExit(f"\nCohort problem: {e}\n")
