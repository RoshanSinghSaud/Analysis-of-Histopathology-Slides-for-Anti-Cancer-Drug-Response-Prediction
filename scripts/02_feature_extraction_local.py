"""
=============================================================================
STEP 2: Feature Extraction — Local Mac (VS Code, one-shot run)
=============================================================================

Model    : Phikon (owkin/phikon) — open access, no HuggingFace token needed
           ViT-B/16 pretrained on TCGA H&E slides → 768-dim embeddings
Device   : Apple Silicon MPS (auto-detected, falls back to CPU)
Input    : preprocessing_manifest.json  (from Step 1)
Output   : output/features/
             TCGA-A2-A0EP_features.pt   → shape [n_tiles, 768]
             TCGA-C8-A131_features.pt   → shape [n_tiles, 768]
             feature_manifest.json      → input to Step 3 (ABMIL)

Install (one-time, in terminal with histopath env active):
    conda activate histopath
    pip install transformers timm tqdm Pillow torch torchvision

Run:
    Just press ▶ in VS Code (no arguments needed)
=============================================================================
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# =============================================================================
# ── CONFIG — edit these if your paths differ ─────────────────────────────────
# =============================================================================

BASE_DIR      = Path(__file__).resolve().parents[1]
MANIFEST_PATH = BASE_DIR / "output" / "preprocessing_manifest.json"
OUTPUT_DIR    = BASE_DIR / "output" / "features"

BATCH_SIZE    = 32       # safe for MPS; raise to 64 if you have >16GB RAM
NUM_WORKERS   = 0        # must be 0 on MPS to avoid multiprocessing issues
MAX_TILES     = None     # set e.g. 500 to do a quick test run; None = all tiles
SKIP_EXISTING = False     # True = resume if interrupted

# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class TileDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, tile_paths: list):
        self.paths = tile_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), str(path)
        except Exception as e:
            log.warning(f"  Bad tile {Path(path).name}: {e} — using zeros")
            return torch.zeros(3, 224, 224), str(path)


# -----------------------------------------------------------------------------
# Model — Phikon (open access, no token needed)
# -----------------------------------------------------------------------------
def load_phikon() -> tuple:
    """
    Phikon: ViT-B/16 trained on 40M+ H&E patches from TCGA.
    Open access at https://huggingface.co/owkin/phikon
    Returns (model, embed_dim=768)
    """
    from transformers import ViTModel

    class Phikon(nn.Module):
        def __init__(self):
            super().__init__()
            log.info("Downloading Phikon weights (~330MB, first run only)…")
            self.vit = ViTModel.from_pretrained(
                "owkin/phikon",
                add_pooling_layer=False,
            )

        def forward(self, x):
            # x: [B, 3, 224, 224]
            out = self.vit(pixel_values=x)
            return out.last_hidden_state[:, 0, :]   # CLS token → [B, 768]

    model = Phikon()
    log.info("✅ Phikon loaded — embed_dim=768")
    return model, 768


# -----------------------------------------------------------------------------
# Extraction loop
# -----------------------------------------------------------------------------
def extract_slide(tile_paths: list, model: nn.Module, device: torch.device) -> torch.Tensor:
    dataset = TileDataset(tile_paths)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,   # pin_memory not supported on MPS
    )

    model.eval()
    all_feats = []

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="  batches", unit="batch", leave=False):
            imgs  = imgs.to(device)
            feats = model(imgs)           # [B, 768]
            all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)   # [N, 768]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # ── Validate paths ────────────────────────────────────────────────────
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Check that MANIFEST_PATH in the CONFIG section is correct."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    log.info(f"Manifest loaded — {len(manifest)} slides found")

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        log.warning("No GPU detected — running on CPU (slow)")

    log.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model, embed_dim = load_phikon()
    model.to(device)

    # ── Per-slide loop ────────────────────────────────────────────────────
    results = {}

    for slide_id, slide_data in manifest.items():
        out_path = OUTPUT_DIR / f"{slide_id}_features.pt"

        # Resume support
        if SKIP_EXISTING and out_path.exists():
            log.info(f"[{slide_id}] Already done — skipping")
            results[slide_id] = {
                "feature_path" : str(out_path),
                "label"        : slide_data.get("label", {}),
                "status"       : "skipped",
            }
            continue

        tile_paths = slide_data.get("tile_paths", [])
        if not tile_paths:
            log.warning(f"[{slide_id}] No tile paths in manifest — skipping")
            continue

        # Optional tile cap (for quick tests)
        if MAX_TILES and len(tile_paths) > MAX_TILES:
            log.info(f"[{slide_id}] Capping {len(tile_paths)} → {MAX_TILES} tiles")
            tile_paths = tile_paths[:MAX_TILES]

        log.info(f"[{slide_id}] Extracting {len(tile_paths)} tiles…")

        try:
            features = extract_slide(tile_paths, model, device)

            # Sanity check shape
            assert features.shape == (len(tile_paths), embed_dim), \
                f"Shape mismatch: got {features.shape}"

            torch.save(features, out_path)
            log.info(f"[{slide_id}] ✅ Saved → {out_path.name}  shape={list(features.shape)}")

            results[slide_id] = {
                "feature_path" : str(out_path),
                "shape"        : list(features.shape),
                "embed_dim"    : embed_dim,
                "n_tiles"      : features.shape[0],
                "label"        : slide_data.get("label", {}),
                "status"       : "done",
            }

        except Exception as e:
            log.error(f"[{slide_id}] ❌ Failed: {e}")
            results[slide_id] = {"status": "failed", "error": str(e)}

    # ── Save feature manifest ─────────────────────────────────────────────
    feat_manifest_path = OUTPUT_DIR / "feature_manifest.json"
    with open(feat_manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    done    = sum(1 for v in results.values() if v["status"] == "done")
    skipped = sum(1 for v in results.values() if v["status"] == "skipped")
    failed  = sum(1 for v in results.values() if v["status"] == "failed")

    log.info("=" * 60)
    log.info(f"  Slides done    : {done}")
    log.info(f"  Slides skipped : {skipped}")
    log.info(f"  Slides failed  : {failed}")
    log.info(f"  Feature manifest → {feat_manifest_path}")
    log.info("=" * 60)
    log.info("Step 2 complete. Next → Step 3: ABMIL attention aggregation.")


if __name__ == "__main__":
    main()
