"""
Step 6 — Explainability: Attention Heatmaps
=============================================
Pipeline : Vision Transformer Based Analysis of Histopathology Images
          for Predicting AntiCancer Drug Response (TCGA-BRCA)

What this does
--------------
  1. Loads per-tile ABMIL attention scores from output/abmil/
  2. Loads tile coordinates from output/features/
  3. Opens the original .svs via OpenSlide to generate a WSI thumbnail
  4. Renders a smooth attention heatmap (Gaussian-blurred, jet colormap) over the thumbnail
  5. Saves output/heatmaps/<slide_id>_heatmap.png
              output/heatmaps/heatmap_manifest.json

Hardware : CPU only — no GPU needed for visualisation
Env      : conda activate histopath
Deps     : pip install matplotlib scipy Pillow openslide-python

Usage
-----
    python 06_attention_heatmap.py
    python 06_attention_heatmap.py --alpha 0.5 --cmap jet
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import gaussian_filter
import openslide

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

BASE_DIR   = Path.home() / "Desktop" / "Research"
SVS_DIR    = BASE_DIR / "data" / "wsi"
TILE_SIZE  = 256


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_attention(abmil_dir: Path, slide_id: str) -> np.ndarray:
    """Load [n_tiles, 1] or [n_tiles] attention scores → 1-D numpy array."""
    for name in (f"{slide_id}_attention_scores.pt", f"{slide_id}_attention.pt"):
        p = abmil_dir / name
        if p.exists():
            t = torch.load(p, map_location="cpu", weights_only=True)
            return t.squeeze().float().numpy()
    raise FileNotFoundError(f"No attention .pt found for {slide_id} in {abmil_dir}")


def load_coords(abmil_dir: Path, slide_id: str) -> np.ndarray | None:
    """Load tile coordinates → [n_tiles, 2] numpy array of (col, row) tile indices."""
    search_dirs = [abmil_dir, abmil_dir.parent / "features"]
    for d in search_dirs:
        for name in (f"{slide_id}_coords.pt", f"{slide_id}_coordinates.pt"):
            p = d / name
            if p.exists():
                t = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(t, dict) and "coords_level0" in t:
                    pixel_coords = t["coords_level0"].numpy()
                    ts = t.get("tile_size", TILE_SIZE)
                    return (pixel_coords // ts).astype(int)
                return t.numpy().astype(int)
    return None


def get_wsi_thumbnail(slide_id: str, target_long_side: int = 2048) -> Image.Image | None:
    """Generate a thumbnail from the .svs file via OpenSlide."""
    svs_files = list(SVS_DIR.glob(f"{slide_id}*.svs"))
    if not svs_files:
        return None
    slide = openslide.OpenSlide(str(svs_files[0]))
    w, h = slide.level_dimensions[0]
    scale = target_long_side / max(w, h)
    thumb_w, thumb_h = int(w * scale), int(h * scale)
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    slide.close()
    return thumb


def attention_to_heatmap(coords: np.ndarray, scores: np.ndarray,
                          canvas_size: tuple, tile_size: int,
                          scale: float, sigma_tiles: float = 1.5) -> np.ndarray:
    """Place attention scores on a grid and Gaussian-blur for a smooth heatmap."""
    W, H = canvas_size
    grid = np.zeros((H, W), dtype=np.float32)

    ts = max(int(tile_size * scale), 1)

    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        norm_scores = (scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(scores) * 0.5

    for (col, row), score in zip(coords, norm_scores):
        x0 = int(col * tile_size * scale)
        y0 = int(row * tile_size * scale)
        x1 = min(x0 + ts, W)
        y1 = min(y0 + ts, H)
        if x0 < W and y0 < H:
            grid[y0:y1, x0:x1] = score

    sigma_px = sigma_tiles * ts
    grid = gaussian_filter(grid, sigma=sigma_px)

    if grid.max() > 0:
        grid /= grid.max()

    return grid


def overlay_heatmap(background: Image.Image, heatmap: np.ndarray,
                    alpha: float = 0.5, cmap_name: str = "jet") -> Image.Image:
    """Blend a heatmap array over a PIL background image."""
    bg = np.array(background.resize((heatmap.shape[1], heatmap.shape[0])), dtype=np.float32)
    cmap = cm.get_cmap(cmap_name)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.float32)
    blended = (1 - alpha) * bg + alpha * colored
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def save_heatmap_figure(background: Image.Image, overlay: Image.Image,
                        slide_id: str, drug: str, pred_label: str,
                        out_path: Path, cmap_name: str = "jet"):
    """Save WSI thumbnail with attention heatmap overlay and colorbar."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0d0d0d")
    fig.suptitle(
        f"{slide_id}  ·  {drug}  ·  Predicted: {pred_label}",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )

    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        ax.axis("off")

    axes[0].imshow(background)
    axes[0].set_title("WSI Thumbnail", color="#aaaaaa", fontsize=10, pad=6)

    axes[1].imshow(overlay)
    axes[1].set_title("ABMIL Attention Heatmap  (red = high attention)",
                       color="#aaaaaa", fontsize=10, pad=6)

    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.03, pad=0.02)
    cbar.set_label("Attention score (normalised)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [SAVED] {out_path}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def run(abmil_dir: str, fusion_dir: str, output_dir: str,
        alpha: float, cmap: str):

    abmil_dir   = Path(abmil_dir)
    fusion_dir  = Path(fusion_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_manifest_path = fusion_dir / "fusion_manifest.json"
    pred_manifest_path   = Path("output/predictions/predictions.json")

    slides = []
    if fusion_manifest_path.exists():
        with open(fusion_manifest_path) as f:
            fm = json.load(f)
        slides = fm if isinstance(fm, list) else [{"slide_id": k, **v} for k, v in fm.items()]
    else:
        for p in sorted(abmil_dir.glob("*_attention_scores.pt")):
            slide_id = p.stem.replace("_attention_scores", "")
            slides.append({"slide_id": slide_id})

    pred_map = {}
    if pred_manifest_path.exists():
        with open(pred_manifest_path) as f:
            preds = json.load(f)
        for r in preds.get("predictions", []):
            pred_map[r["slide_id"]] = r

    print(f"\n{'─'*60}")
    print(f"  Step 6 — Attention Heatmap Generation")
    print(f"  Slides : {len(slides)}")
    print(f"{'─'*60}\n")

    heatmap_records = []

    for entry in slides:
        slide_id = entry.get("slide_id", entry.get("id", "unknown"))
        drug     = entry.get("drug_name", entry.get("drug", "unknown"))
        pred_info = pred_map.get(slide_id, {})
        pred_label = pred_info.get("predicted_label", entry.get("predicted_label", "—"))

        print(f"  Processing: {slide_id}  [{drug}]")

        # 1. Load attention scores
        try:
            scores = load_attention(abmil_dir, slide_id)
        except FileNotFoundError as e:
            print(f"    [SKIP] {e}")
            heatmap_records.append({"slide_id": slide_id, "status": "skipped",
                                    "reason": str(e)})
            continue

        n_tiles = len(scores)
        print(f"    Attention scores : {n_tiles} tiles  "
              f"min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}")

        # 2. Load tile coordinates
        coords = load_coords(abmil_dir, slide_id)
        if coords is None:
            side = int(np.ceil(np.sqrt(n_tiles)))
            coords = np.array([(i % side, i // side) for i in range(n_tiles)], dtype=int)
            print(f"    [WARN] No coordinates found — synthesised {side}x{side} grid layout")

        min_len = min(len(coords), len(scores))
        if len(coords) != len(scores):
            print(f"    [WARN] coords ({len(coords)}) != scores ({len(scores)}) — truncating to {min_len}")
            coords = coords[:min_len]
            scores = scores[:min_len]

        # 3. Build background from WSI thumbnail
        print(f"    Loading WSI thumbnail …")
        max_col = int(coords[:, 0].max()) + 1
        max_row = int(coords[:, 1].max()) + 1
        longest = max(max_col, max_row) * TILE_SIZE
        scale   = min(1.0, 4000 / max(longest, 1))

        background = get_wsi_thumbnail(slide_id)
        if background is None:
            canvas_w = int(max_col * TILE_SIZE * scale)
            canvas_h = int(max_row * TILE_SIZE * scale)
            background = Image.new("RGB", (canvas_w, canvas_h), color=(240, 240, 240))
            print(f"    [WARN] .svs not found — using blank canvas")

        canvas_size = background.size

        # 4. Generate heatmap
        print(f"    Generating heatmap …")
        hmap = attention_to_heatmap(coords, scores, canvas_size,
                                     TILE_SIZE, scale, sigma_tiles=1.5)

        # 5. Overlay
        overlay = overlay_heatmap(background, hmap, alpha=alpha, cmap_name=cmap)

        # 6. Save figure
        heatmap_path = output_dir / f"{slide_id}_heatmap.png"
        save_heatmap_figure(background, overlay, slide_id, drug, pred_label,
                            heatmap_path, cmap_name=cmap)

        # 7. Save raw heatmap array
        np_path = output_dir / f"{slide_id}_heatmap_array.npy"
        np.save(np_path, hmap)

        record = {
            "slide_id"       : slide_id,
            "drug"           : drug,
            "predicted_label": pred_label,
            "n_tiles"        : int(n_tiles),
            "attn_min"       : float(scores.min()),
            "attn_max"       : float(scores.max()),
            "attn_mean"      : float(scores.mean()),
            "canvas_size"    : list(canvas_size),
            "heatmap_png"    : str(heatmap_path),
            "heatmap_array"  : str(np_path),
            "status"         : "ok",
        }
        heatmap_records.append(record)
        print(f"    Done\n")

    manifest_out = {
        "run_timestamp" : datetime.now().isoformat(),
        "alpha"         : alpha,
        "colormap"      : cmap,
        "slides"        : heatmap_records,
    }
    manifest_path = output_dir / "heatmap_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"[SAVED] {manifest_path}")
    print(f"\n[DONE]  Step 6 complete. Heatmaps in → {output_dir}/\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Step 6 — Attention Heatmaps")
    p.add_argument("--abmil_dir",   default="output/abmil",
                   help="Directory with <slide_id>_attention_scores.pt")
    p.add_argument("--fusion_dir",  default="output/fusion",
                   help="Fusion manifest for slide/drug metadata")
    p.add_argument("--output_dir",  default="output/heatmaps",
                   help="Where to save heatmap PNGs and manifest")
    p.add_argument("--alpha",       type=float, default=0.5,
                   help="Heatmap overlay opacity 0–1 (default 0.5)")
    p.add_argument("--cmap",        default="jet",
                   help="Matplotlib colormap (default: jet; try: hot, inferno, RdYlGn)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        abmil_dir  = args.abmil_dir,
        fusion_dir = args.fusion_dir,
        output_dir = args.output_dir,
        alpha      = args.alpha,
        cmap       = args.cmap,
    )
