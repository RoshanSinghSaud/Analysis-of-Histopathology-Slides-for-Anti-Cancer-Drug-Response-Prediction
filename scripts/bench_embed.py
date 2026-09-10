"""
=============================================================================
BENCHMARK: where does the embedding time actually go?
=============================================================================

Gate 1 on the Mac measured 218 ms/tile at the 4,000-tile cap, against 66-70
ms/tile measured in June at the 1,000-tile cap. Same slide, same machine,
roughly 3x worse per tile. Before committing ~50 hours of laptop time to Gate
2 it is worth 15 minutes to find out which of these it is:

  (a) tile DECODING is the wall           -> more workers fix it
  (b) the GPU forward pass is the wall    -> fewer tiles or a rented GPU
  (c) the machine THROTTLES as it heats   -> nothing fixes it on a fanless Air
  (d) memory pressure at batch 32         -> a smaller batch fixes it

These have completely different answers, and guessing between them is how you
lose three days.

WHAT IT DOES
------------
Phase A times tile decoding with NO model at all — pure openslide read_region
plus the timm transform. That is the number that decides whether workers help.

Phase B times the full pipeline at several (workers, batch) settings, on the
SAME tiles each time so the comparison is fair. Each config reports its first
third and last third separately: if the last third is much slower, the machine
is throttling and no setting will save you.

RUN
---
    conda activate histopath
    python scripts/bench_embed.py                  # first slide in data/wsi
    python scripts/bench_embed.py --tiles 900      # quicker, less reliable
    python scripts/bench_embed.py --svs data/wsi/TCGA-A2-A0EP.svs

Nothing is written to output/. This only measures.
=============================================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))


def load_step01():
    """Import 01_preprocess_and_extract.py (its name is not a valid module name)."""
    path = BASE_DIR / "scripts" / "01_preprocess_and_extract.py"
    spec = importlib.util.spec_from_file_location("step01", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step01"] = mod
    spec.loader.exec_module(mod)
    return mod


def ms(x):
    return f"{x * 1000:7.1f}"


def orderings(coords):
    """
    The same tiles, visited in different orders.

    step 01 does `coords.sort()`, which sorts (x, y) tuples — so it walks all
    the way DOWN a column, then moves one column right. SVS files store their
    internal tiles ROW-major, so a column-major walk can defeat both the file's
    layout and openslide's internal cache, turning each read into a fresh seek
    and decode. Row-major (sort by y, then x) follows the file. If that is
    what is costing 3x, this is a one-line fix.

    Row order is scientifically irrelevant here — coords are saved alongside
    the features in the same order, and ABMIL attends over an unordered bag —
    so whichever order is fastest is safe to adopt.
    """
    return {
        "column-major (x,y) — current": sorted(coords),
        "row-major (y,x)":              sorted(coords, key=lambda c: (c[1], c[0])),
    }


def time_decode_only(m, svs, coords, transform, workers):
    """Phase A: read_region + transform, no model. Pure CPU cost per tile."""
    ds = m.TileDataset(svs, coords, 0, transform)
    kw = dict(batch_size=32, num_workers=workers, shuffle=False)
    if workers > 0:
        kw["prefetch_factor"] = 4
    loader = torch.utils.data.DataLoader(ds, **kw)

    t0 = time.time()
    n = 0
    for batch in loader:
        n += batch.shape[0]
    return (time.time() - t0) / max(n, 1)


def time_full(m, svs, coords, model, transform, device, workers, batch):
    """Phase B: the real thing, reported in thirds so throttling shows up."""
    m.BATCH_SIZE, m.NUM_WORKERS = batch, workers
    m.USE_AMP = (device.type == "cuda")

    ds = m.TileDataset(svs, coords, 0, transform)
    kw = dict(batch_size=batch, num_workers=workers, shuffle=False,
              pin_memory=(device.type == "cuda"))
    if workers > 0:
        kw["prefetch_factor"] = 4
    loader = torch.utils.data.DataLoader(ds, **kw)

    # The timer must START BEFORE next(it), because that is where the decode
    # wait happens. Timing only the forward pass reports a fast GPU sitting on
    # an idle line — which is exactly the illusion this benchmark exists to
    # dispel.
    per_batch, seen = [], 0
    t_start = time.time()
    it = iter(loader)
    t_prev = t_start
    with torch.no_grad():
        while True:
            try:
                x = next(it)
            except StopIteration:
                break
            x = x.to(device, non_blocking=(device.type == "cuda"))
            if m.USE_AMP:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
            else:
                out = model(x)
            out = out.float().cpu()
            if device.type == "mps":
                torch.mps.synchronize()
            now = time.time()
            per_batch.append((now - t_prev) / x.shape[0])
            t_prev = now
            seen += x.shape[0]
    total = time.time() - t_start

    # Drop batch 0: it carries worker start-up and the lazy OpenSlide open,
    # which would otherwise masquerade as a huge "first third" and invert the
    # throttling verdict.
    stats = per_batch[1:] if len(per_batch) > 3 else per_batch
    k = max(len(stats) // 3, 1)
    return {
        "per_tile": total / max(seen, 1),
        "first":    statistics.mean(stats[:k]),
        "last":     statistics.mean(stats[-k:]),
        "n":        seen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svs", default=None, help="slide to benchmark (default: first in data/wsi)")
    ap.add_argument("--tiles", type=int, default=1200,
                    help="tiles per config (default 1200; lower = faster, noisier)")
    ap.add_argument("--quick", action="store_true", help="fewer configs")
    args = ap.parse_args()

    m = load_step01()

    svs = Path(args.svs) if args.svs else next(iter(sorted((BASE_DIR / "data/wsi").glob("*.svs"))), None)
    if svs is None:
        raise SystemExit("No .svs found in data/wsi/")
    if not svs.is_absolute():
        svs = BASE_DIR / svs

    import openslide
    import os
    device = m.pick_device()
    n_cpu = os.cpu_count() or 4

    print("=" * 74)
    print(f"  slide   : {svs.name}")
    print(f"  device  : {device.type}   vCPU: {n_cpu}   torch {torch.__version__}")
    print(f"  tiles   : {args.tiles} per config (same tiles every time)")
    print("=" * 74)

    slide = openslide.OpenSlide(str(svs))
    rng = m.slide_rng(svs.stem)
    saved_cap = m.MAX_TILES_PER_SLIDE
    m.MAX_TILES_PER_SLIDE = args.tiles
    coords, n_found, level, _ = m.list_tissue_tiles(slide, rng)
    m.MAX_TILES_PER_SLIDE = saved_cap
    slide.close()
    print(f"  tissue tiles found: {n_found}; benchmarking {len(coords)}\n")

    print("Loading UNI…")
    model, transform = m.load_uni(device)

    # ---- Phase A: decoding only, no model ---------------------------------
    print("\n" + "-" * 74)
    print("  PHASE A — tile decoding only (no model). This is pure CPU work.")
    print("-" * 74)
    worker_opts = [0, 4] if args.quick else [0, 2, 4, min(8, max(n_cpu - 1, 1))]
    worker_opts = sorted(set(w for w in worker_opts if w <= 16))
    # The very first pass over a slide reads from disk; every later pass reads
    # from the OS page cache. Without this warm-up the first config measured
    # looks catastrophically slow and every comparison below is meaningless.
    t_cold0 = time.time()
    _ = time_decode_only(m, svs, coords, transform, 0)
    cold_pass = (time.time() - t_cold0) / max(len(coords), 1)
    print(f"    (cold first pass, disk-backed: {ms(cold_pass)} ms/tile — "
          f"this is what the real run pays)")

    decode = {}
    for w in worker_opts:
        d = time_decode_only(m, svs, coords, transform, w)
        decode[w] = d
        print(f"    workers={w:<2}  {ms(d)} ms/tile")
    best_w = min(decode, key=decode.get)
    if decode[best_w] < decode[0] * 0.7:
        print(f"\n    -> decoding parallelises well ({decode[0]/decode[best_w]:.1f}x at "
              f"{best_w} workers). Use workers on the real run.")
    else:
        print(f"\n    -> workers barely help. Decoding is not the wall, or this "
              f"machine cannot feed them.")

    # ---- Phase A2: does the traversal order matter? -----------------------
    print("\n" + "-" * 74)
    print("  PHASE A2 — same tiles, different visit order (decode only).")
    print("  step 01 currently walks column-major; SVS files store row-major.")
    print("-" * 74)
    order_times = {}
    for label, ordered in orderings(coords).items():
        t = time_decode_only(m, svs, ordered, transform, best_w)
        order_times[label] = t
        print(f"    {label:<32} {ms(t)} ms/tile")
    cur = order_times["column-major (x,y) — current"]
    row = order_times["row-major (y,x)"]
    if row < cur * 0.75:
        print(f"\n    -> ROW-MAJOR IS {cur/row:.1f}x FASTER. This is very likely your 3x.")
        print(f"       Fix: in list_tissue_tiles(), replace  coords.sort()  with")
        print(f"            coords.sort(key=lambda c: (c[1], c[0]))")
    elif row > cur * 1.25:
        print(f"\n    -> column-major (current) is faster. Leave it alone.")
    else:
        print(f"\n    -> traversal order does not matter much here. Not the cause.")

    # ---- Phase B: the full pipeline ---------------------------------------
    print("\n" + "-" * 74)
    print("  PHASE B — full pipeline. 'first'/'last' = mean ms/tile over the")
    print("  first and last third. last >> first means thermal throttling.")
    print("-" * 74)
    configs = [(0, 32), (best_w, 32)] if args.quick else [
        (0, 32), (0, 8), (best_w, 32), (best_w, 16), (best_w, 64)]
    seen_cfg, rows = set(), []
    print(f"    {'workers':>7} {'batch':>6} {'ms/tile':>10} {'first':>9} {'last':>9}  {'drift':>7}")
    for w, b in configs:
        if (w, b) in seen_cfg:
            continue
        seen_cfg.add((w, b))
        try:
            r = time_full(m, svs, coords, model, transform, device, w, b)
        except Exception as e:
            print(f"    {w:>7} {b:>6}   FAILED: {type(e).__name__}: {str(e)[:40]}")
            continue
        drift = r["last"] / r["first"] if r["first"] > 0 else float("nan")
        rows.append(((w, b), r, drift))
        print(f"    {w:>7} {b:>6} {ms(r['per_tile']):>10} {ms(r['first']):>9} "
              f"{ms(r['last']):>9}  {drift:>6.2f}x")

    if not rows:
        raise SystemExit("\nEvery config failed — nothing to conclude.")

    # ---- Verdict ----------------------------------------------------------
    (bw, bb), best, _ = min(rows, key=lambda t: t[1]["per_tile"])
    total_seconds = sum(r["per_tile"] * r["n"] for _, r, _ in rows)
    baseline = next((r for (c, r, _) in rows if c == (0, 32)), best)
    worst_drift = max(d for _, _, d in rows)

    print("\n" + "=" * 74)
    print(f"  FASTEST : workers={bw}, batch={bb} -> {best['per_tile']*1000:.0f} ms/tile")
    speedup = baseline["per_tile"] / best["per_tile"] if best["per_tile"] > 0 else 1.0
    if speedup > 1.05:
        print(f"            {speedup:.1f}x faster than the current default "
              f"(workers=0, batch=32)")
    else:
        print(f"            no better than the current default (workers=0, batch=32)")

    # decode[] is keyed by worker count; use the SAME worker count as the
    # winning full-pipeline config, or the closest one measured.
    dec_key = bw if bw in decode else min(decode, key=lambda w: abs(w - bw))
    dec, full = decode[dec_key], best["per_tile"]
    print(f"  Decoding alone costs {dec*1000:.0f} ms/tile; the full pipeline "
          f"{full*1000:.0f} ms/tile.")
    if dec >= 0.8 * full:
        print(f"  -> DECODING IS THE WALL. The GPU is mostly idle. More vCPUs / more")
        print(f"     workers is the fix; a faster GPU is not.")
    elif dec <= 0.3 * full:
        print(f"  -> THE MODEL FORWARD PASS IS THE WALL. More workers will not help.")
        print(f"     A CUDA GPU with fp16 would; on this Mac, a lower tile cap would.")
    else:
        print(f"  -> Decode and model are comparable; neither alone explains the cost.")

    if total_seconds < 60:
        print(f"  Run too short ({total_seconds:.0f}s) to say anything about thermal")
        print(f"  throttling — re-run with a larger --tiles if that matters.")
    elif worst_drift > 1.25:
        print(f"  ⚠ THROTTLING: slowest config ran {worst_drift:.2f}x slower at the end")
        print(f"    than at the start. A multi-hour run will be WORSE than these")
        print(f"    numbers, not better. Budget accordingly.")
    else:
        print(f"  No significant throttling over this run (max drift {worst_drift:.2f}x).")

    for cap in (1000, 4000, 8000):
        hrs = best["per_tile"] * cap * 206 / 3600
        print(f"  Projected BRCA embed at {cap:>5} tiles/slide, 206 slides: {hrs:6.1f} h"
              + ("   <- current cap" if cap == 4000 else ""))
    print("  (embedding only — the 173.7 GB download is on top of this)")
    print("=" * 74)


if __name__ == "__main__":
    main()
