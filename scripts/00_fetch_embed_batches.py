"""
=============================================================================
STEP 0: FETCH → EMBED → DELETE, in batches
=============================================================================

Walks a GDC manifest in batches, and for each batch:

    download → verify MD5 → embed with step 01 → verify features → delete .svs

The point is that the raw slides are a TRANSIENT. BRCA is 173.7 GB of .svs but
only ~845 MB of features; full Arm A is ~1.42 TB of .svs but ~5 GB of features.
Peak disk is one batch, not the cohort — so a 528 GB laptop can process a
cohort many times larger than its own disk.

SAFETY ORDER — a slide is only deleted after ALL of:
    1. its MD5 matches the manifest
    2. step 01 marked it "done" in the feature manifest
    3. its _features.pt exists and is a [N, EMBED_DIM] tensor with N > 0

Deleting before checking 2 and 3 would mean re-downloading gigabytes to recover
from a bug, so the order matters more than it looks.

RESUMABLE. State lives in output/fetch_state.json. Ctrl-C, close the laptop,
lose power — re-run the same command and it continues from the last completed
slide. Step 01's own SKIP_EXISTING resume means already-embedded slides are
never re-embedded.

USAGE
-----
    # see the plan without downloading anything
    python scripts/00_fetch_embed_batches.py --dry-run

    # the real run (BRCA: 206 slides, 9 batches of 25)
    python scripts/00_fetch_embed_batches.py

    # smaller first batch to time one slide before committing to all of them
    python scripts/00_fetch_embed_batches.py --batch-size 1 --max-batches 1

    # keep the .svs files (only if you have the disk for it)
    python scripts/00_fetch_embed_batches.py --keep-svs

Prerequisite: tools/gdc-client must be executable —  chmod +x tools/gdc-client
TCGA diagnostic slides are open access, so no GDC token is needed.
=============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / "scripts"
MANIFEST     = BASE_DIR / "cohort" / "gdc_manifest_brca.txt"
GDC_CLIENT   = BASE_DIR / "tools" / "gdc-client"
RAW_DIR      = BASE_DIR / "data" / "raw"
WSI_DIR      = BASE_DIR / "data" / "wsi"
FEAT_DIR     = BASE_DIR / "output" / "features"
STATE_PATH   = BASE_DIR / "output" / "fetch_state.json"
EMBED_SCRIPT = SCRIPTS_DIR / "01_preprocess_and_extract.py"

EMBED_DIM = 1024          # must match step 01
GB = 1024 ** 3


# ---------------------------------------------------------------------------
# manifest / state
# ---------------------------------------------------------------------------

def read_manifest(path: Path) -> list[dict]:
    """GDC manifest → [{id, filename, md5, size}]. Tab-separated with a header."""
    lines = path.read_text().splitlines()
    if not lines:
        raise SystemExit(f"Manifest is empty: {path}")

    header = lines[0].split("\t")
    need = {"id", "filename", "md5", "size"}
    if not need.issubset(header):
        raise SystemExit(
            f"{path} does not look like a GDC manifest.\n"
            f"  header: {header}\n  need:   {sorted(need)}"
        )
    ix = {k: header.index(k) for k in header}

    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        f = ln.split("\t")
        rows.append({"id":       f[ix["id"]],
                     "filename": f[ix["filename"]],
                     "md5":      f[ix["md5"]].lower(),
                     "size":     int(f[ix["size"]])})
    return rows


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"embedded": [], "deleted": [], "md5_failures": [], "started": None}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE_PATH)


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: float) -> str:
    return f"{n / GB:.1f} GB"


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60:d}m{s % 60:02d}s"


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def download_batch(batch: list[dict], batch_no: int, quiet: bool) -> None:
    """Write a sub-manifest and hand it to gdc-client."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sub = RAW_DIR / f"_batch_{batch_no:03d}_manifest.txt"
    sub.write_text("id\tfilename\tmd5\tsize\tstate\n" + "".join(
        f"{r['id']}\t{r['filename']}\t{r['md5']}\t{r['size']}\treleased\n" for r in batch))

    cmd = [str(GDC_CLIENT), "download", "-m", str(sub), "-d", str(RAW_DIR)]
    if quiet:
        cmd.append("--log-file")
        cmd.append(str(RAW_DIR / f"_batch_{batch_no:03d}.log"))
    print(f"    $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def collect_and_verify(batch: list[dict], state: dict) -> list[dict]:
    """
    Move each downloaded .svs from data/raw/<uuid>/ into data/wsi/ after its
    MD5 matches. Returns the rows that are ready to embed.
    """
    WSI_DIR.mkdir(parents=True, exist_ok=True)
    ready = []

    for r in batch:
        src = RAW_DIR / r["id"] / r["filename"]
        if not src.exists():
            hits = list((RAW_DIR / r["id"]).glob("*.svs")) if (RAW_DIR / r["id"]).is_dir() else []
            if len(hits) == 1:
                src = hits[0]
            else:
                print(f"    ✗ {r['filename']}: not found after download — skipped")
                continue

        print(f"    md5 {r['filename'][:46]}… ", end="", flush=True)
        got = md5_of(src)
        if got != r["md5"]:
            print(f"MISMATCH\n      expected {r['md5']}\n      got      {got}")
            state["md5_failures"].append(r["filename"])
            save_state(state)
            continue
        print("ok")

        dst = WSI_DIR / r["filename"]
        shutil.move(str(src), str(dst))
        # gdc-client leaves the per-uuid directory behind
        try:
            (RAW_DIR / r["id"]).rmdir()
        except OSError:
            pass
        ready.append(r)

    return ready


def run_embedding() -> None:
    """Run step 01 over whatever is currently in data/wsi. Resume-safe."""
    cmd = [sys.executable, str(EMBED_SCRIPT)]
    print(f"    $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(BASE_DIR))


def feature_ok(slide_stem: str, fmanifest: dict) -> tuple[bool, str]:
    """Both the manifest entry and the tensor on disk must look right."""
    entry = fmanifest.get(slide_stem)
    if entry is None:
        return False, "absent from feature_manifest.json"
    if entry.get("status") != "done":
        return False, f"status={entry.get('status')}"

    fp = entry.get("feature_path")
    if not fp or not Path(fp).exists():
        return False, "feature file missing on disk"

    shape = entry.get("shape") or []
    if len(shape) != 2 or shape[0] < 1 or shape[1] != EMBED_DIM:
        return False, f"shape {shape} (expected [N>0, {EMBED_DIM}])"
    return True, "ok"


def delete_consumed(ready: list[dict], state: dict, keep: bool) -> tuple[int, float]:
    """Delete .svs files whose features verified. Returns (count, bytes freed)."""
    fmanifest = {}
    fm_path = FEAT_DIR / "feature_manifest.json"
    if fm_path.exists():
        fmanifest = json.loads(fm_path.read_text())

    n, freed = 0, 0.0
    for r in ready:
        stem = r["filename"][:-4] if r["filename"].endswith(".svs") else r["filename"]
        ok, why = feature_ok(stem, fmanifest)
        svs = WSI_DIR / r["filename"]

        if not ok:
            print(f"    ! keeping {r['filename'][:46]}… — {why}")
            continue

        if stem not in state["embedded"]:
            state["embedded"].append(stem)

        if keep:
            continue
        if svs.exists():
            size = svs.stat().st_size
            os.remove(svs)
            freed += size
            n += 1
            state["deleted"].append(stem)

    save_state(state)
    return n, freed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST),
                    help=f"GDC manifest to walk (default: {MANIFEST.name})")
    ap.add_argument("--batch-size", type=int, default=25,
                    help="slides per batch; 25 ≈ 21 GB for BRCA (default: 25)")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="stop after N batches (0 = all). Use 1 to time a batch first.")
    ap.add_argument("--keep-svs", action="store_true",
                    help="do not delete raw .svs after embedding (needs the disk)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without downloading")
    ap.add_argument("--quiet-gdc", action="store_true",
                    help="send gdc-client output to a log file instead of the terminal")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = BASE_DIR / manifest_path
    rows = read_manifest(manifest_path)

    state = load_state()
    already = set(state["embedded"])

    def stem_of(r):
        return r["filename"][:-4] if r["filename"].endswith(".svs") else r["filename"]

    todo = [r for r in rows if stem_of(r) not in already]
    total_bytes = sum(r["size"] for r in rows)
    todo_bytes  = sum(r["size"] for r in todo)

    batches = [todo[i:i + args.batch_size]
               for i in range(0, len(todo), args.batch_size)]
    if args.max_batches:
        batches = batches[:args.max_batches]

    print("=" * 70)
    print("  FETCH → EMBED → DELETE")
    print("=" * 70)
    print(f"  manifest        : {manifest_path.name}")
    print(f"  slides total    : {len(rows)}  ({human(total_bytes)})")
    print(f"  already embedded: {len(rows) - len(todo)}")
    print(f"  remaining       : {len(todo)}  ({human(todo_bytes)})")
    print(f"  batch size      : {args.batch_size} slides")
    print(f"  batches to run  : {len(batches)}"
          f"{'  (capped by --max-batches)' if args.max_batches else ''}")
    if batches:
        print(f"  peak disk/batch : ~{human(max(sum(r['size'] for r in b) for b in batches))}")
    print(f"  delete raw .svs : {'NO (--keep-svs)' if args.keep_svs else 'yes, after verification'}")
    print("=" * 70)

    if not batches:
        print("Nothing to do — every slide in the manifest is already embedded.")
        return

    if args.dry_run:
        for i, b in enumerate(batches, 1):
            print(f"  batch {i:>3}: {len(b):>3} slides, "
                  f"{human(sum(r['size'] for r in b))}")
        print("\nDry run — nothing downloaded. Drop --dry-run to start.")
        return

    for tool, path in (("gdc-client", GDC_CLIENT), ("step 01", EMBED_SCRIPT)):
        if not path.exists():
            raise SystemExit(f"{tool} not found at {path}")
    if not os.access(GDC_CLIENT, os.X_OK):
        raise SystemExit(f"gdc-client is not executable. Run:\n    chmod +x {GDC_CLIENT}")

    state["started"] = state.get("started") or time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)

    t_start = time.time()
    embedded_before = len(state["embedded"])
    freed_total = 0.0
    timed_slides = 0          # slides that actually went through step 01 this run

    for i, batch in enumerate(batches, 1):
        gb = sum(r["size"] for r in batch) / GB
        print(f"\n{'─' * 70}\n  BATCH {i}/{len(batches)} — {len(batch)} slides, {gb:.1f} GB\n{'─' * 70}")
        t0 = time.time()

        print("  [1/4] download")
        download_batch(batch, i, args.quiet_gdc)

        print("  [2/4] verify md5 + move into data/wsi")
        ready = collect_and_verify(batch, state)
        if not ready:
            print("    nothing verified in this batch — stopping so you can look")
            break

        print(f"  [3/4] embed {len(ready)} slides (step 01)")
        run_embedding()

        print("  [4/4] verify features, then delete raw slides")
        n, freed = delete_consumed(ready, state, args.keep_svs)
        freed_total += freed
        timed_slides += len(ready)

        dt = time.time() - t0
        per = dt / max(len(ready), 1)
        remaining = sum(len(b) for b in batches[i:])
        print(f"\n  batch {i} done in {hms(dt)}  ({per:.0f}s/slide)  "
              f"freed {human(freed)}")
        if remaining:
            print(f"  {remaining} slides left → ETA {hms(per * remaining)}")

    total = time.time() - t_start
    newly_embedded = len(state["embedded"]) - embedded_before
    print("\n" + "=" * 70)
    print(f"  slides embedded this run : {newly_embedded}"
          f"{f' (of {timed_slides} downloaded)' if timed_slides != newly_embedded else ''}")
    print(f"  disk freed               : {human(freed_total)}")
    print(f"  wall clock               : {hms(total)}"
          f"{f'  ({total / timed_slides:.0f}s/slide)' if timed_slides else ''}")
    print(f"  embedded overall         : {len(state['embedded'])}/{len(rows)}")
    if state["md5_failures"]:
        print(f"  ⚠ MD5 FAILURES ({len(state['md5_failures'])}): "
              f"{', '.join(state['md5_failures'][:5])}")
        print("    Those slides were NOT embedded and NOT deleted. Re-run to retry.")
    print(f"  state                    → {STATE_PATH}")
    print("=" * 70)
    if len(state["embedded"]) >= len(rows):
        print("Cohort fully embedded. Next → python scripts/03_abmil_aggregation.py")
    else:
        print("Re-run the same command to continue with the next batches.")


if __name__ == "__main__":
    main()
