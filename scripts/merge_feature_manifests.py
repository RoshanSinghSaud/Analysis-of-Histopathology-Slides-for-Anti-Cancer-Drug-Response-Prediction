#!/usr/bin/env python3
"""
MERGE FEATURE MANIFESTS  —  Gate 4, trap 3.

The pod's feature_manifest.json holds only the ~1,005 Arm A slides embedded
there. The Mac's holds the 206 Arm B (BRCA) slides from Gate 2. Extracting the
pod's archive over the Mac's output/features/ REPLACES the manifest and orphans
every BRCA feature file: the .pt tensors survive on disk but nothing indexes
them, so the store silently shrinks from 1,211 slides to 1,005.

This script merges the two JSON objects instead, and refuses to write unless
every homogeneity invariant the pre-registration depends on still holds.

Usage
-----
    python scripts/merge_feature_manifests.py --new /path/to/pod_feature_manifest.json
    python scripts/merge_feature_manifests.py --new ... --apply

Without --apply it is a dry run: it reports and writes nothing.

What it checks
--------------
1. No key appears in both manifests with different content (a genuine collision
   would mean a slide was embedded twice, on two devices — a homogeneity break).
2. The merged store has exactly the expected number of entries.
3. Every entry has status == "done".
4. There is exactly ONE provenance tuple (device, fp16, max_tiles, seed) across
   all entries. Two tuples means the store is not homogeneous and §3.5's
   "seeded random sample" claim cannot be backed.
5. Every feature_path and coords_path referenced actually exists on disk.
6. Labels: entries carried over from the BRCA run were labelled against
   cohort_brca.csv, the new ones against cohort_arm_a.csv. Arm B is a strict
   subset of Arm A, so response_binary must agree. Disagreement is a FINDING,
   not a merge detail — it would mean the two arms derive labels differently.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = BASE_DIR / "output" / "features"
MANIFEST = FEATURES_DIR / "feature_manifest.json"
ARM_A_CSV = BASE_DIR / "cohort" / "cohort_arm_a.csv"

EXPECTED_TOTAL = 1211


def provenance(entry: dict) -> tuple:
    return (entry.get("device"), entry.get("fp16"),
            entry.get("max_tiles"), entry.get("seed"))


def barcode(slide_stem: str) -> str:
    return "-".join(slide_stem.split("-")[:3])


def load(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"ERROR: no such manifest: {path}")
    with path.open() as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True,
                    help="the pod's feature_manifest.json (the ~1,005 Arm A slides)")
    ap.add_argument("--existing", default=str(MANIFEST),
                    help=f"the manifest already in place (default: {MANIFEST})")
    ap.add_argument("--expected", type=int, default=EXPECTED_TOTAL,
                    help=f"expected merged entry count (default: {EXPECTED_TOTAL})")
    ap.add_argument("--apply", action="store_true",
                    help="actually write the merged manifest (default: dry run)")
    args = ap.parse_args()

    existing_path = Path(args.existing)
    new_path = Path(args.new)

    existing = load(existing_path)
    new = load(new_path)

    print("=" * 70)
    print("  MERGE FEATURE MANIFESTS")
    print("=" * 70)
    print(f"  existing : {existing_path}  —  {len(existing)} entries")
    print(f"  new      : {new_path}  —  {len(new)} entries")

    problems: list[str] = []

    # ── 1. collisions ────────────────────────────────────────────────────
    overlap = set(existing) & set(new)
    if overlap:
        differing = [k for k in overlap if existing[k] != new[k]]
        print(f"  overlap  : {len(overlap)} slides in both "
              f"({len(differing)} with DIFFERING content)")
        if differing:
            problems.append(
                f"{len(differing)} slide(s) embedded twice with different results — "
                f"e.g. {differing[0]}. The store would not be homogeneous.")
    else:
        print("  overlap  : none")

    merged = {**existing, **new}
    print(f"  merged   : {len(merged)} entries")

    # ── 2. count ─────────────────────────────────────────────────────────
    if len(merged) != args.expected:
        problems.append(
            f"merged count is {len(merged)}, expected {args.expected}")

    # ── 3. status ────────────────────────────────────────────────────────
    bad_status = [k for k, v in merged.items() if v.get("status") != "done"]
    if bad_status:
        problems.append(f"{len(bad_status)} entries not status=done "
                        f"(e.g. {bad_status[0]})")

    # ── 4. provenance homogeneity ────────────────────────────────────────
    tuples = Counter(provenance(v) for v in merged.values())
    print("\n  provenance tuples (device, fp16, max_tiles, seed):")
    for tup, n in tuples.most_common():
        print(f"    {tup}  ×{n}")
    if len(tuples) != 1:
        problems.append(
            f"{len(tuples)} distinct provenance tuples — the feature store is "
            f"NOT homogeneous and the pre-registration's seeded-sample claim "
            f"cannot be backed.")

    # ── 5. files on disk ─────────────────────────────────────────────────
    missing = []
    for k, v in merged.items():
        for field in ("feature_path", "coords_path"):
            rel = v.get(field)
            if not rel:
                missing.append(f"{k}: no {field}")
                continue
            if not (BASE_DIR / rel).is_file():
                missing.append(f"{k}: {rel}")
    if missing:
        problems.append(f"{len(missing)} referenced tensor files are absent "
                        f"(e.g. {missing[0]})")
    else:
        print(f"\n  tensor files : all {2 * len(merged)} present on disk")

    # ── 6. label agreement across the two cohort CSVs ────────────────────
    if ARM_A_CSV.is_file():
        arm_a: dict[str, str] = {}
        with ARM_A_CSV.open() as fh:
            for row in csv.DictReader(fh):
                arm_a[row["submitter_id"]] = row["response_binary"]

        checked = disagree = absent = 0
        examples = []
        for k, v in merged.items():
            lab = v.get("label") or {}
            rb = lab.get("response_binary")
            if rb is None:
                continue
            bc = lab.get("barcode") or barcode(k)
            if bc not in arm_a:
                absent += 1
                continue
            checked += 1
            if str(rb) != str(arm_a[bc]):
                disagree += 1
                if len(examples) < 5:
                    examples.append(f"{bc}: manifest={rb} arm_a={arm_a[bc]}")
        print(f"\n  label check  : {checked} entries compared against "
              f"cohort_arm_a.csv")
        print(f"                 {disagree} disagreements, {absent} barcodes "
              f"absent from Arm A")
        if disagree:
            problems.append(
                "label disagreement between the manifest and cohort_arm_a.csv "
                "(" + "; ".join(examples) + "). Arm B is a strict subset of "
                "Arm A, so this means the two arms derive labels differently. "
                "Investigate before modelling — this is a finding, not a merge "
                "detail.")
        if absent:
            problems.append(
                f"{absent} slide barcodes are not in cohort_arm_a.csv — extra "
                f"downloads, or the wrong manifest.")

    # ── tile-count distribution, for the methods section ─────────────────
    tiles = sorted(v.get("n_tiles", 0) for v in merged.values())
    if tiles:
        cap = max((v.get("max_tiles") or 0) for v in merged.values())
        n = len(tiles)
        below = sum(1 for t in tiles if t < cap)
        print(f"\n  tile counts  : min {tiles[0]}  p25 {tiles[n // 4]}  "
              f"median {tiles[n // 2]}  max {tiles[-1]}  (cap {cap})")
        print(f"                 {below} slides below the cap "
              f"({below / n:.1%}); {n - below} at the cap")
        thin = [k for k, v in merged.items() if v.get("n_tiles", 0) < 500]
        if thin:
            print(f"                 {len(thin)} slides under 500 tiles "
                  f"(thin MIL bags): {', '.join(sorted(thin)[:5])}"
                  f"{' …' if len(thin) > 5 else ''}")

    # ── verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if problems:
        print("  REFUSING TO MERGE — fix these first:\n")
        for p in problems:
            print(f"   ✗ {p}\n")
        print("=" * 70)
        sys.exit(1)

    print("  All checks passed.")
    if not args.apply:
        print("  Dry run — nothing written. Re-run with --apply to commit.")
        print("=" * 70)
        return

    backup = existing_path.with_suffix(".json.bak")
    if existing_path.is_file():
        shutil.copy2(existing_path, backup)
        print(f"  backup   → {backup}")
    with existing_path.open("w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"  written  → {existing_path}  ({len(merged)} entries)")
    print("=" * 70)


if __name__ == "__main__":
    main()
