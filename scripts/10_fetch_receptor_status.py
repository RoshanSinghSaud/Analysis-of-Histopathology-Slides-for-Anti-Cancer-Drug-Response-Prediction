#!/usr/bin/env python3
"""
=============================================================================
10_fetch_receptor_status.py — Gate 0(c): ER / PR / HER2 for the cohort
=============================================================================

§6 of the analysis plan states that BOTH models receive the tissue context —
AJCC stage, and "for breast cases, ER/PR/HER2 receptor status (not in the
harmonized API; parsed from the BCR clinical XMLs)". Until this script ran,
that was true of the plan and false of the code: `load_real()` built its
context from stage and project only, and no receptor value existed anywhere in
the repo.

That matters because §6 leans on the tissue context to answer the sharpest
reviewer objection there is — confounding by indication. The argument is that
drug choice follows tumour type, stage and subtype, all of which are legible in
the slide, so a naive model could score well by inferring which drug was given
and reciting its prognosis; ΔAUROC is safe from that only BECAUSE both arms
already receive that information. In breast cancer, receptor subtype IS the
treatment decision. Leaving it out weakens the defence, in the direction that
hurts the paper.

WHAT THIS DOES
  1. reads cohort/cohort_<tag>.csv and takes its unique patient barcodes
  2. asks the GDC files endpoint for each patient's BCR XML clinical supplement
  3. downloads them into cohort/bcr_xml/  (resumable — existing files are kept)
  4. parses three fields, plus the FISH field used to resolve equivocal HER2
  5. writes cohort/receptor_status_<tag>.csv and prints a coverage report

THE FIELDS, verified against a real TCGA-BRCA XML on 2026-08-28:
    breast_carcinoma_estrogen_receptor_status              -> er_status
    breast_carcinoma_progesterone_receptor_status          -> pr_status
    lab_proc_her2_neu_immunohistochemistry_receptor_status -> her2_ihc
    lab_procedure_her2_neu_in_situ_hybrid_outcome_type     -> her2_fish

  The `metastatic_breast_carcinoma_*` variants of these tags are DELIBERATELY
  IGNORED. They describe a metastatic deposit rather than the primary tumour
  the slide was cut from, and in the file inspected they were empty. Matching
  them loosely would silently mix two different specimens.

  HER2 rule, fixed here before any receptor value reaches a model: HER2 status
  is the IHC result, except that an Equivocal IHC is resolved by FISH when FISH
  is present. This is standard clinical practice (ASCO/CAP) and is written down
  so it is not a choice made later to suit a result.

  Anything absent, blank, or recorded as indeterminate/equivocal-after-FISH
  becomes "unknown". Unknown is a real category and is kept as one — dropping
  those patients would shrink the cohort on a variable that is missing more
  often in older cases.

NOTHING HERE TOUCHES AN OUTCOME. This script reads no label and writes no
label; it only adds inputs both arms receive equally.

Usage
-----
    python scripts/10_fetch_receptor_status.py --tag brca
    python scripts/10_fetch_receptor_status.py --tag arm_a      # breast subset only
    python scripts/10_fetch_receptor_status.py --tag brca --limit 5   # smoke test
=============================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"

ER_TAG = "breast_carcinoma_estrogen_receptor_status"
PR_TAG = "breast_carcinoma_progesterone_receptor_status"
HER2_IHC_TAG = "lab_proc_her2_neu_immunohistochemistry_receptor_status"
HER2_FISH_TAG = "lab_procedure_her2_neu_in_situ_hybrid_outcome_type"

POSITIVE = {"positive"}
NEGATIVE = {"negative"}
EQUIVOCAL = {"equivocal", "indeterminate"}


def strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def norm(value: str | None) -> str:
    """Map a raw BCR string onto positive / negative / equivocal / unknown."""
    v = (value or "").strip().lower()
    if v in POSITIVE:
        return "positive"
    if v in NEGATIVE:
        return "negative"
    if v in EQUIVOCAL:
        return "equivocal"
    return "unknown"


def post_json(url: str, payload: dict, retries: int = 4) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    GDC request failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def find_xml_files(barcodes: list[str], chunk: int = 100) -> dict[str, dict]:
    """barcode -> {file_id, file_name}. One GDC query per chunk of barcodes."""
    found: dict[str, dict] = {}
    for i in range(0, len(barcodes), chunk):
        part = barcodes[i:i + chunk]
        payload = {
            "filters": {"op": "and", "content": [
                {"op": "in", "content": {"field": "cases.submitter_id", "value": part}},
                {"op": "in", "content": {"field": "data_category", "value": ["Clinical"]}},
                {"op": "in", "content": {"field": "data_format", "value": ["BCR XML"]}},
            ]},
            "fields": "file_id,file_name,cases.submitter_id",
            "format": "JSON",
            "size": str(len(part) * 4),
        }
        d = post_json(GDC_FILES, payload)
        for hit in d["data"]["hits"]:
            for case in hit.get("cases", []):
                bc = case["submitter_id"]
                # a case can have more than one clinical supplement; keep the
                # first deterministically so a re-run resolves the same file
                if bc not in found or hit["file_name"] < found[bc]["file_name"]:
                    found[bc] = {"file_id": hit["file_id"], "file_name": hit["file_name"]}
        print(f"    queried {i + len(part):>4}/{len(barcodes)} barcodes · "
              f"{len(found)} matched so far")
    return found


def download(file_id: str, dest: Path, retries: int = 4) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return False
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{GDC_DATA}/{file_id}", timeout=120) as r:
                data = r.read()
            dest.write_bytes(data)
            return True
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return False


def parse_receptors(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    raw: dict[str, str] = {}
    for el in root.iter():
        tag = strip_ns(el.tag)
        if tag in (ER_TAG, PR_TAG, HER2_IHC_TAG, HER2_FISH_TAG):
            txt = (el.text or "").strip()
            if txt and not raw.get(tag):
                raw[tag] = txt

    er = norm(raw.get(ER_TAG))
    pr = norm(raw.get(PR_TAG))
    ihc = norm(raw.get(HER2_IHC_TAG))
    fish = norm(raw.get(HER2_FISH_TAG))

    # HER2 = IHC, with FISH resolving an equivocal IHC (ASCO/CAP). Fixed here,
    # before any receptor value reaches a model.
    if ihc in ("positive", "negative"):
        her2 = ihc
    elif ihc == "equivocal" and fish in ("positive", "negative"):
        her2 = fish
    elif ihc == "unknown" and fish in ("positive", "negative"):
        her2 = fish
    else:
        her2 = "unknown"

    return {"er_status": er, "pr_status": pr,
            "her2_ihc": ihc, "her2_fish": fish, "her2_status": her2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="brca")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N patients — smoke test")
    ap.add_argument("--xml-dir", default=str(BASE_DIR / "cohort" / "bcr_xml"))
    a = ap.parse_args()

    cohort_csv = BASE_DIR / "cohort" / f"cohort_{a.tag}.csv"
    if not cohort_csv.exists():
        print(f"ERROR: {cohort_csv} not found")
        return 1

    with cohort_csv.open() as f:
        rows = list(csv.DictReader(f))
    barcodes = sorted({r["submitter_id"] for r in rows if r.get("submitter_id")})
    projects = Counter(r.get("project", "?") for r in rows)
    if a.limit:
        barcodes = barcodes[:a.limit]

    print("=" * 74)
    print(f"GATE 0(c) — receptor status for cohort '{a.tag}'")
    print("=" * 74)
    print(f"  cohort file       : {cohort_csv.name}")
    print(f"  unique patients   : {len(barcodes)}")
    print(f"  projects          : {dict(projects)}")
    print("  reads NO outcome column; adds inputs both arms receive equally\n")

    print("  [1/3] locating BCR XML clinical supplements on the GDC")
    found = find_xml_files(barcodes)
    missing = [b for b in barcodes if b not in found]
    print(f"    matched {len(found)}/{len(barcodes)}   missing {len(missing)}")

    xml_dir = Path(a.xml_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [2/3] downloading into {xml_dir}")
    fetched = kept = 0
    for n, (bc, meta) in enumerate(sorted(found.items()), 1):
        dest = xml_dir / meta["file_name"]
        if download(meta["file_id"], dest):
            fetched += 1
        else:
            kept += 1
        if n % 25 == 0 or n == len(found):
            print(f"    {n:>4}/{len(found)}   fetched {fetched}, already present {kept}")

    print("\n  [3/3] parsing")
    out_rows, fails = [], []
    for bc, meta in sorted(found.items()):
        try:
            rec = parse_receptors(xml_dir / meta["file_name"])
        except Exception as e:
            fails.append((bc, str(e)))
            continue
        rec["submitter_id"] = bc
        rec["source_file"] = meta["file_name"]
        out_rows.append(rec)
    for bc in missing:
        out_rows.append({"submitter_id": bc, "er_status": "unknown",
                         "pr_status": "unknown", "her2_ihc": "unknown",
                         "her2_fish": "unknown", "her2_status": "unknown",
                         "source_file": ""})

    cols = ["submitter_id", "er_status", "pr_status", "her2_status",
            "her2_ihc", "her2_fish", "source_file"]
    outp = BASE_DIR / "cohort" / f"receptor_status_{a.tag}.csv"
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(out_rows, key=lambda r: r["submitter_id"]):
            w.writerow({c: r.get(c, "") for c in cols})

    print("\n" + "=" * 74)
    print("COVERAGE — report these numbers in the methods")
    print("=" * 74)
    n = len(out_rows)
    for field in ("er_status", "pr_status", "her2_status"):
        dist = Counter(r[field] for r in out_rows)
        known = n - dist.get("unknown", 0)
        print(f"  {field:<12} known {known:>4}/{n} ({100*known/max(n,1):>5.1f}%)   " +
              "  ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    trip = Counter((r["er_status"], r["pr_status"], r["her2_status"]) for r in out_rows)
    complete = sum(v for k, v in trip.items() if "unknown" not in k)
    print(f"\n  all three known   : {complete}/{n} ({100*complete/max(n,1):.1f}%)")
    if fails:
        print(f"\n  ⚠️  {len(fails)} XML files failed to parse:")
        for bc, e in fails[:5]:
            print(f"      {bc}: {e}")
    if missing:
        print(f"\n  ⚠️  {len(missing)} patients had no BCR XML on the GDC "
              f"(recorded as unknown, not dropped)")
        for bc in missing[:5]:
            print(f"      {bc}")

    print(f"\n  written to {outp}")
    print("\n  NEXT: 08_evaluate_ablation.py joins this file into the model's "
          "context vector.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
