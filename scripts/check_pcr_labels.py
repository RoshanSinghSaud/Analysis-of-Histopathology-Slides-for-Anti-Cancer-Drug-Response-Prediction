"""
Scan GDC clinical data and report how many cases have a usable pCR / treatment-
response label, BEFORE downloading the large slide set.

Why this matters
----------------
pCR is a NEOADJUVANT-only pathologic outcome. TCGA is mostly surgery-first, so
most cases have NO pCR value even if they received a drug. This script doesn't
assume — it reads your clinical files and prints exactly what label fields exist
and how many cases actually have them, so you know your real labelled-cohort
size first.

Works on either:
  • the BCR clinical XML files you already download (point at the clinical/ folder), or
  • a GDC "Clinical: TSV" export folder (point at the folder of .tsv files).

The script auto-detects which one it finds.

Edit CLINICAL_DIR below, then run in VS Code (Run ▷) or:  python check_pcr_labels.py
"""

import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# ── EDIT THIS PATH ───────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]
CLINICAL_DIR = BASE_DIR / "downloaded-test-data" / "clinical"
# ─────────────────────────────────────────────────────────────────────────────

BARCODE_RE = re.compile(r"TCGA-\w\w-\w{4}", re.IGNORECASE)

# Field-NAME keywords we treat as candidate response / pCR labels.
RESPONSE_KEYS = [
    "measure_of_response", "primary_therapy_outcome_success",
    "treatment_outcome", "pathologic_complete", "pathologic_response",
    "complete_response", "pcr", "response",
]
NEOADJ_KEYS = ["neoadjuvant"]


def strip_ns(tag):
    return tag.split("}")[-1] if "}" in tag else tag


# =============================================================================
# XML MODE  (BCR clinical XML — the files in your clinical/ folder)
# =============================================================================
def scan_xml(folder):
    xmls = sorted(folder.glob("*.xml"))
    if not xmls:
        return False
    print(f"\nMode: BCR clinical XML  ({len(xmls)} files)\n" + "-" * 60)

    neoadj_yes, has_response, fully_usable = [], [], []
    outcome_dist, measure_dist = Counter(), Counter()

    for xf in xmls:
        try:
            root = ET.parse(xf).getroot()
        except Exception as e:
            print(f"  ⚠️  could not parse {xf.name}: {e}")
            continue

        m = BARCODE_RE.search(xf.stem)
        bc = m.group(0).upper() if m else xf.stem

        neo = None
        outcome = None
        measures = []
        for el in root.iter():
            tag = strip_ns(el.tag).lower()
            txt = (el.text or "").strip()
            if not txt:
                continue
            if "history_of_neoadjuvant_treatment" in tag:
                neo = txt
            elif "primary_therapy_outcome_success" in tag:
                outcome = txt
            elif "measure_of_response" in tag:
                measures.append(txt)

        is_neo = (neo or "").lower().startswith("yes")
        has_resp = bool(outcome) or bool(measures)

        if is_neo:
            neoadj_yes.append(bc)
        if outcome:
            outcome_dist[outcome] += 1
        for mv in measures:
            measure_dist[mv] += 1
        if has_resp:
            has_response.append(bc)
        if is_neo and has_resp:          # the pCR-relevant intersection
            fully_usable.append(bc)

        print(f"  {bc:18}  neoadjuvant={neo or '—':12}  "
              f"outcome={outcome or '—':22}  measure_of_response={measures or '—'}")

    n = len(xmls)
    print("\n" + "=" * 60)
    print("SUMMARY (XML)")
    print(f"  total cases scanned ............... {n}")
    print(f"  neoadjuvant = Yes ................. {len(neoadj_yes)}")
    print(f"  has ANY response value ........... {len(has_response)}")
    print(f"  neoadjuvant + response (pCR-usable) {len(fully_usable)}")
    if outcome_dist:
        print("\n  primary_therapy_outcome_success distribution:")
        for k, v in outcome_dist.most_common():
            print(f"     {v:4}  {k}")
    if measure_dist:
        print("\n  drug measure_of_response distribution:")
        for k, v in measure_dist.most_common():
            print(f"     {v:4}  {k}")
    _verdict(len(fully_usable), n)
    return True


# =============================================================================
# TSV MODE  (GDC "Clinical: TSV" export — a folder of .tsv files)
# =============================================================================
def scan_tsv(folder):
    tsvs = sorted(folder.glob("*.tsv"))
    if not tsvs:
        return False
    print(f"\nMode: GDC clinical TSV  ({len(tsvs)} files)\n" + "-" * 60)

    found_any = False
    for tf in tsvs:
        with open(tf, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if not rows:
            continue
        cols = rows[0].keys()
        resp_cols = [c for c in cols
                     if any(k in c.lower() for k in RESPONSE_KEYS)]
        neo_cols = [c for c in cols
                    if any(k in c.lower() for k in NEOADJ_KEYS)]
        if not resp_cols and not neo_cols:
            continue
        found_any = True
        print(f"\n  {tf.name}  ({len(rows)} rows)")
        for c in neo_cols + resp_cols:
            vals = Counter(
                (r.get(c) or "").strip() or "(blank)" for r in rows)
            non_blank = sum(v for k, v in vals.items()
                            if k not in ("(blank)", "'--", "--", "not reported",
                                         "unknown", "Not Reported"))
            print(f"     [{c}]  non-empty in {non_blank}/{len(rows)} rows")
            for k, v in vals.most_common(8):
                print(f"        {v:5}  {k}")

    if not found_any:
        print("  No response/neoadjuvant columns found in any TSV.")
        print("  → This cohort likely has no pCR label. See verdict below.")
        _verdict(0, 1)
    return True


def _verdict(usable, total):
    print("\n" + "=" * 60)
    if usable == 0:
        print("VERDICT: No usable pCR label found.")
        print("  TCGA is mostly surgery-first, so this is expected. Options:")
        print("   • reframe the label as RECIST treatment response (CR+PR vs")
        print("     SD+PD) and call it that — NOT pCR, or")
        print("   • source pCR from a dedicated neoadjuvant dataset.")
    elif usable < 30:
        print(f"VERDICT: Only {usable} pCR-usable cases — likely too few to")
        print("  train/evaluate a model. Reconsider label or data source.")
    else:
        print(f"VERDICT: {usable} pCR-usable cases. Build your slide manifest")
        print("  from THESE case barcodes only — don't download the rest.")
    print("=" * 60)


def main():
    if not CLINICAL_DIR.is_dir():
        print(f"❌ Folder not found: {CLINICAL_DIR}")
        print("   Fix CLINICAL_DIR at the top of the script.")
        return
    if scan_xml(CLINICAL_DIR):
        return
    if scan_tsv(CLINICAL_DIR):
        return
    print(f"❌ No .xml or .tsv clinical files found in {CLINICAL_DIR}")


if __name__ == "__main__":
    main()
