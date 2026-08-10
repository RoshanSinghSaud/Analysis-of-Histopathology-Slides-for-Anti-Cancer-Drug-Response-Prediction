"""
Check that every .svs slide has a corresponding clinical .xml (and vice-versa),
matched by TCGA patient barcode (TCGA-XX-XXXX).

Edit the two paths below to point at your folders, then run in VS Code
(Run ▷ button) or:  python check_svs_xml_match.py
"""

import re
from pathlib import Path

# ── EDIT THESE TWO PATHS ─────────────────────────────────────────────────────
# Point them at the folders holding your flat .svs and .xml files.
BASE_DIR     = Path(__file__).resolve().parents[1]
WSI_DIR      = BASE_DIR / "data" / "wsi"
CLINICAL_DIR = BASE_DIR / "data" / "clinical"
# ─────────────────────────────────────────────────────────────────────────────

BARCODE_RE = re.compile(r"TCGA-\w\w-\w{4}", re.IGNORECASE)


def barcode(name: str):
    """Extract the TCGA-XX-XXXX patient barcode from a filename, or None."""
    m = BARCODE_RE.search(name)
    return m.group(0).upper() if m else None


def index(folder: Path, pattern: str):
    """Return {barcode: [filenames]} and a list of files with no barcode."""
    by_bc, no_bc = {}, []
    for f in sorted(folder.glob(pattern)):
        bc = barcode(f.name)
        if bc:
            by_bc.setdefault(bc, []).append(f.name)
        else:
            no_bc.append(f.name)
    return by_bc, no_bc


def main():
    for d in (WSI_DIR, CLINICAL_DIR):
        if not d.is_dir():
            print(f"❌ Folder not found: {d}")
            print("   Fix WSI_DIR / CLINICAL_DIR at the top of the script.")
            return

    svs, svs_nobc = index(WSI_DIR, "*.svs")
    xml, xml_nobc = index(CLINICAL_DIR, "*.xml")

    svs_bcs, xml_bcs = set(svs), set(xml)
    matched     = sorted(svs_bcs & xml_bcs)
    svs_only    = sorted(svs_bcs - xml_bcs)   # slide present, no clinical XML
    xml_only    = sorted(xml_bcs - svs_bcs)   # XML present, no slide

    print(f"\n.svs files : {sum(len(v) for v in svs.values())} "
          f"({len(svs_bcs)} unique cases)")
    print(f".xml files : {sum(len(v) for v in xml.values())} "
          f"({len(xml_bcs)} unique cases)")

    print(f"\n✅ MATCHED cases ({len(matched)}):")
    for bc in matched:
        print(f"   {bc}   svs={svs[bc]}   xml={xml[bc]}")

    if svs_only:
        print(f"\n⚠️  SLIDES WITHOUT A CLINICAL XML ({len(svs_only)}):")
        for bc in svs_only:
            print(f"   {bc}   svs={svs[bc]}")

    if xml_only:
        print(f"\n⚠️  CLINICAL XML WITHOUT A SLIDE ({len(xml_only)}):")
        for bc in xml_only:
            print(f"   {bc}   xml={xml[bc]}")

    if svs_nobc:
        print(f"\n⚠️  .svs files with no parseable barcode: {svs_nobc}")
    if xml_nobc:
        print(f"\n⚠️  .xml files with no parseable barcode: {xml_nobc}")

    print("\n" + "=" * 50)
    if not svs_only and not xml_only and not svs_nobc and not xml_nobc:
        print("ALL MATCHED — every slide has a clinical XML. ✅")
    else:
        print("MISMATCHES found — see warnings above. ⚠️")
    print("=" * 50)


if __name__ == "__main__":
    main()
