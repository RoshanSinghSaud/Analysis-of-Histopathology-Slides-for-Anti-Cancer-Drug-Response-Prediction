"""
=============================================================================
COHORT LABELS — the single source of truth for what a patient's label is
=============================================================================

Every script that needs a response label imports it from here, and here reads
it from the cohort CSV produced by `check_drug_outcome_labels.py`.

WHY THIS MODULE EXISTS
----------------------
Until Aug 11 2026, `01_preprocess_and_extract.py` built labels itself by
walking a patient's clinical XML and keeping the LAST `drug_name` and the LAST
`measure_of_response` tag it happened to encounter — with no pairing between
the two. A patient given four drugs with four different outcomes produced one
arbitrary (drug, outcome) guess.

That is exactly what the audit script exists to prevent. The pre-registered
label is a STRICT drug<->outcome pairing taken from the same treatment record,
with mixed-outcome patients excluded and RECIST-only rules applied. Those rules
live in `cohort/cohort_*.csv`. Reading labels from anywhere else means training
on a different study than the one that was registered.

So: never re-derive a label. Read it from the CSV.

THE COHORT CSV SHAPE
--------------------
One row per (patient, drug) pair — 561 rows for 199 BRCA patients. Columns:

    case_id, submitter_id, project, drug, treatment_outcome, response_binary,
    patient_label, stage, primary_diagnosis, treatment_type, treatment_intent,
    days_to_treatment_start, n_diagnostic_slides

`response_binary` and `patient_label` are patient-level and are guaranteed
consistent across a patient's rows, because mixed-outcome patients were
excluded by the audit. `load_cohort()` re-verifies that guarantee rather than
trusting it — if the exclusion ever regresses, this raises instead of silently
training on a coin flip.
=============================================================================
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# TCGA patient barcode, e.g. TCGA-A2-A3XV. Matches inside the long GDC
# filenames too: TCGA-A2-A3XV-01Z-00-DX1.4B58B7CD-....svs
BARCODE_RE = re.compile(r"TCGA-\w\w-\w{4}", re.IGNORECASE)


class CohortError(RuntimeError):
    """Raised when the cohort CSV is missing, malformed or self-inconsistent."""


def patient_barcode(name: str) -> str | None:
    """Extract the TCGA patient barcode from a slide filename, stem or barcode."""
    m = BARCODE_RE.search(name)
    return m.group(0).upper() if m else None


def find_cohort_csv(base_dir: Path, tag: str | None = None) -> Path:
    """
    Locate the cohort CSV in <base_dir>/cohort/.

    With `tag`, looks for exactly cohort_<tag>.csv. Without, accepts a single
    cohort_*.csv and refuses to guess when several exist — picking one silently
    is how an analysis ends up run on the wrong arm.
    """
    cohort_dir = base_dir / "cohort"
    if tag:
        p = cohort_dir / f"cohort_{tag}.csv"
        if not p.exists():
            raise CohortError(f"No cohort CSV for tag '{tag}': {p}")
        return p

    candidates = sorted(
        p for p in cohort_dir.glob("cohort_*.csv")
        if not p.name.endswith("_summary.csv") and "_slides_" not in p.name
    )
    if not candidates:
        raise CohortError(
            f"No cohort_*.csv in {cohort_dir}. Generate one first:\n"
            "    python scripts/check_drug_outcome_labels.py --export-cohort TCGA-BRCA"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise CohortError(
            f"Several cohort CSVs found ({names}). Pass an explicit tag so the "
            "arm being analysed is unambiguous."
        )
    return candidates[0]


def load_cohort(csv_path: Path) -> dict[str, dict]:
    """
    Read the cohort CSV into {patient_barcode -> patient record}.

    Each record:
        barcode            str    TCGA-A2-A3XV
        case_id            str    GDC UUID
        project            str    TCGA-BRCA
        response_binary    int    1 = responder, 0 = non-responder
        patient_label      str    'responder' | 'nonresponder'
        drugs              list   every drug paired to this patient's outcome
        pairs              list   [{drug, treatment_outcome, treatment_type,
                                    treatment_intent, days_to_treatment_start}]
        stage              str    AJCC pathologic stage
        primary_diagnosis  str
        n_drug_pairs       int

    Raises CohortError if a patient carries more than one distinct
    response_binary or patient_label — the mixed-outcome exclusion failing
    silently is the single most dangerous regression possible here.
    """
    if not Path(csv_path).exists():
        raise CohortError(f"Cohort CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise CohortError(f"Cohort CSV is empty: {csv_path}")

    required = {"submitter_id", "drug", "treatment_outcome",
                "response_binary", "patient_label"}
    missing = required - set(rows[0])
    if missing:
        raise CohortError(
            f"Cohort CSV {csv_path} is missing columns: {sorted(missing)}. "
            "Re-export it with the current check_drug_outcome_labels.py."
        )

    cohort: dict[str, dict] = {}
    for r in rows:
        bc = (r["submitter_id"] or "").strip().upper()
        if not bc:
            raise CohortError(f"Row with empty submitter_id in {csv_path}: {r}")

        try:
            resp = int(r["response_binary"])
        except (TypeError, ValueError):
            raise CohortError(
                f"Non-integer response_binary {r['response_binary']!r} for {bc}"
            ) from None

        rec = cohort.setdefault(bc, {
            "barcode":           bc,
            "case_id":           r.get("case_id", ""),
            "project":           r.get("project", ""),
            "response_binary":   resp,
            "patient_label":     r["patient_label"],
            "drugs":             [],
            "pairs":             [],
            "stage":             r.get("stage", ""),
            "primary_diagnosis": r.get("primary_diagnosis", ""),
        })

        # The consistency guarantee, checked rather than assumed.
        if rec["response_binary"] != resp:
            raise CohortError(
                f"Patient {bc} has conflicting response_binary "
                f"({rec['response_binary']} and {resp}) in {csv_path}. "
                "Mixed-outcome patients are supposed to be excluded by the "
                "audit — re-export the cohort before going any further."
            )
        if rec["patient_label"] != r["patient_label"]:
            raise CohortError(
                f"Patient {bc} has conflicting patient_label "
                f"({rec['patient_label']!r} and {r['patient_label']!r})."
            )

        drug = (r["drug"] or "").strip()
        if drug and drug not in rec["drugs"]:
            rec["drugs"].append(drug)
        rec["pairs"].append({
            "drug":                     drug,
            "treatment_outcome":        r.get("treatment_outcome", ""),
            "treatment_type":           r.get("treatment_type", ""),
            "treatment_intent":         r.get("treatment_intent", ""),
            "days_to_treatment_start":  r.get("days_to_treatment_start", ""),
        })

    for rec in cohort.values():
        rec["n_drug_pairs"] = len(rec["pairs"])

    return cohort


def label_for_slide(slide_name: str, cohort: dict[str, dict]) -> tuple[dict, str]:
    """
    Look up a slide's patient record.

    Returns (record, status) with status one of:
        'matched'          the slide's patient is in the cohort
        'no_barcode'       no TCGA barcode could be parsed from the filename
        'not_in_cohort'    barcode parsed, but that patient is not in the cohort

    'not_in_cohort' is not a warning to shrug at: the download manifest is
    generated FROM the cohort, so every downloaded slide should match. A miss
    means extra slides were downloaded, or the CSV and the manifest are from
    different exports.
    """
    bc = patient_barcode(slide_name)
    if bc is None:
        return {}, "no_barcode"
    rec = cohort.get(bc)
    if rec is None:
        return {"barcode": bc}, "not_in_cohort"
    return rec, "matched"


def compact(rec: dict) -> dict:
    """
    The patient record minus the per-pair detail — what gets stored in a
    per-slide manifest, where the full `pairs` list would be repeated for every
    slide of every patient. The CSV remains the place to look up pair detail.
    """
    if not rec:
        return {}
    return {
        "barcode":           rec.get("barcode", ""),
        "response_binary":   rec.get("response_binary"),
        "patient_label":     rec.get("patient_label", ""),
        "drugs":             rec.get("drugs", []),
        "stage":             rec.get("stage", ""),
        "primary_diagnosis": rec.get("primary_diagnosis", ""),
        "n_drug_pairs":      rec.get("n_drug_pairs", 0),
        "label_source":      "cohort_csv",
    }


def summarise(cohort: dict[str, dict]) -> str:
    """One-line description, for logs."""
    n = len(cohort)
    resp = sum(1 for r in cohort.values() if r["response_binary"] == 1)
    pairs = sum(r["n_drug_pairs"] for r in cohort.values())
    drugs = {d for r in cohort.values() for d in r["drugs"]}
    return (f"{n} patients ({resp} responder / {n - resp} non-responder), "
            f"{pairs} drug-outcome pairs, {len(drugs)} distinct drugs")
