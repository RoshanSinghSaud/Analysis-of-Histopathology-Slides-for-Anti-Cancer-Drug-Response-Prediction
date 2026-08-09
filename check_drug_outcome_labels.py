"""
check_drug_outcome_labels.py
============================
GO / NO-GO AUDIT for the treatment-conditioned design.

Question this script answers, with no assumptions:
    How many TCGA patients have ALL THREE of
        (1) a NAMED drug they actually received  (-> can be mapped to SMILES),
        (2) a USABLE outcome label,
        (3) at least one diagnostic H&E whole-slide image (.svs),
    and how many DISTINCT drugs are represented?

(3) matters because the SMILES co-attention branch is only learnable if the
same model sees many different drugs. A cohort of 3000 patients who all got
one regimen gives the drug branch nothing to learn from, and the
with-SMILES vs without-SMILES ablation goes flat.

Everything is read live from the GDC API (harmonized data portal). Nothing is
downloaded except JSON, so this is cheap to re-run.

Outcome definitions used (all reported separately, none silently merged):

  OS   Overall survival. vital_status is Dead or Alive AND the matching time
       field is present and positive (days_to_death / days_to_last_follow_up).
       Most abundant, but it is a PATIENT-level outcome - the drug is one of
       many things influencing it. Interpret as "treatment-conditioned
       prognosis", not "drug efficacy".

  PFI  Progression / recurrence. diagnoses.progression_or_recurrence or
       follow_ups.progression_or_recurrence is an explicit yes/no.

  TRO  diagnoses.treatments.treatment_outcome - the only outcome recorded
       AGAINST A SPECIFIC TREATMENT record, so it is the cleanest
       drug-attributable label. Expect it to be sparse; that is the point of
       measuring it.

Run:  python check_drug_outcome_labels.py
Writes: drug_outcome_audit.json  +  drug_outcome_audit.md  next to this file.
"""

import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

API = "https://api.gdc.cancer.gov"
OUT_DIR = Path(__file__).resolve().parent
PROGRAM = "TCGA"          # set to None to audit every program in the GDC
PAGE = 500

# Values that LOOK like data but carry no information. Anything in here is
# treated as missing rather than as a drug name / label.
NULLISH = {
    "", "not reported", "unknown", "not otherwise specified", "na", "n/a",
    "not applicable", "unspecified", "other", "not available", "none",
    "no drug", "drug not available", "[unknown]", "[not available]",
    "[not applicable]", "[discrepancy]",
}

# treatment_outcome is free-ish text. Only RECIST-interpretable values are a
# real label. "treatment ongoing", "no measurable disease" and "persistent
# disease" are NOT outcomes - they are administrative states - and counting
# them would inflate the cohort with unlabelled patients.
RECIST_RESPONDER = {"complete response", "partial response"}
RECIST_NONRESPONDER = {"progressive disease", "stable disease", "no response"}
RECIST_VALID = RECIST_RESPONDER | RECIST_NONRESPONDER


def is_null(v) -> bool:
    return v is None or str(v).strip().lower() in NULLISH


def post(endpoint: str, payload: dict, retries: int = 4) -> dict:
    """POST to the GDC API with simple retry/backoff."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API}/{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  ! {type(e).__name__}: {e} - retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def program_filter():
    if PROGRAM is None:
        return None
    return {"op": "in", "content": {"field": "project.program.name", "value": [PROGRAM]}}


# ---------------------------------------------------------------------------
# 1. Pull every case with its treatments, demographic and follow-ups
# ---------------------------------------------------------------------------
CASE_FIELDS = ",".join([
    "case_id",
    "submitter_id",
    "project.project_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.progression_or_recurrence",
    "diagnoses.primary_diagnosis",
    "diagnoses.treatments.therapeutic_agents",
    "diagnoses.treatments.treatment_type",
    "diagnoses.treatments.treatment_or_therapy",
    "diagnoses.treatments.treatment_intent_type",
    "diagnoses.treatments.treatment_outcome",
    "diagnoses.treatments.treatment_effect",
    "diagnoses.treatments.days_to_treatment_start",
    "follow_ups.progression_or_recurrence",
    "follow_ups.days_to_progression",
    "follow_ups.days_to_recurrence",
])


def fetch_cases():
    cases, frm = [], 0
    while True:
        payload = {"fields": CASE_FIELDS, "size": PAGE, "from": frm, "format": "JSON"}
        f = program_filter()
        if f:
            payload["filters"] = f
        d = post("cases", payload)["data"]
        hits = d["hits"]
        cases.extend(hits)
        total = d["pagination"]["total"]
        print(f"  cases {len(cases)}/{total}")
        frm += PAGE
        if frm >= total:
            break
    return cases


# ---------------------------------------------------------------------------
# 2. Which cases actually have a diagnostic H&E slide image?
# ---------------------------------------------------------------------------
def fetch_slide_case_ids():
    """case_ids owning at least one .svs slide image, split by slide type.

    Diagnostic Slide = the FFPE H&E section a pathologist read (DX* barcodes).
    Tissue Slide     = frozen section quality-control slide (TS*/BS*), much
                       poorer morphology. Kept separate on purpose - do not
                       train on frozen slides without saying so in the paper.
    """
    out = {}
    for strategy in ("Diagnostic Slide", "Tissue Slide"):
        ids, frm = set(), 0
        while True:
            filters = {"op": "and", "content": [
                {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
                {"op": "in", "content": {"field": "experimental_strategy", "value": [strategy]}},
            ]}
            f = program_filter()
            if f:
                filters["content"].append(
                    {"op": "in", "content": {"field": "cases.project.program.name",
                                             "value": [PROGRAM]}})
            payload = {"filters": filters, "fields": "cases.case_id",
                       "size": 1000, "from": frm, "format": "JSON"}
            d = post("files", payload)["data"]
            for h in d["hits"]:
                for c in h.get("cases", []) or []:
                    if c.get("case_id"):
                        ids.add(c["case_id"])
            total = d["pagination"]["total"]
            frm += 1000
            print(f"  {strategy}: {min(frm, total)}/{total} files -> {len(ids)} cases")
            if frm >= total:
                break
        out[strategy] = ids
    return out


# ---------------------------------------------------------------------------
# 3. Per-case extraction
# ---------------------------------------------------------------------------
def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def extract(case):
    """Return a flat record of what this case can and cannot supply."""
    rec = {
        "case_id": case.get("case_id"),
        "submitter_id": case.get("submitter_id"),
        "project": (case.get("project") or {}).get("project_id"),
        "drugs": set(),
        "drug_records": [],     # (drug, intent, outcome, effect) per treatment
        "os_event": None,       # 1 = died, 0 = censored
        "os_time": None,
        "pfi_event": None,
        "tro_labels": [],       # per-treatment treatment_outcome values
    }

    diagnoses = as_list(case.get("diagnoses"))

    # --- drugs -------------------------------------------------------------
    for dx in diagnoses:
        for tx in as_list(dx.get("treatments")):
            agents = [a for a in as_list(tx.get("therapeutic_agents")) if not is_null(a)]
            # therapeutic_agents is sometimes a single comma/plus separated string
            split = []
            for a in agents:
                for part in str(a).replace("+", ",").replace(";", ",").split(","):
                    part = part.strip().lower()
                    if part and not is_null(part):
                        split.append(part)
            outcome = tx.get("treatment_outcome")
            effect = tx.get("treatment_effect")
            for drug in split:
                rec["drugs"].add(drug)
                rec["drug_records"].append({
                    "drug": drug,
                    "treatment_type": tx.get("treatment_type"),
                    "intent": tx.get("treatment_intent_type"),
                    "outcome": None if is_null(outcome) else outcome,
                    "effect": None if is_null(effect) else effect,
                    "days_to_start": tx.get("days_to_treatment_start"),
                })
            if not is_null(outcome) and str(outcome).strip().lower() in RECIST_VALID:
                rec["tro_labels"].append(str(outcome).strip().lower())

    # --- overall survival --------------------------------------------------
    demo = case.get("demographic") or {}
    vs = (demo.get("vital_status") or "").strip().lower()
    if vs == "dead":
        t = demo.get("days_to_death")
        if t is not None and float(t) > 0:
            rec["os_event"], rec["os_time"] = 1, float(t)
    elif vs == "alive":
        fups = [dx.get("days_to_last_follow_up") for dx in diagnoses]
        fups = [float(t) for t in fups if t is not None and float(t) > 0]
        if fups:
            rec["os_event"], rec["os_time"] = 0, max(fups)

    # --- progression / recurrence -----------------------------------------
    pr_vals = [dx.get("progression_or_recurrence") for dx in diagnoses]
    pr_vals += [fu.get("progression_or_recurrence") for fu in as_list(case.get("follow_ups"))]
    pr_vals = [str(v).strip().lower() for v in pr_vals if not is_null(v)]
    if any(v == "yes" for v in pr_vals):
        rec["pfi_event"] = 1
    elif any(v == "no" for v in pr_vals):
        rec["pfi_event"] = 0

    return rec


# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------
def main():
    scope = PROGRAM or "ALL GDC PROGRAMS"
    print(f"\nAuditing {scope} via the GDC API\n" + "=" * 64)

    print("\n[1/3] Fetching cases + treatments ...")
    cases = fetch_cases()

    print("\n[2/3] Fetching slide availability ...")
    slides = fetch_slide_case_ids()
    dx_slides = slides["Diagnostic Slide"]
    ts_slides = slides["Tissue Slide"]

    print("\n[3/3] Cross-tabulating ...")
    recs = [extract(c) for c in cases]

    n_total = len(recs)
    has_drug = [r for r in recs if r["drugs"]]
    has_os = [r for r in recs if r["os_event"] is not None]
    has_pfi = [r for r in recs if r["pfi_event"] is not None]
    has_tro = [r for r in recs if r["tro_labels"]]
    has_dx = [r for r in recs if r["case_id"] in dx_slides]

    def triple(outcome_key, pool):
        return [r for r in pool if r["drugs"] and r["case_id"] in dx_slides]

    usable_os = triple("os", has_os)
    usable_pfi = triple("pfi", has_pfi)
    usable_tro = triple("tro", has_tro)

    # STRICT pairing: the RECIST outcome and the drug must come from the SAME
    # treatment record, so the outcome is actually attributable to that drug.
    # The looser case-level version above lets any drug on the case pair with
    # any outcome on the case, which is not defensible in a paper. GDC's own
    # portal filters use the strict (nested) semantics; we match that and
    # report both so the gap is visible instead of hidden.
    def strict_pairs(r):
        return [d for d in r["drug_records"]
                if d["outcome"] and str(d["outcome"]).strip().lower() in RECIST_VALID]

    usable_tro_strict = [r for r in usable_tro if strict_pairs(r)]

    drug_counter = Counter()
    for r in usable_os:
        for d in r["drugs"]:
            drug_counter[d] += 1

    # Binary responder label from the per-treatment RECIST outcome. This is the
    # closest drug-attributable replacement for pCR. A patient is a responder if
    # ANY of their treatments reached CR/PR and none is contradictory; patients
    # with both responder and non-responder records are counted as "mixed" and
    # reported separately rather than silently assigned.
    tro_drug_counter = Counter()
    resp, nonresp, mixed = 0, 0, 0
    n_pairs = 0
    for r in usable_tro_strict:
        pairs = strict_pairs(r)
        n_pairs += len(pairs)
        s = {str(d["outcome"]).strip().lower() for d in pairs}
        hit = bool(s & RECIST_RESPONDER)
        miss = bool(s & RECIST_NONRESPONDER)
        if hit and miss:
            mixed += 1
        elif hit:
            resp += 1
        elif miss:
            nonresp += 1
        for d in pairs:
            tro_drug_counter[d["drug"]] += 1

    per_project = defaultdict(lambda: {"usable_os": 0, "drugs": set()})
    for r in usable_os:
        p = per_project[r["project"]]
        p["usable_os"] += 1
        p["drugs"].update(r["drugs"])

    mono = sum(1 for r in usable_os if len(r["drugs"]) == 1)
    combo = sum(1 for r in usable_os if len(r["drugs"]) > 1)

    # ---- machine-readable ------------------------------------------------
    result = {
        "scope": scope,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "cases_total": n_total,
            "with_named_drug": len(has_drug),
            "with_os_label": len(has_os),
            "with_pfi_label": len(has_pfi),
            "with_treatment_outcome_label": len(has_tro),
            "with_diagnostic_slide": len(has_dx),
            "with_tissue_slide_only": len([r for r in recs
                                           if r["case_id"] in ts_slides
                                           and r["case_id"] not in dx_slides]),
            "USABLE_drug_and_OS_and_slide": len(usable_os),
            "USABLE_drug_and_PFI_and_slide": len(usable_pfi),
            "USABLE_drug_and_TREATMENTOUTCOME_and_slide": len(usable_tro_strict),
            "USABLE_drug_and_TREATMENTOUTCOME_and_slide_loose": len(usable_tro),
        },
        "drug_variation": {
            "distinct_drugs_in_usable_os_cohort": len(drug_counter),
            "drugs_with_ge_20_patients": sum(1 for v in drug_counter.values() if v >= 20),
            "drugs_with_ge_50_patients": sum(1 for v in drug_counter.values() if v >= 50),
            "monotherapy_patients": mono,
            "combination_patients": combo,
            "top_40_drugs": drug_counter.most_common(40),
        },
        "recist_response_cohort": {
            "patients_strict_same_treatment_record": len(usable_tro_strict),
            "patients_loose_case_level": len(usable_tro),
            "drug_outcome_pairs_strict": n_pairs,
            "responder_CR_or_PR": resp,
            "nonresponder_SD_or_PD": nonresp,
            "mixed_records_excluded": mixed,
            "distinct_drugs": len(tro_drug_counter),
            "drugs_with_ge_20_patients": sum(1 for v in tro_drug_counter.values() if v >= 20),
            "top_20_drugs": tro_drug_counter.most_common(20),
        },
        "per_project": {
            k: {"usable_os": v["usable_os"], "distinct_drugs": len(v["drugs"])}
            for k, v in sorted(per_project.items(),
                               key=lambda kv: -kv[1]["usable_os"])
        },
    }

    (OUT_DIR / "drug_outcome_audit.json").write_text(json.dumps(result, indent=2))

    # ---- human-readable ---------------------------------------------------
    c = result["counts"]
    dv = result["drug_variation"]
    L = []
    L.append(f"# TCGA drug + outcome audit ({scope})")
    L.append(f"\nGenerated {result['generated_utc']} from the live GDC API.\n")
    L.append("## Funnel\n")
    L.append("| Step | Patients |")
    L.append("|---|---:|")
    L.append(f"| Cases in scope | {c['cases_total']} |")
    L.append(f"| ...with >=1 NAMED therapeutic agent | {c['with_named_drug']} |")
    L.append(f"| ...with a usable OS label | {c['with_os_label']} |")
    L.append(f"| ...with an explicit progression/recurrence label | {c['with_pfi_label']} |")
    L.append(f"| ...with a per-treatment treatment_outcome | {c['with_treatment_outcome_label']} |")
    L.append(f"| ...with >=1 diagnostic (FFPE H&E) slide | {c['with_diagnostic_slide']} |")
    L.append("")
    L.append("## The number that decides the project\n")
    L.append("| Cohort (drug AND outcome AND diagnostic slide) | Patients |")
    L.append("|---|---:|")
    L.append(f"| drug + overall survival + slide | **{c['USABLE_drug_and_OS_and_slide']}** |")
    L.append(f"| drug + progression/recurrence + slide | {c['USABLE_drug_and_PFI_and_slide']} |")
    L.append(f"| drug + RECIST treatment outcome + slide (strict pairing) | "
             f"**{c['USABLE_drug_and_TREATMENTOUTCOME_and_slide']}** |")
    L.append("")
    L.append("## Drug variation in the OS cohort\n")
    L.append(f"- distinct drugs: **{dv['distinct_drugs_in_usable_os_cohort']}**")
    L.append(f"- drugs seen in >=20 patients: {dv['drugs_with_ge_20_patients']}")
    L.append(f"- drugs seen in >=50 patients: {dv['drugs_with_ge_50_patients']}")
    L.append(f"- monotherapy patients: {dv['monotherapy_patients']}")
    L.append(f"- combination patients: {dv['combination_patients']}")
    L.append("\n| Drug | Patients |\n|---|---:|")
    for d, n in dv["top_40_drugs"]:
        L.append(f"| {d} | {n} |")
    rc = result["recist_response_cohort"]
    L.append("\n## RECIST response cohort (the pCR replacement)\n")
    L.append("Per-treatment `treatment_outcome`, restricted to RECIST-interpretable "
             "values only. Administrative states ('treatment ongoing', "
             "'no measurable disease', 'persistent disease') are NOT counted as labels.\n")
    L.append(f"- **STRICT** (drug and outcome on the SAME treatment record) "
             f"+ diagnostic slide: **{rc['patients_strict_same_treatment_record']}** patients, "
             f"{rc['drug_outcome_pairs_strict']} drug-outcome pairs")
    L.append(f"- loose (any drug on the case paired with any outcome on the case): "
             f"{rc['patients_loose_case_level']} - do NOT use this as the headline")
    L.append(f"- responders (CR or PR): {rc['responder_CR_or_PR']}")
    L.append(f"- non-responders (SD, PD or no response): {rc['nonresponder_SD_or_PD']}")
    L.append(f"- mixed / contradictory records (excluded): {rc['mixed_records_excluded']}")
    L.append(f"- distinct drugs: {rc['distinct_drugs']} "
             f"({rc['drugs_with_ge_20_patients']} seen in >=20 patients)")
    L.append("\n| Drug | Patients |\n|---|---:|")
    for d, n in rc["top_20_drugs"]:
        L.append(f"| {d} | {n} |")
    L.append("\n## Per cancer type (OS cohort)\n")
    L.append("| Project | Usable patients | Distinct drugs |\n|---|---:|---:|")
    for k, v in result["per_project"].items():
        L.append(f"| {k} | {v['usable_os']} | {v['distinct_drugs']} |")
    L.append("")
    L.append("## Caveats to carry into the paper\n")
    L.append("- OS is a patient-level outcome; the drug is one of many causes. "
             "Frame results as treatment-conditioned prediction, never as proof "
             "of drug efficacy.")
    L.append("- TCGA drug names are free text harmonized by the GDC; brand/generic "
             "and salt forms (e.g. 'doxorubicin' vs 'doxorubicin hydrochloride') "
             "appear separately and must be normalized before SMILES lookup.")
    L.append("- Combination therapy is common. Decide explicitly whether the drug "
             "branch takes one SMILES, a set, or a pooled regimen embedding.")
    L.append("- Diagnostic (FFPE) and Tissue (frozen) slides are counted separately. "
             "Frozen sections have much worse morphology; state which you used.")
    L.append("- Treatment start dates are frequently missing, so immortal-time bias "
             "cannot always be excluded. Report how many patients had a usable "
             "days_to_treatment_start.")

    (OUT_DIR / "drug_outcome_audit.md").write_text("\n".join(L))

    print("\n" + "=" * 64)
    print(f"USABLE (drug + OS + diagnostic slide) : {c['USABLE_drug_and_OS_and_slide']}")
    print(f"distinct drugs in that cohort         : {dv['distinct_drugs_in_usable_os_cohort']}")
    print(f"drugs with >=20 patients              : {dv['drugs_with_ge_20_patients']}")
    print(f"drug + RECIST outcome + slide (STRICT): {len(usable_tro_strict)}"
          f"  ({resp} responder / {nonresp} non-responder / {mixed} mixed)")
    print(f"  ... same, loose case-level pairing   : {len(usable_tro)}")
    print(f"distinct drugs in RECIST cohort       : {len(tro_drug_counter)}")
    print("=" * 64)
    print(f"\nWrote {OUT_DIR/'drug_outcome_audit.md'}")
    print(f"Wrote {OUT_DIR/'drug_outcome_audit.json'}")


if __name__ == "__main__":
    main()
