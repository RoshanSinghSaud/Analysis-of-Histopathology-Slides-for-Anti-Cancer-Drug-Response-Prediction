#!/usr/bin/env python3
"""
14_audit_mixed_outcomes.py — who are the 36 Arm A patients excluded as "mixed"?

Worklist step 4 (29 Sep 2026). Read-only: fetches the Arm A cases from the GDC
API (the same source and the same strict_pairs / patient_label rules as
check_drug_outcome_labels.py), confirms the frozen cohort CSV is reproduced
exactly, and classifies each mixed-outcome patient by treatment timing:

  sequential                         >= 2 distinct days_to_treatment_start, and
                                     every start date carries ONE outcome class;
                                     the conflict is ACROSS lines of therapy
  sequential_with_same_day_conflict  >= 2 start dates, and one date also
                                     carries both outcome classes
  same_start_conflict                a single start date with both classes
  unknown_timing                     start dates missing

Writes cohort/mixed_outcome_audit_arm_a.json. Nothing in the cohort changes.
Cache of the GDC pull: ~/.cache/histopath/arm_a_cases.json (delete to refresh).
"""
import sys, json, importlib.util, os, time, csv
from collections import Counter, defaultdict
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[1])
sys.path.insert(0, 'scripts')
spec = importlib.util.spec_from_file_location("cdol", "scripts/check_drug_outcome_labels.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
os.makedirs(os.path.expanduser('~/.cache/histopath'), exist_ok=True)
out = os.path.expanduser('~/.cache/histopath/arm_a_cases.json')
st = json.load(open(out)) if os.path.exists(out) else {"cases": [], "frm": 0, "total": None}
t0 = time.time()
flt = {"op": "in", "content": {"field": "project.project_id", "value": m.ARM_A}}
while True:
    if st["total"] is not None and st["frm"] >= st["total"]:
        break
    d = m.post("cases", {"fields": m.CASE_FIELDS, "size": 500, "from": st["frm"], "format": "JSON", "filters": flt})["data"]
    st["cases"].extend(d["hits"]); st["total"] = d["pagination"]["total"]; st["frm"] += 500
    json.dump(st, open(out, "w"))
    print(len(st["cases"]), st["total"], flush=True)

# ---- classify ----

recs = [m.extract(c) for c in st["cases"]]
cand = [r for r in recs if r["drugs"] and m.strict_pairs(r)]
# diagnostic-slide membership, only for candidates
ids = [r["case_id"] for r in cand]
dx = set()
for i in range(0, len(ids), 200):
    f = {"op": "and", "content": [
        {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
        {"op": "in", "content": {"field": "cases.case_id", "value": ids[i:i+200]}}]}
    d = m.post("files", {"filters": f, "fields": "cases.case_id", "size": 2000, "format": "JSON"})["data"]
    for h in d["hits"]:
        for c in h.get("cases") or []:
            dx.add(c["case_id"])
cohort = [r for r in cand if r["case_id"] in dx]
labs = Counter(m.patient_label(m.strict_pairs(r)) for r in cohort)
print("cohort-eligible with labels:", labs)
kept_ids = {row["submitter_id"] for row in csv.DictReader(open("cohort/cohort_arm_a.csv"))}
nonmixed = {r["submitter_id"] for r in cohort if m.patient_label(m.strict_pairs(r)) in ("responder","nonresponder")}
print("non-mixed now:", len(nonmixed), " in frozen cohort csv:", len(kept_ids), " symmetric diff:", len(nonmixed ^ kept_ids))
mixed = [r for r in cohort if m.patient_label(m.strict_pairs(r)) == "mixed"]
rows = []
cls = Counter()
for r in mixed:
    pairs = m.strict_pairs(r)
    by_day = defaultdict(set)
    for p in pairs:
        by_day[p["days_to_start"]].add(str(p["outcome"]).strip().lower())
    days = [d for d in by_day if d is not None]
    lab_of = lambda s: "R" if s & m.RECIST_RESPONDER else "N"
    if None in by_day and len(days) < 2:
        c = "unknown_timing"
    elif len(days) >= 2:
        # conflicting outcomes attach to different start dates?
        per = {d: by_day[d] for d in days}
        within_conflict = any((s & m.RECIST_RESPONDER) and (s & m.RECIST_NONRESPONDER) for s in per.values())
        c = "sequential" if not within_conflict else "sequential_with_same_day_conflict"
    else:
        c = "same_start_conflict"
    cls[c] += 1
    ordered = sorted(((d, sorted({p["drug"] for p in pairs if p["days_to_start"] == d}), sorted(by_day[d])) for d in by_day), key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
    first = [o for o in ordered if o[0] is not None]
    rows.append({"submitter_id": r["submitter_id"], "project": r["project"], "class": c,
                 "n_start_dates": len(days),
                 "first_line_outcome": ",".join(first[0][2]) if first else "",
                 "timeline": " | ".join(f"d{d}: {'+'.join(dr)} -> {','.join(oc)}" for d, dr, oc in ordered)})
print("mixed:", len(mixed), cls)
print("mixed by project:", Counter(r["project"] for r in mixed))
json.dump({'generated_by':'scripts/14_audit_mixed_outcomes.py','n_mixed':len(mixed),'classes':dict(cls),'by_project':dict(Counter(r['project'] for r in mixed)),'frozen_cohort_reproduced':len(nonmixed ^ kept_ids)==0,'patients':rows}, open('cohort/mixed_outcome_audit_arm_a.json','w'), indent=1)
for x in rows: print(x["submitter_id"], x["project"], x["class"], "::", x["timeline"][:220])
