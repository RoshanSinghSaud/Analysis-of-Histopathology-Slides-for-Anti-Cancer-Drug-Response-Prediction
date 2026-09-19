#!/usr/bin/env python3
"""
=============================================================================
12_context_only_baseline.py — THE CONTEXT-ONLY REFERENCE ARM
=============================================================================

What does the tissue context ALONE predict, with no image and no drug?

This is the floor every reported AUROC in the paper must clear. §6 gives BOTH
arms of the primary comparison the tissue context (AJCC stage, cancer type,
ER/PR/HER2) precisely so that DeltaAUROC isolates chemistry rather than
confounding by indication. The unavoidable consequence is that both arms start
from whatever the context alone is worth — and on a pan-cancer cohort that is a
great deal, because cancer type nearly determines the label (TCGA-TGCT is
overwhelmingly responder, TCGA-LGG overwhelmingly non-responder).

An absolute AUROC reported without this number beside it is uninterpretable. A
reviewer WILL ask what cancer type alone achieves; the answer belongs in the
manuscript, not in the response to review.

WHAT THIS IS NOT
----------------
No model is trained. No image is read. No drug is used. This is a lookup table:
the observed non-responder RATE per context level, learned on the training folds
only, smoothed toward the training prior, and applied to held-out patients.
Because it is a fitted-and-evaluated predictor on real labels, it is logged to
`output/eval/run_log.jsonl` exactly like any other configuration (§8).

IT DOES NOT BIAS DeltaAUROC. Both arms hold the same context, so the primary
contrast remains clean — that was the point of giving both arms context. What
this number does is (a) compress the space in which an effect can appear and
(b) create a serious misinterpretation risk for any absolute AUROC.

PARITY WITH THE HARNESS — this is the point of the file
-------------------------------------------------------
Every choice below mirrors `08_evaluate_ablation.py` so the numbers are
comparable rather than merely adjacent:

  * the analysable cohort is built by the same four inclusion criteria, including
    the fourth (a resolvable small-molecule structure), so this runs on the same
    981 Arm A patients the ablation will;
  * samples are (patient, drug) pairs, as there;
  * StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed), grouped by
    patient and stratified by the same "{y}|{project}" key;
  * predictions are pooled out-of-fold and averaged per patient, so every
    patient is scored exactly once;
  * y = 1 - response_binary. THE NON-RESPONDER IS THE POSITIVE CLASS;
  * the whole procedure is repeated over 5 seeds and the spread is reported.

USAGE
-----
    python scripts/12_context_only_baseline.py --tag arm_a
    python scripts/12_context_only_baseline.py --tag arm_a --seeds 5 --smoothing 10
    python scripts/12_context_only_baseline.py --tag brca --dry-run

Pre-registered as a declared reference arm (§10). Report its AUROC beside every
absolute AUROC in the paper.
=============================================================================
"""
from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))

log = logging.getLogger("ctx")

# The context blocks §6 specifies, evaluated alone and in combination. Keys are
# the run-log "arm" names; values are the patient-record fields combined into
# one lookup key.
PREDICTORS = {
    "context_project":            ("project",),
    "context_stage":              ("stage",),
    "context_project_stage":      ("project", "stage"),
    "context_project_stage_receptor": ("project", "stage", "receptor"),
}

# Additive (empirical-Bayes) smoothing toward the TRAINING prior. A level seen
# n times gets (n_pos + m * prior) / (n + m). With m = 0 a level seen once and
# happening to be a non-responder predicts 1.0, which is pure memorisation of a
# single patient; m = 10 keeps rare levels near the prior while leaving the
# large cancer-type groups essentially untouched. Fixed here rather than tuned:
# this is a reference arm, and tuning it would make it a model.
DEFAULT_SMOOTHING = 10.0


def build_samples(tag: str, features_dir: Path, require_features: bool = True):
    """
    The analysable cohort, by the same four inclusion criteria as the ablation.

    Returns (samples, meta) where each sample is a dict with patient, project,
    stage, receptor, drug and y. One sample per (patient, drug) pair.
    """
    import cohort_labels as CL

    csv_path = CL.find_cohort_csv(BASE_DIR, tag=tag)
    cohort = CL.load_cohort(csv_path)
    log.info(f"  cohort: {csv_path.name} — {CL.summarise(cohort)}")

    # --- criterion 3: the patient has at least one embedded diagnostic slide ---
    by_patient: dict[str, int] = defaultdict(int)
    if require_features:
        manifest_path = features_dir / "feature_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No feature manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        for slide, entry in manifest.items():
            if entry.get("status") != "done":
                continue
            bc = CL.patient_barcode(slide)
            if bc is not None:
                by_patient[bc] += 1

    # --- criterion 4: the drug resolves to a small-molecule structure --------
    resolvable, resolution_mode = _resolvable_drugs(
        {d for r in cohort.values() for d in r["drugs"]})

    # --- the ER/PR/HER2 context block (§6), when the arm has one ------------
    receptors: dict[str, dict] = {}
    rec_csv = BASE_DIR / "cohort" / f"receptor_status_{tag}.csv"
    if rec_csv.exists():
        with rec_csv.open() as f:
            for r in _csv.DictReader(f):
                receptors[r["submitter_id"]] = r

    samples = []
    skipped_no_feat = skipped_no_struct = 0
    for bc, rec in cohort.items():
        if require_features and not by_patient.get(bc):
            skipped_no_feat += 1
            continue

        rr = receptors.get(bc, {})
        receptor_key = "|".join(
            (rr.get(f) or "unknown").strip().lower() if
            (rr.get(f) or "unknown").strip().lower() in ("positive", "negative", "unknown")
            else "unknown"
            for f in ("er_status", "pr_status", "her2_status"))

        y = 1 - int(rec["response_binary"])          # NON-RESPONDER is positive
        for drug in rec["drugs"]:
            if drug not in resolvable:
                skipped_no_struct += 1
                continue
            samples.append({
                "patient":  bc,
                "project":  rec.get("project") or "unknown",
                "stage":    rec.get("stage") or "unknown",
                "receptor": receptor_key,
                "drug":     drug,
                "y":        y,
            })

    patients = {s["patient"] for s in samples}
    n_pos = len({s["patient"] for s in samples if s["y"] == 1})
    meta = {
        "cohort_csv":            csv_path.name,
        "n_patients_in_cohort":  len(cohort),
        "n_patients_analysable": len(patients),
        "n_samples":             len(samples),
        "n_nonresponder":        n_pos,
        "n_responder":           len(patients) - n_pos,
        "n_drugs":               len({s["drug"] for s in samples}),
        "skipped_no_features":   skipped_no_feat,
        "skipped_no_structure":  skipped_no_struct,
        "receptor_context":      bool(receptors),
        "structure_resolution":  resolution_mode,
    }
    return samples, meta


def _resolvable_drugs(drug_names):
    """
    Which drug names have a usable small-molecule structure — criterion 4.

    Mirrors `08_evaluate_ablation.py::build_fingerprints`: a name resolves only
    if the verified map gives it a non-null SMILES AND RDKit can parse that
    SMILES. If RDKit is unavailable the non-null check alone is used and the
    mode is recorded in the run log, because the two can differ and a silent
    difference in n would be exactly the kind of thing this project logs.
    """
    blob = json.loads((BASE_DIR / "drugs" / "drug_smiles_map.json").read_text())
    dmap = blob["map"]

    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        parse = lambda smi: Chem.MolFromSmiles(smi) is not None
        mode = "rdkit"
    except ImportError:
        parse = lambda smi: True
        mode = "smiles_non_null_only"

    out = set()
    for name in sorted(drug_names):
        entry = dmap.get(name.lower().strip())
        if entry is None or entry.get("smiles") is None:
            continue
        if parse(entry["smiles"]):
            out.add(name)
    return out, mode


def rate_table(rows, keys, smoothing):
    """Non-responder rate per context level, learned on TRAINING rows only."""
    prior = float(np.mean([r["y"] for r in rows])) if rows else 0.5
    agg: dict[str, list] = defaultdict(lambda: [0, 0])     # level -> [n, n_pos]
    for r in rows:
        cell = agg["|".join(str(r[k]) for k in keys)]
        cell[0] += 1
        cell[1] += r["y"]
    table = {lv: (n_pos + smoothing * prior) / (n + smoothing)
             for lv, (n, n_pos) in agg.items()}
    return table, prior


def run_cv(samples, keys, n_splits, seed, smoothing):
    """
    Pooled out-of-fold scores, one per patient — the harness's own procedure
    with the model replaced by a lookup table.
    """
    groups = np.array([s["patient"] for s in samples])
    strat = np.array([f"{s['y']}|{s['project']}" for s in samples])
    key_ix = {k: i for i, k in enumerate(sorted(set(strat)))}
    y_strat = np.array([key_ix[k] for k in strat])

    oof: dict[str, list] = defaultdict(list)
    labels = {s["patient"]: s["y"] for s in samples}

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(samples, y_strat, groups):
        table, prior = rate_table([samples[i] for i in tr], keys, smoothing)
        for i in te:
            s = samples[i]
            lv = "|".join(str(s[k]) for k in keys)
            oof[s["patient"]].append(table.get(lv, prior))   # unseen level -> prior

    pats = sorted(oof)
    y = np.array([labels[p] for p in pats])
    score = np.array([float(np.mean(oof[p])) for p in pats])
    return y, score, pats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="cohort tag, e.g. arm_a or brca. REAL LABELS.")
    ap.add_argument("--seeds", type=int, default=5, help="§4 requires 5")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoothing", type=float, default=DEFAULT_SMOOTHING,
                    help="additive smoothing toward the training prior")
    ap.add_argument("--features-dir", default=str(BASE_DIR / "output" / "features"))
    ap.add_argument("--no-require-features", action="store_true",
                    help="do NOT require an embedded slide (diagnostic only; "
                         "changes n and is recorded in the run log)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the cohort and exit without reading a label")
    ap.add_argument("--note", default="", help="free text stored in the run log")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info(f"\nCONTEXT-ONLY REFERENCE ARM — tag={a.tag}")
    samples, meta = build_samples(a.tag, Path(a.features_dir),
                                  require_features=not a.no_require_features)

    log.info(f"  analysable: {meta['n_patients_analysable']} patients "
             f"({meta['n_responder']} responder / {meta['n_nonresponder']} non-responder), "
             f"{meta['n_samples']} (patient, drug) samples, {meta['n_drugs']} drugs")
    log.info(f"  dropped: {meta['skipped_no_features']} without an embedded slide, "
             f"{meta['skipped_no_structure']} (patient, drug) records without a structure")
    log.info(f"  receptor context available: {meta['receptor_context']}")

    if a.dry_run:
        log.info("\n  --dry-run: no label was read. Exiting.\n")
        return

    prevalence = meta["n_nonresponder"] / max(meta["n_patients_analysable"], 1)
    log.info(f"  non-responder prevalence (the no-skill AUPRC baseline): {prevalence:.3f}")
    log.info(f"  majority class: {max(prevalence, 1 - prevalence):.1%}\n")

    seeds = list(range(1, a.seeds + 1))
    results = {}
    for arm, keys in PREDICTORS.items():
        if "receptor" in keys and not meta["receptor_context"]:
            continue
        aurocs = []
        for seed in seeds:
            y, score, _ = run_cv(samples, keys, a.folds, seed, a.smoothing)
            aurocs.append(float(roc_auc_score(y, score)))
        results[arm] = {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_min":  float(np.min(aurocs)),
            "auroc_max":  float(np.max(aurocs)),
            "auroc_per_seed": aurocs,
            "predictor_fields": list(keys),
        }

    print(f"\n{'predictor':<34} {'cross-validated AUROC':>22}  {'spread over seeds':>20}")
    print("-" * 80)
    for arm, r in results.items():
        pretty = arm.replace("context_", "").replace("_", " + ")
        print(f"{pretty:<34} {r['auroc_mean']:>22.3f}  "
              f"{r['auroc_min']:.3f} – {r['auroc_max']:.3f}")
    print()
    print("  No model was trained. No image or drug was read. This is the floor")
    print("  every absolute AUROC in the paper must be reported against.\n")

    payload = {
        "source":        f"context_only(tag={a.tag})",
        "purpose":       "context_only_reference_arm_S10",
        "arms":          sorted(results),
        "seeds":         seeds,
        "folds":         a.folds,
        "hparams":       {"smoothing": a.smoothing, "model": "none"},
        "drug_encoder":  "none",
        "cohort":        meta,
        "results":       results,
        "note":          a.note,
    }
    payload["receptor_context"] = bool(meta["receptor_context"])
    payload["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    payload["config_hash"] = hashlib.sha256(
        json.dumps({k: payload[k] for k in ("source", "arms", "seeds", "folds",
                                            "hparams", "code_sha256", "drug_encoder")},
                   sort_keys=True, default=str).encode()).hexdigest()[:12]

    p = BASE_DIR / "output" / "eval" / "run_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    log.info(f"Run logged to {p} (config {payload['config_hash']})")


if __name__ == "__main__":
    main()
