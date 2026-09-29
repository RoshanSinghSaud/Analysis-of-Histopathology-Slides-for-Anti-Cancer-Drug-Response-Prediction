#!/usr/bin/env python3
"""
=============================================================================
13_drug_identity_check.py — FREE CHECK: does drug IDENTITY add anything
                             above cancer type, before any model is trained?
=============================================================================

Worklist step 3 (28 Sep 2026). Gates the regimen-representation decision
(worklist step 9) and the permutation-null spend (step 23). No image, no
encoder, no GPU — seconds on a laptop.

The question: if a lookup table / additive model that KNOWS which molecules a
patient received cannot beat cancer type alone, no architecture choice will
rescue the drug branch, and a null is the honest expectation.

WHAT IS COMPARED (all on the same 979 analysable Arm A patients, the same
StratifiedGroupKFold splits as 08_ and 12_, 5 seeds, pooled out-of-fold AUROC,
non-responder = positive class):

  A. Lookup tables (12_'s rate_table, smoothing 10), cells counted ONCE PER
     PATIENT so multi-drug patients are not over-weighted:
       project                    -> the reference floor
       project + molecule         -> identity at the level the model sees it
                                     (salts merged by InChIKey), row-level,
                                     averaged per patient
       project + regimen          -> the whole regimen (sorted molecule set)
  B. Additive L2 logistic regression, one row per patient, FIXED C = 1.0
     (never tuned — this is a check, not a model):
       project one-hot
       project one-hot + multi-hot molecules
  C. SHUFFLE CONTROL for every drug-bearing predictor: regimens permuted among
     patients WITHIN the same project, 20 permutations x 5 seeds. This keeps
     each project's drug mix and every cell's size, and destroys only the
     drug-label link beyond project. It is the right comparator, because adding
     ANY high-cardinality key to a lookup table costs AUROC through sparsity
     (the receptor row in 12_ scored 0.794 < 0.800 for exactly that reason), so
     "project + drug < project" alone would NOT mean "drug carries nothing".
     The signal is (real - shuffled), not (real - project).

It reads REAL LABELS with drug information, so it is logged to
output/eval/run_log.jsonl like every other configuration, and must be recorded
in the pre-registration's deviations log as run BEFORE the ablation.

USAGE
    python scripts/13_drug_identity_check.py --tag arm_a
    python scripts/13_drug_identity_check.py --tag arm_a --perms 20 --seeds 5
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))
log = logging.getLogger("drugcheck")

_spec = importlib.util.spec_from_file_location(
    "ctx12", BASE_DIR / "scripts" / "12_context_only_baseline.py")
ctx12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctx12)

SMOOTHING = 10.0
LOGREG_C = 1.0


def attach_molecules(samples):
    """molecule = parent InChIKey from the verified map (what 08_ embeds)."""
    dmap = json.loads((BASE_DIR / "drugs" / "drug_smiles_map.json").read_text())["map"]
    for s in samples:
        ik = dmap[s["drug"].lower().strip()]["inchikey"]
        assert ik, s["drug"]
        s["molecule"] = ik
    reg = defaultdict(set)
    for s in samples:
        reg[s["patient"]].add(s["molecule"])
    for s in samples:
        s["regimen"] = "+".join(sorted(reg[s["patient"]]))
    return samples


def splits(samples, n_splits, seed):
    """Identical to 12_.run_cv / 08_: rows grouped by patient, strat y|project."""
    groups = np.array([s["patient"] for s in samples])
    strat = np.array([f"{s['y']}|{s['project']}" for s in samples])
    ix = {k: i for i, k in enumerate(sorted(set(strat)))}
    y_strat = np.array([ix[k] for k in strat])
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Returned as PATIENT sets, computed once on the REAL samples, so the
    # shuffle control is evaluated on exactly the same patient folds (a
    # regimen shuffle changes rows-per-patient, which would otherwise move
    # the row-level splitter and contaminate the comparison).
    return [({samples[i]["patient"] for i in tr}, {samples[i]["patient"] for i in te})
            for tr, te in skf.split(samples, y_strat, groups)]


def _rows(samples, pats):
    return [i for i, s in enumerate(samples) if s["patient"] in pats]


def lookup_auroc(samples, folds, keys):
    labels = {s["patient"]: s["y"] for s in samples}
    oof = defaultdict(list)
    for trp, tep in folds:
        tr, te = _rows(samples, trp), _rows(samples, tep)
        # one count per (patient, cell): multi-drug patients not over-weighted
        seen, rows = set(), []
        for i in tr:
            s = samples[i]
            lv = "|".join(str(s[k]) for k in keys)
            if (s["patient"], lv) not in seen:
                seen.add((s["patient"], lv))
                rows.append(s)
        table, prior = ctx12.rate_table(rows, keys, SMOOTHING)
        for i in te:
            s = samples[i]
            oof[s["patient"]].append("|".join(str(s[k]) for k in keys))
        for pt in {samples[i]["patient"] for i in te}:
            # mean over the patient's DISTINCT cells (two salt names of one
            # molecule are one cell), exactly as the patient is the eval unit
            oof[pt] = [float(np.mean([table.get(lv, prior) for lv in sorted(set(oof[pt]))]))]
    pats = sorted(oof)
    return roc_auc_score([labels[p] for p in pats], [oof[p][0] for p in pats])


def logreg_auroc(samples, folds, use_drugs):
    pts = sorted({s["patient"] for s in samples})
    pix = {p: i for i, p in enumerate(pts)}
    proj = sorted({s["project"] for s in samples})
    mols = sorted({s["molecule"] for s in samples})
    X = np.zeros((len(pts), len(proj) + (len(mols) if use_drugs else 0)))
    y = np.zeros(len(pts))
    for s in samples:
        i = pix[s["patient"]]
        X[i, proj.index(s["project"])] = 1
        y[i] = s["y"]
        if use_drugs:
            for m in s["regimen"].split("+"):
                X[i, len(proj) + mols.index(m)] = 1
    score = np.full(len(pts), np.nan)
    for trs, tes in folds:
        trp = sorted(pix[p] for p in trs)
        tep = sorted(pix[p] for p in tes)
        clf = LogisticRegression(C=LOGREG_C, max_iter=5000).fit(X[trp], y[trp])
        score[tep] = clf.predict_proba(X[tep])[:, 1]
    assert not np.isnan(score).any()
    return roc_auc_score(y, score)


def shuffle_within_project(samples, rng):
    """Permute whole regimens among patients of the same project."""
    by_proj = defaultdict(list)
    first = {}
    for s in samples:
        if s["patient"] not in first:
            first[s["patient"]] = s
            by_proj[s["project"]].append(s["patient"])
    pat_rows = defaultdict(list)
    for s in samples:
        pat_rows[s["patient"]].append(s)
    out = []
    for proj, pats in by_proj.items():
        donors = list(pats)
        rng.shuffle(donors)
        for p, d in zip(pats, donors):
            base = pat_rows[p][0]
            for dr in pat_rows[d]:          # recipient takes donor's regimen
                out.append({**base, "drug": dr["drug"], "molecule": dr["molecule"],
                            "regimen": dr["regimen"]})
    return out


PREDICTORS = {
    "lookup_project":          ("lookup", ("project",)),
    "lookup_project_molecule": ("lookup", ("project", "molecule")),
    "lookup_project_regimen":  ("lookup", ("project", "regimen")),
    "logreg_project":          ("logreg", False),
    "logreg_project_molecules": ("logreg", True),
}


def evaluate(samples, fold_sets):
    res = {k: [] for k in PREDICTORS}
    for folds in fold_sets:
        for name, (kind, arg) in PREDICTORS.items():
            res[name].append(lookup_auroc(samples, folds, arg) if kind == "lookup"
                             else logreg_auroc(samples, folds, arg))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    samples, meta = ctx12.build_samples(a.tag, BASE_DIR / "output" / "features")
    samples = attach_molecules(samples)
    pts = {s["patient"] for s in samples}
    n_mol = defaultdict(set)
    for s in samples:
        n_mol[s["patient"]].add(s["molecule"])
    single = sum(1 for p in pts if len(n_mol[p]) == 1)
    rows_multi = sum(1 for s in samples if len(n_mol[s["patient"]]) > 1)
    regimens = {s["regimen"] for s in samples}
    meta.update({"n_molecules": len({s["molecule"] for s in samples}),
                 "n_single_molecule_patients": single,
                 "n_multi_molecule_patients": len(pts) - single,
                 "n_rows_from_multi_molecule_patients": rows_multi,
                 "n_distinct_regimens": len(regimens)})
    log.info(f"\nDRUG-IDENTITY FREE CHECK — tag={a.tag}")
    log.info(f"  {len(pts)} patients, {len(samples)} rows, {meta['n_molecules']} molecules, "
             f"{len(regimens)} distinct regimens")
    log.info(f"  single-molecule patients {single}, multi {len(pts)-single}; "
             f"rows from multi-molecule patients {rows_multi}/{len(samples)}")

    seeds = list(range(1, a.seeds + 1))
    fold_sets = [splits(samples, a.folds, seed) for seed in seeds]
    real = evaluate(samples, fold_sets)

    rng = np.random.default_rng(20260929)
    shuf = {k: [] for k in PREDICTORS}
    for _ in range(a.perms):
        r = evaluate(shuffle_within_project(samples, rng), fold_sets)
        for k in PREDICTORS:
            shuf[k].append(float(np.mean(r[k])))

    print(f"\n{'predictor':<28}{'real AUROC':>11}{'seed range':>17}"
          f"{'shuffled mean':>15}{'shuf 95%':>17}{'real-shuf':>11}{'p(perm)':>9}")
    print("-" * 108)
    results = {}
    for k in PREDICTORS:
        rm = float(np.mean(real[k]))
        sm = np.array(shuf[k])
        p = (1 + int((sm >= rm).sum())) / (1 + len(sm))
        lo, hi = np.percentile(sm, [2.5, 97.5])
        results[k] = {"auroc_mean": rm, "auroc_per_seed": real[k],
                      "shuffled_mean": float(sm.mean()), "shuffled_sd": float(sm.std()),
                      "shuffled_2p5": float(lo), "shuffled_97p5": float(hi),
                      "p_perm": p}
        print(f"{k:<28}{rm:>11.3f}{min(real[k]):>9.3f}–{max(real[k]):.3f}"
              f"{sm.mean():>15.3f}{lo:>9.3f}–{hi:.3f}{rm - sm.mean():>+11.3f}{p:>9.3f}")
    print("\n  project-only rows are unaffected by the shuffle (they read no drug);"
          "\n  their 'shuffled' column is a sanity check and should equal 'real'.\n")

    payload = {"source": f"drug_identity_check(tag={a.tag})",
               "purpose": "free_check_drug_identity_worklist_step3",
               "arms": sorted(results), "seeds": seeds, "folds": a.folds,
               "hparams": {"smoothing": SMOOTHING, "logreg_C": LOGREG_C,
                           "perms": a.perms, "perm_rng_seed": 20260929,
                           "model": "lookup + fixed-C logistic; no image"},
               "drug_encoder": "none (identity only)", "cohort": meta,
               "results": results, "note": a.note,
               "receptor_context": bool(meta["receptor_context"])}
    payload["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    payload["config_hash"] = hashlib.sha256(json.dumps(
        {k: payload[k] for k in ("source", "arms", "seeds", "folds", "hparams", "code_sha256")},
        sort_keys=True, default=str).encode()).hexdigest()[:12]
    p = BASE_DIR / "output" / "eval" / "run_log.jsonl"
    with p.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    log.info(f"Run logged to {p} (config {payload['config_hash']})")


if __name__ == "__main__":
    main()
