#!/usr/bin/env python3
"""
=============================================================================
15_secondary_analyses.py — the pre-registered secondaries that need no retraining
=============================================================================

Two subcommands:

  tiles   (no model, no predictions) — the TILE-COUNT analysis, §10 2026-09-19.
          Threshold-free primary: Mann–Whitney U on each patient's number of
          qualifying tissue tiles (n_tiles_found summed over the patient's
          diagnostic slides) by response class, plus a logistic regression of
          non-response on log(n_tiles). Reads labels; computes no ΔAUROC; can
          be run at any time.

              python scripts/15_secondary_analyses.py tiles --tag arm_a

  preds   (reads output/eval/preds_<config>.json written by 08_) — computed
          from the PRIMARY run's per-patient out-of-fold predictions:

            * within-cancer-type ΔAUROC (with_smiles − without_smiles), per
              project and pooled by inverse variance (§10 2026-09-19)
            * single- vs multi-molecule ΔAUROC with bootstrap CIs
              (§10 2026-09-29)
            * the tile-count exclusion staircase: ΔAUROC on patients whose
              thinnest slide has ≥ 0 / 500 / 1,000 / 2,000 tiles
              (§10 2026-09-19; evaluation subsets of the primary predictions —
              the retrained 1,000-tile sensitivity of §3.5 is separate, in 08_)

              python scripts/15_secondary_analyses.py preds output/eval/preds_<cfg>.json

CONVENTIONS, fixed here before any Arm A ΔAUROC exists (§10 2026-09-29):
  * Positive class = NON-RESPONDER (y = 1), as everywhere else.
  * Scores are averaged over the run's seeds per patient ("seed-ensembled")
    before any subgroup AUROC, DeLong variance or bootstrap is computed; the
    mean of per-seed deltas is reported alongside for comparison with §7.
  * Within-cancer-type pooling uses DeLong variances of the per-project delta
    and includes only projects with ≥ 5 patients in EACH class (a 3-patient
    class gives a degenerate variance and an unbounded weight). Every project
    is still reported individually.
  * Bootstrap: 2,000 resamples, stratified by class within each subgroup,
    resampling patients, seed 0.
=============================================================================
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, norm
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))
log = logging.getLogger("secondary")

N_BOOT = 2000
MIN_PER_CLASS_FOR_POOLING = 5
STAIRCASE = (0, 500, 1000, 2000)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# patient tables
# ---------------------------------------------------------------------------
def patient_tiles(tag):
    """Per analysable patient: y, project, total qualifying tiles, thinnest slide."""
    ctx12 = _load("ctx12", "12_context_only_baseline.py")
    import cohort_labels as CL
    samples, meta = ctx12.build_samples(tag, BASE_DIR / "output" / "features")
    man = json.loads((BASE_DIR / "output" / "features" / "feature_manifest.json").read_text())
    found, capped = defaultdict(int), defaultdict(list)
    for slide, e in man.items():
        if e.get("status") != "done":
            continue
        bc = CL.patient_barcode(slide)
        if bc is None:
            continue
        found[bc] += int(e.get("n_tiles_found", e.get("n_tiles", 0)))
        capped[bc].append(int(e.get("n_tiles", 0)))
    out = {}
    for s in samples:
        out[s["patient"]] = {"y": s["y"], "project": s["project"],
                             "tiles_found": found[s["patient"]],
                             "min_slide_tiles": min(capped[s["patient"]])}
    return out, samples, meta


def logistic_1d(x, y, iters=50):
    """Unpenalised logistic y ~ a + b x by Newton-Raphson; Wald SE for b."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    se = math.sqrt(cov[1, 1])
    z = beta[1] / se
    return {"coef_log_tiles": float(beta[1]), "se": se,
            "odds_ratio_per_e_fold": float(math.exp(beta[1])),
            "or_ci95": [float(math.exp(beta[1] - 1.96 * se)),
                        float(math.exp(beta[1] + 1.96 * se))],
            "wald_p": float(2 * norm.sf(abs(z)))}


def cmd_tiles(a):
    pts, _, meta = patient_tiles(a.tag)
    y = np.array([v["y"] for v in pts.values()])
    t = np.array([v["tiles_found"] for v in pts.values()], float)
    if (t <= 0).any():
        raise SystemExit(f"{int((t <= 0).sum())} patients with zero tiles — check the manifest")
    u = mannwhitneyu(t[y == 1], t[y == 0], alternative="two-sided")
    res = {
        "n_patients": int(y.size), "n_nonresponder": int(y.sum()),
        "median_tiles_nonresponder": float(np.median(t[y == 1])),
        "median_tiles_responder": float(np.median(t[y == 0])),
        "mann_whitney_U": float(u.statistic), "mann_whitney_p": float(u.pvalue),
        "auroc_tiles_for_nonresponse": float(u.statistic / ((y == 1).sum() * (y == 0).sum())),
        "logistic_log_tiles": logistic_1d(np.log(t), y.astype(float)),
        "tile_measure": "n_tiles_found summed over the patient's diagnostic slides",
    }
    print(f"\nTILE-COUNT ANALYSIS — tag={a.tag}, {res['n_patients']} patients")
    print(f"  median qualifying tiles: non-responder {res['median_tiles_nonresponder']:.0f}"
          f" · responder {res['median_tiles_responder']:.0f}")
    print(f"  Mann–Whitney U = {res['mann_whitney_U']:.0f}, p = {res['mann_whitney_p']:.4g}"
          f"  (AUROC of tile count alone = {res['auroc_tiles_for_nonresponse']:.3f})")
    lg = res["logistic_log_tiles"]
    print(f"  logistic on log(tiles): OR per e-fold {lg['odds_ratio_per_e_fold']:.3f} "
          f"[{lg['or_ci95'][0]:.3f}, {lg['or_ci95'][1]:.3f}], p = {lg['wald_p']:.4g}\n")
    out = BASE_DIR / "output" / "eval" / f"tile_count_analysis_{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    log.info(f"written {out}")


# ---------------------------------------------------------------------------
# prediction-based secondaries
# ---------------------------------------------------------------------------
def delong_delta(y, sa, sb):
    """(delta, variance) of AUC(sa) - AUC(sb), paired, via 08_'s DeLong components."""
    ev = _EV
    yb = np.asarray(y).astype(bool)
    m, n = int(yb.sum()), int((~yb).sum())
    v01 = np.empty((2, m)); v10 = np.empty((2, n)); auc = np.empty(2)
    for k, s in enumerate((np.asarray(sa, float), np.asarray(sb, float))):
        v01[k], v10[k], auc[k] = ev._structural_components(s[yb], s[~yb])
    S = np.cov(v01, ddof=1) / m + np.cov(v10, ddof=1) / n
    d = np.array([1.0, -1.0])
    return float(auc[0] - auc[1]), float(d @ S @ d)


def boot_delta(y, sa, sb, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        out[i] = roc_auc_score(y[ix], sa[ix]) - roc_auc_score(y[ix], sb[ix])
    return out


def ensembled(block, arm):
    """patient -> (y, mean score over seeds); and per-seed dicts."""
    per_seed = {}
    for seed, arms in block.items():
        a = arms[arm]
        per_seed[seed] = dict(zip(a["patients"], zip(a["y"], a["score"])))
    pats = sorted(next(iter(per_seed.values())))
    for d in per_seed.values():
        assert sorted(d) == pats, "seeds scored different patient sets"
    y = {p: next(iter(per_seed.values()))[p][0] for p in pats}
    s = {p: float(np.mean([d[p][1] for d in per_seed.values()])) for p in pats}
    return pats, y, s, per_seed


def subset_delta(pats_sel, y, sa, sb, per_a, per_b):
    yy = np.array([y[p] for p in pats_sel])
    if yy.min() == yy.max():
        return None
    A = np.array([sa[p] for p in pats_sel]); B = np.array([sb[p] for p in pats_sel])
    d, var = delong_delta(yy, A, B)
    seed_ds = [roc_auc_score(yy, [per_a[k][p][1] for p in pats_sel]) -
               roc_auc_score(yy, [per_b[k][p][1] for p in pats_sel]) for k in per_a]
    return {"n": len(pats_sel), "n_nonresponder": int(yy.sum()),
            "auroc_with": float(roc_auc_score(yy, A)), "auroc_without": float(roc_auc_score(yy, B)),
            "delta_ensembled": d, "delong_var": var,
            "delta_mean_of_seeds": float(np.mean(seed_ds)), "yy": yy, "A": A, "B": B}


def cmd_preds(a):
    global _EV
    _EV = _load("ev08", "08_evaluate_ablation.py")
    blob = json.loads(Path(a.preds).read_text())
    prim = blob["primary"]
    arms = next(iter(prim.values())).keys()
    if not {"with_smiles", "without_smiles"} <= set(arms):
        raise SystemExit(f"need with_smiles and without_smiles; file has {sorted(arms)}")
    pats, y, sa, per_a = ensembled(prim, "with_smiles")
    _, _, sb, per_b = ensembled(prim, "without_smiles")
    tag = "arm_a" if "arm_a" in blob["source"] else blob["source"].split("tag=")[-1].rstrip(")")
    info, samples, _ = patient_tiles(tag)
    assert set(info) == set(pats), "prediction patients differ from the analysable cohort"
    rng = np.random.default_rng(0)
    res = {"preds_file": Path(a.preds).name, "config_hash": blob.get("config_hash")}
    strip = lambda r: {k: v for k, v in r.items() if k not in ("yy", "A", "B")}

    # --- within cancer type ------------------------------------------------
    by_proj = defaultdict(list)
    for p in pats:
        by_proj[info[p]["project"]].append(p)
    rows, num, den = {}, 0.0, 0.0
    for proj in sorted(by_proj):
        r = subset_delta(by_proj[proj], y, sa, sb, per_a, per_b)
        if r is None:
            rows[proj] = {"n": len(by_proj[proj]), "note": "one class only"}
            continue
        n1 = r["n_nonresponder"]; n0 = r["n"] - n1
        pooled = min(n1, n0) >= MIN_PER_CLASS_FOR_POOLING and r["delong_var"] > 0
        r = strip(r); r["in_pool"] = pooled
        rows[proj] = r
        if pooled:
            num += r["delta_ensembled"] / r["delong_var"]; den += 1 / r["delong_var"]
    pooled_d = num / den if den else float("nan")
    se = math.sqrt(1 / den) if den else float("nan")
    res["within_cancer_type"] = {
        "per_project": rows, "pooled_delta": pooled_d, "pooled_se": se,
        "pooled_ci95": [pooled_d - 1.96 * se, pooled_d + 1.96 * se],
        "pooled_p": float(2 * norm.sf(abs(pooled_d / se))) if den else None,
        "rule": f"pool projects with >= {MIN_PER_CLASS_FOR_POOLING} patients per class"}

    # --- single vs multi molecule -------------------------------------------
    d13 = _load("d13", "13_drug_identity_check.py")
    mols = defaultdict(set)
    for s in d13.attach_molecules([dict(x) for x in samples]):
        mols[s["patient"]].add(s["molecule"])
    groups = {"single": [p for p in pats if len(mols[p]) == 1],
              "multi": [p for p in pats if len(mols[p]) >= 2]}
    sm, boots = {}, {}
    for g, ps in groups.items():
        r = subset_delta(ps, y, sa, sb, per_a, per_b)
        boots[g] = boot_delta(r["yy"], r["A"], r["B"], rng)
        r = strip(r); r["delta_ci95"] = [float(v) for v in np.percentile(boots[g], [2.5, 97.5])]
        sm[g] = r
    diff = boots["single"] - boots["multi"]
    sm["single_minus_multi"] = {
        "delta": sm["single"]["delta_ensembled"] - sm["multi"]["delta_ensembled"],
        "ci95": [float(v) for v in np.percentile(diff, [2.5, 97.5])],
        "prediction_committed": "larger in single (§10 2026-09-29)"}
    res["single_vs_multi"] = sm

    # --- staircase ------------------------------------------------------------
    st = {}
    for thr in STAIRCASE:
        ps = [p for p in pats if info[p]["min_slide_tiles"] >= thr]
        r = subset_delta(ps, y, sa, sb, per_a, per_b)
        st[f">={thr}"] = strip(r) if r else {"n": len(ps), "note": "one class only"}
    res["tile_staircase"] = st

    # --- print ------------------------------------------------------------------
    print(f"\nSECONDARY ANALYSES from {res['preds_file']} (with_smiles − without_smiles)")
    print("\nWithin cancer type:")
    for proj, r in rows.items():
        if "delta_ensembled" in r:
            print(f"  {proj:<11} n={r['n']:>4} ({r['n_nonresponder']:>3} NR)  "
                  f"Δ {r['delta_ensembled']:+.4f}  {'pooled' if r['in_pool'] else 'reported only'}")
        else:
            print(f"  {proj:<11} n={r['n']:>4}  {r['note']}")
    w = res["within_cancer_type"]
    print(f"  POOLED Δ {w['pooled_delta']:+.4f}  95% CI [{w['pooled_ci95'][0]:+.4f}, "
          f"{w['pooled_ci95'][1]:+.4f}]  p = {w['pooled_p']}")
    print("\nSingle vs multi molecule:")
    for g in ("single", "multi"):
        r = sm[g]
        print(f"  {g:<6} n={r['n']:>4}  Δ {r['delta_ensembled']:+.4f}  "
              f"[{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}]")
    d = sm["single_minus_multi"]
    print(f"  single − multi {d['delta']:+.4f}  [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    print("\nTile staircase (evaluation subsets of the primary predictions):")
    for k, r in st.items():
        if "delta_ensembled" in r:
            print(f"  thinnest slide {k:>6}: n={r['n']:>4}  Δ {r['delta_ensembled']:+.4f}")
    out = BASE_DIR / "output" / "eval" / f"secondary_{blob.get('config_hash', 'x')}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwritten {out}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiles"); t.add_argument("--tag", required=True)
    p = sub.add_parser("preds"); p.add_argument("preds")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    {"tiles": cmd_tiles, "preds": cmd_preds}[a.cmd](a)


if __name__ == "__main__":
    main()
