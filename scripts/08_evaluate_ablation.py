#!/usr/bin/env python3
"""
=============================================================================
08_evaluate_ablation.py — THE EXPERIMENT
=============================================================================

Everything in scripts 01_-07_ turns slides into vectors. THIS script is the
study: it runs the with-SMILES vs without-SMILES comparison that the paper's
single claim rests on, exactly as pre-registered in
`docs/analysis_plan_preregistration.md` (v2.2).

WHAT IT DOES
------------
  * 5-fold cross-validation, **grouped by patient** (§4). Every record for a
    patient lands in exactly one fold. Leakage here invalidates the endpoint.
  * Stratified by response class AND cancer type (§4).
  * An inner validation split inside each training fold, for early stopping.
  * Four arms on IDENTICAL folds:
        without_smiles   histology + tissue context only
        with_smiles      + frozen molecular embedding, mean-pooled over the regimen
        shuffled_drug    + the regimen of a RANDOM patient  (capacity control)
        onehot_drug      + one-hot MOLECULE IDs, pooled     (identity control)
    One training row per PATIENT from 2026-09-29 (see TRAINING_UNIT).
  * Pooled out-of-fold predictions: every patient scored exactly once, by the
    fold that did not train on them (§4, §5 — this is the power decision).
  * AUROC + AUPRC with the NON-RESPONDER as the positive class (§7), bootstrap
    95% CIs resampling PATIENTS, Brier score.
  * DeLong's test for two correlated ROC curves — the primary endpoint (§1).
  * The whole procedure repeated over 5 seeds; mean Δ and spread reported (§4).
  * The pre-registered thin-slide sensitivity analysis (§3.5).
  * Every configuration trained against labels is appended to a run log (§8).

THE RULE THIS SCRIPT EXISTS TO PROTECT
--------------------------------------
Build and debug on `--synthetic`. Never iterate on real labels. Watching a real
AUROC move while fixing bugs is how a pre-registration quietly stops being one.
`--selftest` proves the harness works without touching a single real slide:
it plants a known effect and checks the harness recovers it, then plants NO
effect and checks the false-positive rate lands near alpha.

USAGE
-----
    python scripts/08_evaluate_ablation.py --selftest          # validate harness
    python scripts/08_evaluate_ablation.py --synthetic --delta-signal 0.9
    python scripts/08_evaluate_ablation.py --tag brca --seeds 5 # REAL LABELS
    python scripts/08_evaluate_ablation.py --tag brca --thin-slide-sensitivity

Positive class is the NON-RESPONDER throughout. The model predicts
P(non-responder); AUPRC's no-skill baseline is the non-responder prevalence and
is printed with every result.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))

log = logging.getLogger("eval")

# Set by load_real(): did the §6 ER/PR/HER2 context actually reach the model?
# Recorded in every run-log entry so a run without it is self-identifying.
RECEPTOR_CONTEXT_USED = False

# =============================================================================
# Hyperparameters
# -----------------------------------------------------------------------------
# §6: "Hyperparameters are selected ONCE, on the without-SMILES model, and then
# frozen and reused unchanged for the with-SMILES model." These are that frozen
# set. Changing them is a deviation and must be logged in §10. They are written
# here as a plain dict so the exact values land in every run-log entry.
# =============================================================================
HPARAMS = {
    "hidden_dim":     256,   # ABMIL hidden / patient-embedding dim (matches 03_)
    "attn_dim":       128,   # gated-attention internal dim (matches 03_)
    "drug_feat_dim":  32,    # FROZEN drug embedding width (see reduce_fingerprints)
    "drug_proj_dim":  16,    # drug branch projection. Deliberately SMALL: a
                             # 1024-bit fingerprint into a wide layer adds ~131k
                             # trainable parameters at n~1k and overfits, which
                             # would show up as a NEGATIVE delta and be misread
                             # as "chemistry does not help". §6 gives the drug
                             # branch no tuning budget, so this is fixed a priori.
    # FROZEN 2026-08-28 by the single tuning pass §6 permits
    # (scripts/09_tune_baseline.py, 12 configs x 3 seeds x 5 folds, 83 min,
    #  selected on mean pos-weighted INNER-validation loss, baseline arm only).
    # The pass was UNINFORMATIVE and that is recorded, not hidden: all twelve
    # configurations landed at or above 0.6931 -- the loss of predicting 0.5 for
    # every patient -- so none of them beat having no information at all. Best
    # 0.7043, worst 0.7218; the whole grid spans 0.0175 against a fold-to-fold
    # sd of 0.357. Do NOT re-tune these after seeing a delta. See §10.
    "dropout":        0.25,   # tuned, grid {0.25, 0.45}
    "lr":             3e-4,
    "weight_decay":   1.0,    # tuned, grid {1e-2, 1e-1, 1.0} -- EDGE of grid
    "max_epochs":     40,     # tuned, grid {15, 40} -- EDGE of grid
    "patience":       8,     # early-stopping patience on inner-val LOSS (§10)
    "batch_size":     16,    # bags per optimiser step (gradient accumulation)
    "train_tiles":    512,   # tiles sampled per bag per epoch (augmentation)
    "eval_tiles":     2048,  # tiles used at inference, seeded and fixed
    "inner_val_frac": 0.15,
}

FINGERPRINT_BITS = 1024      # Morgan width for the legacy morgan_pca encoder
                             # (was: must match 04_, now scripts/archive/)

# §6's frozen PRETRAINED molecular encoder. Built once, offline and label-free,
# by scripts/11_build_drug_embeddings.py; see that file for the tokenizer guard
# that decided which model. `morgan_pca` is the earlier stand-in, kept so every
# result produced before 2026-08-29 stays reproducible from this one file.
DRUG_ENCODERS = {
    "chemberta": "drug_embeddings_chemberta_zinc.npz",
    "morgan_pca": None,
}
DEFAULT_DRUG_ENCODER = "chemberta"
THIN_SLIDE_TILES = 1000      # §3.5 pre-registered threshold
N_BOOTSTRAP      = 2000      # §7
ALPHA            = 0.05      # §1


# =============================================================================
# DeLong's test for two correlated ROC curves  (DeLong 1988; Sun & Xu 2014)
# -----------------------------------------------------------------------------
# This implementation was validated against the pre-registered power simulation
# in §5: it reproduced the Arm B power row (0.12 / 0.25 / 0.45 vs the plan's
# 0.12 / 0.25 / 0.46) to within simulation error.
# =============================================================================
def _structural_components(x: np.ndarray, y: np.ndarray):
    """Midrank-based V10 / V01 components for one score vector."""
    m, n = x.size, y.size
    ox, oy = np.argsort(x), np.argsort(y)
    xs, ys = x[ox], y[oy]
    vx = (np.searchsorted(ys, xs, "left") + np.searchsorted(ys, xs, "right")) / 2.0 / n
    v01 = np.empty(m); v01[ox] = vx
    wy = 1.0 - (np.searchsorted(xs, ys, "left") + np.searchsorted(xs, ys, "right")) / 2.0 / m
    v10 = np.empty(n); v10[oy] = wy
    return v01, v10, float(v01.mean())


def delong_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray):
    """
    Paired comparison of two AUCs on the same samples.

    Returns (auc_a, auc_b, delta, z, p_two_sided). `delta` is auc_a - auc_b,
    so pass the WITH-SMILES scores as `score_a` to make a positive delta mean
    "the drug branch helped", matching §1's stated direction.
    """
    y = np.asarray(y_true).astype(bool)
    pos, neg = y, ~y
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present.")

    v01 = np.empty((2, m)); v10 = np.empty((2, n)); aucs = np.empty(2)
    for k, s in enumerate((np.asarray(score_a, float), np.asarray(score_b, float))):
        v01[k], v10[k], aucs[k] = _structural_components(s[pos], s[neg])

    S = np.cov(v01, ddof=1) / m + np.cov(v10, ddof=1) / n
    d = np.array([1.0, -1.0])
    var = float(d @ S @ d)
    delta = float(aucs[0] - aucs[1])
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), delta, 0.0, 1.0
    z = delta / math.sqrt(var)
    return float(aucs[0]), float(aucs[1]), delta, float(z), float(2 * norm.sf(abs(z)))


# =============================================================================
# Metrics
# =============================================================================
def bootstrap_ci(y: np.ndarray, s: np.ndarray, fn, n_boot=N_BOOTSTRAP, seed=0):
    """Stratified bootstrap CI, resampling PATIENTS (each row here IS a patient)."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, pos_idx.size, replace=True),
                              rng.choice(neg_idx, neg_idx.size, replace=True)])
        try:
            vals.append(fn(y[idx], s[idx]))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))


def score_report(y: np.ndarray, s: np.ndarray, seed=0) -> dict:
    """All §7 statistics for one set of pooled out-of-fold patient predictions."""
    prevalence = float(y.mean())          # non-responder prevalence = AUPRC baseline
    return {
        "n":                 int(y.size),
        "n_positive":        int(y.sum()),
        "prevalence":        prevalence,
        "auroc":             float(roc_auc_score(y, s)),
        "auroc_ci95":        bootstrap_ci(y, s, roc_auc_score, seed=seed),
        "auprc":             float(average_precision_score(y, s)),
        "auprc_ci95":        bootstrap_ci(y, s, average_precision_score, seed=seed),
        "auprc_baseline":    prevalence,
        "brier":             float(brier_score_loss(y, s)),
    }


# =============================================================================
# Sample container
# =============================================================================
@dataclass
class Sample:
    """One (patient, drug) record — the training unit. The EVAL unit is the patient."""
    patient:   str
    project:   str
    drug:      str
    y:         int                       # 1 = NON-RESPONDER (positive class)
    context:   np.ndarray                # tissue context: stage, cancer type, receptors
    bag_paths: list = field(default_factory=list)   # real mode: .pt files to pool
    bag:       object = None                        # synthetic mode: in-memory tensor
    min_tiles: int = 10**9               # smallest slide in this bag (thin-slide filter)
    oracle:    tuple = None              # synthetic only: (true logit, true logit without drug)
    regimen:   tuple = ()                # drug names whose vectors are pooled; () -> (drug,)
    negative_start: bool = False         # a record for this drug has days_to_treatment_start < 0


# =============================================================================
# Unit of training — ONE ROW PER PATIENT (§10, 2026-09-29)
# =============================================================================
# The label is the outcome of a REGIMEN, not of one drug: every drug row of a
# multi-drug patient carries the same slide and the same label, and no Arm A
# patient has conflicting labels across drugs (0 of 1,031). One row per
# (patient, drug) therefore gave 626 multi-drug patients 83% of the training
# rows while adding no within-patient contrast at all. From 2026-09-29 the
# training unit is the patient: one sample per patient, the drug input pooled
# (mean) over the DISTINCT MOLECULES of the regimen. The evaluation unit was
# always the patient, so AUROC, folds and power are unchanged.
#
# 'pair' reproduces the pre-2026-09-29 behaviour exactly (a one-element regimen
# pools to itself) and exists only as a regression baseline for --selftest.
TRAINING_UNIT = "patient"

# -----------------------------------------------------------------------------
# DRUG-GATE VARIANTS — EXPLORATORY ONLY (§10, 2026-09-30)
# -----------------------------------------------------------------------------
# 'frozen_prereg' is the pre-registered model and the default: gate initialised
# at 0, trained with the same lr (3e-4) and weight decay as every other weight.
# Adam moves a parameter by roughly lr per step, and early stopping selected
# epochs 4-6 on Arm A (~60 steps per epoch under the patient unit), so the gate
# cannot have moved much beyond ~0.1: the drug enters the head scaled down
# about tenfold. The primary result is consistent with that (with_smiles and
# shuffled_drug nearly identical fold by fold). The variants below exist to
# measure whether the null is a property of the DATA or of this THROTTLE:
#   fast_gate  gate still starts at 0 (so the null-safety of the design is
#              kept at initialisation) but has its own lr = 100 x base and no
#              weight decay, so it can open within tens of steps
#   open_init  gate starts at 1 with the base optimiser, i.e. no throttle at all
# Selected on SYNTHETIC data only, by the rule declared in §10, before any real
# label is read under them. Never the primary.
GATE_VARIANTS = ("frozen_prereg", "fast_gate", "open_init")
GATE_VARIANT = "frozen_prereg"
GATE_LR_MULT = 100.0
GATE_LOG: list = []           # (seed, fold, arm, final gate value), every drug-arm fit

# Digest of THIS file, taken at import (2026-09-29). It used to be read when the
# run finished, so editing the file during a long run logged the hash of code
# that never ran. Hashing at start ties the run log to the code that executed.
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def _load_drug_molecules() -> dict:
    """name -> parent InChIKey from the verified map; unknown names map to themselves."""
    path = BASE_DIR / "drugs" / "drug_smiles_map.json"
    if not path.exists():
        return {}
    dmap = json.loads(path.read_text()).get("map", {})
    return {n: e["inchikey"] for n, e in dmap.items() if e.get("inchikey")}


DRUG_MOLECULE = _load_drug_molecules()


NO_MOLECULE = ""      # a patient with no resolvable small molecule (biologic secondary)


def molecule_of(name: str) -> str:
    return DRUG_MOLECULE.get(name, name)


def _dedupe_by_molecule(names) -> tuple:
    """One name per distinct molecule (salt aliases share one structure)."""
    seen, out = set(), []
    for n in sorted(names):
        if n == NO_MOLECULE:
            continue
        m = molecule_of(n)
        if m not in seen:
            seen.add(m)
            out.append(n)
    return tuple(out)


def to_training_unit(samples, unit=None):
    """Collapse (patient, drug) records to one sample per patient under 'patient'."""
    unit = unit or TRAINING_UNIT
    if unit == "pair":
        return [s if s.regimen else replace(s, regimen=(s.drug,)) for s in samples]
    if unit != "patient":
        raise ValueError(f"unknown training unit {unit!r}")
    by_patient: dict = {}
    for s in samples:
        by_patient.setdefault(s.patient, []).append(s)
    out = []
    for pid in sorted(by_patient):
        rows = by_patient[pid]
        assert len({r.y for r in rows}) == 1, f"conflicting labels for {pid}"
        names = []
        for r in rows:
            names.extend(r.regimen or (r.drug,))
        reg = _dedupe_by_molecule(names)
        out.append(replace(rows[0], drug="+".join(reg), regimen=reg))
    return out


# =============================================================================
# Synthetic data — the ONLY place to debug this harness
# =============================================================================
def make_synthetic(n_patients=300, n_drugs=12, tiles=200, embed_dim=64,
                   delta_signal=0.9, seed=0, n_projects=3, interaction=False):
    """
    Build a fake cohort with a KNOWN, tunable drug effect.

    Response depends on:
        morphology       — recoverable from the bag by attention
        tissue context   — available to BOTH arms
        drug interaction — scaled by `delta_signal`; ONLY the drug branch can
                           see it, so it is exactly what ΔAUROC should measure

    delta_signal = 0.0 is the null case: the with-SMILES model has nothing extra
    to learn, so the harness should NOT reject, at close to the alpha rate.
    """
    rng = np.random.default_rng(seed)
    morph_dir = rng.standard_normal(embed_dim); morph_dir /= np.linalg.norm(morph_dir)
    drug_vecs = rng.standard_normal((n_drugs, FINGERPRINT_BITS)) > 0.7   # sparse, FP-like
    drug_effect = rng.standard_normal(n_drugs)                           # the hidden truth

    samples = []
    for i in range(n_patients):
        pid = f"SYN-{i:04d}"
        proj = f"PROJ-{i % n_projects}"
        m = float(rng.standard_normal())                     # morphology score
        ctx = rng.standard_normal(4)
        n_drug = 1 + int(rng.random() < 0.25)                # some patients get 2 drugs
        drugs = rng.choice(n_drugs, size=n_drug, replace=False)

        # bag: noise + the morphology direction planted in a minority of tiles
        bag = rng.standard_normal((tiles, embed_dim)).astype(np.float32) * 0.5
        signal_tiles = rng.random(tiles) < 0.3
        bag[signal_tiles] += (m * morph_dir).astype(np.float32)

        drug_term = float(drug_effect[drugs].mean())
        # additive main effect by default: the drug branch alone can learn it, so a
        # failure to recover it is a HARNESS fault, not a capacity limit. The
        # interaction form (drug x morphology) is the harder, more realistic case
        # and is available for stress-testing, but it is not what validates plumbing.
        extra = drug_term * m if interaction else drug_term
        logit = 1.1 * m + 0.6 * ctx[0] + delta_signal * extra
        y = int(rng.random() < 1.0 / (1.0 + math.exp(-logit)))

        for d in drugs:
            s = Sample(patient=pid, project=proj, drug=f"drug_{d}", y=y,
                       context=np.concatenate([ctx, np.eye(n_projects)[i % n_projects]]).astype(np.float32),
                       min_tiles=tiles,
                       oracle=(float(logit), float(1.1 * m + 0.6 * ctx[0])))
            s.bag = torch.from_numpy(bag)
            samples.append(s)

    fps = {f"drug_{d}": torch.from_numpy(drug_vecs[d].astype(np.float32))
           for d in range(n_drugs)}
    return samples, fps, embed_dim


# =============================================================================
# Real data
# =============================================================================
BIOLOGIC_INFO: dict = {}      # filled by load_real(include_biologics=True); logged


def load_real(tag: str, features_dir: Path, include_biologics: bool = False):
    """
    Build samples from the verified feature store + the cohort CSV.

    Labels come from cohort_labels.load_cohort and NOWHERE else (see that
    module's docstring). Bags pool every diagnostic slide of a patient, per the
    slide policy in §3.
    """
    import cohort_labels as CL

    csv_path = CL.find_cohort_csv(BASE_DIR, tag=tag)
    cohort = CL.load_cohort(csv_path)
    log.info(f"  cohort: {csv_path.name} — {CL.summarise(cohort)}")

    manifest_path = features_dir / "feature_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No feature manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    # group slides by patient
    by_patient = defaultdict(list)
    for slide, entry in manifest.items():
        if entry.get("status") != "done":
            continue
        bc = CL.patient_barcode(slide)
        if bc is None:
            log.warning(f"  no barcode parsed from {slide} — skipped")
            continue
        by_patient[bc].append(entry)

    # drug fingerprints, from the VERIFIED map; never zero-filled
    fps = build_fingerprints({d for r in cohort.values() for d in r["drugs"]})

    stages = sorted({(r.get("stage") or "unknown") for r in cohort.values()})
    projects = sorted({(r.get("project") or "unknown") for r in cohort.values()})
    stage_ix = {s: i for i, s in enumerate(stages)}
    proj_ix = {p: i for i, p in enumerate(projects)}

    # -----------------------------------------------------------------------
    # §6 tissue context, part two: ER / PR / HER2 (Gate 0(c)).
    #
    # §6 promises BOTH models receive receptor status for breast cases. It is
    # the load-bearing half of the answer to confounding by indication: drug
    # choice follows type, stage and subtype, all legible in the slide, so
    # DeltaAUROC only isolates chemistry BECAUSE both arms already hold that
    # information. In breast cancer, receptor subtype IS the treatment
    # decision. Produced by scripts/10_fetch_receptor_status.py.
    #
    # Three receptors x {positive, negative, unknown} = 9 dimensions. "unknown"
    # is a real level, not a dropped patient: missingness is higher in older
    # cases, so excluding them would select on collection era. For a non-breast
    # cohort every patient is "unknown" and the block is constant, which the
    # network can ignore at no cost.
    # -----------------------------------------------------------------------
    global RECEPTOR_CONTEXT_USED
    rec_csv = BASE_DIR / "cohort" / f"receptor_status_{tag}.csv"
    receptors: dict = {}
    if rec_csv.exists():
        import csv as _csv
        with rec_csv.open() as _f:
            for _r in _csv.DictReader(_f):
                receptors[_r["submitter_id"]] = _r
        RECEPTOR_CONTEXT_USED = True
    else:
        RECEPTOR_CONTEXT_USED = False
        log.warning(
            f"  ⚠️  {rec_csv.name} NOT FOUND — the model will run WITHOUT the "
            f"ER/PR/HER2 context §6 requires.\n"
            f"      Build it first:  python scripts/10_fetch_receptor_status.py "
            f"--tag {tag}\n"
            f"      The run log records receptor_context=false so this run can "
            f"never be mistaken for a compliant one.")

    REC_FIELDS = ("er_status", "pr_status", "her2_status")
    REC_LEVELS = ("positive", "negative", "unknown")
    rec_ix = {(f, lv): i * len(REC_LEVELS) + j
              for i, f in enumerate(REC_FIELDS) for j, lv in enumerate(REC_LEVELS)}
    n_rec = len(REC_FIELDS) * len(REC_LEVELS)
    if receptors:
        _known = sum(1 for v in receptors.values()
                     if all(v.get(f) not in (None, "", "unknown") for f in REC_FIELDS))
        log.info(f"  receptor context: {len(receptors)} patients, "
                 f"{_known} with all three known ({100 * _known / max(len(receptors), 1):.1f}%)")

    # -----------------------------------------------------------------------
    # BIOLOGIC-INCLUSIVE SECONDARY (§10 2026-09-19, design fixed 2026-09-29).
    # Biologics have no structure, so they are given to BOTH arms as binary
    # flags in the CONTEXT block -- one per named agent with >= 10 patients
    # (bevacizumab, trastuzumab, pertuzumab are the candidates named in the
    # plan) plus one "any other biologic". Because both arms hold them, the
    # contrast still measures small-molecule structure, now on the scoped
    # cohort. Patients with no resolvable small molecule (the biologic-only
    # regimens) stay in with a ZERO molecular input.
    # -----------------------------------------------------------------------
    bio_flags, bio_of = [], {}
    if include_biologics:
        dmap = json.loads((BASE_DIR / "drugs" / "drug_smiles_map.json").read_text())["map"]
        is_bio = lambda d: (dmap.get(d.lower().strip()) or {}).get("kind") == "biologic"
        for bc, rec in cohort.items():
            bio_of[bc] = {d for d in rec["drugs"] if is_bio(d)}
        named = []
        for agent in ("bevacizumab", "trastuzumab", "pertuzumab"):
            n = sum(1 for s in bio_of.values() if any(agent in d for d in s))
            if n >= 10:
                named.append(agent)
            BIOLOGIC_INFO[f"n_patients_{agent}"] = n
        bio_flags = named + ["other_biologic"]
        BIOLOGIC_INFO["flags"] = bio_flags
        log.info(f"  biologic context flags (both arms): {bio_flags} — "
                 + ", ".join(f"{k.replace('n_patients_', '')} {v}"
                             for k, v in BIOLOGIC_INFO.items() if k.startswith("n_")))

    samples, skipped_no_feat, skipped_no_fp = [], 0, 0
    n_zero_molecule = 0
    for bc, rec in cohort.items():
        entries = by_patient.get(bc)
        if not entries:
            skipped_no_feat += 1
            continue
        ctx = np.zeros(len(stages) + len(projects) + n_rec + len(bio_flags), dtype=np.float32)
        if bio_flags:
            _boff = len(stages) + len(projects) + n_rec
            for d in bio_of.get(bc, ()):
                hit = [f for f in bio_flags[:-1] if f in d]
                ctx[_boff + bio_flags.index(hit[0] if hit else "other_biologic")] = 1.0
        ctx[stage_ix[rec.get("stage") or "unknown"]] = 1.0
        ctx[len(stages) + proj_ix[rec.get("project") or "unknown"]] = 1.0
        _rr = receptors.get(bc, {})
        _off = len(stages) + len(projects)
        for _f in REC_FIELDS:
            _lv = (_rr.get(_f) or "unknown").strip().lower()
            if _lv not in REC_LEVELS:          # e.g. a stray "equivocal"
                _lv = "unknown"
            ctx[_off + rec_ix[(_f, _lv)]] = 1.0

        paths = [BASE_DIR / e["feature_path"] for e in entries]
        min_tiles = min(int(e.get("n_tiles", 0)) for e in entries)
        y = 1 - int(rec["response_binary"])          # NON-RESPONDER is positive

        neg_drugs = set()
        for pr in rec.get("pairs", []):
            try:
                if float(pr.get("days_to_treatment_start")) < 0:
                    neg_drugs.add(pr.get("drug"))
            except (TypeError, ValueError):
                pass                                   # missing day -> not negative
        n_before = len(samples)
        for drug in rec["drugs"]:
            if drug not in fps:
                skipped_no_fp += 1
                continue
            samples.append(Sample(patient=bc, project=rec.get("project", "unknown"),
                                  drug=drug, y=y, context=ctx,
                                  bag_paths=paths, min_tiles=min_tiles,
                                  negative_start=drug in neg_drugs))
        if include_biologics and len(samples) == n_before:
            # no resolvable small molecule: kept, with a zero molecular input
            samples.append(Sample(patient=bc, project=rec.get("project", "unknown"),
                                  drug=NO_MOLECULE, y=y, context=ctx,
                                  bag_paths=paths, min_tiles=min_tiles))
            n_zero_molecule += 1
    if include_biologics:
        BIOLOGIC_INFO["n_patients_zero_molecule"] = n_zero_molecule
        log.info(f"  {n_zero_molecule} patients kept with no small molecule (zero drug input)")

    if skipped_no_feat:
        log.warning(f"  {skipped_no_feat} cohort patients have no embedded slide — "
                    "expected 0 once the arm is fully embedded")
    if skipped_no_fp:
        log.warning(f"  {skipped_no_fp} (patient, drug) records dropped: no verified "
                    "structure for that drug (biologic or unresolved)")

    embed_dim = int(next(iter(manifest.values()))["embed_dim"])
    return samples, fps, embed_dim


def build_fingerprints(drug_names) -> dict:
    """Morgan fingerprints from the verified map. Raises rather than zero-filling."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    blob = json.loads((BASE_DIR / "drugs" / "drug_smiles_map.json").read_text())
    dmap = blob["map"]

    fps, unresolved = {}, []
    for name in sorted(drug_names):
        entry = dmap.get(name.lower().strip())
        if entry is None or entry.get("smiles") is None:
            unresolved.append(name)
            continue
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None:
            unresolved.append(name)
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FINGERPRINT_BITS)
        arr = np.zeros(FINGERPRINT_BITS, dtype=np.float32)
        for b in bv.GetOnBits():
            arr[b] = 1.0
        if arr.sum() == 0:
            raise ValueError(f"'{name}' produced an all-zero fingerprint — "
                             "indistinguishable from 'no drug'. Fix the map.")
        fps[name] = torch.from_numpy(arr)
    if unresolved:
        log.warning(f"  {len(unresolved)} drugs have no small-molecule structure "
                    f"and are excluded: {', '.join(unresolved[:8])}"
                    f"{' …' if len(unresolved) > 8 else ''}")
    return fps


def _frozen_pca(X: np.ndarray, k: int) -> np.ndarray:
    """
    Label-free PCA to k components, unit-variance scaled.

    Shared by BOTH drug encoders so they are interchangeable at fixed width.
    It sees a matrix of molecules and nothing else — no labels, no patients, no
    fold indices — so it cannot leak outcome information, and it is a
    deterministic function of the drug set.
    """
    k = int(min(k, X.shape[0] - 1, X.shape[1]))
    if k < 1:
        return X.astype(np.float32)
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # deterministic, no labels
    Z = Xc @ Vt[:k].T
    return Z / (Z.std(0, keepdims=True) + 1e-8)


def load_pretrained_drug_embeddings(drug_names, filename: str) -> dict:
    """
    The frozen pretrained encoder's output, read from the artefact on disk.

    Raises rather than falling back. A missing molecule here means the artefact
    and the cohort have drifted apart, and silently substituting fingerprints —
    or a zero vector — for the affected drugs would produce a run whose drug
    branch is a quiet mixture of two encoders. Rebuild the artefact instead:

        python scripts/11_build_drug_embeddings.py --tag <tag>
    """
    path = BASE_DIR / "drugs" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"pretrained drug embeddings not found: {path}\n"
            f"    build them first:  python scripts/11_build_drug_embeddings.py --tag <tag>")

    blob = np.load(path, allow_pickle=True)
    names = [str(n) for n in blob["names"]]
    X = blob["X"].astype(np.float64)
    table = {n: X[i] for i, n in enumerate(names)}

    missing = sorted(set(drug_names) - set(table))
    if missing:
        raise KeyError(
            f"{len(missing)} drug(s) in this cohort have no pretrained embedding: "
            f"{', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}\n"
            f"    the artefact was built for a different drug set — rebuild it.")

    prov_path = path.with_suffix(".json")
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    log.info(f"  drug encoder: pretrained {prov.get('model_id', filename)} "
             f"@ {str(prov.get('revision', '?'))[:12]} "
             f"({prov.get('pooling', '?')}-pooled, {X.shape[1]}-d)")
    return {n: table[n] for n in sorted(drug_names)}, prov


def build_drug_features(fps: dict, k: int, encoder: str) -> tuple[dict, dict]:
    """
    THE DRUG BRANCH INPUT. One frozen, label-free vector per drug, width k.

    Both encoders end at the SAME width k = HPARAMS['drug_feat_dim'] = 32, and
    that is the point of the design rather than an incidental detail:

      * The trainable parameter count of the drug branch is Linear(k, 16), so
        swapping the encoder changes WHAT the model is told about a molecule
        without changing HOW MUCH capacity it gets to fit it. The capacity bias
        documented in §10 (a wide drug branch produces a negative ΔAUROC out of
        parameter count alone, measured at -0.066 on data with no drug effect)
        is a function of that width, so holding it fixed is what keeps the two
        encoders comparable and keeps the §6 hyperparameter freeze honest.

      * Feeding ChemBERTa's raw 768 dimensions in instead would have multiplied
        the drug branch's trainable parameters by 24 and reintroduced exactly
        the bias the PCA was added to remove — while looking like an upgrade.

    Returns (features, provenance).
    """
    names = sorted(fps)
    if encoder == "morgan_pca":
        X = np.stack([fps[n].numpy() for n in names]).astype(np.float64)
        prov = {"encoder": "morgan_pca", "input_dim": int(X.shape[1])}
        log.info(f"  drug encoder: {X.shape[1]}-bit Morgan -> frozen PCA "
                 f"({len(names)} molecules)  [LEGACY STAND-IN]")
    elif encoder in DRUG_ENCODERS:
        table, prov = load_pretrained_drug_embeddings(names, DRUG_ENCODERS[encoder])
        X = np.stack([table[n] for n in names])
        prov = {"encoder": encoder, "input_dim": int(X.shape[1]), **prov}
    else:
        raise ValueError(f"unknown drug encoder '{encoder}'")

    Z = _frozen_pca(X, k)
    prov["output_dim"] = int(Z.shape[1])
    prov["n_drugs"] = len(names)
    log.info(f"                {prov['input_dim']}-d -> {Z.shape[1]}-d frozen PCA, "
             f"{len(names)} drugs")
    return ({n: torch.from_numpy(Z[i].astype(np.float32)) for i, n in enumerate(names)},
            prov)


def reduce_fingerprints(fps: dict, k: int) -> dict:
    """
    Compress Morgan fingerprints to a FROZEN, low-dimensional drug embedding.

    WHY THIS EXISTS. A raw 1024-bit fingerprint feeding a trainable layer gives
    the with-SMILES model ~65k extra trainable parameters. At n ~ 1k patients
    those parameters fit noise, and the ablation acquires a systematic NEGATIVE
    bias -- measured at dAUROC = -0.066 on synthetic data containing NO drug
    effect at all. A study that reports "chemistry hurts" when nothing is there
    cannot interpret its own null, and the pre-registered null-result fallback
    (section 8) depends on exactly that interpretation.

    The reduction is a PCA over the fingerprint matrix alone. It sees no labels,
    no patients and no folds -- only chemistry -- so it cannot leak outcome
    information, and it is deterministic given the drug set.

    SUPERSEDED 2026-08-29. This was the stand-in for the frozen pretrained
    encoder §6 requires; that encoder now exists (scripts/11_, ChemBERTa) and is
    the default. This function is retained ONLY so `--drug-encoder morgan_pca`
    can reproduce every result recorded before that date, and is reached through
    build_drug_features(). Do not add features here.
    """
    names = sorted(fps)
    X = np.stack([fps[n].numpy() for n in names]).astype(np.float64)
    if int(min(k, len(names) - 1, X.shape[1])) < 1:
        return dict(fps)
    Z = _frozen_pca(X, k)
    log.info(f"  drug encoder: {X.shape[1]}-bit Morgan -> {Z.shape[1]}-d frozen PCA "
             f"({len(names)} molecules)")
    return {n: torch.from_numpy(Z[i].astype(np.float32)) for i, n in enumerate(names)}


# =============================================================================
# Model - identical in every respect except the drug branch (section 6)
# =============================================================================
class ResponseModel(nn.Module):
    """Gated-attention MIL over tiles, plus tissue context, plus (optionally) a drug."""

    def __init__(self, embed_dim: int, ctx_dim: int, drug_dim: int | None, hp: dict):
        super().__init__()
        h, a, p = hp["hidden_dim"], hp["attn_dim"], hp["dropout"]
        self.feature_proj = nn.Sequential(nn.Linear(embed_dim, h), nn.ReLU(), nn.Dropout(p))
        self.attn_V = nn.Linear(h, a)      # tanh branch
        self.attn_U = nn.Linear(h, a)      # sigmoid gate
        self.attn_w = nn.Linear(a, 1)

        self.use_drug = drug_dim is not None
        head_in = h + ctx_dim
        if self.use_drug:
            self.drug_proj = nn.Sequential(
                nn.Linear(drug_dim, hp["drug_proj_dim"]), nn.ReLU(),
                nn.Dropout(p), nn.LayerNorm(hp["drug_proj_dim"]))
            # ZERO-INITIALISED GATE. At initialisation the drug branch contributes
            # exactly nothing, so the with-SMILES model IS the without-SMILES model
            # and can only move away from it if the drug reduces validation loss.
            # Without this the extra ~65k parameters fit noise, and the ablation
            # carries a systematic NEGATIVE bias under the null - measured at
            # dAUROC = -0.066 on synthetic data with NO planted effect, which would
            # have been reported as "chemistry hurts". See selftest [2/2].
            self.drug_gate = nn.Parameter(
                torch.ones(1) if GATE_VARIANT == "open_init" else torch.zeros(1))
            head_in += hp["drug_proj_dim"]

        self.head = nn.Sequential(
            nn.Linear(head_in, h // 2), nn.ReLU(), nn.Dropout(p), nn.Linear(h // 2, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tiles, ctx, drug=None, mask=None):
        """
        BATCHED. tiles [B, N, D] zero-padded, mask [B, N] True on real tiles,
        ctx [B, C], drug [B, drug_dim] or None.  Returns logits [B].

        The mask is load-bearing. Bags are padded to the longest in the batch,
        and a padded row must contribute NOTHING to the attention pool. Setting
        its logit to the dtype minimum before the softmax makes its weight
        exactly zero. Omitting the mask silently averages in padding and
        corrupts every result without raising anything -- which is why
        --selftest is the regression test for this function.
        """
        h = self.feature_proj(tiles)                                   # [B, N, h]
        A = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h)))
        if mask is not None:                                           # [B, N, 1]
            A = A.masked_fill(~mask.unsqueeze(-1), torch.finfo(A.dtype).min)
        a = torch.softmax(A, dim=1)                                    # [B, N, 1]
        z = (a * h).sum(dim=1)                                         # [B, h]
        parts = [z, ctx]
        if self.use_drug:
            parts.append(self.drug_gate * self.drug_proj(drug))
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)          # [B]


# =============================================================================
# Bag loading
# =============================================================================
def _load_part(path):
    """
    Memory-map one slide's feature tensor. Nothing is read from disk until rows
    are indexed, and the OS page cache -- not this process -- decides what stays
    in RAM. Falls back to a plain load for a file saved in the legacy format.
    """
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except (RuntimeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=True)


def get_tiles(s: Sample, n_tiles: int, rng: np.random.Generator, cache: dict) -> torch.Tensor:
    """
    Pool all of a patient's slides, then take a seeded random subsample.

    LAZY LOADER (worklist step 24, 2026-09-29). This used to torch.cat every
    slide into one float32 bag and keep it in `cache` for the whole run. Arm A
    is 4.69 M tiles = 19.2 GB in float32, on an 8 GB machine. Now each slide is
    memory-mapped, the SAME index draw is made over the concatenated row space,
    and only the selected rows are read and cast. The returned tensor is
    BIT-IDENTICAL to the old one -- same rng call, same sorted indices, same
    row order, same cast -- so no result changes; only memory does. `cache` is
    no longer used for real bags (kept for the call signature).
    """
    if s.bag is not None:
        bag = s.bag
        if bag.shape[0] <= n_tiles:
            return bag
        idx = rng.choice(bag.shape[0], n_tiles, replace=False)
        return bag[torch.from_numpy(np.sort(idx))]

    parts = [_load_part(p) for p in s.bag_paths]
    sizes = [int(p.shape[0]) for p in parts]
    total = sum(sizes)
    if total <= n_tiles:
        return torch.cat([p.float() for p in parts])
    idx = np.sort(rng.choice(total, n_tiles, replace=False))
    offsets = np.cumsum([0] + sizes)
    out = []
    for j, p in enumerate(parts):
        lo, hi = offsets[j], offsets[j + 1]
        sel = idx[(idx >= lo) & (idx < hi)] - lo
        if len(sel):
            out.append(p[torch.from_numpy(sel)].float())
    return torch.cat(out)


def patient_seed(patient_id: str) -> int:
    """
    A STABLE per-patient seed for the eval-time tile subsample.

    Python's built-in hash() on a str is salted per process unless
    PYTHONHASHSEED is set, so the previous `abs(hash(s.patient))` drew a
    DIFFERENT set of evaluation tiles on every run of this script. Within one
    run it was consistent across arms, so no comparison was ever unfair -- but
    the numbers in the paper could not be regenerated from the recorded seed,
    which §3.5 requires. blake2b is deterministic across processes, machines and
    Python versions. Label-blind and identical for every arm.
    """
    return int.from_bytes(
        hashlib.blake2b(str(patient_id).encode(), digest_size=4).digest(), "big")


def collate_bags(bags, ctxs, drugs, device):
    """
    Pad a list of [n_i, D] bags into one [B, N, D] tensor plus a [B, N] mask.

    This is the whole point of the batching refactor: one matmul over B bags
    instead of B small ones. Measured on the Arm B smoke run, the one-bag-at-a-
    time loop reached ~65 GFLOPS, about 2% of an M-series GPU -- the other 98%
    was Python and kernel-launch overhead, not arithmetic. A faster card does
    not fix that; batching does.
    """
    B, D = len(bags), bags[0].shape[1]
    N = max(b.shape[0] for b in bags)
    tiles = torch.zeros(B, N, D, dtype=torch.float32)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for i, b in enumerate(bags):
        n = b.shape[0]
        tiles[i, :n] = b
        mask[i, :n] = True
    ctx = torch.stack(ctxs)
    drug = None if drugs[0] is None else torch.stack(drugs)
    return (tiles.to(device), mask.to(device), ctx.to(device),
            None if drug is None else drug.to(device))


# =============================================================================
# Training one fold
# =============================================================================
def train_fold(train_s, val_s, arm, fps, embed_dim, hp, seed, cache, device, shuffle_map):
    """Train one model on one fold. Early stops on POS-WEIGHTED inner-val LOSS (§10)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    drug_dim = drug_dim_for(arm, fps)
    model = ResponseModel(embed_dim, train_s[0].context.size, drug_dim, hp).to(device)
    if GATE_VARIANT == "fast_gate" and drug_dim is not None:
        gate = [model.drug_gate]
        rest = [p_ for n_, p_ in model.named_parameters() if n_ != "drug_gate"]
        opt = torch.optim.AdamW([
            {"params": rest, "lr": hp["lr"], "weight_decay": hp["weight_decay"]},
            {"params": gate, "lr": hp["lr"] * GATE_LR_MULT, "weight_decay": 0.0}])
    else:
        # the pre-registered optimiser, constructed exactly as before
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    n_pos = sum(s.y for s in train_s)
    pos_weight = torch.tensor([(len(train_s) - n_pos) / max(n_pos, 1)], device=device)
    # reduction="sum" then divide by batch_size reproduces EXACTLY the arithmetic
    # of the old accumulate-then-step loop, including its treatment of a short
    # final batch. The refactor is meant to change speed, not the objective.
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    best_loss, best_state, bad, sel_auc = float("inf"), None, 0, 0.5
    for epoch in range(hp["max_epochs"]):
        model.train()
        order = rng.permutation(len(train_s))
        epoch_loss = 0.0
        for start in range(0, len(order), hp["batch_size"]):
            chunk = [train_s[ix] for ix in order[start:start + hp["batch_size"]]]
            # tiles are drawn per sample in the SAME order as before, so the rng
            # stream this function consumes is unchanged
            tiles, mask, ctx, drug = collate_bags(
                [get_tiles(s, hp["train_tiles"], rng, cache) for s in chunk],
                [torch.as_tensor(s.context, dtype=torch.float32) for s in chunk],
                [drug_vec(s, arm, fps, shuffle_map, "cpu") for s in chunk],
                device)
            yb = torch.tensor([float(s.y) for s in chunk], device=device)
            loss = lossf(model(tiles, ctx, drug, mask), yb) / hp["batch_size"]
            epoch_loss += float(loss.detach()) * hp["batch_size"]
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        epoch_loss /= max(len(order), 1)
        yv, sv = predict(model, val_s, arm, fps, hp, seed, cache, device, shuffle_map)
        auc = roc_auc_score(yv, sv) if len(set(yv)) > 1 else 0.5

        # §10 (2026-08-27): EARLY STOPPING ON VALIDATION LOSS, NOT AUROC.
        # An inner-val fold in Arm B is ~23 patients with 2-3 non-responders, so
        # its AUROC takes only a handful of discrete values and cannot tell an
        # epoch-3 model from an epoch-30 one. With the model memorising inside
        # five epochs, the restored epoch was effectively a coin flip.
        #
        # The loss is POS-WEIGHTED with the training fold's own pos_weight --
        # the same objective the model is optimised against. Unweighted BCE on
        # a fold that is 90% responders is minimised by predicting "responder"
        # for everyone, which is exactly the degenerate model early stopping is
        # supposed to reject.
        yv_a, sv_a = np.asarray(yv, dtype=float), np.clip(np.asarray(sv, dtype=float), 1e-9, 1 - 1e-9)
        w = np.where(yv_a == 1, float(pos_weight), 1.0)
        terms = -(yv_a * np.log(sv_a) + (1 - yv_a) * np.log(1 - sv_a))
        vloss = float(np.sum(w * terms) / np.sum(w))

        if hp.get("log_epochs"):
            log.info(f"      epoch {epoch+1:>3}  train_loss {epoch_loss:.4f}  "
                     f"val_loss {vloss:.4f}  val_auroc {auc:.3f}"
                     f"{'  *best' if vloss < best_loss - 1e-6 else ''}")
        if vloss < best_loss - 1e-6:
            best_loss, bad, sel_auc = vloss, 0, auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    # sel_auc is the inner-val AUROC AT the selected epoch -- reported for
    # continuity with earlier runs, but it selects nothing any more.
    return model, sel_auc, best_loss


def _molecule_index(fps) -> dict:
    """Column of each distinct molecule in the one-hot identity control."""
    return {m: i for i, m in enumerate(sorted({molecule_of(n) for n in fps}))}


def drug_dim_for(arm, fps):
    if arm == "without_smiles":
        return None
    if arm == "onehot_drug":
        # keyed on the MOLECULE, not the name (§10, 2026-09-29), so the identity
        # control partitions drugs exactly as with_smiles does: salt aliases
        # are one identity in both arms.
        return len(_molecule_index(fps))
    return int(next(iter(fps.values())).shape[0])


def drug_vec(s: Sample, arm, fps, shuffle_map, device):
    """
    The drug branch input for this arm. None for the histology-only arm.

    The shuffled-drug control reads from `shuffle_map`, a FIXED per-seed
    assignment built once in run_cv. Drawing a fresh random drug at every
    forward pass would instead train the model to ignore the drug branch
    entirely, which quietly turns the capacity control into a second copy of
    the baseline and destroys the thing it exists to detect.
    """
    if arm == "without_smiles":
        return None
    names = shuffle_map[id(s)] if arm == "shuffled_drug" else (s.regimen or (s.drug,))
    names = _dedupe_by_molecule(names)
    if not names:                            # biologic-only / no-structure patient
        return torch.zeros(drug_dim_for(arm, fps)).to(device)
    if arm == "onehot_drug":
        idx = _molecule_index(fps)
        v = torch.zeros(len(idx))
        for n in names:                      # mean of one-hot vectors: 1/k each
            v[idx[molecule_of(n)]] += 1.0 / len(names)
        return v.to(device)
    if len(names) == 1:                      # bit-identical to the pre-pooling path
        return fps[names[0]].to(device)
    return torch.stack([fps[n] for n in names]).mean(dim=0).to(device)


@torch.no_grad()
def predict(model, samples, arm, fps, hp, seed, cache, device, shuffle_map):
    """Deterministic inference: eval-time tiles are drawn from a FIXED seed."""
    model.eval()
    # eval bags are 4x the training bags, so they get their own (smaller) batch
    bs = hp.get("eval_batch_size", max(1, hp["batch_size"] // 2))
    ys, ss = [], []
    for start in range(0, len(samples), bs):
        chunk = samples[start:start + bs]
        bags, ctxs, drugs = [], [], []
        for s in chunk:
            # tile subsample is seeded PER SAMPLE so inference is deterministic and
            # identical across arms — the arms must differ only in the drug branch
            rng = np.random.default_rng(patient_seed(s.patient) + seed)
            bags.append(get_tiles(s, hp["eval_tiles"], rng, cache))
            ctxs.append(torch.as_tensor(s.context, dtype=torch.float32))
            drugs.append(drug_vec(s, arm, fps, shuffle_map, "cpu"))
        tiles, mask, ctx, drug = collate_bags(bags, ctxs, drugs, device)
        ss.extend(torch.sigmoid(model(tiles, ctx, drug, mask)).cpu().reshape(-1).tolist())
        ys.extend(s.y for s in chunk)
    return np.array(ys), np.array(ss)


# =============================================================================
# Cross-validation
# =============================================================================
def run_cv(samples, fps, embed_dim, arms, hp, seed, device, n_splits=5, val_log=None):
    """
    5-fold, grouped by patient, stratified by (response class x cancer type).

    Returns {arm: (y_patient, score_patient, patient_ids)} — pooled out-of-fold
    predictions with every patient scored exactly ONCE.

    If `val_log` is a list, one record per (fold, arm) is appended to it giving
    the INNER-validation loss and AUROC at the selected epoch. That is the only
    channel the baseline tuning grid (§10, 2026-08-27) is allowed to read: it
    never sees the out-of-fold predictions this function returns.
    """
    samples = to_training_unit(samples)
    groups = np.array([s.patient for s in samples])
    strat = np.array([f"{s.y}|{s.project}" for s in samples])
    # StratifiedGroupKFold needs integer-ish y; map the combined key
    keys = {k: i for i, k in enumerate(sorted(set(strat)))}
    y_strat = np.array([keys[k] for k in strat])

    cache: dict = {}
    # fixed random drug per sample for the capacity control (§6), one draw per seed
    _srng = np.random.default_rng(seed + 99991)
    if TRAINING_UNIT == "pair":
        # unchanged from the pre-2026-09-29 harness: one random drug per record
        drug_names = sorted(fps)
        shuffle_map = {id(s): (drug_names[int(_srng.integers(len(drug_names)))],)
                       for s in samples}
    else:
        # §10 2026-09-29: under the patient unit the capacity control shuffles
        # WHOLE REGIMENS — each patient receives the regimen of a randomly drawn
        # patient, so regimen sizes and co-occurrence structure are preserved and
        # only the regimen-label link is broken.
        regimens = [s.regimen for s in samples]
        shuffle_map = {id(s): regimens[int(_srng.integers(len(regimens)))]
                       for s in samples}
    oof = {arm: defaultdict(list) for arm in arms}   # arm -> patient -> [probs]
    labels = {s.patient: s.y for s in samples}

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(samples, y_strat, groups), 1):
        # inner validation split, also grouped by patient
        tr_patients = sorted({groups[i] for i in tr})
        rng = np.random.default_rng(seed * 1000 + fold)
        rng.shuffle(tr_patients)
        n_val = max(1, int(len(tr_patients) * hp["inner_val_frac"]))
        val_p = set(tr_patients[:n_val])

        train_s = [samples[i] for i in tr if groups[i] not in val_p]
        val_s   = [samples[i] for i in tr if groups[i] in val_p]
        test_s  = [samples[i] for i in te]

        leak = {s.patient for s in train_s} & {s.patient for s in test_s}
        assert not leak, f"PATIENT LEAKAGE in fold {fold}: {sorted(leak)[:5]}"

        for arm in arms:
            model, val_auc, val_loss = train_fold(train_s, val_s, arm, fps, embed_dim,
                                                  hp, seed, cache, device, shuffle_map)
            _, probs = predict(model, test_s, arm, fps, hp, seed, cache, device, shuffle_map)
            for s, p in zip(test_s, probs):
                oof[arm][s.patient].append(float(p))
            if val_log is not None:
                val_log.append({"seed": seed, "fold": fold, "arm": arm,
                                "inner_val_loss": val_loss, "inner_val_auroc": val_auc})
            g = float(model.drug_gate.detach().cpu()) if getattr(model, "use_drug", False) else None
            if g is not None:
                GATE_LOG.append({"seed": seed, "fold": fold, "arm": arm, "gate": g})
            log.info(f"    fold {fold} · {arm:<15} inner-val loss {val_loss:.4f} "
                     f"(AUROC {val_auc:.3f})" + (f"  gate {g:+.4f}" if g is not None else ""))

    out = {}
    for arm in arms:
        pats = sorted(oof[arm])
        out[arm] = (np.array([labels[p] for p in pats]),
                    np.array([float(np.mean(oof[arm][p])) for p in pats]),
                    pats)
    return out


def permutation_null(samples, fps, embed_dim, arms, hp, device, n_perm, n_splits, seed0=5000,
                     out=None, observed=None, curtail=False):
    """
    Empirical null distribution for the primary delta, by permuting LABELS.

    WHY THIS IS NEEDED, and it is not optional. DeLong's test conditions on the
    two score vectors as if they came from fixed models. Here they do not: every
    fold retrains both arms, so the scores carry training randomness that
    DeLong's variance estimate does not model. The test is therefore
    ANTI-CONSERVATIVE on pooled out-of-fold predictions -- measured on synthetic
    data with no drug effect, single-seed DeLong rejected roughly half the time
    instead of 5% of the time.

    Permuting the labels and re-running the WHOLE pipeline rebuilds the null the
    honest way: every source of variance the real run has, minus the signal. The
    p-value is then the fraction of permuted deltas at least as extreme as the
    observed one. This is the pre-registered label-permutation control doing
    double duty as the calibrated test.

    Labels are permuted at the PATIENT level, so a patient keeps one label across
    all of their (patient, drug) records and the grouping structure is preserved.

    COST: n_perm complete cross-validation runs. Budget accordingly, run it once,
    and log it. Do not tune anything against it.
    """
    import copy
    rng = np.random.default_rng(seed0)
    patients = sorted({s.patient for s in samples})
    base_labels = {p: next(s.y for s in samples if s.patient == p) for p in patients}

    deltas = out if out is not None else []   # caller's list is filled in place
    for i in range(n_perm):
        shuffled = list(base_labels.values())
        rng.shuffle(shuffled)
        newlab = dict(zip(patients, shuffled))
        perm = []
        for s in samples:
            s2 = copy.copy(s)
            s2.y = newlab[s.patient]
            perm.append(s2)
        # Only the two arms of the primary contrast are trained (2026-09-29):
        # each arm is trained independently from its own seed, so dropping the
        # other two changes neither of these and halves the cost.
        null_arms = ["without_smiles", "with_smiles"]
        res = {1: run_cv(perm, fps, embed_dim, null_arms, hp, seed0 + i, device, n_splits)}
        rep = summarise_run(res, null_arms, [1])
        d = rep["contrasts"]["with_smiles_vs_without_smiles"]["delta_auroc_mean"]
        deltas.append(d)
        log.info(f"    permutation {i+1}/{n_perm}: delta {d:+.4f}")
        # CURTAILED TEST (Besag & Clifford 1991), added 2026-09-30. Once
        # H = floor(alpha * (n_perm + 1)) permuted deltas are at least as
        # extreme as the observed one, the fixed-N p-value is already
        # >= (H + 1) / (n_perm + 1) > alpha whatever the remaining runs give,
        # so the fixed-N DECISION ("do not reject") is determined. Stopping
        # there cannot change the pre-registered decision; it only saves the
        # compute. The reject direction can never be curtailed this way.
        if curtail and observed is not None:
            H = int(math.floor(ALPHA * (n_perm + 1)))
            h = sum(1 for x in deltas if abs(x) >= abs(observed))
            if h >= H:
                log.info(f"    curtailed after {i+1} runs: {h} permuted deltas >= |observed|; "
                         f"fixed-N p >= {(h + 1) / (n_perm + 1):.4f} > alpha")
                break
    return deltas


def permutation_p(observed: float, null_deltas: list) -> float:
    """Two-sided permutation p-value, with the +1 correction (Phipson & Smyth)."""
    b = sum(1 for d in null_deltas if abs(d) >= abs(observed))
    return (b + 1) / (len(null_deltas) + 1)


# =============================================================================
# Reporting
# =============================================================================
def summarise_run(results_by_seed, arms, seeds):
    """Per-arm metrics per seed, plus the primary endpoint against without_smiles."""
    report = {"arms": {}, "contrasts": {}, "seeds": seeds}

    for arm in arms:
        rows = []
        for seed in seeds:
            y, s, _ = results_by_seed[seed][arm]
            rows.append(score_report(y, s, seed=seed))
        report["arms"][arm] = {
            "per_seed": rows,
            "auroc_mean": float(np.mean([r["auroc"] for r in rows])),
            "auroc_sd":   float(np.std([r["auroc"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "auprc_mean": float(np.mean([r["auprc"] for r in rows])),
        }

    # Every arm against the pre-registered baseline, PLUS the capacity-matched
    # contrast. with_smiles vs shuffled_drug holds the parameter count fixed and
    # changes only whether the molecule is the RIGHT one - it is the contrast that
    # isolates chemistry from capacity (SS6).
    pairs = [(arm, "without_smiles") for arm in arms if arm != "without_smiles"]
    if "with_smiles" in arms and "shuffled_drug" in arms:
        pairs.append(("with_smiles", "shuffled_drug"))
    # Structure vs identity -- the most direct test of the study's actual claim
    # ("structure beats name"). Declared as a secondary contrast on 2026-09-29,
    # before any Arm A result (pre-registration SS10). Reported, never the primary.
    if "with_smiles" in arms and "onehot_drug" in arms:
        pairs.append(("with_smiles", "onehot_drug"))

    for arm, base in pairs:
        deltas, ps, zs = [], [], []
        for seed in seeds:
            y_a, s_a, p_a = results_by_seed[seed][arm]
            y_b, s_b, p_b = results_by_seed[seed][base]
            assert p_a == p_b, "arms scored different patient sets — folds diverged"
            assert (y_a == y_b).all(), "label mismatch between arms"
            auc_a, auc_b, d, z, p = delong_test(y_a, s_a, s_b)
            deltas.append(d); ps.append(p); zs.append(z)
        report["contrasts"][f"{arm}_vs_{base}"] = {
            "capacity_matched": base == "shuffled_drug",
            "delta_auroc_per_seed": deltas,
            "delta_auroc_mean": float(np.mean(deltas)),
            "delta_auroc_sd":   float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "delong_p_per_seed": ps,
            "delong_z_per_seed": zs,
            "n_seeds_significant": int(sum(p < ALPHA for p in ps)),
        }
    return report


def print_report(rep, title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    print(f"{'arm':<18}{'AUROC (mean±sd)':>22}{'AUPRC':>10}{'baseline':>11}")
    for arm, a in rep["arms"].items():
        r0 = a["per_seed"][0]
        print(f"{arm:<18}{a['auroc_mean']:>14.4f} ± {a['auroc_sd']:<5.4f}"
              f"{a['auprc_mean']:>10.4f}{r0['auprc_baseline']:>11.4f}")
    print(f"\n{'contrast':<32}{'ΔAUROC':>12}{'sd':>9}{'DeLong p (seed 1)':>20}{'sig':>6}")
    for name, c in rep["contrasts"].items():
        print(f"{name:<32}{c['delta_auroc_mean']:>12.4f}{c['delta_auroc_sd']:>9.4f}"
              f"{c['delong_p_per_seed'][0]:>20.4g}"
              f"{c['n_seeds_significant']:>4}/{len(c['delong_p_per_seed'])}")
    print()


def append_run_log(entry: dict):
    """§8: every configuration trained against labels is logged and reported."""
    p = BASE_DIR / "output" / "eval" / "run_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return p


def overfit_check(samples, fps, embed_dim, hp, device, n_patients=50, epochs=60):
    """
    CAN THIS MODEL LEARN AT ALL? A capability check, not an experiment.

    Trains the BASELINE arm (without_smiles) on a small set of patients with all
    regularisation switched off and no early stopping, then scores it on THE SAME
    patients. Memorising 50 examples is easy for a network with ~300k parameters,
    so a healthy training loop drives training AUROC towards 1.0 within a few
    dozen epochs.

    WHY THIS IS THE RIGHT NEXT TEST. The first real BRCA run put every arm at
    AUROC ~0.53 against a chance level of 0.500, with seed-to-seed spread (0.047)
    a hundred times larger than the primary delta (-0.0005). That result cannot
    distinguish two very different worlds:

        (a) the training loop is broken - gradients, loss, or bag pooling - or
        (b) the loop is fine and 195 patients with 22 non-responders simply do
            not carry a learnable signal.

    This test separates them. Cannot memorise 50 patients  -> world (a), a bug.
    Memorises easily but generalises at 0.5                -> world (b), and the
    answer waits for Arm A's 354 non-responders.

    IT CANNOT CONTAMINATE THE PRE-REGISTRATION. It touches one arm only, never
    computes a with-versus-without difference, and reports performance on
    TRAINING data, which is not an estimate of anything and can never be quoted
    as a result. Nothing here is tuned against the primary endpoint.
    """
    rng = np.random.default_rng(0)
    by_patient = defaultdict(list)
    for s in samples:
        by_patient[s.patient].append(s)

    # keep the class balance representative rather than convenient
    pos = [p for p, ss in by_patient.items() if ss[0].y == 1]
    neg = [p for p, ss in by_patient.items() if ss[0].y == 0]
    n_pos = max(2, min(len(pos), n_patients // 4))
    n_neg = min(len(neg), n_patients - n_pos)
    chosen = list(rng.choice(pos, n_pos, replace=False)) + \
             list(rng.choice(neg, n_neg, replace=False))
    subset = [s for p in chosen for s in by_patient[p]]

    print(f"\n{'=' * 74}")
    print("OVERFIT CAPABILITY CHECK - can the model memorise a small training set?")
    print(f"{'=' * 74}")
    print(f"  patients          : {len(chosen)}  ({n_pos} non-responder / {n_neg} responder)")
    print(f"  (patient,drug) rows: {len(subset)}")
    print("  arm               : without_smiles only")
    print("  regularisation    : OFF (dropout 0, weight decay 0)")
    print("  early stopping    : OFF")
    print("  scored on         : THE TRAINING DATA ITSELF - not a performance estimate\n")

    hp2 = dict(hp, dropout=0.0, weight_decay=0.0, max_epochs=epochs, patience=10**9)
    torch.manual_seed(0)
    model = ResponseModel(embed_dim, subset[0].context.size, None, hp2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp2["lr"], weight_decay=0.0)
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    cache, shuffle_map = {}, {}

    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(subset))
        tot = 0.0
        for start in range(0, len(order), hp2["batch_size"]):
            chunk = [subset[ix] for ix in order[start:start + hp2["batch_size"]]]
            tiles, mask, ctx, _ = collate_bags(
                [get_tiles(s, hp2["train_tiles"], rng, cache) for s in chunk],
                [torch.as_tensor(s.context, dtype=torch.float32) for s in chunk],
                [None] * len(chunk), device)
            yb = torch.tensor([float(s.y) for s in chunk], device=device)
            loss = lossf(model(tiles, ctx, None, mask), yb) / hp2["batch_size"]
            tot += float(loss.detach()) * hp2["batch_size"]
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        tot /= len(order)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            y, s_ = predict(model, subset, "without_smiles", fps, hp2, 0, cache, device, shuffle_map)
            auc = roc_auc_score(y, s_) if len(set(y)) > 1 else float("nan")
            history.append((epoch + 1, tot, auc))
            print(f"  epoch {epoch+1:>3}   train_loss {tot:.4f}   train_AUROC {auc:.3f}")

    best = max(h[2] for h in history)
    first, last = history[0][1], history[-1][1]
    print(f"\n  best training AUROC : {best:.3f}")
    print(f"  loss {first:.4f} -> {last:.4f}  "
          f"({'falling' if last < first * 0.95 else 'NOT falling'})")

    if best > 0.90:
        verdict = ("PASS - the training loop works. It can fit data when the signal is\n"
                   "         there, so the ~0.53 test AUROC is about the DATA, not a bug.\n"
                   "         195 patients with 22 non-responders is likely too little.\n"
                   "         Do not tune. The answer waits for Arm A (354 non-responders).")
    elif best > 0.70:
        verdict = ("PARTIAL - it learns, but cannot memorise 50 patients, which it should\n"
                   "         manage easily. Suspect under-training: too few epochs, learning\n"
                   "         rate too low, or the 512-tile resample adding too much noise.")
    else:
        verdict = ("FAIL - the model cannot fit even 50 patients it is looking straight at.\n"
                   "         This is a BUG, not a data limitation. Check, in order: gradients\n"
                   "         reaching feature_proj, the attention softmax dimension, label\n"
                   "         orientation, and whether get_tiles returns the right bag.")
    print(f"\n  VERDICT: {verdict}\n{'=' * 74}\n")
    return best


# =============================================================================
# Self-test - the reason this harness can be trusted before it sees real labels
# =============================================================================
def _null_dataset(width_mode: str, seed: int, n_patients: int, k: int):
    """
    One null-case dataset (no drug effect) with the drug branch wired the way a
    named configuration wires it.

      'legacy'      raw 1024-bit synthetic fingerprints straight into the drug
                    branch. This is what selftest() has always used, and the
                    recorded reference numbers of 2026-08-28 belong to it.
      'production'  the same synthetic fingerprints, but reduced to k dims by
                    the same frozen PCA the real runs use.
      'chemberta'   REAL pretrained embeddings for REAL molecules, reduced to k.
                    Returns None if the artefact has not been built.

    WHY THIS EXISTS (2026-08-29). Until today selftest() called run_cv on the
    unreduced fingerprints and never touched the reduction, so the pass/fail
    bias gate had never once exercised the width the study actually runs at.
    It was not giving a false pass — 1024 -> 16 is a WIDER, more bias-prone drug
    branch than 32 -> 16, so clearing the gate there is the harder test — but it
    was measuring a configuration no result was ever produced from, while the
    §10 record reads as though it validated the PCA. Both are now measured.
    """
    if width_mode == "chemberta":
        path = BASE_DIR / "drugs" / DRUG_ENCODERS["chemberta"]
        if not path.exists():
            return None
        blob = np.load(path, allow_pickle=True)
        real_names = [str(n) for n in blob["names"]]
        real_X = blob["X"].astype(np.float64)
        samples, _, ed = make_synthetic(n_patients=n_patients, n_drugs=len(real_names),
                                        delta_signal=0.0, seed=seed)
        # relabel the synthetic drugs with real molecules, keeping the assignment
        # deterministic. The LABELS still contain no drug effect, so any non-zero
        # delta is the drug branch fitting chemistry-shaped noise — which is
        # precisely the thing that must not happen.
        remap = {f"drug_{i}": real_names[i] for i in range(len(real_names))}
        for s in samples:
            s.drug = remap[s.drug]
        Z = _frozen_pca(real_X, k)
        fps = {n: torch.from_numpy(Z[i].astype(np.float32))
               for i, n in enumerate(real_names)}
        return samples, fps, ed

    samples, fps, ed = make_synthetic(n_patients=n_patients, delta_signal=0.0,
                                      seed=seed)
    if width_mode == "production":
        fps = reduce_fingerprints(fps, k)
    return samples, fps, ed


def _null_block(title, width_mode, device, hp, arms, n_rep, k, n_patients):
    """Run n_rep null replicates and report mean delta, MC SE and rejections."""
    print(f"\n{title}")
    built = _null_dataset(width_mode, 100, n_patients, k)
    if built is None:
        print("  SKIPPED — pretrained drug embeddings not built. Run:")
        print("      python scripts/11_build_drug_embeddings.py --tag arm_a")
        return None, True
    rejects, ds = 0, []
    for r in range(n_rep):
        s2, f2, e2 = _null_dataset(width_mode, 100 + r, n_patients, k)
        dim = int(next(iter(f2.values())).shape[0])
        res2 = {r: run_cv(s2, f2, e2, arms, hp, seed=r + 1, device=device)}
        rep2 = summarise_run(res2, arms, [r])
        c = rep2["contrasts"]["with_smiles_vs_without_smiles"]
        ds.append(c["delta_auroc_mean"])
        rejects += int(c["delong_p_per_seed"][0] < ALPHA)
        print(f"  null replicate {r+1}/{n_rep} (drug dim {dim}): "
              f"ΔAUROC {c['delta_auroc_mean']:+.4f}  p={c['delong_p_per_seed'][0]:.3f}")
    bias = float(np.mean(ds))
    se = float(np.std(ds, ddof=1) / math.sqrt(n_rep)) if n_rep > 1 else float("nan")
    ok = abs(bias) <= 0.03
    print(f"  mean delta under the null : {bias:+.4f}  (MC standard error {se:.4f})")
    print(f"  DeLong rejections         : {rejects}/{n_rep}  (nominal ~{ALPHA*n_rep:.1f})")
    print(f"  bias check                : {'PASS' if ok else 'FAIL'} "
          f"(|{bias:+.4f}| vs tolerance 0.03)")
    return (bias, se, rejects), ok


def _chemberta_synthetic(seed: int, n_patients: int, delta_signal: float, k: int):
    """Synthetic labels with a KNOWN drug effect, carried by the REAL frozen molecule vectors."""
    path = BASE_DIR / "drugs" / DRUG_ENCODERS["chemberta"]
    blob = np.load(path, allow_pickle=True)
    names = [str(n) for n in blob["names"]]
    samples, _, ed = make_synthetic(n_patients=n_patients, n_drugs=len(names),
                                    delta_signal=delta_signal, seed=seed)
    remap = {f"drug_{i}": names[i] for i in range(len(names))}
    for s in samples:
        s.drug = remap[s.drug]
    Z = _frozen_pca(blob["X"].astype(np.float64), k)
    fps = {n: torch.from_numpy(Z[i].astype(np.float32)) for i, n in enumerate(names)}
    return samples, fps, ed


def _oracle_delta(samples) -> float:
    """AUROC(true logit) - AUROC(true logit without the drug term), per patient."""
    per = {s.patient: (s.y, s.oracle) for s in samples}
    y = np.array([v[0] for v in per.values()])
    full = np.array([v[1][0] for v in per.values()])
    nod = np.array([v[1][1] for v in per.values()])
    return float(roc_auc_score(y, full) - roc_auc_score(y, nod))


def sensitivity_check(device, n_patients=979, n_null=6, target=0.05):
    """
    EXPLORATORY (§10, 2026-09-30). CAN THE DRUG BRANCH SEE A REAL EFFECT?

    For each gate variant, on synthetic labels carried by the real 93 molecule
    vectors, at Arm A's size and with the PRODUCTION epochs / patience / batch /
    lr (only the tile counts are reduced, for speed):

      planted  a drug effect whose ORACLE ΔAUROC is ~= the SESOI (0.05); the
               harness's recovery is reported as a fraction of that oracle
      null     n_null datasets with no drug effect; mean ΔAUROC and MC SE

    Pre-declared selection rule (§10): a variant PASSES if
      |null bias| <= 0.01 or within 2 MC SE of zero,  AND
      planted recovery >= 50% of the oracle ΔAUROC.
    The first passing variant in the order fast_gate, open_init is the one
    carried to the single exploratory real-data run. frozen_prereg is measured
    as the reference. No real label is read here.
    """
    global GATE_VARIANT
    k = HPARAMS["drug_feat_dim"]
    hp = dict(HPARAMS, train_tiles=128, eval_tiles=200)
    arms = ["without_smiles", "with_smiles"]

    cands = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    od = {c: _oracle_delta(_chemberta_synthetic(11, n_patients, c, k)[0]) for c in cands}
    ds = min(cands, key=lambda c: abs(od[c] - target))
    print(f"\nplanted strength {ds} -> oracle ΔAUROC {od[ds]:+.4f} (target {target})")

    out = {"n_patients": n_patients, "n_null": n_null, "delta_signal": ds,
           "oracle_delta": od[ds], "oracle_by_strength": od, "hparams": hp,
           "gate_lr_mult": GATE_LR_MULT, "variants": {}}
    for v in GATE_VARIANTS:
        GATE_VARIANT = v
        print(f"\n=== gate variant: {v} ===")
        GATE_LOG.clear()
        s, f, e = _chemberta_synthetic(11, n_patients, ds, k)
        r = summarise_run({1: run_cv(s, f, e, arms, hp, seed=1, device=device)}, arms, [1])
        dp = r["contrasts"]["with_smiles_vs_without_smiles"]["delta_auroc_mean"]
        g_planted = [x["gate"] for x in GATE_LOG]
        nulls, g_null = [], []
        for rr in range(n_null):
            GATE_LOG.clear()
            s2, f2, e2 = _chemberta_synthetic(200 + rr, n_patients, 0.0, k)
            r2 = summarise_run({1: run_cv(s2, f2, e2, arms, hp, seed=rr + 1, device=device)},
                               arms, [1])
            nulls.append(r2["contrasts"]["with_smiles_vs_without_smiles"]["delta_auroc_mean"])
            g_null += [x["gate"] for x in GATE_LOG]
            print(f"  null {rr + 1}/{n_null}: ΔAUROC {nulls[-1]:+.4f}")
        bias = float(np.mean(nulls))
        se = float(np.std(nulls, ddof=1) / math.sqrt(n_null)) if n_null > 1 else float("nan")
        rec = dp / od[ds] if od[ds] else float("nan")
        passed = (abs(bias) <= 0.01 or abs(bias) <= 2 * se) and rec >= 0.5
        out["variants"][v] = {"planted_delta": dp, "recovery_fraction": rec,
                              "null_deltas": nulls, "null_bias": bias, "null_se": se,
                              "gate_planted_mean_abs": float(np.mean(np.abs(g_planted))),
                              "gate_null_mean_abs": float(np.mean(np.abs(g_null))),
                              "passed": bool(passed)}
        print(f"  planted Δ {dp:+.4f} = {rec:.0%} of oracle · null bias {bias:+.4f} "
              f"(SE {se:.4f}) · mean |gate| planted {np.mean(np.abs(g_planted)):.3f} "
              f"· {'PASS' if passed else 'fail'}")
    GATE_VARIANT = "frozen_prereg"
    chosen = next((v for v in ("fast_gate", "open_init") if out["variants"][v]["passed"]), None)
    out["chosen_variant"] = chosen
    print(f"\nSELECTED for the exploratory real-data run: {chosen or 'NONE — no variant passed'}")
    out["code_sha256"] = CODE_SHA256
    pth = BASE_DIR / "output" / "eval" / f"gate_sensitivity_{CODE_SHA256}.json"
    pth.write_text(json.dumps(out, indent=1, default=float))
    print(f"written {pth}\n")
    return 0


def selftest(device):
    """
    Four checks:

      1. POWER   — plant a real drug effect; the harness must recover a positive
                   ΔAUROC. If it cannot, a real null would be meaningless.
      2. NULL    — plant NO drug effect across several independent datasets; the
                   rejection rate must sit near alpha, not above it. This is what
                   proves the harness is not manufacturing significance.
                   Runs at the LEGACY 1024-d drug width, unchanged, so the
                   numbers recorded on 2026-08-28 remain reproducible.
      3. NULL at the width the study actually runs at (drug_feat_dim).
      4. NULL with the real pretrained molecular embeddings, which is the only
         one of the four that exercises the exact drug branch the primary
         endpoint will use.

    Only the BIAS figures are pass/fail. The DeLong rejection rate is reported
    as a warning: its cause is understood (see DEFECT 2 / --permutation-null)
    and failing on it would invite loosening the threshold.
    """
    hp = dict(HPARAMS, max_epochs=12, patience=4, train_tiles=128, eval_tiles=200)
    arms = ["without_smiles", "with_smiles"]
    ok = True

    print("\n[1/4] SIGNAL — planted drug effect, harness must find it")
    samples, fps, ed = make_synthetic(n_patients=300, delta_signal=1.4, seed=1)
    res = {1: run_cv(samples, fps, ed, arms, hp, seed=1, device=device)}
    rep = summarise_run(res, arms, [1])
    print_report(rep, "SELFTEST 1 — planted effect")
    d = rep["contrasts"]["with_smiles_vs_without_smiles"]["delta_auroc_mean"]
    print(f"  planted effect recovered: ΔAUROC = {d:+.4f}  "
          f"{'PASS' if d > 0 else 'FAIL — harness cannot see a real effect'}")
    ok &= d > 0

    print("\n[2/4] NULL — no drug effect, rejection rate must sit near alpha")
    rejects, ds = 0, []
    n_rep = 6
    for r in range(n_rep):
        s2, f2, e2 = make_synthetic(n_patients=240, delta_signal=0.0, seed=100 + r)
        res2 = {r: run_cv(s2, f2, e2, arms, hp, seed=r + 1, device=device)}
        rep2 = summarise_run(res2, arms, [r])
        c = rep2["contrasts"]["with_smiles_vs_without_smiles"]
        p = c["delong_p_per_seed"][0]
        ds.append(c["delta_auroc_mean"])
        rejects += int(p < ALPHA)
        print(f"  null replicate {r+1}/{n_rep}: ΔAUROC {c['delta_auroc_mean']:+.4f}  p={p:.3f}")
    bias = float(np.mean(ds))
    se = float(np.std(ds, ddof=1) / math.sqrt(n_rep)) if n_rep > 1 else float("nan")
    print(f"\n  mean delta under the null : {bias:+.4f}  (MC standard error {se:.4f})")
    print(f"  DeLong rejections         : {rejects}/{n_rep}  (nominal ~{ALPHA*n_rep:.1f})")

    # CHECK 1 - BIAS. This one is pass/fail. A systematically non-zero delta when
    # no drug effect exists means the two arms are not exchangeable, and the study
    # could report "chemistry hurts" from nothing but parameter count. That would
    # also make the pre-registered null-result fallback (section 8) uninterpretable.
    bias_ok = abs(bias) <= 0.03
    print(f"  bias check                : {'PASS' if bias_ok else 'FAIL'} "
          f"(|{bias:+.4f}| vs tolerance 0.03)")
    if not bias_ok:
        print("      The drug branch is fitting noise. Shrink drug_feat_dim /")
        print("      drug_proj_dim, or swap in a smaller frozen molecular encoder.")

    # CHECK 2 - CALIBRATION. Deliberately NOT pass/fail, because we now know why it
    # fails: DeLong conditions on the scores as if the models were fixed, but every
    # fold retrains them. The excess rejections are that unmodelled variance, not a
    # bug. The fix is --permutation-null, which rebuilds the null with the
    # retraining variance included. Reported loudly so it can never be forgotten.
    if rejects > max(1, ALPHA * n_rep * 2):
        print("\n  NOTE, and this matters for the paper: DeLong rejected "
              f"{rejects}/{n_rep} times under a true null.")
        print("      DeLong's variance assumes FIXED models. Cross-validation "
              "retrains per fold, so its")
        print("      p-values are anti-conservative here. Do not report a single-seed "
              "DeLong p as the")
        print("      primary result. Use --permutation-null for the calibrated "
              "p-value, and report")
        print("      the across-seed spread the plan already requires.")

    ok &= bias_ok

    # ------------------------------------------------------------------------
    # Checks 3 and 4, added 2026-08-29 with the pretrained encoder. Checks 1 and
    # 2 above are deliberately left byte-identical so the reference numbers
    # recorded on 2026-08-28 still reproduce from this file.
    # ------------------------------------------------------------------------
    k = HPARAMS["drug_feat_dim"]
    _, ok3 = _null_block(
        f"[3/4] NULL at PRODUCTION width — synthetic fingerprints reduced to {k}-d",
        "production", device, hp, arms, n_rep=6, k=k, n_patients=240)
    ok &= ok3

    _, ok4 = _null_block(
        f"[4/4] NULL with the REAL pretrained encoder — ChemBERTa -> {k}-d PCA",
        "chemberta", device, hp, arms, n_rep=6, k=k, n_patients=300)
    ok &= ok4

    print(f"\n{'=' * 74}\nSELFTEST {'PASSED' if ok else 'FAILED'}\n{'=' * 74}\n")
    return 0 if ok else 1


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="cohort tag, e.g. brca or arm_a. REAL LABELS.")
    ap.add_argument("--synthetic", action="store_true", help="run on fake vectors")
    ap.add_argument("--selftest", action="store_true", help="validate the harness, then exit")
    ap.add_argument("--delta-signal", type=float, default=0.9,
                    help="synthetic only: size of the planted drug effect (0 = null)")
    ap.add_argument("--seeds", type=int, default=5, help="§4 requires 5")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--arms", default="without_smiles,with_smiles,shuffled_drug,onehot_drug")
    ap.add_argument("--drug-encoder", default=DEFAULT_DRUG_ENCODER,
                    choices=sorted(DRUG_ENCODERS),
                    help="frozen molecular encoder for the drug branch. "
                         "'chemberta' is §6's pretrained encoder and the default; "
                         "'morgan_pca' is the pre-2026-08-29 stand-in, kept only "
                         "to reproduce earlier runs.")
    ap.add_argument("--thin-slide-sensitivity", action="store_true",
                    help="§3.5: repeat excluding patients whose thinnest slide is "
                         f"under {THIN_SLIDE_TILES} tiles")
    ap.add_argument("--include-biologics", action="store_true",
                    help="§10 biologic-inclusive SECONDARY: scoped cohort, biologic "
                         "flags in the context of BOTH arms, zero molecular input "
                         "for patients with no small molecule. Never the primary.")
    ap.add_argument("--negative-start-sensitivity", action="store_true",
                    help="§10 2026-09-19: repeat excluding (patient, drug) records "
                         "with a negative days_to_treatment_start")
    ap.add_argument("--features-dir", default=str(BASE_DIR / "output" / "features"))
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--note", default="", help="free text stored in the run log")
    ap.add_argument("--overfit-check", action="store_true",
                    help="Can the model learn at all? Trains the BASELINE arm on ~50 "
                         "patients with regularisation off and scores it on the same "
                         "patients. Diagnostic only - never computes a with/without "
                         "difference, so it cannot contaminate the primary endpoint.")
    ap.add_argument("--log-epochs", action="store_true",
                    help="Print train loss, val loss and val AUROC every epoch.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load the cohort and the feature store, print what was "
                         "assembled, and EXIT before any model is built. Use this to "
                         "verify the data wiring without a model ever seeing a label.")
    ap.add_argument("--permutation-null", type=int, default=0, metavar="N",
                    help="Re-run the whole CV with PATIENT labels permuted, N times, "
                         "to build a calibrated null for the primary delta. DeLong "
                         "alone is anti-conservative here because it does not model "
                         "the retraining variance of cross-validation. Costs N full "
                         "CV runs. N=200 is the usual minimum for a 0.05 threshold.")
    ap.add_argument("--unit", default="patient", choices=["patient", "pair"],
                    help="training unit. 'patient' (one row per patient, drug input "
                         "pooled over the regimen) is the pre-registered unit from "
                         "2026-09-29; 'pair' is the old per-(patient, drug) unit, kept "
                         "only as a regression baseline.")
    ap.add_argument("--gate-variant", default="frozen_prereg", choices=list(GATE_VARIANTS),
                    help="EXPLORATORY: drug-gate variant (§10 2026-09-30). The default is "
                         "the pre-registered model; anything else is labelled in the run "
                         "log source and can never be the primary.")
    ap.add_argument("--sensitivity-check", action="store_true",
                    help="EXPLORATORY: synthetic test of whether each gate variant can "
                         "recover a planted effect of the SESOI's size (no real labels)")
    ap.add_argument("--curtail-null", action="store_true",
                    help="stop the permutation null once the fixed-N 'do not reject' "
                         "decision is mathematically determined (Besag & Clifford 1991). "
                         "Cannot change the decision; declared in §10 2026-09-30.")
    ap.add_argument("--quick", action="store_true",
                    help="SMOKE TEST ONLY — shrinks epochs and tiles so a run finishes "
                         "in minutes. Never use for a reported result; the run log "
                         "records it so a quick run can never be mistaken for a real one.")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device(a.device)
    global TRAINING_UNIT
    TRAINING_UNIT = a.unit

    global GATE_VARIANT
    GATE_VARIANT = a.gate_variant
    if a.sensitivity_check:
        sys.exit(sensitivity_check(device))
    if a.selftest:
        sys.exit(selftest(device))

    if not a.synthetic and not a.tag:
        ap.error("pass --tag <cohort> for real labels, or --synthetic. "
                 "Debug on --synthetic first; see the module docstring.")

    hp = dict(HPARAMS)
    if a.log_epochs:
        hp["log_epochs"] = True
    if a.quick:
        hp.update(max_epochs=8, patience=3, train_tiles=128, eval_tiles=256)
        log.warning("  --quick: reduced hyperparameters. SMOKE TEST ONLY.")

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if "without_smiles" not in arms:
        ap.error("without_smiles is the baseline for the primary endpoint — keep it")

    t0 = time.time()
    if a.synthetic:
        log.info("Loading SYNTHETIC data (no real labels touched)")
        samples, fps, embed_dim = make_synthetic(delta_signal=a.delta_signal, seed=0)
        source = f"synthetic(delta_signal={a.delta_signal})"
    else:
        log.info(f"Loading REAL data — cohort tag '{a.tag}'")
        log.warning("  These are real labels. Every run from here is logged (§8).")
        samples, fps, embed_dim = load_real(a.tag, Path(a.features_dir),
                                            include_biologics=a.include_biologics)
        source = (f"real(tag={a.tag},biologic_inclusive)" if a.include_biologics
                  else f"real(tag={a.tag})")
    if GATE_VARIANT != "frozen_prereg":
        source = source[:-1] + f",EXPLORATORY_gate={GATE_VARIANT})"

    # Synthetic data has no real molecules behind its fake fingerprints, so the
    # pretrained artefact cannot apply to it; the PCA path is the only coherent
    # choice there and the self-test relies on that.
    enc = "morgan_pca" if a.synthetic else a.drug_encoder
    if a.synthetic and a.drug_encoder != "morgan_pca":
        log.info("  (synthetic data: drug encoder forced to morgan_pca)")
    fps, drug_encoder_prov = build_drug_features(fps, hp["drug_feat_dim"], enc)

    n_pat = len({s.patient for s in samples})
    n_pos = len({s.patient for s in samples if s.y == 1})
    log.info(f"  {len(samples)} (patient, drug) samples · {n_pat} patients "
             f"({n_pos} non-responder / {n_pat - n_pos} responder) · "
             f"{len(fps)} drugs · embed_dim {embed_dim}")

    if a.overfit_check:
        # NOTE: fps was already reduced above. Do not reduce twice - a second PCA
        # on already-reduced features re-standardises them. Harmless for this
        # check (it uses without_smiles, so no drug input at all) but wrong if
        # the drug arms are ever wired in here.
        overfit_check(samples, fps, embed_dim, hp, device)
        return

    if a.dry_run:
        from collections import Counter
        per_proj = Counter(s.project for s in samples)
        thin = len({s.patient for s in samples if s.min_tiles < THIN_SLIDE_TILES})
        drugs = Counter(s.drug for s in samples)
        print("\nDRY RUN — data assembled, nothing trained, no label seen by a model")
        print(f"  patients            : {n_pat}  ({n_pos} non-responder / {n_pat-n_pos} responder)")
        print(f"  (patient, drug) rows: {len(samples)}")
        print(f"  training unit       : {TRAINING_UNIT} -> "
              f"{len(to_training_unit(samples))} training rows")
        print(f"  distinct molecules  : {len({molecule_of(n) for n in fps})}")
        print(f"  drugs with structure: {len(fps)}")
        print(f"  drug embedding dim  : {int(next(iter(fps.values())).shape[0])}")
        print(f"  tissue context dim  : {samples[0].context.size}")
        print(f"  thin-slide patients : {thin} (under {THIN_SLIDE_TILES} tiles)")
        print(f"  negative-start recs : {sum(s.negative_start for s in samples)}")
        print(f"  rows per project    : {dict(per_proj.most_common())}")
        print(f"  patients per project: "
              f"{dict(Counter(s.project for s in to_training_unit(samples)).most_common())}")
        print(f"  top drugs (rows)    : {dict(drugs.most_common(8))}")
        print("\n  Check these against your cohort summary JSON before running for real.\n")
        return

    seeds = list(range(1, a.seeds + 1))
    results = {}
    for seed in seeds:
        log.info(f"\n  === seed {seed}/{len(seeds)} ===")
        results[seed] = run_cv(samples, fps, embed_dim, arms, hp, seed,
                               device, n_splits=a.folds)

    rep = summarise_run(results, arms, seeds)
    print_report(rep, f"PRIMARY ANALYSIS — {source}")

    payload = {"source": source, "arms": arms, "seeds": seeds, "folds": a.folds,
               "hparams": hp, "quick_smoke_test": bool(a.quick),
               "drug_encoder": drug_encoder_prov,
               "n_samples": len(samples), "n_patients": n_pat,
               "training_unit": TRAINING_UNIT,
               "biologics": dict(BIOLOGIC_INFO) or None,
               "n_training_rows": len(to_training_unit(samples)),
               "device": a.device, "note": a.note, "report": rep,
               "elapsed_s": round(time.time() - t0, 1),
               "gate_variant": GATE_VARIANT, "gate_values": GATE_LOG}

    # §8 requires that every configuration trained against labels be logged and
    # reported. Hyperparameters alone do NOT identify a configuration: on
    # 2026-08-28 the pre-batch and batched builds of this file produced AUROC
    # 0.5912 and 0.5783 on identical data, seeds and hparams, and both logged
    # the same config hash. The digest of this source file is therefore part of
    # the run's identity — and so is the drug encoder, added 2026-08-29: two runs
    # differing only in --drug-encoder share every hyperparameter and every byte
    # of this file, so without it they would collide in exactly the same way.
    payload["receptor_context"] = bool(RECEPTOR_CONTEXT_USED)
    payload["code_sha256"] = CODE_SHA256
    payload["config_hash"] = hashlib.sha256(
        json.dumps({k: payload[k] for k in ("source", "arms", "seeds", "folds",
                                            "hparams", "code_sha256",
                                            "drug_encoder", "training_unit")},
                   sort_keys=True, default=str).encode()).hexdigest()[:12]
    extra_preds: dict = {}

    def _save(status: str):
        """
        Write the predictions and a checkpoint of the payload NOW (2026-09-30).
        Until this change nothing was written until the permutation null had
        finished — days after the primary result had already been printed — so
        stopping a run lost its record. The run log itself still gets exactly
        one line per run: on completion, or on Ctrl-C with status "interrupted".
        """
        def _preds(res):
            return {str(seed): {arm: {"patients": list(pats), "y": [int(v) for v in yv],
                                      "score": [float(v) for v in sv]}
                                for arm, (yv, sv, pats) in by_arm.items()}
                    for seed, by_arm in res.items()}
        preds = {"config_hash": payload["config_hash"], "source": source,
                 "training_unit": TRAINING_UNIT, "primary": _preds(results)}
        for k, res in extra_preds.items():
            preds[k] = _preds(res)
        pp = BASE_DIR / "output" / "eval" / f"preds_{payload['config_hash']}.json"
        pp.write_text(json.dumps(preds))
        payload["predictions_file"] = pp.name
        payload["status"] = status
        payload["elapsed_s"] = round(time.time() - t0, 1)
        if payload.get("permutation_null"):
            payload["permutation_null"]["n_completed"] = len(payload["permutation_null"]["null_deltas"])
        ck = BASE_DIR / "output" / "eval" / f"checkpoint_{payload['config_hash']}.json"
        ck.write_text(json.dumps(payload, default=str))
        return pp, ck

    pp, ck = _save("primary_done")
    log.info(f"Primary predictions written to {pp}; checkpoint {ck.name}")

    try:
        if a.thin_slide_sensitivity and not a.synthetic:
            kept = [s for s in samples if s.min_tiles >= THIN_SLIDE_TILES]
            dropped = n_pat - len({s.patient for s in kept})
            log.info(f"\n  === thin-slide sensitivity: excluding {dropped} patients "
                     f"with a slide under {THIN_SLIDE_TILES} tiles ===")
            res2 = {seed: run_cv(kept, fps, embed_dim, arms, hp, seed, device,
                                 n_splits=a.folds) for seed in seeds}
            extra_preds["sensitivity_thin_slide"] = res2
            rep2 = summarise_run(res2, arms, seeds)
            print_report(rep2, f"SENSITIVITY (thin slides excluded) — {source}")
            payload["sensitivity_thin_slide"] = {"n_patients_dropped": dropped, "report": rep2}

        if a.negative_start_sensitivity and not a.synthetic:
            # §10 2026-09-19: repeat the primary without the (patient, drug) records
            # whose days_to_treatment_start is negative. Records are removed BEFORE
            # the regimen is formed, so a patient keeps their other drugs and drops
            # out only if nothing is left.
            kept_n = [s for s in samples if not s.negative_start]
            n_rec = len(samples) - len(kept_n)
            dropped_n = n_pat - len({s.patient for s in kept_n})
            log.info(f"\n  === treatment-timing sensitivity: {n_rec} records with negative "
                     f"days_to_treatment_start removed ({dropped_n} patients lost) ===")
            res3 = {seed: run_cv(kept_n, fps, embed_dim, arms, hp, seed, device,
                                 n_splits=a.folds) for seed in seeds}
            extra_preds["sensitivity_negative_start"] = res3
            rep3 = summarise_run(res3, arms, seeds)
            print_report(rep3, f"SENSITIVITY (negative treatment start removed) — {source}")
            payload["sensitivity_negative_start"] = {"n_records_removed": n_rec,
                                                     "n_patients_dropped": dropped_n,
                                                     "report": rep3}

        if a.permutation_null and "with_smiles" in arms:
            obs = rep["contrasts"]["with_smiles_vs_without_smiles"]["delta_auroc_mean"]
            log.info(f"\n  === permutation null: {a.permutation_null} runs "
                     f"(observed delta = {obs:+.4f}) ===")
            nulls = []           # filled as it goes, so an interrupt keeps the partial null
            payload["permutation_null"] = {"n": a.permutation_null, "observed": obs,
                                           "null_deltas": nulls, "p": None,
                                           "n_completed": 0}
            permutation_null(samples, fps, embed_dim, arms, hp, device,
                             a.permutation_null, a.folds, out=nulls,
                             observed=obs, curtail=a.curtail_null)
            h = sum(1 for x in nulls if abs(x) >= abs(obs))
            if len(nulls) < a.permutation_null:
                # curtailed: report the Besag–Clifford p and the bound on the fixed-N p
                pv = h / len(nulls)
                payload["permutation_null"].update(
                    curtailed=True, exceedances=h,
                    fixed_n_p_lower_bound=(h + 1) / (a.permutation_null + 1),
                    p_method="Besag-Clifford sequential Monte Carlo (h/L)")
            else:
                pv = permutation_p(obs, nulls)
            print(f"\nPERMUTATION NULL ({len(nulls)} of {a.permutation_null} runs"
                  f"{', curtailed' if len(nulls) < a.permutation_null else ''})")
            print(f"  observed delta AUROC : {obs:+.4f}")
            print(f"  null mean / sd       : {np.mean(nulls):+.4f} / {np.std(nulls, ddof=1):.4f}")
            print(f"  permutation p        : {pv:.4f}   "
                  f"({'reject' if pv < ALPHA else 'do not reject'} at alpha={ALPHA})")
            print("  This p-value, not the DeLong one, is the calibrated test when the "
                  "scores come from models retrained per fold.\n")
            payload["permutation_null"].update(p=pv, n_completed=len(nulls))

    except KeyboardInterrupt:
        pp, ck = _save("interrupted")
        p = append_run_log(payload)
        log.warning(f"\nINTERRUPTED — partial run logged to {p} (config "
                    f"{payload['config_hash']}, status 'interrupted'); predictions in {pp.name}")
        raise SystemExit(130)

    pp, ck = _save("complete")
    log.info(f"Per-patient predictions written to {pp}")
    p = append_run_log(payload)
    log.info(f"Run logged to {p} (config {payload['config_hash']})")


if __name__ == "__main__":
    main()
