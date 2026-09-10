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
        with_smiles      + Morgan fingerprint of the administered drug
        shuffled_drug    + a fingerprint of a RANDOM drug   (capacity control)
        onehot_drug      + a one-hot drug ID                (identity control)
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
from dataclasses import dataclass, field, asdict
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
    "dropout":        0.25,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "max_epochs":     40,
    "patience":       8,     # early-stopping patience on inner-val AUROC
    "batch_size":     16,    # bags per optimiser step (gradient accumulation)
    "train_tiles":    512,   # tiles sampled per bag per epoch (augmentation)
    "eval_tiles":     2048,  # tiles used at inference, seeded and fixed
    "inner_val_frac": 0.15,
}

FINGERPRINT_BITS = 1024      # must match 04_cross_modal_fusion.py
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
                       min_tiles=tiles)
            s.bag = torch.from_numpy(bag)
            samples.append(s)

    fps = {f"drug_{d}": torch.from_numpy(drug_vecs[d].astype(np.float32))
           for d in range(n_drugs)}
    return samples, fps, embed_dim


# =============================================================================
# Real data
# =============================================================================
def load_real(tag: str, features_dir: Path):
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

    samples, skipped_no_feat, skipped_no_fp = [], 0, 0
    for bc, rec in cohort.items():
        entries = by_patient.get(bc)
        if not entries:
            skipped_no_feat += 1
            continue
        ctx = np.zeros(len(stages) + len(projects), dtype=np.float32)
        ctx[stage_ix[rec.get("stage") or "unknown"]] = 1.0
        ctx[len(stages) + proj_ix[rec.get("project") or "unknown"]] = 1.0

        paths = [BASE_DIR / e["feature_path"] for e in entries]
        min_tiles = min(int(e.get("n_tiles", 0)) for e in entries)
        y = 1 - int(rec["response_binary"])          # NON-RESPONDER is positive

        for drug in rec["drugs"]:
            if drug not in fps:
                skipped_no_fp += 1
                continue
            samples.append(Sample(patient=bc, project=rec.get("project", "unknown"),
                                  drug=drug, y=y, context=ctx,
                                  bag_paths=paths, min_tiles=min_tiles))

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

    This is a stand-in for the frozen pretrained molecular encoder section 6
    requires (a chemical LM or molecular GNN). Swapping in real embeddings means
    replacing this function and nothing else: the interface is
    {drug_name: vector}.
    """
    names = sorted(fps)
    X = np.stack([fps[n].numpy() for n in names]).astype(np.float64)
    k = int(min(k, len(names) - 1, X.shape[1]))
    if k < 1:
        return dict(fps)
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # deterministic, no labels
    Z = Xc @ Vt[:k].T
    Z = Z / (Z.std(0, keepdims=True) + 1e-8)            # unit-variance components
    log.info(f"  drug encoder: {X.shape[1]}-bit Morgan -> {k}-d frozen PCA "
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
            self.drug_gate = nn.Parameter(torch.zeros(1))
            head_in += hp["drug_proj_dim"]

        self.head = nn.Sequential(
            nn.Linear(head_in, h // 2), nn.ReLU(), nn.Dropout(p), nn.Linear(h // 2, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tiles, ctx, drug=None):
        h = self.feature_proj(tiles)                                   # [N, h]
        A = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h)))
        a = torch.softmax(A, dim=0)                                    # [N, 1]
        z = (a * h).sum(dim=0)                                         # [h]
        parts = [z, ctx]
        if self.use_drug:
            parts.append(self.drug_gate * self.drug_proj(drug))
        return self.head(torch.cat(parts)).squeeze(-1)                 # scalar logit


# =============================================================================
# Bag loading
# =============================================================================
def get_tiles(s: Sample, n_tiles: int, rng: np.random.Generator, cache: dict) -> torch.Tensor:
    """Pool all of a patient's slides, then take a seeded random subsample."""
    if s.bag is not None:
        bag = s.bag
    else:
        key = tuple(str(p) for p in s.bag_paths)
        bag = cache.get(key)
        if bag is None:
            bag = torch.cat([torch.load(p, map_location="cpu").float() for p in s.bag_paths])
            cache[key] = bag
    if bag.shape[0] <= n_tiles:
        return bag
    idx = rng.choice(bag.shape[0], n_tiles, replace=False)
    return bag[torch.from_numpy(np.sort(idx))]


# =============================================================================
# Training one fold
# =============================================================================
def train_fold(train_s, val_s, arm, fps, embed_dim, hp, seed, cache, device, shuffle_map):
    """Train one model on one training fold. Early stops on inner-val AUROC."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    drug_dim = drug_dim_for(arm, fps)
    model = ResponseModel(embed_dim, train_s[0].context.size, drug_dim, hp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    n_pos = sum(s.y for s in train_s)
    pos_weight = torch.tensor([(len(train_s) - n_pos) / max(n_pos, 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(hp["max_epochs"]):
        model.train()
        order = rng.permutation(len(train_s))
        opt.zero_grad()
        epoch_loss = 0.0
        for i, ix in enumerate(order):
            s = train_s[ix]
            logit = model(get_tiles(s, hp["train_tiles"], rng, cache).to(device),
                          torch.as_tensor(s.context, dtype=torch.float32).to(device),
                          drug_vec(s, arm, fps, shuffle_map, device))
            loss = lossf(logit.view(1), torch.tensor([float(s.y)], device=device)) / hp["batch_size"]
            epoch_loss += float(loss.detach()) * hp["batch_size"]
            loss.backward()
            if (i + 1) % hp["batch_size"] == 0 or i == len(order) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); opt.zero_grad()

        epoch_loss /= max(len(order), 1)
        yv, sv = predict(model, val_s, arm, fps, hp, seed, cache, device, shuffle_map)
        auc = roc_auc_score(yv, sv) if len(set(yv)) > 1 else 0.5
        if hp.get("log_epochs"):
            # Diagnostic only. Val AUROC on ~23 patients with 2-3 positives is a
            # near-coin-flip; val LOSS is continuous and is the honest signal of
            # whether anything is being learned at all.
            vloss = float(np.mean([
                -(y * np.log(max(p, 1e-9)) + (1 - y) * np.log(max(1 - p, 1e-9)))
                for y, p in zip(yv, sv)]))
            log.info(f"      epoch {epoch+1:>3}  train_loss {epoch_loss:.4f}  "
                     f"val_loss {vloss:.4f}  val_auroc {auc:.3f}"
                     f"{'  *best' if auc > best_auc + 1e-5 else ''}")
        if auc > best_auc + 1e-5:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def drug_dim_for(arm, fps):
    if arm == "without_smiles":
        return None
    if arm == "onehot_drug":
        return len(fps)
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
    name = shuffle_map[id(s)] if arm == "shuffled_drug" else s.drug
    if arm == "onehot_drug":
        v = torch.zeros(len(fps))
        v[sorted(fps).index(name)] = 1.0
        return v.to(device)
    return fps[name].to(device)


@torch.no_grad()
def predict(model, samples, arm, fps, hp, seed, cache, device, shuffle_map):
    """Deterministic inference: eval-time tiles are drawn from a FIXED seed."""
    model.eval()
    ys, ss = [], []
    for s in samples:
        # tile subsample is seeded PER SAMPLE so inference is deterministic and
        # identical across arms — the arms must differ only in the drug branch
        rng = np.random.default_rng(abs(hash(s.patient)) % (2**32) + seed)
        logit = model(get_tiles(s, hp["eval_tiles"], rng, cache).to(device),
                      torch.as_tensor(s.context, dtype=torch.float32).to(device),
                      drug_vec(s, arm, fps, shuffle_map, device))
        ys.append(s.y); ss.append(float(torch.sigmoid(logit)))
    return np.array(ys), np.array(ss)


# =============================================================================
# Cross-validation
# =============================================================================
def run_cv(samples, fps, embed_dim, arms, hp, seed, device, n_splits=5):
    """
    5-fold, grouped by patient, stratified by (response class x cancer type).

    Returns {arm: (y_patient, score_patient, patient_ids)} — pooled out-of-fold
    predictions with every patient scored exactly ONCE.
    """
    groups = np.array([s.patient for s in samples])
    strat = np.array([f"{s.y}|{s.project}" for s in samples])
    # StratifiedGroupKFold needs integer-ish y; map the combined key
    keys = {k: i for i, k in enumerate(sorted(set(strat)))}
    y_strat = np.array([keys[k] for k in strat])

    cache: dict = {}
    # fixed random drug per sample for the capacity control (§6), one draw per seed
    _srng = np.random.default_rng(seed + 99991)
    drug_names = sorted(fps)
    shuffle_map = {id(s): drug_names[int(_srng.integers(len(drug_names)))] for s in samples}
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
            model, val_auc = train_fold(train_s, val_s, arm, fps, embed_dim,
                                        hp, seed, cache, device, shuffle_map)
            _, probs = predict(model, test_s, arm, fps, hp, seed, cache, device, shuffle_map)
            for s, p in zip(test_s, probs):
                oof[arm][s.patient].append(float(p))
            log.info(f"    fold {fold} · {arm:<15} inner-val AUROC {val_auc:.3f}")

    out = {}
    for arm in arms:
        pats = sorted(oof[arm])
        out[arm] = (np.array([labels[p] for p in pats]),
                    np.array([float(np.mean(oof[arm][p])) for p in pats]),
                    pats)
    return out


def permutation_null(samples, fps, embed_dim, arms, hp, device, n_perm, n_splits, seed0=5000):
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

    deltas = []
    for i in range(n_perm):
        shuffled = list(base_labels.values())
        rng.shuffle(shuffled)
        newlab = dict(zip(patients, shuffled))
        perm = []
        for s in samples:
            s2 = copy.copy(s)
            s2.y = newlab[s.patient]
            perm.append(s2)
        res = {1: run_cv(perm, fps, embed_dim, arms, hp, seed0 + i, device, n_splits)}
        rep = summarise_run(res, arms, [1])
        d = rep["contrasts"]["with_smiles_vs_without_smiles"]["delta_auroc_mean"]
        deltas.append(d)
        log.info(f"    permutation {i+1}/{n_perm}: delta {d:+.4f}")
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
            "capacity_matched": base != "without_smiles",
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
    lossf = nn.BCEWithLogitsLoss()
    cache, shuffle_map = {}, {}

    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(subset))
        opt.zero_grad(); tot = 0.0
        for i, ix in enumerate(order):
            s = subset[ix]
            logit = model(get_tiles(s, hp2["train_tiles"], rng, cache).to(device),
                          torch.as_tensor(s.context, dtype=torch.float32).to(device),
                          None)
            loss = lossf(logit.view(1), torch.tensor([float(s.y)], device=device)) / hp2["batch_size"]
            tot += float(loss.detach()) * hp2["batch_size"]
            loss.backward()
            if (i + 1) % hp2["batch_size"] == 0 or i == len(order) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); opt.zero_grad()
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
def selftest(device):
    """
    Two checks, both on synthetic vectors:

      1. POWER   — plant a real drug effect; the harness must recover a positive
                   ΔAUROC. If it cannot, a real null would be meaningless.
      2. NULL    — plant NO drug effect across several independent datasets; the
                   rejection rate must sit near alpha, not above it. This is what
                   proves the harness is not manufacturing significance.
    """
    hp = dict(HPARAMS, max_epochs=12, patience=4, train_tiles=128, eval_tiles=200)
    arms = ["without_smiles", "with_smiles"]
    ok = True

    print("\n[1/2] SIGNAL — planted drug effect, harness must find it")
    samples, fps, ed = make_synthetic(n_patients=300, delta_signal=1.4, seed=1)
    res = {1: run_cv(samples, fps, ed, arms, hp, seed=1, device=device)}
    rep = summarise_run(res, arms, [1])
    print_report(rep, "SELFTEST 1 — planted effect")
    d = rep["contrasts"]["with_smiles_vs_without_smiles"]["delta_auroc_mean"]
    print(f"  planted effect recovered: ΔAUROC = {d:+.4f}  "
          f"{'PASS' if d > 0 else 'FAIL — harness cannot see a real effect'}")
    ok &= d > 0

    print("\n[2/2] NULL — no drug effect, rejection rate must sit near alpha")
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
    ap.add_argument("--thin-slide-sensitivity", action="store_true",
                    help="§3.5: repeat excluding patients whose thinnest slide is "
                         f"under {THIN_SLIDE_TILES} tiles")
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
    ap.add_argument("--quick", action="store_true",
                    help="SMOKE TEST ONLY — shrinks epochs and tiles so a run finishes "
                         "in minutes. Never use for a reported result; the run log "
                         "records it so a quick run can never be mistaken for a real one.")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device(a.device)

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
        samples, fps, embed_dim = load_real(a.tag, Path(a.features_dir))
        source = f"real(tag={a.tag})"

    fps = reduce_fingerprints(fps, hp["drug_feat_dim"])

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
        print(f"  drugs with structure: {len(fps)}")
        print(f"  drug embedding dim  : {int(next(iter(fps.values())).shape[0])}")
        print(f"  tissue context dim  : {samples[0].context.size}")
        print(f"  thin-slide patients : {thin} (under {THIN_SLIDE_TILES} tiles)")
        print(f"  per project         : {dict(per_proj.most_common())}")
        print(f"  top drugs           : {dict(drugs.most_common(8))}")
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
               "n_samples": len(samples), "n_patients": n_pat,
               "device": a.device, "note": a.note, "report": rep,
               "elapsed_s": round(time.time() - t0, 1)}

    if a.thin_slide_sensitivity and not a.synthetic:
        kept = [s for s in samples if s.min_tiles >= THIN_SLIDE_TILES]
        dropped = n_pat - len({s.patient for s in kept})
        log.info(f"\n  === thin-slide sensitivity: excluding {dropped} patients "
                 f"with a slide under {THIN_SLIDE_TILES} tiles ===")
        res2 = {seed: run_cv(kept, fps, embed_dim, arms, hp, seed, device,
                             n_splits=a.folds) for seed in seeds}
        rep2 = summarise_run(res2, arms, seeds)
        print_report(rep2, f"SENSITIVITY (thin slides excluded) — {source}")
        payload["sensitivity_thin_slide"] = {"n_patients_dropped": dropped, "report": rep2}

    if a.permutation_null and "with_smiles" in arms:
        obs = rep["contrasts"]["with_smiles_vs_without_smiles"]["delta_auroc_mean"]
        log.info(f"\n  === permutation null: {a.permutation_null} runs "
                 f"(observed delta = {obs:+.4f}) ===")
        nulls = permutation_null(samples, fps, embed_dim, arms, hp, device,
                                 a.permutation_null, a.folds)
        pv = permutation_p(obs, nulls)
        print(f"\nPERMUTATION NULL ({a.permutation_null} runs)")
        print(f"  observed delta AUROC : {obs:+.4f}")
        print(f"  null mean / sd       : {np.mean(nulls):+.4f} / {np.std(nulls, ddof=1):.4f}")
        print(f"  permutation p        : {pv:.4f}   "
              f"({'reject' if pv < ALPHA else 'do not reject'} at alpha={ALPHA})")
        print("  This p-value, not the DeLong one, is the calibrated test when the "
              "scores come from models retrained per fold.\n")
        payload["permutation_null"] = {"n": a.permutation_null, "observed": obs,
                                       "null_deltas": nulls, "p": pv}

    payload["config_hash"] = hashlib.sha256(
        json.dumps({k: payload[k] for k in ("source", "arms", "seeds", "folds", "hparams")},
                   sort_keys=True, default=str).encode()).hexdigest()[:12]
    p = append_run_log(payload)
    log.info(f"Run logged to {p} (config {payload['config_hash']})")


if __name__ == "__main__":
    main()
