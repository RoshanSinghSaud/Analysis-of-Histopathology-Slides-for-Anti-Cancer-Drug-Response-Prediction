#!/usr/bin/env python3
"""
=============================================================================
09_tune_baseline.py — THE ONE PERMITTED TUNING PASS
=============================================================================

§6 of the analysis plan: "Hyperparameters are selected ONCE, on the
without-SMILES model, and then frozen and reused unchanged for the with-SMILES
model." This script is that one pass. Running it a second time after seeing a
delta would be the exact post-hoc selection the pre-registration exists to
prevent.

WHAT IS FIXED IN ADVANCE, and was written into §10 of
docs/analysis_plan_preregistration.md on 2026-08-27 BEFORE this file existed:

  * the grid            dropout {0.25, 0.45}
                        weight_decay {1e-2, 1e-1, 1.0}
                        max_epochs {15, 40}          -> 12 configurations

    The weight_decay RANGE was widened on 2026-08-28, before any real tuning
    run, and logged in §10. AdamW's decoupled decay multiplies each weight by
    (1 - lr*wd) per step; at lr=3e-4 over a full ~1,320-step BRCA run the
    originally declared range leaves 99.996% to 99.6% of every weight, i.e. it
    does nothing. Measured on synthetic vectors: mean inner-val loss 0.4884 at
    wd=1e-4 and 0.4884 at wd=1e-2. The new range retains 99.6% / 96% / 67%.
    Label-blind, no real outcome was involved, and the grid keeps its shape.
  * the selection rule  mean POS-WEIGHTED INNER-validation loss, pooled over
                        folds and seeds. Lowest wins.
  * the arm             without_smiles ONLY
  * what gets no budget drug_feat_dim, drug_proj_dim, lr, hidden_dim, attn_dim

WHY THE SELECTION RULE IS SAFE. `run_cv` returns pooled out-of-fold predictions
-- the thing the primary endpoint is computed from. This script never binds that
return value. It reads only the `val_log` sink, which carries inner-validation
numbers from a split taken INSIDE each training fold. The outer test fold is
never consulted, so nothing here can select a configuration for scoring well on
the data the paper will be judged on.

The out-of-fold AUROC is nevertheless recorded in the JSON output, marked as
unused, so that a reviewer can verify no selection tracked it. Acting on it
would be a §10 deviation and it must not happen.

Usage
-----
    python scripts/09_tune_baseline.py --tag brca --device mps            # 3 seeds
    python scripts/09_tune_baseline.py --tag brca --device mps --seeds 1  # faster
    python scripts/09_tune_baseline.py --tag brca --dry-run               # print the grid
=============================================================================
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))

import importlib
ev = importlib.import_module("08_evaluate_ablation")

log = logging.getLogger("tune")

# =============================================================================
# THE PRE-DECLARED GRID. Do not edit. Changing it is a §10 deviation.
# =============================================================================
GRID = {
    "dropout":      [0.25, 0.45],
    "weight_decay": [1e-2, 1e-1, 1.0],   # widened 2026-08-28, see docstring + §10
    "max_epochs":   [15, 40],
}
ARM = "without_smiles"


def grid_configs():
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, combo))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="brca", help="cohort tag. REAL LABELS.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--features-dir", default=str(BASE_DIR / "output" / "features"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and exit before any model is built")
    ap.add_argument("--synthetic", action="store_true",
                    help="SMOKE TEST the driver on fake vectors — no real label is "
                         "touched and the winner it prints means nothing. Use this "
                         "to check the script runs before spending an hour on it.")
    ap.add_argument("--quick", action="store_true",
                    help="SMOKE TEST ONLY — shrink tiles so the grid finishes in "
                         "minutes. A result from this must never be frozen.")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configs = list(grid_configs())

    print("=" * 74)
    print("BASELINE TUNING — the ONE pass §6 permits")
    print("=" * 74)
    print(f"  arm               : {ARM} only")
    print(f"  configurations    : {len(configs)}")
    print(f"  seeds x folds     : {a.seeds} x {a.folds}")
    print(f"  total trainings   : {len(configs) * a.seeds * a.folds}")
    print(f"  selection metric  : mean POS-WEIGHTED INNER-validation loss (lower is better)")
    print(f"  NEVER consulted   : out-of-fold predictions, outer-fold AUROC")
    for i, c in enumerate(configs, 1):
        print(f"    {i:>2}. " + "  ".join(f"{k}={v}" for k, v in c.items()))
    print()
    if a.dry_run:
        print("  --dry-run: exiting before any model is built.")
        return 0

    device = torch.device(a.device)
    if a.synthetic:
        print("  ⚠️  --synthetic: SMOKE TEST. No real label is touched and the "
              "winner below means NOTHING.\n")
        samples, fps, embed_dim = ev.make_synthetic(n_patients=120, tiles=120, seed=0)
    else:
        samples, fps, embed_dim = ev.load_real(a.tag, Path(a.features_dir))
    fps = ev.reduce_fingerprints(fps, ev.HPARAMS["drug_feat_dim"])
    seeds = list(range(1, a.seeds + 1))

    results, t0 = [], time.time()
    for i, cfg in enumerate(configs, 1):
        hp = dict(ev.HPARAMS, **cfg)
        if a.quick or a.synthetic:
            hp.update(train_tiles=64, eval_tiles=96)
        # patience stays at the frozen HPARAMS value. §10 declared three knobs
        # and patience is not one of them; deriving it from max_epochs would be
        # a fourth, undeclared knob. max_epochs is a CAP, exactly as declared.
        label = "  ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"\n[{i}/{len(configs)}] {label}   (patience {hp['patience']})")

        val_log, oof_auroc, t1 = [], [], time.time()
        for seed in seeds:
            out = ev.run_cv(samples, fps, embed_dim, [ARM], hp, seed, device,
                            n_splits=a.folds, val_log=val_log)
            # recorded for audit ONLY -- see the module docstring
            y, sc, _ = out[ARM]
            oof_auroc.append(float(ev.roc_auc_score(y, sc)) if len(set(y.tolist())) > 1 else float("nan"))

        losses = [r["inner_val_loss"] for r in val_log]
        rec = {
            "config": cfg,
            "patience": hp["patience"],
            "mean_inner_val_loss": float(np.mean(losses)),
            "sd_inner_val_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            "n_inner_val_points": len(losses),
            "seconds": round(time.time() - t1, 1),
            "oof_auroc_RECORDED_FOR_AUDIT_NOT_USED_FOR_SELECTION": oof_auroc,
        }
        results.append(rec)
        print(f"     mean inner-val loss {rec['mean_inner_val_loss']:.4f} "
              f"± {rec['sd_inner_val_loss']:.4f}   ({rec['seconds']:.0f}s)")
        # checkpoint after every configuration. This run is ~90 minutes and is
        # permitted ONCE; losing it to a crash at config 11 would be a bad way
        # to discover that the results were only written at the end.
        _partial = BASE_DIR / "output" / "eval" / f"tuning_baseline_{a.tag}.partial.json"
        _partial.parent.mkdir(parents=True, exist_ok=True)
        _partial.write_text(json.dumps(
            {"completed": len(results), "of": len(configs), "results": results}, indent=2))

        if a.synthetic:
            continue
        ev.append_run_log({
            "source": f"tuning_grid(tag={a.tag})", "purpose": "baseline_tuning_grid_S6",
            "arms": [ARM], "seeds": seeds, "folds": a.folds, "hparams": hp,
            "quick_smoke_test": bool(a.quick or a.synthetic), "device": a.device,
            "receptor_context": bool(getattr(ev, "RECEPTOR_CONTEXT_USED", False)),
            "selection_metric": "mean_pos_weighted_inner_val_loss",
            "mean_inner_val_loss": rec["mean_inner_val_loss"],
            "code_sha256_tuner": ev.hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "code_sha256": ev.hashlib.sha256(
                (BASE_DIR / "scripts" / "08_evaluate_ablation.py").read_bytes()).hexdigest()[:16],
            "elapsed_s": rec["seconds"],
        })

    ranked = sorted(results, key=lambda r: r["mean_inner_val_loss"])
    best = ranked[0]

    print("\n" + "=" * 74)
    print("RANKING — by mean pos-weighted inner-validation loss (lower is better)")
    print("=" * 74)
    print(f"{'rank':>4}  {'dropout':>7} {'wd':>7} {'epochs':>6}   {'loss':>8} {'sd':>7}")
    for r, rec in enumerate(ranked, 1):
        c = rec["config"]
        print(f"{r:>4}  {c['dropout']:>7} {c['weight_decay']:>7} {c['max_epochs']:>6}   "
              f"{rec['mean_inner_val_loss']:>8.4f} {rec['sd_inner_val_loss']:>7.4f}")

    edges = [k for k, v in GRID.items()
             if best["config"][k] in (min(v), max(v)) and len(v) > 1]
    print("\n" + "=" * 74)
    print("WINNER — freeze these into HPARAMS and record them in §10")
    print("=" * 74)
    for k, v in best["config"].items():
        print(f"  {k:<14} = {v}")
    print(f"  patience       = {best['patience']}")
    print(f"  mean inner-val loss = {best['mean_inner_val_loss']:.4f} "
          f"± {best['sd_inner_val_loss']:.4f} over {best['n_inner_val_points']} fold-seed points")
    margin = ranked[1]["mean_inner_val_loss"] - best["mean_inner_val_loss"]
    print(f"  margin over 2nd     = {margin:.4f}  "
          f"({'CLEAR' if margin > best['sd_inner_val_loss'] / np.sqrt(best['n_inner_val_points']) else 'WITHIN NOISE — report this'})")
    if edges:
        print(f"\n  ⚠️  the winner sits on an EDGE of the grid for: {', '.join(edges)}")
        print("      §10 requires this be REPORTED, not quietly widened.")

    outp = BASE_DIR / "output" / "eval" / f"tuning_baseline_{a.tag}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "tag": a.tag, "arm": ARM, "grid": GRID, "seeds": seeds, "folds": a.folds,
        "selection_metric": "mean_pos_weighted_inner_val_loss",
        "winner": best, "ranking": ranked,
        "winner_on_grid_edge_for": edges,
        "total_seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"\n  written to {outp}")
    print(f"  total {(time.time() - t0) / 60:.1f} min\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
