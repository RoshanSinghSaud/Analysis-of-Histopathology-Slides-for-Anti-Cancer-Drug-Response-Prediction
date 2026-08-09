"""
Step 5 — Drug Response Prediction
==================================
Pipeline : Vision Transformer Based Analysis of Histopathology Images
          for Predicting AntiCancer Drug Response (TCGA-BRCA)

Input  : output/fusion/fusion_manifest.json
         output/fusion/<slide_id>_fused.pt   →  [1, 1280] float32 tensor
Output : output/predictions/predictions.json
         output/predictions/predictions_report.txt

Model  : 3-layer MLP  (1280 → 512 → 128 → 2)
Mode   : validation / inference  (no training data required for 2 slides)
         Weights are randomly initialised and then evaluated in "demo" mode.
         Swap in a real checkpoint via --checkpoint <path> to use trained weights.

Hardware : CPU only (Apple Silicon M-series via MPS falls back to CPU for safety)
Env      : conda activate histopath  (Python 3.10)

Usage
-----
    python step5_predict_drug_response.py
    python step5_predict_drug_response.py --fusion_dir output/fusion \
                                          --output_dir output/predictions \
                                          --checkpoint my_model.pt
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  1.  Model definition
# ─────────────────────────────────────────────

class DrugResponseMLP(nn.Module):
    """
    Simple 3-layer MLP for binary drug response classification.

    Input  : fused vector  [B, input_dim]   (default 1280 = 256 hist + 1024 drug)
    Output : logits        [B, num_classes]  (0 = No Response, 1 = Complete Response)
    """

    def __init__(self, input_dim: int = 1280, hidden_dims=(512, 128), num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  2.  Inference helpers
# ─────────────────────────────────────────────

LABEL_MAP = {0: "No Response", 1: "Complete Response"}
LABEL_REVERSE = {"nr": 0, "cr": 1, "complete response": 1, "no response": 0,
                  "partial response": 1, "stable disease": 0}


def load_fused_tensor(pt_path: str) -> torch.Tensor:
    """Load a [1, 1280] fused feature tensor from disk."""
    t = torch.load(pt_path, map_location="cpu", weights_only=True)
    if t.dim() == 1:
        t = t.unsqueeze(0)          # [1280] → [1, 1280]
    if t.dim() != 2:
        raise ValueError(f"Unexpected tensor shape {t.shape} in {pt_path}")
    return t.float()


def ground_truth_label(entry: dict) -> int | None:
    """Try to extract a ground-truth label from a manifest entry (best-effort)."""
    for key in ("response_label", "label", "clinical_response", "drug_response"):
        val = entry.get(key)
        if val is not None:
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                return LABEL_REVERSE.get(val.lower().strip())
    return None


def predict_single(model: nn.Module, tensor: torch.Tensor):
    """Return (predicted_class, confidence, probabilities)."""
    model.eval()
    with torch.no_grad():
        logits = model(tensor)                     # [1, 2]
        probs  = F.softmax(logits, dim=-1)         # [1, 2]
        pred   = torch.argmax(probs, dim=-1).item()
        conf   = probs[0, pred].item()
        prob_list = probs[0].tolist()
    return pred, conf, prob_list


# ─────────────────────────────────────────────
#  3.  Main
# ─────────────────────────────────────────────

def run(fusion_dir: str, output_dir: str, checkpoint: str | None,
        input_dim: int, num_classes: int):

    fusion_dir  = Path(fusion_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = fusion_dir / "fusion_manifest.json"
    if not manifest_path.exists():
        sys.exit(f"[ERROR] fusion_manifest.json not found at {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # ── Build model ────────────────────────────────────────────────────────
    model = DrugResponseMLP(input_dim=input_dim, num_classes=num_classes)

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Support both raw state-dict and {"model_state_dict": ...} wrappers
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        print(f"[INFO]  Loaded weights from {ckpt_path}")
    else:
        print("[INFO]  No checkpoint supplied — using random weights (demo mode).")
        print("[INFO]  Predictions are ILLUSTRATIVE only until real weights are loaded.")

    model.eval()

    # ── Iterate manifest entries ───────────────────────────────────────────
    # Support both list-of-entries and dict-of-slide_id → entry formats
    if isinstance(manifest, dict):
        entries = [{"slide_id": k, **v} for k, v in manifest.items()]
    elif isinstance(manifest, list):
        entries = manifest
    else:
        sys.exit("[ERROR] fusion_manifest.json must be a JSON list or object.")

    results = []
    correct = 0
    total_with_gt = 0

    print(f"\n{'─'*60}")
    print(f"  Step 5 — Drug Response Prediction")
    print(f"  Slides : {len(entries)}")
    print(f"  Model  : DrugResponseMLP  ({input_dim}→512→128→{num_classes})")
    print(f"{'─'*60}\n")

    for entry in entries:
        slide_id = entry.get("slide_id", entry.get("id", "unknown"))

        # Locate the fused .pt file (try multiple naming conventions)
        candidates = [
            fusion_dir / entry.get("fused_path", ""),
            fusion_dir / f"{slide_id}_fused.pt",
            fusion_dir / f"{slide_id}.pt",
        ]
        pt_path = None
        for c in candidates:
            if c.exists():
                pt_path = c
                break

        if pt_path is None:
            print(f"[WARN]  {slide_id}: fused .pt not found — skipping")
            results.append({"slide_id": slide_id, "status": "skipped",
                             "reason": "fused .pt file not found"})
            continue

        try:
            tensor = load_fused_tensor(str(pt_path))
        except Exception as e:
            print(f"[WARN]  {slide_id}: failed to load tensor — {e}")
            results.append({"slide_id": slide_id, "status": "error", "reason": str(e)})
            continue

        pred, conf, probs = predict_single(model, tensor)
        gt = ground_truth_label(entry)

        match_str = ""
        if gt is not None:
            total_with_gt += 1
            if pred == gt:
                correct += 1
                match_str = "✓"
            else:
                match_str = "✗"

        drug = entry.get("drug_name", entry.get("drug", "unknown drug"))
        print(f"  {slide_id}")
        print(f"    Drug       : {drug}")
        print(f"    Prediction : {LABEL_MAP[pred]}  (conf {conf:.1%})  {match_str}")
        print(f"    P(NR)={probs[0]:.4f}   P(CR)={probs[1]:.4f}")
        if gt is not None:
            print(f"    Ground truth: {LABEL_MAP[gt]}")
        print()

        record = {
            "slide_id"        : slide_id,
            "drug"            : drug,
            "predicted_class" : pred,
            "predicted_label" : LABEL_MAP[pred],
            "confidence"      : round(conf, 6),
            "prob_no_response": round(probs[0], 6),
            "prob_complete_response": round(probs[1], 6),
            "ground_truth_class": gt,
            "ground_truth_label": LABEL_MAP[gt] if gt is not None else None,
            "correct"         : (pred == gt) if gt is not None else None,
            "tensor_shape"    : list(tensor.shape),
            "pt_file"         : str(pt_path),
            "status"          : "ok",
        }
        results.append(record)

    # ── Summary ───────────────────────────────────────────────────────────
    processed = [r for r in results if r.get("status") == "ok"]
    print(f"{'─'*60}")
    print(f"  Processed : {len(processed)} / {len(entries)} slides")
    if total_with_gt > 0:
        acc = correct / total_with_gt
        print(f"  Accuracy  : {correct}/{total_with_gt}  ({acc:.1%})  [random weights — not meaningful]")
    print(f"{'─'*60}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "run_timestamp"   : datetime.now().isoformat(),
        "checkpoint"      : str(checkpoint) if checkpoint else "none (random weights)",
        "input_dim"       : input_dim,
        "num_classes"     : num_classes,
        "label_map"       : LABEL_MAP,
        "total_slides"    : len(entries),
        "processed_slides": len(processed),
        "accuracy"        : round(correct / total_with_gt, 4) if total_with_gt else None,
        "predictions"     : results,
    }

    pred_json = output_dir / "predictions.json"
    with open(pred_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVED] {pred_json}")

    # ── Save human-readable report ────────────────────────────────────────
    report_path = output_dir / "predictions_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Drug Response Prediction Report\n")
        f.write(f"  Generated : {output['run_timestamp']}\n")
        f.write(f"  Checkpoint: {output['checkpoint']}\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            if r.get("status") != "ok":
                f.write(f"SKIPPED  {r['slide_id']}  ({r.get('reason','')})\n\n")
                continue
            f.write(f"Slide    : {r['slide_id']}\n")
            f.write(f"Drug     : {r['drug']}\n")
            f.write(f"Predicted: {r['predicted_label']}  (conf {r['confidence']:.1%})\n")
            f.write(f"Probs    : NR={r['prob_no_response']:.4f}  CR={r['prob_complete_response']:.4f}\n")
            if r["ground_truth_label"]:
                verdict = "CORRECT" if r["correct"] else "WRONG"
                f.write(f"GT       : {r['ground_truth_label']}  → {verdict}\n")
            f.write("\n")
        if total_with_gt:
            f.write(f"Accuracy : {correct}/{total_with_gt}\n")
    print(f"[SAVED] {report_path}")
    print("\n[DONE]  Step 5 complete.\n")


# ─────────────────────────────────────────────
#  4.  CLI entry-point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Step 5 — Drug Response Prediction")
    p.add_argument("--fusion_dir",  default="output/fusion",
                   help="Folder containing fusion_manifest.json and *_fused.pt files")
    p.add_argument("--output_dir",  default="output/predictions",
                   help="Where to save predictions.json and predictions_report.txt")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to a trained model checkpoint (.pt). Omit for demo/random mode.")
    p.add_argument("--input_dim",   type=int, default=1280,
                   help="Fused vector dimension (default 1280 = 256 hist + 1024 drug)")
    p.add_argument("--num_classes", type=int, default=2,
                   help="Number of output classes (default 2: NR / CR)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        fusion_dir  = args.fusion_dir,
        output_dir  = args.output_dir,
        checkpoint  = args.checkpoint,
        input_dim   = args.input_dim,
        num_classes = args.num_classes,
    )
