"""
=============================================================================
STEP 7: pCR Prediction — Pathologic Complete Response
=============================================================================

From the pre-treatment biopsy image + drug/regimen molecule (SMILES), predict
whether this patient will achieve pCR after neoadjuvant chemotherapy — yes or no.

KEY REFRAME
-----------
Same multi-modal inputs as the original pipeline (histology + drug fingerprint),
but with a cleaner, better-labeled binary target:

    pCR = 1  →  no residual invasive cancer at surgery  (pathologic complete response)
    pCR = 0  →  residual disease remains                (non-pCR)

The drug molecule still enters as an input feature (fused with histology via
Step 4). What changes is ONLY THE LABEL — from the sparse, inconsistent
per-drug "measure_of_response" categories to a single clean binary definition.
All patients are pooled into one target; the drug is an input, not a
data-splitting variable.

pCR LABEL EXTRACTION
--------------------
From TCGA clinical XML, in priority order:
  1. Pathologic staging  ypT0/ypTis + ypN0  →  pCR
  2. residual_tumor = R0                    →  pCR
  3. measure_of_response = "Complete Response" for neoadjuvant patients → pCR

Patients without neoadjuvant treatment are flagged — their "Complete Response"
to adjuvant chemo is NOT the same as pCR, but can be used as a proxy for
pipeline validation.

Input  : output/fusion/fusion_manifest.json  (fused vectors [1, 1280] = 256 hist + 1024 drug)
         data/clinical/*.xml                 (pCR labels)
Output : output/pcr_prediction/
           pcr_predictions.json
           pcr_report.txt

Run:
    conda activate histopath
    python 07_pcr_prediction.py
=============================================================================
"""

import argparse
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# ── CONFIG ───────────────────────────────────────────────────────────────────
# =============================================================================

BASE_DIR    = Path(__file__).resolve().parents[1]
XML_DIR     = BASE_DIR / "data" / "clinical"
FUSION_DIR  = BASE_DIR / "output" / "fusion"
OUTPUT_DIR  = BASE_DIR / "output" / "pcr_prediction"

FUSION_MANIFEST = FUSION_DIR / "fusion_manifest.json"

# Model input dim (must match Step 4 fused vector: 256 hist + 1024 drug = 1280)
FUSED_DIM = 1280

# Training config (activates at MIN_SLIDES labelled patients)
MIN_SLIDES_TO_TRAIN = 10
LEARNING_RATE       = 1e-4
NUM_EPOCHS          = 30
DROPOUT             = 0.3

# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BARCODE_RE = re.compile(r"TCGA-\w\w-\w{4}", re.IGNORECASE)


# =============================================================================
# 1. pCR LABEL EXTRACTION FROM CLINICAL XML
# =============================================================================

def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def parse_pcr_label(xml_path: Path) -> dict:
    """
    Extract pCR-relevant fields from a TCGA clinical XML.

    Returns dict with:
      - patient_id: TCGA barcode
      - neoadjuvant: bool or None
      - pathologic_T, pathologic_N: staging (yp-prefix = post-neoadjuvant)
      - residual_tumor: R0/R1/R2 if available
      - chemo_responses: list of measure_of_response for chemo drugs
      - pcr_label: 1 (pCR), 0 (non-pCR), or None (indeterminate)
      - pcr_source: how the label was derived
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    info = {
        "xml_file": xml_path.name,
        "patient_id": None,
        "neoadjuvant": None,
        "pathologic_T": None,
        "pathologic_N": None,
        "residual_tumor": None,
        "chemo_responses": [],
        "all_responses": [],
        "pcr_label": None,
        "pcr_source": None,
    }

    current_therapy_type = None
    current_response = None

    for elem in root.iter():
        tag = _strip_ns(elem.tag).lower()
        text = (elem.text or "").strip()
        if not text:
            continue

        if "bcr_patient_barcode" in tag or "patient_barcode" in tag:
            info["patient_id"] = text.upper()
        elif "history_of_neoadjuvant_treatment" in tag:
            info["neoadjuvant"] = text.lower() == "yes"
        elif tag == "pathologic_t":
            info["pathologic_T"] = text
        elif tag == "pathologic_n":
            info["pathologic_N"] = text
        elif "residual_tumor" in tag:
            info["residual_tumor"] = text
        elif "therapy_type" in tag:
            current_therapy_type = text.lower()
        elif "measure_of_response" in tag or "treatment_best_response" in tag:
            current_response = text
            info["all_responses"].append(text)
            if current_therapy_type and "chemo" in current_therapy_type:
                info["chemo_responses"].append(text)

    if not info["patient_id"]:
        m = BARCODE_RE.search(xml_path.stem)
        if m:
            info["patient_id"] = m.group(0).upper()

    # --- Derive pCR label ---
    info["pcr_label"], info["pcr_source"] = _derive_pcr(info)

    return info


def _derive_pcr(info: dict) -> tuple[int | None, str]:
    """
    Derive binary pCR label from clinical fields.

    Priority:
      1. yp-prefixed staging: ypT0/ypTis + ypN0 = pCR
      2. residual_tumor R0 = pCR (for neoadjuvant patients)
      3. All chemo responses = "Complete Response" for neoadjuvant patients
      4. Fallback: use measure_of_response even for adjuvant (proxy, flagged)
    """
    pT = (info["pathologic_T"] or "").strip()
    pN = (info["pathologic_N"] or "").strip()

    # 1. yp-staging (post-neoadjuvant pathologic staging)
    if pT.lower().startswith("yp"):
        t_clean = pT.lower().replace("yp", "")
        n_clean = pN.lower().replace("yp", "") if pN.lower().startswith("yp") else pN.lower()
        is_t0 = t_clean in ("t0", "tis", "0", "is")
        is_n0 = n_clean.startswith("n0") or n_clean == "0"
        if is_t0 and is_n0:
            return 1, "yp-staging (ypT0/ypTis + ypN0)"
        else:
            return 0, f"yp-staging (residual: {pT} {pN})"

    # 2. Residual tumor field (neoadjuvant patients)
    rt = info["residual_tumor"]
    if rt and info["neoadjuvant"]:
        if rt.upper() == "R0":
            return 1, "residual_tumor R0 + neoadjuvant"
        else:
            return 0, f"residual_tumor {rt} + neoadjuvant"

    # 3. Chemo response for neoadjuvant patients
    if info["neoadjuvant"] and info["chemo_responses"]:
        all_cr = all(r.lower() in ("complete response", "complete remission")
                     for r in info["chemo_responses"])
        if all_cr:
            return 1, "neoadjuvant + all chemo Complete Response"
        else:
            return 0, f"neoadjuvant + mixed chemo responses"

    # 4. Fallback: adjuvant chemo response as proxy (flagged)
    if info["chemo_responses"]:
        all_cr = all(r.lower() in ("complete response", "complete remission")
                     for r in info["chemo_responses"])
        if all_cr:
            return 1, "PROXY: adjuvant chemo Complete Response (not true pCR)"
        any_progressive = any("progressive" in r.lower() for r in info["chemo_responses"])
        any_stable = any("stable" in r.lower() for r in info["chemo_responses"])
        if any_progressive or any_stable:
            return 0, "PROXY: adjuvant chemo non-complete response"

    # 5. Last resort: any response field
    if info["all_responses"]:
        all_cr = all(r.lower() in ("complete response", "complete remission")
                     for r in info["all_responses"])
        if all_cr:
            return 1, "PROXY: all treatments Complete Response"
        return 0, "PROXY: mixed treatment responses"

    return None, "indeterminate — no response data"


def load_all_pcr_labels(xml_dir: Path) -> dict:
    """Return {patient_barcode -> pcr_info dict}."""
    labels = {}
    for xf in sorted(xml_dir.glob("*.xml")):
        info = parse_pcr_label(xf)
        bc = info.get("patient_id")
        if bc:
            labels[bc] = info
        else:
            log.warning(f"[XML] No barcode in {xf.name}")
    return labels


# =============================================================================
# 2. pCR CLASSIFIER MODEL
# =============================================================================

class PCRClassifier(nn.Module):
    """
    Binary classifier: fused vector [1, 1280] → pCR probability.

    Input is the histology + drug fingerprint fusion from Step 4.
    """

    def __init__(self, input_dim: int = 1280, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 3. TRAINING
# =============================================================================

def run_training(slide_data: list[dict], model: nn.Module, device: torch.device):
    """
    Train pCR classifier on labelled slide embeddings.
    Skipped if fewer than MIN_SLIDES_TO_TRAIN labelled slides.
    """
    trainable = [s for s in slide_data if s["pcr_label"] is not None]

    if len(trainable) < MIN_SLIDES_TO_TRAIN:
        log.info(
            f"Only {len(trainable)} labelled slides — skipping training "
            f"(need {MIN_SLIDES_TO_TRAIN}+). Running inference with random weights."
        )
        return

    log.info(f"Training pCR classifier on {len(trainable)} slides for {NUM_EPOCHS} epochs")

    n_pcr = sum(1 for s in trainable if s["pcr_label"] == 1)
    n_non = len(trainable) - n_pcr
    log.info(f"  pCR: {n_pcr}  |  non-pCR: {n_non}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        correct = 0

        for s in trainable:
            embed = s["embedding"].to(device)
            y = torch.tensor([s["pcr_label"]], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(embed)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (logits.argmax(-1) == y).sum().item()

        acc = correct / len(trainable)
        log.info(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  loss={epoch_loss:.4f}  acc={acc:.2%}")

    weights_path = OUTPUT_DIR / "pcr_classifier_weights.pt"
    torch.save(model.state_dict(), weights_path)
    log.info(f"Trained weights saved → {weights_path}")


# =============================================================================
# 4. INFERENCE
# =============================================================================

LABEL_MAP = {0: "non-pCR (residual disease)", 1: "pCR (complete response)"}


def run_inference(slide_data: list[dict], model: nn.Module, device: torch.device) -> list[dict]:
    model.eval()
    results = []

    with torch.no_grad():
        for s in slide_data:
            embed = s["embedding"].to(device)
            logits = model(embed)
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
            conf = probs[0, pred].item()

            gt = s["pcr_label"]
            match_str = ""
            if gt is not None:
                match_str = " CORRECT" if pred == gt else " WRONG"

            log.info(
                f"  [{s['slide_id']}] "
                f"pred={LABEL_MAP[pred]}  conf={conf:.1%}  "
                f"gt={LABEL_MAP[gt] if gt is not None else 'unknown'}"
                f"{match_str}"
            )

            results.append({
                "slide_id": s["slide_id"],
                "drug_name": s.get("drug_name", "unknown"),
                "predicted_class": pred,
                "predicted_label": LABEL_MAP[pred],
                "confidence": round(conf, 6),
                "prob_non_pcr": round(probs[0, 0].item(), 6),
                "prob_pcr": round(probs[0, 1].item(), 6),
                "ground_truth_class": gt,
                "ground_truth_label": LABEL_MAP[gt] if gt is not None else None,
                "correct": (pred == gt) if gt is not None else None,
                "pcr_source": s["pcr_source"],
                "neoadjuvant": s["neoadjuvant"],
                "pathologic_T": s["pathologic_T"],
                "pathologic_N": s["pathologic_N"],
                "fused_shape": list(embed.shape),
                "status": "ok",
            })

    return results


# =============================================================================
# 5. MAIN
# =============================================================================

def main(checkpoint: str | None = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 64)
    log.info("  STEP 7: pCR PREDICTION (Pathologic Complete Response)")
    log.info("  Target: pCR yes/no from histology + drug fingerprint")
    log.info("=" * 64)

    # ── Load fusion manifest ─────────────────────────────────────────────
    if not FUSION_MANIFEST.exists():
        raise FileNotFoundError(
            f"Fusion manifest not found: {FUSION_MANIFEST}\n"
            "Run Step 4 (04_cross_modal_fusion.py) first."
        )

    with open(FUSION_MANIFEST) as f:
        fusion_manifest = json.load(f)
    log.info(f"Fusion manifest: {len(fusion_manifest)} slides")

    # ── Load pCR labels from XML ─────────────────────────────────────────
    pcr_labels = load_all_pcr_labels(XML_DIR)
    log.info(f"Clinical XMLs: {len(pcr_labels)} patients parsed")

    for bc, info in pcr_labels.items():
        pcr_str = "pCR" if info["pcr_label"] == 1 else (
            "non-pCR" if info["pcr_label"] == 0 else "unknown")
        neo_str = "neoadjuvant" if info["neoadjuvant"] else "adjuvant"
        log.info(f"  {bc}: {pcr_str} ({neo_str}) — {info['pcr_source']}")

    # ── Match slides to pCR labels and load fused vectors ────────────────
    slide_data = []

    for slide_id, slide_info in fusion_manifest.items():
        fused_path = slide_info.get("fused_path")
        if not fused_path or not Path(fused_path).exists():
            log.warning(f"  [{slide_id}] Fused vector not found — skipping")
            continue

        fused = torch.load(fused_path, map_location="cpu", weights_only=True)
        if fused.dim() == 1:
            fused = fused.unsqueeze(0)

        drug_name = slide_info.get("drug_name", "unknown")

        m = BARCODE_RE.search(slide_id)
        bc = m.group(0).upper() if m else None
        pcr_info = pcr_labels.get(bc, {})

        slide_data.append({
            "slide_id": slide_id,
            "embedding": fused,
            "drug_name": drug_name,
            "pcr_label": pcr_info.get("pcr_label"),
            "pcr_source": pcr_info.get("pcr_source", "no XML match"),
            "neoadjuvant": pcr_info.get("neoadjuvant"),
            "pathologic_T": pcr_info.get("pathologic_T"),
            "pathologic_N": pcr_info.get("pathologic_N"),
        })

    log.info(f"Matched {len(slide_data)} slides with fused vectors")

    labelled = sum(1 for s in slide_data if s["pcr_label"] is not None)
    n_pcr = sum(1 for s in slide_data if s["pcr_label"] == 1)
    n_non = sum(1 for s in slide_data if s["pcr_label"] == 0)
    log.info(f"  Labelled: {labelled}  (pCR={n_pcr}, non-pCR={n_non})")

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # ── Build model ───────────────────────────────────────────────────────
    model = PCRClassifier(input_dim=FUSED_DIM, dropout=DROPOUT).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"PCRClassifier: {total_params:,} parameters  ({FUSED_DIM} → 512 → 128 → 2)")

    if checkpoint:
        ckpt = Path(checkpoint)
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=True))
            log.info(f"Loaded checkpoint: {ckpt}")
        else:
            log.warning(f"Checkpoint not found: {ckpt} — using random weights")

    # ── Train (if enough data) ───────────────────────────────────────────
    run_training(slide_data, model, device)

    # ── Inference ─────────────────────────────────────────────────────────
    log.info("Running pCR inference on all slides")
    results = run_inference(slide_data, model, device)

    # ── Summary ───────────────────────────────────────────────────────────
    total_with_gt = sum(1 for r in results if r["ground_truth_class"] is not None)
    correct = sum(1 for r in results if r.get("correct") is True)

    log.info("=" * 64)
    log.info(f"  Slides processed  : {len(results)}")
    if total_with_gt > 0:
        acc = correct / total_with_gt
        log.info(f"  Accuracy          : {correct}/{total_with_gt} ({acc:.1%})")
    neo_count = sum(1 for r in results if r["neoadjuvant"] is True)
    proxy_count = sum(1 for r in results
                      if r["pcr_source"] and "PROXY" in r["pcr_source"])
    log.info(f"  Neoadjuvant cases : {neo_count}")
    log.info(f"  Proxy labels      : {proxy_count} (adjuvant response used as stand-in)")
    log.info("=" * 64)

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "target": "pCR (pathologic complete response) — binary",
        "model": "PCRClassifier (histology + drug fingerprint fusion)",
        "input_dim": FUSED_DIM,
        "checkpoint": str(checkpoint) if checkpoint else "none (random weights)",
        "total_slides": len(results),
        "labelled_slides": total_with_gt,
        "neoadjuvant_slides": neo_count,
        "proxy_labels": proxy_count,
        "accuracy": round(correct / total_with_gt, 4) if total_with_gt else None,
        "predictions": results,
    }

    pred_json = OUTPUT_DIR / "pcr_predictions.json"
    with open(pred_json, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"[SAVED] {pred_json}")

    # ── Save report ───────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "pcr_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 64 + "\n")
        f.write("  pCR Prediction Report\n")
        f.write(f"  Generated : {output['run_timestamp']}\n")
        f.write(f"  Checkpoint: {output['checkpoint']}\n")
        f.write(f"  Target    : pCR (yes/no) from histology + drug fingerprint\n")
        f.write("=" * 64 + "\n\n")

        for r in results:
            f.write(f"Slide       : {r['slide_id']}\n")
            f.write(f"Drug        : {r.get('drug_name', 'unknown')}\n")
            f.write(f"Neoadjuvant : {'Yes' if r['neoadjuvant'] else 'No (adjuvant)'}\n")
            f.write(f"Staging     : {r['pathologic_T'] or '—'} / {r['pathologic_N'] or '—'}\n")
            f.write(f"Predicted   : {r['predicted_label']}  (conf {r['confidence']:.1%})\n")
            f.write(f"P(non-pCR)  : {r['prob_non_pcr']:.4f}\n")
            f.write(f"P(pCR)      : {r['prob_pcr']:.4f}\n")
            if r["ground_truth_label"]:
                verdict = "CORRECT" if r["correct"] else "WRONG"
                f.write(f"Ground truth: {r['ground_truth_label']}  →  {verdict}\n")
            f.write(f"Label source: {r['pcr_source']}\n")
            f.write("\n")

        if total_with_gt:
            f.write(f"Accuracy: {correct}/{total_with_gt}\n")
            f.write(f"Neoadjuvant cases: {neo_count}\n")
            f.write(f"Proxy labels (adjuvant): {proxy_count}\n")

    log.info(f"[SAVED] {report_path}")
    log.info("Done. Step 7 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7 — pCR Prediction")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained classifier weights (.pt)")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)
