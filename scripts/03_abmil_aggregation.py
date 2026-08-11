"""
=============================================================================
STEP 3: ABMIL — Attention-Based Multiple Instance Learning Aggregation
=============================================================================

Converts tile features → ONE embedding per PATIENT + per-tile attention scores.

Input  : output/features/feature_manifest.json   (per slide, from step 01)
         cohort/cohort_*.csv                     (labels — see cohort_labels.py)

Output : output/abmil/
           TCGA-A2-A0EP_patient_embedding.pt   [1, 256]
           TCGA-A2-A0EP_attention_scores.pt    [N_tiles_all_slides, 1]
           abmil_manifest.json                 input to Step 4

THE UNIT OF ANALYSIS IS THE PATIENT, NOT THE SLIDE
--------------------------------------------------
Fixed Aug 11 2026. This script used to emit one embedding per .svs file. The
cohort has 206 BRCA slides for 199 patients — 5 patients have 2 slides and 1
has 3 — so six patients entered the analysis two or three times. That breaks
the patient-level grouping the 5-fold CV design depends on, which is the
highest-leverage decision in the whole study.

All of a patient's tiles are now concatenated into ONE MIL bag. The bag is all
of their tissue; picking a "best slide" would need an arbitrary rule a reviewer
would rightly challenge.

Because the bag spans several slides, the attention vector spans them too. The
manifest therefore records `slide_order` — the slide_id, tile count and offset
of each slide's block within the bag — so step 06 can slice attention back
apart and map each score onto the right slide's coordinates. Without that
bookkeeping, heatmaps for multi-slide patients would silently plot one slide's
attention onto another's pixels.

Run:
    conda activate histopath
    python scripts/03_abmil_aggregation.py
=============================================================================
"""

import json
import logging
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cohort_labels import (CohortError, find_cohort_csv, load_cohort,
                           patient_barcode, summarise)

# =============================================================================
# ── CONFIG ───────────────────────────────────────────────────────────────────
# =============================================================================

BASE_DIR         = Path(__file__).resolve().parents[1]
FEATURE_MANIFEST = BASE_DIR / "output" / "features" / "feature_manifest.json"
OUTPUT_DIR       = BASE_DIR / "output" / "abmil"

# Which cohort arm to label against. None = auto-detect a single cohort_*.csv
# in cohort/, and refuse to guess if there are several.
COHORT_TAG = None

# ABMIL architecture dims
INPUT_DIM  = 1024    # UNI embed dim (match Step 1)
HIDDEN_DIM = 256     # projection dim — patient embedding will be this size
ATTN_DIM   = 128     # attention network internal dim

# Training config (only used when n_patients >= MIN_PATIENTS_TO_TRAIN)
MIN_PATIENTS_TO_TRAIN = 10   # won't attempt training below this
LEARNING_RATE         = 1e-4
NUM_EPOCHS            = 20

# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# ABMIL Model
# =============================================================================

class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning (Ilse et al. 2018).

    Architecture:
        tile features [N, input_dim]      ← N = ALL tiles of ALL of one
            → feature projection  [N, hidden_dim]      patient's slides
            → attention network   [N, 1]  (gated attention)
            → weighted sum        [1, hidden_dim]  ← patient embedding
            → classifier head     [1, n_classes]

    The attention scores tell you WHICH TILES the model focused on.
    These are saved and used in Step 6 for heatmap generation.
    """

    def __init__(
        self,
        input_dim:  int = 1024,
        hidden_dim: int = 256,
        attn_dim:   int = 128,
        n_classes:  int = 2,
        dropout:    float = 0.25,
    ):
        super().__init__()

        # 1. Project tile features to lower dim
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. Gated attention network (two parallel branches)
        #    A(h) = softmax( V tanh(Wh) ⊙ U sigmoid(Wh) )
        self.attn_V = nn.Linear(hidden_dim, attn_dim)   # tanh branch
        self.attn_U = nn.Linear(hidden_dim, attn_dim)   # sigmoid gate
        self.attn_w = nn.Linear(attn_dim, 1)            # scalar score

        # 3. Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [N, input_dim]  — all tile features for ONE PATIENT

        Returns:
            logits         : [1, n_classes]
            patient_embed  : [1, hidden_dim]
            attn_scores    : [N, 1]  (after softmax — sum to 1 across the bag)
        """
        h = self.feature_proj(x)                        # [N, hidden_dim]

        a_V = torch.tanh(self.attn_V(h))                # [N, attn_dim]
        a_U = torch.sigmoid(self.attn_U(h))             # [N, attn_dim]
        a   = self.attn_w(a_V * a_U)                    # [N, 1]
        a   = F.softmax(a, dim=0)                       # [N, 1]  sums to 1

        patient_embed = (a * h).sum(dim=0, keepdim=True)  # [1, hidden_dim]
        logits = self.classifier(patient_embed)           # [1, n_classes]

        return logits, patient_embed, a


# =============================================================================
# Patient grouping — the fix for one-bag-per-slide
# =============================================================================

def group_slides_by_patient(manifest: dict) -> "OrderedDict[str, list]":
    """
    {patient_barcode -> [(slide_id, entry), ...]}, slides sorted by slide_id
    and patients sorted by barcode so the ordering is deterministic across runs.

    Entries whose filename yields no TCGA barcode are dropped with a warning —
    they cannot be attributed to a patient, so they cannot be labelled.
    """
    groups: dict[str, list] = {}
    for slide_id, entry in manifest.items():
        bc = patient_barcode(slide_id) or patient_barcode(
            str(entry.get("label", {}).get("barcode", "")))
        if bc is None:
            log.warning(f"[{slide_id}] no TCGA barcode in slide id — dropped")
            continue
        groups.setdefault(bc, []).append((slide_id, entry))

    for bc in groups:
        groups[bc].sort(key=lambda t: t[0])
    return OrderedDict(sorted(groups.items()))


def load_patient_bag(slides: list, device: torch.device):
    """
    Concatenate every slide's tile features into one bag.

    Returns (x, slide_order) where x is [N_total, INPUT_DIM] and slide_order is
    [{slide_id, n_tiles, offset}] describing each slide's contiguous block, so
    attention can be sliced back apart per slide in step 06.

    Raises FileNotFoundError if any of the patient's feature files is missing —
    a partial bag is silently a different patient, so this must not be tolerated.
    """
    parts, slide_order, offset = [], [], 0

    for slide_id, entry in slides:
        feat_path = entry.get("feature_path")
        if not feat_path or not Path(feat_path).exists():
            raise FileNotFoundError(
                f"{slide_id}: feature file missing ({feat_path}). The bag for "
                f"this patient would be incomplete, which silently changes who "
                f"the patient is. Re-run step 01 for this slide."
            )
        t = torch.load(feat_path, map_location="cpu", weights_only=True)
        if t.ndim != 2 or t.shape[1] != INPUT_DIM:
            raise ValueError(
                f"{slide_id}: features are {tuple(t.shape)}, expected "
                f"[N, {INPUT_DIM}]. Check INPUT_DIM against the extractor used "
                f"in step 01 (UNI=1024, Phikon=768)."
            )
        parts.append(t)
        slide_order.append({"slide_id": slide_id,
                            "n_tiles": int(t.shape[0]),
                            "offset": offset})
        offset += int(t.shape[0])

    x = torch.cat(parts, dim=0).to(device)
    return x, slide_order


def split_attention_by_slide(attn: torch.Tensor, slide_order: list) -> dict:
    """
    Inverse of the concatenation: {slide_id -> attention block for that slide}.
    Step 06 needs this to map scores onto the right slide's tile coordinates.
    """
    out = {}
    for s in slide_order:
        o, n = s["offset"], s["n_tiles"]
        out[s["slide_id"]] = attn[o:o + n]
    return out


# =============================================================================
# Labels
# =============================================================================

def build_patient_labels(groups, cohort) -> dict:
    """
    {patient -> int}. 1 = responder, 0 = non-responder, -1 = not in cohort.

    Labels come from the cohort CSV only. This script deliberately does not
    read the `label` block written into the feature manifest, so that an old
    manifest built by the pre-Aug-11 XML-scraping code cannot quietly supply a
    label that the study was never registered on.
    """
    labels = {}
    for bc in groups:
        rec = cohort.get(bc)
        labels[bc] = int(rec["response_binary"]) if rec else -1
    return labels


# =============================================================================
# Inference — forward pass, save embeddings + attention scores
# =============================================================================

def run_inference(groups, labels, cohort, model, device) -> dict:
    model.eval()
    results = {}

    with torch.no_grad():
        for bc, slides in tqdm(groups.items(), desc="Patients"):
            try:
                x, slide_order = load_patient_bag(slides, device)
            except (FileNotFoundError, ValueError) as e:
                log.warning(f"[{bc}] skipped — {e}")
                continue

            logits, patient_embed, attn_scores = model(x)
            probs = F.softmax(logits, dim=-1)
            pred  = logits.argmax(dim=-1).item()

            embed_path = OUTPUT_DIR / f"{bc}_patient_embedding.pt"
            attn_path  = OUTPUT_DIR / f"{bc}_attention_scores.pt"
            torch.save(patient_embed.cpu(), embed_path)
            torch.save(attn_scores.cpu(),   attn_path)

            rec = cohort.get(bc, {})
            log.info(
                f"[{bc}] slides={len(slide_order)} tiles={x.shape[0]:,}  "
                f"embed={list(patient_embed.shape)}  pred={pred} "
                f"prob={probs[0][pred]:.3f}  true={labels[bc]}"
            )

            results[bc] = {
                "patient_barcode":       bc,
                "patient_embedding_path": str(embed_path),
                "attention_scores_path":  str(attn_path),
                "embed_shape":           list(patient_embed.shape),
                "n_slides":              len(slide_order),
                "n_tiles":               int(x.shape[0]),
                "slide_order":           slide_order,
                "pred_class":            pred,
                "pred_prob":             probs[0][pred].item(),
                "true_label":            labels[bc],
                "patient_label":         rec.get("patient_label", "not_in_cohort"),
                "drugs":                 rec.get("drugs", []),
                "stage":                 rec.get("stage", ""),
                "primary_diagnosis":     rec.get("primary_diagnosis", ""),
                "status":                "done",
            }

    return results


# =============================================================================
# Training loop
# =============================================================================

def run_training(groups, labels, model, device):
    """
    Supervised training of ABMIL with cross-entropy loss.

    ⚠️  THIS IS NOT THE PRE-REGISTERED EVALUATION. It trains on every patient
    and then step 4/5 score the same patients, so the accuracy printed here is
    TRAINING accuracy and means nothing. The registered design is 5-fold CV
    with pooled out-of-fold predictions, patient-level grouping and 5 seeds
    (24% power versus 91% for a single split). Never quote a number from this
    function in the paper.
    """
    trainable = {bc: sl for bc, sl in groups.items() if labels[bc] != -1}

    if len(trainable) < MIN_PATIENTS_TO_TRAIN:
        log.info(
            f"Only {len(trainable)} labelled patients — skipping training "
            f"(need {MIN_PATIENTS_TO_TRAIN}+). Running inference only."
        )
        return

    log.warning("Training on ALL patients — the accuracy below is TRAINING "
                "accuracy, not a result. See this function's docstring.")
    log.info(f"Training ABMIL on {len(trainable)} patients for {NUM_EPOCHS} epochs…")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss, correct, seen = 0.0, 0, 0

        for bc, slides in trainable.items():
            try:
                x, _ = load_patient_bag(slides, device)
            except (FileNotFoundError, ValueError) as e:
                log.warning(f"[{bc}] skipped in training — {e}")
                continue
            y = torch.tensor([labels[bc]], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, _, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (logits.argmax(-1) == y).sum().item()
            seen       += 1

        acc = correct / seen if seen else float("nan")
        log.info(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  loss={epoch_loss:.4f}  "
                 f"train_acc={acc:.2%} (not a result)")

    weights_path = OUTPUT_DIR / "abmil_weights.pt"
    torch.save(model.state_dict(), weights_path)
    log.info(f"Trained weights saved → {weights_path}")


# =============================================================================
# Main
# =============================================================================

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        log.info("Using Apple Silicon MPS")
        return torch.device("mps")
    log.warning("No GPU — running on CPU")
    return torch.device("cpu")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {FEATURE_MANIFEST}\n"
            "Run step 01 (scripts/01_preprocess_and_extract.py) first."
        )

    with open(FEATURE_MANIFEST) as f:
        manifest = json.load(f)
    manifest = {k: v for k, v in manifest.items()
                if v.get("status") in ("done", "skipped")}
    log.info(f"Loaded {len(manifest)} slides from the feature manifest")

    cohort_csv = find_cohort_csv(BASE_DIR, COHORT_TAG)
    cohort = load_cohort(cohort_csv)
    log.info(f"Cohort {cohort_csv.name}: {summarise(cohort)}")

    groups = group_slides_by_patient(manifest)
    labels = build_patient_labels(groups, cohort)

    multi = {bc: len(sl) for bc, sl in groups.items() if len(sl) > 1}
    outside = [bc for bc, v in labels.items() if v == -1]
    log.info(f"Grouped {len(manifest)} slides → {len(groups)} patients "
             f"({len(multi)} with >1 slide)")
    if multi:
        log.info("  multi-slide patients: " +
                 ", ".join(f"{bc}×{n}" for bc, n in sorted(multi.items())))
    if outside:
        log.warning(f"  {len(outside)} patients are NOT in {cohort_csv.name} and "
                    f"will be label -1 / excluded from training: "
                    f"{', '.join(outside[:10])}"
                    f"{' …' if len(outside) > 10 else ''}")

    device = pick_device()
    log.info(f"Device: {device}")

    model = ABMIL(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  attn_dim=ATTN_DIM, n_classes=2).to(device)
    log.info(f"ABMIL built — {sum(p.numel() for p in model.parameters()):,} parameters")

    run_training(groups, labels, model, device)

    log.info("Running inference (forward pass for all patients)…")
    results = run_inference(groups, labels, cohort, model, device)

    abmil_manifest_path = OUTPUT_DIR / "abmil_manifest.json"
    with open(abmil_manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    n_resp = sum(1 for r in results.values() if r["true_label"] == 1)
    n_non  = sum(1 for r in results.values() if r["true_label"] == 0)
    log.info("=" * 60)
    log.info(f"  Patients processed : {len(results)}")
    log.info(f"  Labels             : {n_resp} responder / {n_non} non-responder "
             f"/ {len(results) - n_resp - n_non} unlabelled")
    log.info(f"  Total tiles pooled : {sum(r['n_tiles'] for r in results.values()):,}")
    log.info(f"  ABMIL manifest     → {abmil_manifest_path}")
    log.info("=" * 60)
    log.info("Step 3 complete. Next → Step 4: Cross-Modal Fusion.")


if __name__ == "__main__":
    try:
        main()
    except CohortError as e:
        raise SystemExit(f"\nCohort problem: {e}\n")
