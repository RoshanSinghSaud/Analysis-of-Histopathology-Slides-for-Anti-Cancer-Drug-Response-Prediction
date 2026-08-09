"""
=============================================================================
STEP 3: ABMIL — Attention-Based Multiple Instance Learning Aggregation
=============================================================================

Converts per-slide tile features → single slide embedding + attention scores.

Input  : output/features/feature_manifest.json
           TCGA-A2-A0EP_features.pt  [10036, 768]
           TCGA-C8-A131_features.pt  [33252, 768]

Output : output/abmil/
           TCGA-A2-A0EP_slide_embedding.pt   [1, 256]  slide-level vector
           TCGA-A2-A0EP_attention_scores.pt  [10036, 1] per-tile weights
           TCGA-C8-A131_slide_embedding.pt   [1, 256]
           TCGA-C8-A131_attention_scores.pt  [33252, 1]
           abmil_manifest.json               input to Step 4

NOTE: With only 2 slides we run in INFERENCE / VALIDATION MODE.
      The ABMIL network is initialized with random weights and does a
      forward pass — this validates shapes and the full pipeline.
      Training is stubbed and activates automatically once you have 200+ slides.

Run:
    conda activate histopath
    python 03_abmil_aggregation.py
=============================================================================
"""

import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# =============================================================================
# ── CONFIG ───────────────────────────────────────────────────────────────────
# =============================================================================

FEATURE_MANIFEST = Path.home() / "Desktop/Research/output/features/feature_manifest.json"
OUTPUT_DIR       = Path.home() / "Desktop/Research/output/abmil"

# ABMIL architecture dims
INPUT_DIM  = 1024    # UNI embed dim (match Step 1)
HIDDEN_DIM = 256    # projection dim — slide embedding will be this size
ATTN_DIM   = 128    # attention network internal dim

# Training config (only used when n_slides >= MIN_SLIDES_TO_TRAIN)
MIN_SLIDES_TO_TRAIN = 10   # won't attempt training below this
LEARNING_RATE       = 1e-4
NUM_EPOCHS          = 20
LABEL_MAP           = {    # drug response → integer class
    "Complete Response"  : 1,
    "Partial Response"   : 1,   # group as responder
    "Stable Disease"     : 0,
    "Clinical Progressive Disease": 0,
    "unknown"            : -1,  # excluded from training
}

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
        tile features [N, input_dim]
            → feature projection  [N, hidden_dim]
            → attention network   [N, 1]  (gated attention)
            → weighted sum        [1, hidden_dim]  ← slide embedding
            → classifier head     [1, n_classes]

    The attention scores tell you WHICH TILES the model focused on.
    These are saved and used in Step 6 for heatmap generation.
    """

    def __init__(
        self,
        input_dim:  int = 768,
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
            x : [N, input_dim]  — all tile features for one slide

        Returns:
            logits       : [1, n_classes]
            slide_embed  : [1, hidden_dim]
            attn_scores  : [N, 1]  (after softmax — sum to 1)
        """
        # Project
        h = self.feature_proj(x)                        # [N, hidden_dim]

        # Gated attention scores
        a_V = torch.tanh(self.attn_V(h))               # [N, attn_dim]
        a_U = torch.sigmoid(self.attn_U(h))            # [N, attn_dim]
        a   = self.attn_w(a_V * a_U)                   # [N, 1]
        a   = F.softmax(a, dim=0)                      # [N, 1]  sums to 1

        # Weighted aggregation
        slide_embed = (a * h).sum(dim=0, keepdim=True) # [1, hidden_dim]

        # Classification
        logits = self.classifier(slide_embed)           # [1, n_classes]

        return logits, slide_embed, a


# =============================================================================
# Helpers
# =============================================================================

def get_label(label_dict: dict) -> int:
    """Convert manifest label dict → integer class. Returns -1 if unknown."""
    response = label_dict.get("drug_response", "unknown")
    return LABEL_MAP.get(response, -1)


def load_features(feature_path: str, device: torch.device) -> torch.Tensor:
    """Load a .pt feature tensor and move to device."""
    t = torch.load(feature_path, map_location="cpu", weights_only=True)
    return t.to(device)


# =============================================================================
# Inference — forward pass, save embeddings + attention scores
# =============================================================================

def run_inference(manifest: dict, model: nn.Module, device: torch.device) -> dict:
    model.eval()
    results = {}

    with torch.no_grad():
        for slide_id, slide_data in tqdm(manifest.items(), desc="Slides"):
            feat_path = slide_data.get("feature_path")
            if not feat_path or not Path(feat_path).exists():
                log.warning(f"[{slide_id}] Feature file missing — skipping")
                continue

            x = load_features(feat_path, device)       # [N, 768]
            logits, slide_embed, attn_scores = model(x)

            probs = F.softmax(logits, dim=-1)           # [1, 2]
            pred  = logits.argmax(dim=-1).item()

            # Save outputs
            embed_path = OUTPUT_DIR / f"{slide_id}_slide_embedding.pt"
            attn_path  = OUTPUT_DIR / f"{slide_id}_attention_scores.pt"
            torch.save(slide_embed.cpu(),  embed_path)
            torch.save(attn_scores.cpu(),  attn_path)

            label    = get_label(slide_data.get("label", {}))
            response = slide_data.get("label", {}).get("drug_response", "unknown")

            log.info(
                f"[{slide_id}] tiles={x.shape[0]:,}  "
                f"embed={list(slide_embed.shape)}  "
                f"pred_class={pred}  prob={probs[0][pred]:.3f}  "
                f"true_label='{response}'"
            )

            results[slide_id] = {
                "slide_embedding_path" : str(embed_path),
                "attention_scores_path": str(attn_path),
                "embed_shape"          : list(slide_embed.shape),
                "n_tiles"              : x.shape[0],
                "pred_class"           : pred,
                "pred_prob"            : probs[0][pred].item(),
                "true_label"           : label,
                "drug_response"        : response,
                "label"                : slide_data.get("label", {}),
                "status"               : "done",
            }

    return results


# =============================================================================
# Training loop (activates at 200+ slides)
# =============================================================================

def run_training(manifest: dict, model: nn.Module, device: torch.device):
    """
    Full supervised training of ABMIL with cross-entropy loss.
    Skipped automatically when n_slides < MIN_SLIDES_TO_TRAIN.
    """
    # Filter slides with valid labels
    trainable = {
        sid: d for sid, d in manifest.items()
        if get_label(d.get("label", {})) != -1
    }

    if len(trainable) < MIN_SLIDES_TO_TRAIN:
        log.info(
            f"Only {len(trainable)} labelled slides — skipping training "
            f"(need {MIN_SLIDES_TO_TRAIN}+). Running inference only."
        )
        return

    log.info(f"Training ABMIL on {len(trainable)} slides for {NUM_EPOCHS} epochs…")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        correct    = 0

        for slide_id, slide_data in trainable.items():
            feat_path = slide_data.get("feature_path")
            label     = get_label(slide_data.get("label", {}))

            x = load_features(feat_path, device)
            y = torch.tensor([label], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, _, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (logits.argmax(-1) == y).sum().item()

        acc = correct / len(trainable)
        log.info(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  loss={epoch_loss:.4f}  acc={acc:.2%}")

    # Save trained weights
    weights_path = OUTPUT_DIR / "abmil_weights.pt"
    torch.save(model.state_dict(), weights_path)
    log.info(f"Trained weights saved → {weights_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load feature manifest ─────────────────────────────────────────────
    if not FEATURE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {FEATURE_MANIFEST}\n"
            "Run Step 2 (02_feature_extraction_local.py) first."
        )

    with open(FEATURE_MANIFEST) as f:
        manifest = json.load(f)

    # Filter to done slides only
    manifest = {k: v for k, v in manifest.items() if v.get("status") in ("done", "skipped")}
    log.info(f"Loaded {len(manifest)} slides from feature manifest")

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        log.warning("No GPU — running on CPU")

    log.info(f"Device: {device}")

    # ── Build model ───────────────────────────────────────────────────────
    model = ABMIL(
        input_dim  = INPUT_DIM,
        hidden_dim = HIDDEN_DIM,
        attn_dim   = ATTN_DIM,
        n_classes  = 2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"ABMIL model built — {total_params:,} parameters")

    # ── Train (skipped for < 10 slides) ──────────────────────────────────
    run_training(manifest, model, device)

    # ── Inference — always runs ───────────────────────────────────────────
    log.info("Running inference (forward pass for all slides)…")
    results = run_inference(manifest, model, device)

    # ── Save ABMIL manifest ───────────────────────────────────────────────
    abmil_manifest_path = OUTPUT_DIR / "abmil_manifest.json"
    with open(abmil_manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"  Slides processed : {len(results)}")
    for sid, r in results.items():
        log.info(
            f"  {sid} → embed {r['embed_shape']}  "
            f"attn [{r['n_tiles']}, 1]  "
            f"pred={r['pred_class']}  true='{r['drug_response']}'"
        )
    log.info(f"  ABMIL manifest   → {abmil_manifest_path}")
    log.info("=" * 60)
    log.info("Step 3 complete. Next → Step 4: Cross-Modal Fusion.")


if __name__ == "__main__":
    main()
