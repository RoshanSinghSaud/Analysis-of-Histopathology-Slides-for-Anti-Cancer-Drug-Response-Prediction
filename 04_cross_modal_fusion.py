"""
=============================================================================
STEP 4: Cross-Modal Fusion — Histology + Drug Molecular Fingerprint
=============================================================================

Combines two modalities per slide:
  1. Histology  : [1, 256]  slide embedding from Step 3 (ABMIL)
  2. Drug       : [1, 1024] Morgan fingerprint computed from SMILES (PubChem)

Fusion strategies (set FUSION_MODE in CONFIG):
  - "concat"      : simple concatenation → [1, 1280]  (default, always works)
  - "mlp"         : concat → small MLP  → [1, 256]   (learnable, needs 10+ slides)
  - "cross_attn"  : cross-attention     → [1, 256]   (best, needs 50+ slides)

Input  : output/abmil/abmil_manifest.json
Output : output/fusion/
           TCGA-A2-A0EP_fused.pt        fused vector [1, 256 or 1280]
           TCGA-C8-A131_fused.pt
           drug_smiles_cache.json       drug name → SMILES (reused at 200 cases)
           fusion_manifest.json         input to Step 5

Run:
    conda activate histopath
    conda install -c conda-forge rdkit   ← one-time install
    python 04_cross_modal_fusion.py
=============================================================================
"""

import json
import logging
import time
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ── CONFIG ───────────────────────────────────────────────────────────────────
# =============================================================================

ABMIL_MANIFEST = Path.home() / "Desktop/Research/output/abmil/abmil_manifest.json"
OUTPUT_DIR     = Path.home() / "Desktop/Research/output/fusion"

# Fusion mode: "concat" | "mlp" | "cross_attn"
# For 2 slides use "concat" — no training needed, shapes still correct
# For 200 slides switch to "mlp" or "cross_attn"
FUSION_MODE    = "concat"

# Drug fingerprint settings
FINGERPRINT_BITS = 1024   # Morgan fingerprint length (standard)
MORGAN_RADIUS    = 2      # radius 2 = ECFP4 (standard in cheminformatics)

# MLP fusion dims (only used if FUSION_MODE = "mlp" or "cross_attn")
HIST_DIM    = 256         # must match HIDDEN_DIM from Step 3
DRUG_DIM    = FINGERPRINT_BITS
FUSED_DIM   = 256         # output dim of learnable fusion

# Verified drug -> structure map, produced by normalize_drugs_to_smiles.py.
# There is deliberately NO hardcoded SMILES dictionary here. The one that used
# to live at this spot had 9 of its 11 entries pointing at the wrong molecule
# (its "anastrozole" was C9H10N2; the real drug is C17H19N5), and because it was
# consulted before PubChem those errors could never be corrected.
DRUG_MAP_PATH = Path.home() / "Desktop/Research/drug_smiles_map.json"

# What to do with a drug that has no small-molecule structure (antibodies,
# vaccines, oncolytic viruses) or that could not be resolved at all.
#   "exclude" - raise, so the caller drops the patient from the cohort  <- default
#   "flag"    - return None and let the caller route it to a separate embedding
# Zero-filling is NOT an option. A zero fingerprint is byte-identical to "no
# drug given", which silently converts those patients into without-SMILES
# controls inside the with-SMILES model and drags the headline ablation to null
# for a reason that appears in no log.
UNRESOLVED_POLICY = "exclude"

# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Drug SMILES acquisition
# =============================================================================

_DRUG_MAP = None


class UnresolvedDrugError(RuntimeError):
    """Raised when a drug has no verified structure. Never silently ignored."""


def load_drug_map() -> dict:
    """Load the verified name -> structure map, once."""
    global _DRUG_MAP
    if _DRUG_MAP is None:
        if not DRUG_MAP_PATH.exists():
            raise FileNotFoundError(
                f"{DRUG_MAP_PATH} not found.\n"
                "Run normalize_drugs_to_smiles.py first - the drug branch must "
                "not be fed unverified structures."
            )
        blob = json.loads(DRUG_MAP_PATH.read_text())
        _DRUG_MAP = blob["map"]
        log.info(f"  Loaded {len(_DRUG_MAP)} drug strings -> "
                 f"{blob['n_distinct_molecules']} distinct molecules "
                 f"({blob['n_biologic']} biologics, {blob['n_unresolved']} unresolved)")
    return _DRUG_MAP


def resolve_smiles(drug_name: str, cache: dict | None = None) -> str:
    """drug name -> canonical parent SMILES, or raise.

    No network call and no guessing happens here. Every structure was resolved
    by PubChem CID and salt-stripped ahead of time by normalize_drugs_to_smiles.py,
    and every one of them is listed in drug_resolution_report.md for inspection.
    """
    dmap = load_drug_map()
    key = drug_name.lower().strip()
    entry = dmap.get(key)

    if entry is None:
        raise UnresolvedDrugError(
            f"'{drug_name}' is not in {DRUG_MAP_PATH.name}. Re-run "
            "normalize_drugs_to_smiles.py so this drug is resolved and reviewed."
        )
    if entry["smiles"] is None:
        raise UnresolvedDrugError(
            f"'{drug_name}' has no small-molecule structure (kind="
            f"{entry['kind']}). Decide explicitly whether to exclude it or give "
            "it a separate embedding - do not zero-fill it."
        )
    return entry["smiles"]


# =============================================================================
# Molecular fingerprint computation
# =============================================================================

def smiles_to_fingerprint(smiles: str) -> torch.Tensor | None:
    """
    Convert SMILES string → Morgan fingerprint tensor [1, FINGERPRINT_BITS].
    Uses RDKit (conda install -c conda-forge rdkit).
    Morgan radius=2 = ECFP4, the standard in drug discovery.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning(f"  RDKit could not parse SMILES: {smiles[:50]}")
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=MORGAN_RADIUS,
            nBits=FINGERPRINT_BITS,
        )
        # Convert RDKit bit vector → numpy → torch
        import numpy as np
        arr = np.zeros((FINGERPRINT_BITS,), dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)

        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1, 1024]

    except ImportError:
        raise ImportError(
            "RDKit not installed.\n"
            "Run: conda install -c conda-forge rdkit"
        )
    except Exception as e:
        log.warning(f"  Fingerprint computation failed: {e}")
        return None


def get_drug_fingerprint(drug_name: str, cache: dict | None = None) -> torch.Tensor:
    """drug name -> [1, FINGERPRINT_BITS] Morgan fingerprint.

    Raises instead of returning zeros. See UNRESOLVED_POLICY above for why.
    """
    smiles = resolve_smiles(drug_name)          # raises if unverified
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        raise UnresolvedDrugError(
            f"RDKit could not build a fingerprint for '{drug_name}' "
            f"(SMILES: {smiles}). This is a bug in the drug map, not a patient "
            "to be zero-filled."
        )
    if int(fp.sum().item()) == 0:
        raise UnresolvedDrugError(
            f"'{drug_name}' produced an all-zero fingerprint, which is "
            "indistinguishable from 'no drug'. Refusing to continue."
        )
    return fp


# =============================================================================
# Fusion modules
# =============================================================================

class ConcatFusion(nn.Module):
    """
    Simplest fusion: just concatenate histology + drug vectors.
    Output dim = HIST_DIM + DRUG_DIM = 256 + 1024 = 1280
    No parameters, no training needed.
    """
    def forward(self, hist: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        return torch.cat([hist, drug], dim=-1)   # [1, 1280]


class MLPFusion(nn.Module):
    """
    Learnable fusion: concat → MLP → lower-dim fused vector.
    Output dim = FUSED_DIM = 256
    Recommended for 10–100 slides.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIST_DIM + DRUG_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, FUSED_DIM),
            nn.ReLU(),
        )

    def forward(self, hist: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hist, drug], dim=-1)   # [1, 1280]
        return self.net(x)                     # [1, 256]


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: histology queries drug features (and vice versa).
    Best at capturing interactions between modalities.
    Output dim = FUSED_DIM = 256
    Recommended for 50+ slides.
    """
    def __init__(self, n_heads: int = 4):
        super().__init__()

        # Project both modalities to same dim
        self.hist_proj = nn.Linear(HIST_DIM, FUSED_DIM)
        self.drug_proj = nn.Linear(DRUG_DIM, FUSED_DIM)

        # Cross-attention: hist attends to drug
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=FUSED_DIM,
            num_heads=n_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(FUSED_DIM)
        self.ffn  = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM * 2),
            nn.GELU(),
            nn.Linear(FUSED_DIM * 2, FUSED_DIM),
        )

    def forward(self, hist: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        # Project to common dim
        h = self.hist_proj(hist)   # [1, 256]
        d = self.drug_proj(drug)   # [1, 256]

        # Add sequence dim for attention: [1, 1, 256]
        h = h.unsqueeze(1)
        d = d.unsqueeze(1)

        # Hist queries drug (what drug info is relevant to this tumor?)
        attn_out, _ = self.cross_attn(query=h, key=d, value=d)

        # Residual + FFN
        out = self.norm(h + attn_out)       # [1, 1, 256]
        out = out + self.ffn(out)
        return out.squeeze(1)               # [1, 256]


FUSION_MODULES = {
    "concat"     : ConcatFusion,
    "mlp"        : MLPFusion,
    "cross_attn" : CrossAttentionFusion,
}


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ABMIL manifest ───────────────────────────────────────────────
    if not ABMIL_MANIFEST.exists():
        raise FileNotFoundError(
            f"ABMIL manifest not found: {ABMIL_MANIFEST}\n"
            "Run Step 3 (03_abmil_aggregation.py) first."
        )

    with open(ABMIL_MANIFEST) as f:
        abmil_manifest = json.load(f)

    log.info(f"ABMIL manifest loaded — {len(abmil_manifest)} slides")

    # ── Load SMILES cache (persist across runs) ───────────────────────────
    cache_path = OUTPUT_DIR / "drug_smiles_cache.json"
    smiles_cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            smiles_cache = json.load(f)
        log.info(f"SMILES cache loaded — {len(smiles_cache)} drugs cached")

    # ── Build fusion model ────────────────────────────────────────────────
    if FUSION_MODE not in FUSION_MODULES:
        raise ValueError(f"Unknown FUSION_MODE '{FUSION_MODE}'. Choose: concat | mlp | cross_attn")

    fusion_model = FUSION_MODULES[FUSION_MODE]()
    fusion_model.eval()

    fused_dim = (
        HIST_DIM + DRUG_DIM if FUSION_MODE == "concat" else FUSED_DIM
    )

    log.info(f"Fusion mode : {FUSION_MODE}")
    log.info(f"Hist dim    : {HIST_DIM}")
    log.info(f"Drug dim    : {DRUG_DIM} (Morgan fingerprint, radius={MORGAN_RADIUS})")
    log.info(f"Fused dim   : {fused_dim}")

    # ── Per-slide fusion ──────────────────────────────────────────────────
    results = {}

    for slide_id, slide_data in abmil_manifest.items():

        log.info(f"\n[{slide_id}]")

        # 1. Load slide embedding from Step 3
        embed_path = slide_data.get("slide_embedding_path")
        if not embed_path or not Path(embed_path).exists():
            log.warning(f"  Slide embedding not found — skipping")
            continue

        hist_embed = torch.load(embed_path, map_location="cpu", weights_only=True)
        # hist_embed: [1, 256]
        log.info(f"  Histology embedding : {list(hist_embed.shape)}")

        # 2. Get drug name from label
        label      = slide_data.get("label", {})
        drug_name  = label.get("drug_name", "unknown")
        log.info(f"  Drug                : {drug_name}")

        # 3. Compute drug fingerprint
        drug_fp = get_drug_fingerprint(drug_name, smiles_cache)
        # drug_fp: [1, 1024]
        log.info(f"  Drug fingerprint    : {list(drug_fp.shape)}  "
                 f"active_bits={int(drug_fp.sum().item())}/{FINGERPRINT_BITS}")

        # 4. Fuse
        with torch.no_grad():
            fused = fusion_model(hist_embed, drug_fp)
        # fused: [1, fused_dim]
        log.info(f"  Fused vector        : {list(fused.shape)}")

        # 5. Save fused vector
        fused_path = OUTPUT_DIR / f"{slide_id}_fused.pt"
        torch.save(fused.cpu(), fused_path)

        results[slide_id] = {
            "fused_path"         : str(fused_path),
            "fused_shape"        : list(fused.shape),
            "fusion_mode"        : FUSION_MODE,
            "hist_dim"           : HIST_DIM,
            "drug_dim"           : DRUG_DIM,
            "fused_dim"          : fused_dim,
            "drug_name"          : drug_name,
            "drug_smiles"        : smiles_cache.get(drug_name.lower().strip(), "unknown"),
            "drug_active_bits"   : int(drug_fp.sum().item()),
            "true_label"         : slide_data.get("true_label", -1),
            "drug_response"      : slide_data.get("drug_response", "unknown"),
            "label"              : label,
            "status"             : "done",
        }

    # ── Save SMILES cache ─────────────────────────────────────────────────
    with open(cache_path, "w") as f:
        json.dump(smiles_cache, f, indent=2)
    log.info(f"\nSMILES cache saved → {cache_path}")

    # ── Save fusion manifest ──────────────────────────────────────────────
    fusion_manifest_path = OUTPUT_DIR / "fusion_manifest.json"
    with open(fusion_manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"  Slides fused : {len(results)}")
    for sid, r in results.items():
        log.info(
            f"  {sid}"
            f"\n    drug        = {r['drug_name']}"
            f"\n    active bits = {r['drug_active_bits']}/{FINGERPRINT_BITS}"
            f"\n    fused shape = {r['fused_shape']}"
            f"\n    response    = {r['drug_response']}"
        )
    log.info(f"\n  Fusion manifest → {fusion_manifest_path}")
    log.info("=" * 60)
    log.info("Step 4 complete. Next → Step 5: Drug Response Prediction.")


if __name__ == "__main__":
    main()
