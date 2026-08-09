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

# Manual SMILES fallback — used if PubChem API is unreachable
# Add drugs here as you encounter them in your dataset
MANUAL_SMILES = {
    "anastrozole" : "CC(C)(C#N)c1ccncc1",
    "arimidex"    : "CC(C)(C#N)c1ccncc1",   # brand name for anastrozole
    "tamoxifen"   : "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "taxol"       : "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(=O)O)(C2(O)C)C)(O)OC(=O)c2ccccc2)OC(=O)C)O)OC(=O)c2ccccc2",
    "paclitaxel"  : "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(=O)O)(C2(O)C)C)(O)OC(=O)c2ccccc2)OC(=O)C)O)OC(=O)c2ccccc2",
    "doxorubicin" : "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(=O)CO)Cc4c(O)c3C(=O)c12",
    "adriamycin"  : "COc1cccc2C(=O)c3c(O)c4CC(O)(CC(=O)CO)Cc4c(O)c3C(=O)c12",
    "trastuzumab" : None,   # antibody — no small-molecule SMILES; handled separately
    "herceptin"   : None,
    "cyclophosphamide": "ClCCN(CCCl)P1(=O)NCCCO1",
    "capecitabine": "CCOC(=O)N1CCC(OC(=O)c2cnc(F)cn2)C1",
    "letrozole"   : "N#Cc1cccc(C(c2ccc(C#N)cc2)n2ccnc2)c1",
    "exemestane"  : "O=C1CC[C@H]2[C@@H]3CC[C@](C=C)(C(=O)C)[C@@H]3CC[C@@H]12",
}

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

def fetch_smiles_pubchem(drug_name: str) -> str | None:
    """
    Query PubChem REST API for the canonical SMILES of a drug by name.
    Free, no API key needed.
    Returns SMILES string or None if not found.
    """
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(drug_name)}/property/IsomericSMILES/JSON"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            smiles = r.json()["PropertyTable"]["Properties"][0]["IsomericSMILES"]
            log.info(f"  PubChem → '{drug_name}': {smiles[:60]}{'…' if len(smiles) > 60 else ''}")
            return smiles
        else:
            log.warning(f"  PubChem returned {r.status_code} for '{drug_name}'")
            return None
    except Exception as e:
        log.warning(f"  PubChem query failed for '{drug_name}': {e}")
        return None


def resolve_smiles(drug_name: str, cache: dict) -> str | None:
    """
    Resolve SMILES for a drug name using:
      1. In-memory cache (already fetched this session)
      2. Manual fallback dict (hardcoded common drugs)
      3. PubChem API
    Returns SMILES string or None if drug is a biologic (antibody, etc.)
    """
    key = drug_name.lower().strip()

    # 1. Cache hit
    if key in cache:
        return cache[key]

    # 2. Manual fallback
    if key in MANUAL_SMILES:
        smiles = MANUAL_SMILES[key]
        cache[key] = smiles
        if smiles is None:
            log.info(f"  '{drug_name}' is a biologic — no SMILES fingerprint")
        return smiles

    # 3. PubChem API
    log.info(f"  Querying PubChem for '{drug_name}'…")
    smiles = fetch_smiles_pubchem(drug_name)
    time.sleep(0.2)   # be polite to PubChem API

    if smiles is None:
        # Try brand name → generic name fallback via PubChem synonym search
        log.info(f"  Retrying PubChem with synonym search…")
        smiles = fetch_smiles_pubchem(drug_name + " drug")
        time.sleep(0.2)

    cache[key] = smiles
    return smiles


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


def get_drug_fingerprint(drug_name: str, cache: dict) -> torch.Tensor | None:
    """Full pipeline: drug name → [1, 1024] fingerprint tensor."""
    smiles = resolve_smiles(drug_name, cache)

    if smiles is None:
        log.warning(f"  No SMILES for '{drug_name}' — will use zero fingerprint")
        return torch.zeros(1, FINGERPRINT_BITS, dtype=torch.float32)

    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return torch.zeros(1, FINGERPRINT_BITS, dtype=torch.float32)

    return fp   # [1, 1024]


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
