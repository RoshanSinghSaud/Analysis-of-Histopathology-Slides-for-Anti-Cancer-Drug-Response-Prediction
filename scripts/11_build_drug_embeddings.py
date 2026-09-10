#!/usr/bin/env python3
"""
11_build_drug_embeddings.py — the FROZEN pretrained molecular encoder (§6).

Turns each drug's verified SMILES into a fixed vector using a pretrained
chemical language model, and writes the result to a file. Nothing here ever
sees a patient, a label or a fold: this script reads `drugs/drug_smiles_map.json`
and writes `drugs/drug_embeddings_<model>.npz`, and that is the whole of its
contact with the study.

WHY THIS IS A SEPARATE SCRIPT, RUN ONCE
---------------------------------------
The encoder is frozen by §6, so running it inside the cross-validation loop
would recompute an identical answer ~5 x 4 x n_seeds times and would put a
transformer download on the critical path of the primary endpoint. Computing it
once into a versioned artefact with recorded provenance is both cheaper and more
auditable: the file that fed the ablation can be diffed, archived and shipped
with the paper.

THE TOKENIZER GUARD — READ THIS BEFORE CHANGING --model
-------------------------------------------------------
A SMILES tokenizer that lacks a token for a symbol may DROP that symbol
silently. This is not hypothetical. Measured on this project's own 94 Arm A
molecules on 2026-08-29:

    DeepChem/ChemBERTa-77M-MTR  and  -77M-MLM  and  ChemBERTa-10M-MTR
        71 of 94 molecules do not survive a tokenize -> decode round trip.
        'Cl' collapses to 'C', '[nH]' to 'n', stereocentres '[C@H]' to 'C',
        and '[Pt+2]' to 'P'. These are token-id collisions, not display
        artefacts: carmustine  O=NN(CCCl)C(=O)NCCCl  and its dechloro
        analogue O=NN(CCC)C(=O)NCCC  encode to IDENTICAL input_ids.

Under such an encoder cisplatin, carboplatin and oxaliplatin — three of the
most-administered agents in TCGA — become near-indistinguishable inputs, and
the with-SMILES arm is handed a degraded chemistry signal while still paying
full parameter cost. The resulting null would be uninterpretable, and nothing
in the training loop would raise.

So this script REFUSES to emit embeddings unless every molecule survives a
round trip and no two distinct molecules collide. Do not weaken these checks to
make a different model load; pick a model that passes them.

seyonec/ChemBERTa-zinc-base-v1 passes both at 94/94 and is the pre-registered
choice. ibm/MoLFormer-XL-both-10pct also passes, but loading it requires
executing code shipped in its Hugging Face repository (trust_remote_code=True),
which is a deliberate decision and is not enabled here.

USAGE
-----
    python scripts/11_build_drug_embeddings.py --tag arm_a
    python scripts/11_build_drug_embeddings.py            # every resolvable drug

Output:
    drugs/drug_embeddings_chemberta_zinc.npz    names + [n, 768] float32
    drugs/drug_embeddings_chemberta_zinc.json   provenance
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

log = logging.getLogger("drugenc")

# Pinned so the artefact is reproducible. A moving 'main' would let the encoder
# change under the study without the run log noticing.
DEFAULT_MODEL    = "seyonec/ChemBERTa-zinc-base-v1"
DEFAULT_REVISION = "761d6a18cf99db371e0b43baf3e2d21b3e865a20"
SHORT_NAME       = "chemberta_zinc"


# =============================================================================
# Inputs
# =============================================================================
def load_smiles(tag: str | None) -> dict:
    """
    {drug_name: smiles} for the drugs in scope.

    Names are taken from the cohort CSV when --tag is given, so the artefact
    covers exactly the analysable drug set and a missing molecule is caught
    here rather than at training time. The SMILES themselves come only from the
    verified map — never from the CSV, never guessed.
    """
    blob = json.loads((BASE_DIR / "drugs" / "drug_smiles_map.json").read_text())
    dmap = blob["map"]

    if tag:
        csv_path = BASE_DIR / "cohort" / f"cohort_{tag}.csv"
        if not csv_path.exists():
            sys.exit(f"no cohort CSV for tag '{tag}': {csv_path}")
        with csv_path.open() as fh:
            names = sorted({r["drug"] for r in csv.DictReader(fh)})
        log.info(f"  {len(names)} distinct drug strings in cohort_{tag}.csv")
    else:
        names = sorted(dmap)
        log.info(f"  {len(names)} drug strings in the verified map")

    out, unresolved = {}, []
    for name in names:
        entry = dmap.get(name.lower().strip())
        if entry is None or entry.get("smiles") is None:
            unresolved.append(name)
            continue
        out[name] = entry["smiles"]

    if unresolved:
        # Expected and pre-registered: biologics have no small-molecule
        # structure. They are excluded from the analysis, not zero-filled.
        log.info(f"  {len(unresolved)} drugs have no resolvable structure and are "
                 f"excluded (biologics / unresolved): {', '.join(unresolved[:6])}"
                 f"{' …' if len(unresolved) > 6 else ''}")
    log.info(f"  {len(out)} molecules to encode")
    if not out:
        sys.exit("nothing to encode")
    return out


# =============================================================================
# The guard
# =============================================================================
def check_tokenizer(tok, smiles: dict) -> None:
    """
    Abort unless the tokenizer is information-preserving on THIS drug set.

    Two independent checks, because they fail differently:

      round trip   catches symbols the tokenizer drops or rewrites.
      collision    catches two distinct molecules that encode identically,
                   which is the failure that actually corrupts the ablation.

    A round-trip failure without a collision is still a refusal: it means the
    model is reading a molecule that is not the one in the map.
    """
    lossy, ids_seen = [], {}
    collisions = []
    for name, smi in smiles.items():
        ids = tuple(tok(smi)["input_ids"])
        rt = tok.decode(ids, skip_special_tokens=True).replace(" ", "")
        if rt != smi:
            lossy.append((name, smi, rt))
        if ids in ids_seen and ids_seen[ids][1] != smi:
            collisions.append((ids_seen[ids][0], name, ids_seen[ids][1], smi))
        ids_seen.setdefault(ids, (name, smi))

    if lossy or collisions:
        print("\n" + "=" * 78)
        print("TOKENIZER REJECTED — this model cannot represent this drug set")
        print("=" * 78)
        if lossy:
            print(f"\n{len(lossy)}/{len(smiles)} molecules do not survive a "
                  f"tokenize -> decode round trip:\n")
            for n, s, r in lossy[:12]:
                print(f"  {n}\n      in : {s}\n      out: {r}")
            if len(lossy) > 12:
                print(f"  … and {len(lossy) - 12} more")
        if collisions:
            print(f"\n{len(collisions)} DISTINCT molecules encode to identical "
                  f"token ids — the model literally cannot tell them apart:\n")
            for a, b, sa, sb in collisions[:12]:
                print(f"  {a}  ==  {b}\n      {sa}\n      {sb}")
        print("\nDo not relax this check. Choose a model whose tokenizer covers "
              "these symbols.\n")
        sys.exit(2)

    log.info(f"  tokenizer guard: PASS — {len(smiles)}/{len(smiles)} round-trip "
             f"exact, 0 collisions")


# =============================================================================
# Encoding
# =============================================================================
def embed(smiles: dict, model_id: str, revision: str, pooling: str, device: str):
    """
    Frozen forward pass. eval() + no_grad, so dropout is off and the result is
    a deterministic function of (model, revision, pooling, SMILES) alone.

    TWO DELIBERATE CHOICES, both measured on 2026-08-29, both about determinism:

    1. ONE MOLECULE PER FORWARD PASS, never a padded batch. Padding perturbs the
       pooled vector at ~1.7e-6 (measured: the same SMILES encoded alone versus
       encoded beside a long one is not bitwise equal). Tiny, but it makes the
       artefact depend on batch composition, so the recorded sha256 would not
       reproduce and two molecules with identical SMILES could receive different
       vectors. 82 short sequences through a 44M-parameter model costs seconds;
       there is nothing to gain by batching here.

    2. EACH DISTINCT SMILES IS ENCODED ONCE and shared by every name that maps
       to it. In this cohort 94 drug names cover only 82 molecules — salt and
       free-base forms of the same agent ('imatinib' / 'imatinib mesylate',
       'vincristine' / 'vincristine sulfate', 'nab-paclitaxel' / 'paclitaxel')
       resolve to one structure in the verified map. They are the same molecule,
       so they must get the same vector, exactly.

    Mean pooling averages the real token positions only, excluding the special
    <s>/</s> tokens: those add a constant that would survive the downstream PCA
    as a spurious near-constant direction.

    Weights are loaded with use_safetensors=False on purpose. This checkpoint
    ships only pytorch_model.bin, and transformers will otherwise chase a
    community safetensors-conversion pull request (refs/pr/N) whose contents are
    NOT the pinned revision — silently defeating the pin.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    check_tokenizer(tok, smiles)

    model = AutoModel.from_pretrained(model_id, revision=revision,
                                      use_safetensors=False)
    model.eval().to(device)

    uniq = sorted(set(smiles.values()))
    log.info(f"  {len(smiles)} names -> {len(uniq)} distinct molecules to encode")

    by_smiles = {}
    with torch.no_grad():
        for smi in uniq:
            enc = tok(smi, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state[0]          # [T, H], no padding
            if pooling == "cls":
                pooled = hidden[0]
            else:
                ids = enc["input_ids"][0].tolist()
                keep = [i for i, t in enumerate(ids) if t not in tok.all_special_ids]
                if not keep:
                    sys.exit(f"every token of {smi!r} is a special token")
                pooled = hidden[keep].mean(0)
            by_smiles[smi] = pooled.float().cpu().numpy()

    names = sorted(smiles)
    X = np.stack([by_smiles[smiles[n]] for n in names]).astype(np.float32)

    # A degenerate encoder would be worse than the fingerprints it replaces and
    # must not reach the ablation silently. Distinctness is checked over
    # MOLECULES, not names: shared salt forms are supposed to collide.
    if not np.isfinite(X).all():
        sys.exit("encoder produced non-finite values")
    U = np.stack([by_smiles[s] for s in uniq])
    if len(uniq) > 1:
        dup = len(uniq) - len({U[i].tobytes() for i in range(len(uniq))})
        if dup:
            sys.exit(f"{dup} DISTINCT molecules received byte-identical embeddings")
        if float(U.std(0).max()) < 1e-6:
            sys.exit("encoder output is constant across molecules")
    log.info(f"  distinctness: PASS — {len(uniq)}/{len(uniq)} molecules have "
             f"distinct vectors")
    return names, X, len(uniq)


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="cohort tag (arm_a, brca). Omit for the whole map.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help="pinned commit sha; a moving branch is not reproducible")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--device", default="cpu",
                    help="cpu is fine — this is ~100 short sequences, once")
    ap.add_argument("--short-name", default=SHORT_NAME,
                    help="filename stem for the artefact")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info(f"Frozen molecular encoder: {a.model} @ {a.revision[:12]} "
             f"({a.pooling}-pooled)")

    smiles = load_smiles(a.tag)
    names, X, n_uniq = embed(smiles, a.model, a.revision, a.pooling, a.device)

    out_dir = Path(a.out_dir) if a.out_dir else BASE_DIR / "drugs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"drug_embeddings_{a.short_name}"

    np.savez(stem.with_suffix(".npz"),
             names=np.array(names, dtype=object), X=X)

    import torch, transformers
    payload = json.dumps({n: smiles[n] for n in names}, sort_keys=True).encode()
    prov = {
        "model_id":        a.model,
        "revision":        a.revision,
        "pooling":         a.pooling,
        "embed_dim":       int(X.shape[1]),
        "n_drug_names":    len(names),
        "n_molecules":     n_uniq,
        "batching":        "none — one molecule per forward pass (see embed())",
        "weights_file":    "pytorch_model.bin (use_safetensors=False, pin-safe)",
        "cohort_tag":      a.tag,
        "smiles_sha256":   hashlib.sha256(payload).hexdigest(),
        "embeddings_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
        "tokenizer_guard": "round-trip exact and collision-free on all molecules",
        "torch":           torch.__version__,
        "transformers":    transformers.__version__,
        "numpy":           np.__version__,
        "script":          Path(__file__).name,
    }
    stem.with_suffix(".json").write_text(json.dumps(prov, indent=2) + "\n")

    log.info(f"\n  wrote {stem.with_suffix('.npz').name}  [{X.shape[0]}, {X.shape[1]}] float32")
    log.info(f"  wrote {stem.with_suffix('.json').name}  (provenance)")
    log.info(f"  embeddings sha256 {prov['embeddings_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
