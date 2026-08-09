"""
normalize_drugs_to_smiles.py
============================
Turn the free-text drug names in TCGA into verified, canonical molecular
structures - and refuse to guess.

Why this script exists
----------------------
`diagnoses.treatments.therapeutic_agents` is free text typed at dozens of
institutions. The same molecule appears as a generic name, a brand name, and
several salt forms. If those are fed to the model as-is, one drug becomes
several unrelated inputs and the drug branch learns noise.

Three rules this script follows, each one fixing a real defect found in
04_cross_modal_fusion.py on 2026-08-09:

  1. NO HARDCODED SMILES. Every structure comes from PubChem, keyed by CID.
     The previous MANUAL_SMILES dict had 9 of 11 entries pointing at the wrong
     molecule (e.g. "anastrozole" was C9H10N2; the real drug is C17H19N5), and
     because it was consulted BEFORE PubChem those errors were permanent.

  2. RESOLVE BY CID, NOT BY STRING. PubChem's name->CID lookup already handles
     brand names (arimidex and anastrozole are both CID 2187). Salt forms it
     does NOT collapse - doxorubicin is CID 31703, doxorubicin hydrochloride is
     CID 443939 - so we strip counterions ourselves with RDKit and group on the
     parent structure's InChIKey.

  3. NEVER SILENTLY RETURN ZEROS. A drug that cannot be resolved is recorded
     with an explicit status and must be dealt with by a human. The old code
     returned a zero fingerprint, which is indistinguishable from "no drug" -
     silently converting those patients into without-SMILES controls inside the
     with-SMILES model, and driving the headline result to null for a reason
     that would never appear in any log.

Input : drug_names_all.json  (list of distinct lowercase drug strings)
        drug_vocabulary.json (per-drug patient counts, for the report)
Output: drug_smiles_map.json      <- the pipeline consumes this
        drug_resolution_report.md <- a human reads this before training anything
        pubchem_cache.json        <- so re-runs cost nothing

Run:  python normalize_drugs_to_smiles.py
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.SaltRemover import SaltRemover
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit is required:  conda install -c conda-forge rdkit")

HERE = Path(__file__).resolve().parent
NAMES_IN = HERE / "drug_names_all.json"
VOCAB_IN = HERE / "drug_vocabulary.json"
CACHE = HERE / "pubchem_cache.json"
MAP_OUT = HERE / "drug_smiles_map.json"
REPORT_OUT = HERE / "drug_resolution_report.md"

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SLEEP = 0.22          # PubChem asks for <=5 requests/second
MW_SMALL_MOLECULE_MAX = 1500.0

# Biologics have no meaningful small-molecule structure. Recognised by the
# WHO/USAN stem in the generic name. These are FLAGGED, never zero-filled.
BIOLOGIC_PATTERNS = [
    r"mab$", r"mab ",  # monoclonal antibody      - trastuzumab, bevacizumab
    r"cept$",          # receptor fusion protein  - aflibercept
    r"ase$",           # enzyme                   - asparaginase
    r"^interferon", r"^interleukin", r"^peginterferon",
    r"^filgrastim", r"^pegfilgrastim", r"^epoetin", r"^darbepoetin",
    r"^aldesleukin", r"^denileukin", r"^sipuleucel", r"^bcg",
    r"^bacillus calmette", r"vaccine", r"^immunoglobulin",
    # found empirically in the TCGA vocabulary on 2026-08-09
    r"^recombinant ", r"monoclonal antibody", r"immunotoxin", r"anticalin",
    r"oncolytic", r"^sargramostim", r"^vitespen", r"besudotox",
    r"^talimogene", r"^pelareorep", r"^lontucirev", r"^astuprotimut",
    r"antigen$", r"antigen ", r"^poly iclc", r"^polysaccharide-k",
    r"colony-stimulating factor", r"parvovec$", r"repvec$",
]

# Strings that are not a specific molecule at all. These patients cannot be
# given a drug input honestly, so their drug-outcome pairs are EXCLUDED from
# the cohort. They are not zero-filled and they are not guessed at.
PLACEHOLDER_PATTERNS = [
    r"^clinical trial", r"^hormone therapy$", r"^chemotherapy",
    r"^taxane compound$", r"^ras inhibitor$", r"^amino acid injection$",
    r"^total androgen blockade$", r"^thyroid extract$", r"^mistletoe extract$",
    r"^gonadotropin-releasing hormone analog$", r"^other",
    r"^unspecified", r"^combination", r"^antineoplastic$",
]

# Formulation wrappers around a real molecule. The MOLECULE is what the drug
# branch should see: 'pegylated liposomal doxorubicin hydrochloride' is
# doxorubicin in a lipid carrier, not a different chemical. Stripping these is
# a rewrite of the NAME only - the structure still comes from PubChem, and
# every rewrite is recorded in the report so it can be checked by eye.
FORMULATION_PREFIXES = [
    "pegylated liposomal", "liposomal pegylated", "nanoparticle albumin-bound",
    "albumin-bound", "pegylated", "liposomal", "micellar", "therapeutic",
]

# Delivery-format words appended to a real molecule, e.g. 'carmustine implant'
# is carmustine in a polymer wafer. Same reasoning as the prefixes.
FORMULATION_SUFFIXES = [
    "implant", "wafer", "injection", "tablets", "tablet", "capsules",
    "capsule", "solution", "suspension", "cream", "ointment",
]


def _matches(name: str, patterns) -> bool:
    n = name.strip().lower()
    return any(re.search(p, n) for p in patterns)


def is_biologic_name(name: str) -> bool:
    return _matches(name, BIOLOGIC_PATTERNS)


def is_placeholder_name(name: str) -> bool:
    return _matches(name, PLACEHOLDER_PATTERNS)


def name_variants(name: str):
    """Conservative rewrites to retry when the raw string is not in PubChem.

    Only leading formulation words are removed, and only as whole phrases. We
    deliberately do NOT try to salvage strings like
    'glutathione pegylated liposomal doxorubicin hydrochloride formulation
    2b3-101' - guessing there could silently attach the wrong molecule to a
    patient, which is the exact failure this whole script exists to prevent.
    """
    n = name.strip().lower()
    out = []
    for pref in FORMULATION_PREFIXES:
        if n.startswith(pref + " "):
            stripped = n[len(pref) + 1:].strip()
            if stripped and stripped not in out:
                out.append(stripped)
    for suf in FORMULATION_SUFFIXES:
        if n.endswith(" " + suf):
            stripped = n[: -(len(suf) + 1)].strip()
            if stripped and stripped not in out:
                out.append(stripped)
    return out


# ---------------------------------------------------------------------------
# PubChem
# ---------------------------------------------------------------------------
def _get(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                      # genuinely not in PubChem
            if e.code in (429, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt + 1)     # back off, PubChem throttles
                continue
            return None
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1 + attempt)
    return None


def name_to_cid(name: str):
    url = f"{PUBCHEM}/compound/name/{urllib.parse.quote(name)}/cids/JSON"
    d = _get(url)
    time.sleep(SLEEP)
    if not d:
        return None
    cids = (d.get("IdentifierList") or {}).get("CID") or []
    return cids[0] if cids else None


def cid_to_props(cid: int):
    """NOTE: PubChem returns the key 'SMILES', NOT 'IsomericSMILES'.

    The old code indexed the response with ["IsomericSMILES"], which now raises
    KeyError, was swallowed by a bare except, and made EVERY lookup return None.
    That single line is why the drug branch was receiving zero vectors.
    """
    props = "SMILES,MolecularFormula,MolecularWeight,IUPACName,Title"
    url = f"{PUBCHEM}/compound/cid/{cid}/property/{props}/JSON"
    d = _get(url)
    time.sleep(SLEEP)
    if not d:
        return None
    rows = (d.get("PropertyTable") or {}).get("Properties") or []
    if not rows:
        return None
    p = rows[0]
    smiles = p.get("SMILES") or p.get("ConnectivitySMILES") or p.get("IsomericSMILES")
    if not smiles:
        return None
    return {
        "cid": cid,
        "smiles_raw": smiles,
        "formula_raw": p.get("MolecularFormula"),
        "mw_raw": p.get("MolecularWeight"),
        "pubchem_title": p.get("Title"),
        "iupac": p.get("IUPACName"),
    }


# ---------------------------------------------------------------------------
# Chemistry: strip counterions, canonicalise, key on the parent structure
# ---------------------------------------------------------------------------
_remover = SaltRemover()


def parent_structure(smiles: str):
    """Return the drug's parent molecule, free of salts and counterions.

    'doxorubicin hydrochloride' and 'doxorubicin' must land on the same key, so
    the model treats 287 + 197 patients as one drug rather than two unrelated
    ones. Strategy: RDKit's SaltRemover, then keep the largest remaining
    covalent fragment (handles counterions SaltRemover's table doesn't list,
    e.g. the calcium in leucovorin calcium).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        stripped = _remover.StripMol(mol, dontRemoveEverything=True)
    except Exception:
        stripped = mol
    if stripped is None or stripped.GetNumAtoms() == 0:
        stripped = mol
    frags = Chem.GetMolFrags(stripped, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
        stripped = frags[0]
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        return None
    return {
        "smiles": Chem.MolToSmiles(stripped, canonical=True),
        "inchikey": Chem.MolToInchiKey(stripped),
        "formula": rdMolDescriptors.CalcMolFormula(stripped),
        "mw": round(Descriptors.MolWt(stripped), 2),
        "heavy_atoms": stripped.GetNumHeavyAtoms(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    names = json.loads(NAMES_IN.read_text())
    vocab = json.loads(VOCAB_IN.read_text()) if VOCAB_IN.exists() else {}
    recist_counts = vocab.get("recist_cohort_patient_counts", {})
    os_counts = vocab.get("os_cohort_patient_counts", {})

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    # Only successful structure lookups are worth caching. Re-deriving the
    # classification statuses is cheap and lets classification rules be edited
    # without a stale cache silently freezing the old answer.
    cache = {k: v for k, v in cache.items()
             if v.get("status") in ("resolved", "review_mw_too_large")}
    print(f"Resolving {len(names)} distinct drug strings "
          f"({len(cache)} already cached)\n")

    entries = {}
    for i, name in enumerate(names, 1):
        if name in cache:
            entries[name] = cache[name]
            continue

        rec = {"name": name, "status": None, "cid": None}

        if is_placeholder_name(name):
            rec["status"] = "not_a_specific_drug"
            rec["note"] = "free-text placeholder, not a molecule; excluded from cohort"
        elif is_biologic_name(name):
            rec["status"] = "biologic_no_smiles"
            rec["note"] = "matched a biologic naming stem; no small-molecule structure"
        else:
            cid = name_to_cid(name)
            if cid is None:
                # retry with formulation wrappers stripped, recording the rewrite
                for alt in name_variants(name):
                    cid = name_to_cid(alt)
                    if cid is not None:
                        rec["resolved_via_name"] = alt
                        break
            if cid is None:
                rec["status"] = "unresolved_not_in_pubchem"
            else:
                props = cid_to_props(cid)
                if props is None:
                    rec["status"] = "unresolved_no_structure_for_cid"
                    rec["cid"] = cid
                else:
                    par = parent_structure(props["smiles_raw"])
                    if par is None:
                        rec["status"] = "unresolved_rdkit_parse_failed"
                        rec.update(props)
                    else:
                        rec.update(props)
                        rec.update({
                            "parent_smiles": par["smiles"],
                            "inchikey": par["inchikey"],
                            "formula": par["formula"],
                            "mw": par["mw"],
                            "salt_stripped": par["smiles"] !=
                                             Chem.MolToSmiles(
                                                 Chem.MolFromSmiles(props["smiles_raw"]),
                                                 canonical=True),
                        })
                        if par["mw"] > MW_SMALL_MOLECULE_MAX:
                            rec["status"] = "review_mw_too_large"
                        else:
                            rec["status"] = "resolved"

        entries[name] = rec
        cache[name] = rec
        if i % 20 == 0 or i == len(names):
            CACHE.write_text(json.dumps(cache, indent=2))
            print(f"  {i}/{len(names)}  last: {name} -> {rec['status']}")

    CACHE.write_text(json.dumps(cache, indent=2))

    # ---- group distinct strings onto one parent structure ------------------
    by_key = defaultdict(list)
    for name, rec in entries.items():
        if rec["status"] in ("resolved", "review_mw_too_large"):
            by_key[rec["inchikey"]].append(name)

    groups = []
    for key, members in by_key.items():
        ref = entries[members[0]]
        # canonical label = the member with the most patients, which is almost
        # always the plain generic name rather than a salt form
        label = max(members, key=lambda n: recist_counts.get(n, 0) + os_counts.get(n, 0))
        groups.append({
            "canonical_name": label,
            "inchikey": key,
            "cid": ref["cid"],
            "smiles": ref["parent_smiles"],
            "formula": ref["formula"],
            "mw": ref["mw"],
            "pubchem_title": ref.get("pubchem_title"),
            "aliases": sorted(members),
            "n_strings_collapsed": len(members),
            "recist_patients": sum(recist_counts.get(n, 0) for n in members),
            "os_patients": sum(os_counts.get(n, 0) for n in members),
        })
    groups.sort(key=lambda g: -g["recist_patients"])

    biologics = sorted(n for n, r in entries.items() if r["status"] == "biologic_no_smiles")
    unresolved = sorted(n for n, r in entries.items() if r["status"].startswith("unresolved"))
    placeholders = sorted(n for n, r in entries.items() if r["status"] == "not_a_specific_drug")
    rewrites = sorted((n, r["resolved_via_name"]) for n, r in entries.items()
                      if r.get("resolved_via_name"))

    # ---- the map the pipeline actually consumes ---------------------------
    smiles_map = {}
    for g in groups:
        for a in g["aliases"]:
            smiles_map[a] = {
                "canonical_name": g["canonical_name"],
                "smiles": g["smiles"],
                "inchikey": g["inchikey"],
                "cid": g["cid"],
                "kind": "small_molecule",
            }
    for n in biologics:
        smiles_map[n] = {"canonical_name": n, "smiles": None, "inchikey": None,
                         "cid": None, "kind": "biologic"}
    for n in unresolved:
        smiles_map[n] = {"canonical_name": n, "smiles": None, "inchikey": None,
                         "cid": None, "kind": "unresolved"}
    for n in placeholders:
        smiles_map[n] = {"canonical_name": n, "smiles": None, "inchikey": None,
                         "cid": None, "kind": "not_a_specific_drug"}

    MAP_OUT.write_text(json.dumps({
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "PubChem PUG-REST, resolved by CID, salts stripped with RDKit",
        "n_input_strings": len(names),
        "n_distinct_molecules": len(groups),
        "n_biologic": len(biologics),
        "n_unresolved": len(unresolved),
        "n_not_a_specific_drug": len(placeholders),
        "name_rewrites": dict(rewrites),
        "map": smiles_map,
        "molecules": groups,
    }, indent=2))

    # ---- the report a human eyeballs --------------------------------------
    collapsed = [g for g in groups if g["n_strings_collapsed"] > 1]
    n_res = sum(1 for r in entries.values()
                if r["status"] in ("resolved", "review_mw_too_large"))
    L = []
    L.append("# Drug name -> structure resolution report\n")
    L.append(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
             "from PubChem PUG-REST. Every structure is fetched by CID; none is "
             "hardcoded.\n")
    L.append("## Summary\n")
    L.append("| | count |\n|---|---:|")
    L.append(f"| distinct free-text drug strings in | {len(names)} |")
    L.append(f"| resolved to a structure | {n_res} |")
    L.append(f"| **distinct molecules out** | **{len(groups)}** |")
    L.append(f"| strings collapsed onto another (salt/brand variants) | {n_res - len(groups)} |")
    L.append(f"| biologics (no small-molecule structure) | {len(biologics)} |")
    L.append(f"| free-text placeholders, not a molecule | {len(placeholders)} |")
    L.append(f"| unresolved - MUST be handled explicitly | {len(unresolved)} |")

    L.append("\n## Name rewrites applied before lookup\n")
    L.append("Formulation wrappers removed so the molecule could be found. "
             "The structure still came from PubChem - only the search string "
             "was changed. Check each row.\n")
    if rewrites:
        L.append("| Original string | Looked up as |\n|---|---|")
        for a, b in rewrites:
            L.append(f"| {a} | {b} |")
    else:
        L.append("_none_")

    L.append("\n## Free-text placeholders - excluded from the cohort\n")
    L.append("These name no specific molecule, so no honest drug input exists. "
             "Their drug-outcome pairs are dropped rather than guessed at.\n")
    if placeholders:
        L.append("| String | RECIST pts | OS pts |\n|---|---:|---:|")
        for n in placeholders:
            L.append(f"| {n} | {recist_counts.get(n,0)} | {os_counts.get(n,0)} |")
    else:
        L.append("_none_")

    L.append("\n## Variants collapsed onto one molecule\n")
    L.append("These are the strings that would otherwise have been fed to the model "
             "as separate, unrelated drugs.\n")
    if collapsed:
        L.append("| Canonical | CID | Formula | Collapsed strings | RECIST pts |")
        L.append("|---|---:|---|---|---:|")
        for g in collapsed:
            L.append(f"| {g['canonical_name']} | {g['cid']} | {g['formula']} | "
                     f"{'; '.join(g['aliases'])} | {g['recist_patients']} |")
    else:
        L.append("_none_")

    L.append("\n## Biologics - no small-molecule structure exists\n")
    L.append("Antibodies and proteins have no meaningful SMILES. They must be "
             "excluded from the cohort or given a separate embedding - and that "
             "choice must be stated in the paper. They must never be silently "
             "zero-filled, which would turn them into without-SMILES controls "
             "inside the with-SMILES model.\n")
    if biologics:
        L.append("| Drug | RECIST pts | OS pts |\n|---|---:|---:|")
        for n in biologics:
            L.append(f"| {n} | {recist_counts.get(n,0)} | {os_counts.get(n,0)} |")
    else:
        L.append("_none_")

    L.append("\n## Unresolved - blocking\n")
    if unresolved:
        L.append("| Drug | status | RECIST pts | OS pts |\n|---|---|---:|---:|")
        for n in unresolved:
            L.append(f"| {n} | {entries[n]['status']} | "
                     f"{recist_counts.get(n,0)} | {os_counts.get(n,0)} |")
    else:
        L.append("_none_")

    L.append("\n## Every resolved molecule\n")
    L.append("Check these by eye before training. A wrong-but-plausible structure "
             "is the failure mode that does not announce itself.\n")
    L.append("| Canonical | CID | PubChem title | Formula | MW | RECIST pts | OS pts | SMILES |")
    L.append("|---|---:|---|---|---:|---:|---:|---|")
    for g in groups:
        L.append(f"| {g['canonical_name']} | {g['cid']} | {g['pubchem_title']} | "
                 f"{g['formula']} | {g['mw']} | {g['recist_patients']} | "
                 f"{g['os_patients']} | `{g['smiles']}` |")
    REPORT_OUT.write_text("\n".join(L))

    print("\n" + "=" * 62)
    print(f"distinct strings in       : {len(names)}")
    print(f"distinct molecules out    : {len(groups)}")
    print(f"variants collapsed        : {n_res - len(groups)}")
    print(f"biologics flagged         : {len(biologics)}")
    print(f"placeholders excluded     : {len(placeholders)}")
    print(f"name rewrites applied     : {len(rewrites)}")
    print(f"unresolved (blocking)     : {len(unresolved)}")
    print("=" * 62)
    print(f"\nWrote {MAP_OUT.name} and {REPORT_OUT.name}")


if __name__ == "__main__":
    main()
