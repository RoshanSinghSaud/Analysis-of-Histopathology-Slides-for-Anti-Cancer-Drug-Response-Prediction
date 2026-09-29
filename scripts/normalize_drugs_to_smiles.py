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
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit is required:  conda install -c conda-forge rdkit")

DRUGS_DIR = Path(__file__).resolve().parents[1] / "drugs"
DRUGS_DIR.mkdir(parents=True, exist_ok=True)
NAMES_IN = DRUGS_DIR / "drug_names_all.json"
VOCAB_IN = DRUGS_DIR / "drug_vocabulary.json"
CACHE = DRUGS_DIR / "pubchem_cache.json"
MAP_OUT = DRUGS_DIR / "drug_smiles_map.json"
REPORT_OUT = DRUGS_DIR / "drug_resolution_report.md"

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
    # Added 2026-09-19. A string ENDING in a target class is a class, not an
    # agent: 'braf inhibitor' and 'tyrosine kinase inhibitor' were each being
    # resolved to one arbitrary member of the class (CID 53438230 and CID
    # 24956525), which puts a molecule the patient never received into the arm
    # that exists to measure molecular structure. 'ras inhibitor' was already
    # listed by name; this generalises it. Names carrying a compound CODE after
    # the class -- 'akt inhibitor mk2206', 'aurora kinase/vegfr2 inhibitor
    # cyc116' -- do NOT match and stay 'unresolved', which is correct: those
    # name a specific agent that PubChem simply does not have.
    r"inhibitor$", r"agonist$", r"antagonist$", r"^chemotherapeutic agent",
]

# Free-text strings that name the WRONG molecule or no approved molecule, with
# the name that should be resolved instead. Added 2026-09-19 from the 10 Sep
# pharmacist-review audit. These are NAME rewrites, never hardcoded structures:
# the replacement is still resolved through PubChem by CID, per rule 1 at the
# top of this file. The expected molecular formula is a REFUSAL GATE -- if
# PubChem returns anything else, the script stops rather than silently
# substituting a second wrong molecule for the first.
NAME_CORRECTIONS = {
    # 'pv-10' resolved to CID 69247748, alpha-pyrrolidinononanophenone, a
    # synthetic cathinone stimulant. The oncology agent PV-10 is rose bengal
    # disodium, a halogenated xanthene dye. Wrong molecule, wrong class, wrong
    # elements. Expected parent after counterion stripping: the free acid,
    # 4,5,6,7-tetrachloro-2',4',5',7'-tetraiodofluorescein.
    "pv-10": ("rose bengal", "C20H4Cl4I4O5"),
    # 'halichondrin b' is a marine natural product that was never an approved
    # drug. The TCGA string is read as eribulin, its synthetic analogue and an
    # approved agent. LOWER CONFIDENCE than the correction above: it is an
    # interpretation of what the submitting institution meant, not a
    # misresolution, and it is reported as such.
    "halichondrin b": ("eribulin", "C40H59NO11"),
}

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
_uncharger = rdMolStandardize.Uncharger()

# Metals that are part of the DRUG ITSELF, not a counterion. PubChem stores
# coordination complexes as several disconnected fragments - cisplatin is
# 'N.N.Cl[Pt]Cl' and carboplatin is 'ligand.[NH2-].[NH2-].[Pt+2]' - so the
# usual "keep the largest fragment" rule silently destroys them. It reduced
# cisplatin (435 patients) to Cl2Pt and carboplatin (240 patients) to its
# organic ligand with the platinum thrown away. Anything containing one of
# these is kept whole.
CORE_METALS = {"Pt", "As", "Ru", "Ti", "Gd", "Tc", "Au", "Sb", "Bi", "Cr", "Co"}

# Metals that appear as salt counterions and SHOULD be stripped, e.g. the
# calcium in leucovorin calcium.
COUNTERION_METALS = {"Na", "K", "Ca", "Mg", "Li", "Zn", "Al", "Ba"}


def has_core_metal(mol) -> bool:
    return any(a.GetSymbol() in CORE_METALS for a in mol.GetAtoms())


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

    # Coordination complexes (platinum drugs, arsenic trioxide) are kept whole.
    if has_core_metal(mol):
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        return {
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "inchikey": Chem.MolToInchiKey(mol),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "mw": round(Descriptors.MolWt(mol), 2),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "metal_complex": True,
        }

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

    # Neutralise the remaining fragment before keying. Added 2026-09-19.
    #
    # Counterion stripping alone does NOT merge every salt, because PubChem
    # often stores the salt as the bare ION with no counterion in the SMILES at
    # all: 'leucovorin calcium' (CID 135403647) is the DIANION, with no calcium
    # present, so SaltRemover finds nothing to remove and GetMolFrags sees one
    # fragment. The deprotonated form then survives as its own parent structure,
    # and one drug becomes two unrelated inputs -- exactly the defect this
    # function exists to prevent. The InChIKeys differ only in the protonation
    # layer (…-ABLWVSNPSA-N vs …-ABLWVSNPSA-L), which is the signature.
    #
    # Measured across the whole 274-string vocabulary on 2026-09-19, this merges
    # FIVE pairs and nothing else: leucovorin/leucovorin calcium,
    # pemetrexed/pemetrexed disodium, pamidronic acid/pamidronate disodium,
    # clodronic acid/clodronate disodium, levothyroxine/levothyroxine sodium.
    # 178 distinct parent structures become 173.
    #
    # It correctly does NOT merge dexamethasone with dexamethasone phosphate:
    # that pair is an ester prodrug, a genuinely different molecule
    # (C22H29FO5 vs C22H30FO8P), not a salt. A hand-written merge list had it
    # wrong; the rule gets it right.
    #
    # Metal complexes never reach this point -- they return above -- so the
    # anionic nitrogen notation in carboplatin and oxaliplatin is untouched.
    try:
        stripped = _uncharger.uncharge(stripped)
        Chem.SanitizeMol(stripped)
    except Exception:
        pass

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
        rec = {"name": name, "status": None, "cid": None}

        # ------------------------------------------------------------------
        # CLASSIFICATION RUNS BEFORE THE CACHE IS CONSULTED. Reordered
        # 2026-09-19. The cache comment above promises that "re-deriving the
        # classification statuses is cheap and lets classification rules be
        # edited without a stale cache silently freezing the old answer" -- but
        # the cache lookup used to come FIRST, so a name already cached as
        # 'resolved' never reached these tests and an edit to
        # PLACEHOLDER_PATTERNS had no effect on it. That is precisely how
        # 'braf inhibitor' and 'tyrosine kinase inhibitor' would have survived
        # the 2026-09-19 pattern additions. The promise is now kept.
        # ------------------------------------------------------------------
        if is_placeholder_name(name):
            rec["status"] = "not_a_specific_drug"
            rec["note"] = "free-text placeholder, not a molecule; excluded from cohort"
            entries[name] = rec
            cache.pop(name, None)
            continue
        if is_biologic_name(name):
            rec["status"] = "biologic_no_smiles"
            rec["note"] = "matched a biologic naming stem; no small-molecule structure"
            entries[name] = rec
            cache.pop(name, None)
            continue

        # A declared correction always bypasses the cache -- the cached answer
        # is the wrong molecule, which is the whole reason the entry exists.
        correction = NAME_CORRECTIONS.get(name)

        if correction is None and name in cache:
            # The cache holds the expensive part -- the PubChem round trip --
            # but NOT the cheap local chemistry derived from it. Added
            # 2026-09-19: re-derive the parent structure on every run from the
            # cached raw SMILES, so an edit to parent_structure() takes effect
            # for cached names too.
            #
            # This is the same defect the classification reordering above fixes,
            # one layer down, and it bit on 2026-09-19: the charge-neutralisation
            # rule was added, the script re-run, and leucovorin/leucovorin
            # calcium stayed unmerged because both were cached as 'resolved'
            # with an inchikey computed under the OLD rule. A cache that freezes
            # derived values is a cache that silently vetoes bug fixes.
            hit = dict(cache[name])
            if hit.get("smiles_raw"):
                par = parent_structure(hit["smiles_raw"])
                if par is not None:
                    hit.update({
                        "parent_smiles": par["smiles"],
                        "inchikey":      par["inchikey"],
                        "formula":       par["formula"],
                        "mw":            par["mw"],
                    })
            entries[name] = hit
            cache[name] = hit
            continue

        if correction is not None:
            lookup_name, expect_formula = correction
            rec["corrected_from"] = name
            rec["resolved_via_name"] = lookup_name
            rec["correction_expected_formula"] = expect_formula
        else:
            lookup_name, expect_formula = name, None

        cid = name_to_cid(lookup_name)
        if cid is None:
            # retry with formulation wrappers stripped, recording the rewrite
            for alt in name_variants(lookup_name):
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
                    # A declared correction must land on the molecule it
                    # declared. If PubChem returns anything else the script
                    # STOPS: substituting a second wrong molecule for the
                    # first, silently, is the failure this table exists to
                    # prevent, and a guard that refuses beats one that warns.
                    if expect_formula and par["formula"] != expect_formula:
                        sys.exit(
                            f"\nNAME_CORRECTIONS refused for {name!r}:\n"
                            f"  looked up   : {lookup_name!r} -> CID {cid}\n"
                            f"  expected    : {expect_formula}\n"
                            f"  PubChem gave: {par['formula']} "
                            f"({par['mw']} g/mol, {rec.get('title','?')})\n"
                            "Resolve by hand and update NAME_CORRECTIONS "
                            "before re-running.")
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
