# TCGA drug + outcome audit (TCGA)

Generated 2026-08-09T10:40:51Z from the live GDC API.

## Funnel

| Step | Patients |
|---|---:|
| Cases in scope | 11428 |
| ...with >=1 NAMED therapeutic agent | 4985 |
| ...with a usable OS label | 11064 |
| ...with an explicit progression/recurrence label | 3320 |
| ...with a per-treatment treatment_outcome | 2595 |
| ...with >=1 diagnostic (FFPE H&E) slide | 9752 |

## The number that decides the project

| Cohort (drug AND outcome AND diagnostic slide) | Patients |
|---|---:|
| drug + overall survival + slide | **3894** |
| drug + progression/recurrence + slide | 1513 |
| drug + RECIST treatment outcome + slide (strict pairing) | **1565** |

## Drug variation in the OS cohort

- distinct drugs: **273**
- drugs seen in >=20 patients: 63
- drugs seen in >=50 patients: 32
- monotherapy patients: 1599
- combination patients: 2295

| Drug | Patients |
|---|---:|
| tamoxifen | 833 |
| cisplatin | 718 |
| paclitaxel | 615 |
| cyclophosphamide | 578 |
| carboplatin | 561 |
| temozolomide | 492 |
| fluorouracil | 480 |
| docetaxel | 291 |
| doxorubicin | 287 |
| gemcitabine | 271 |
| anastrozole | 247 |
| bevacizumab | 217 |
| oxaliplatin | 209 |
| doxorubicin hydrochloride | 197 |
| etoposide | 181 |
| capecitabine | 146 |
| leucovorin | 145 |
| leucovorin calcium | 94 |
| bleomycin | 91 |
| vincristine | 80 |
| lomustine | 78 |
| letrozole | 76 |
| trastuzumab | 75 |
| epirubicin | 75 |
| irinotecan | 74 |
| gemcitabine hydrochloride | 68 |
| pemetrexed disodium | 61 |
| dexamethasone | 59 |
| mitotane | 56 |
| leuprolide acetate | 54 |
| irinotecan hydrochloride | 54 |
| interferon | 50 |
| methotrexate | 49 |
| sorafenib | 47 |
| exemestane | 45 |
| dacarbazine | 45 |
| pemetrexed | 44 |
| vinorelbine | 43 |
| erlotinib hydrochloride | 43 |
| carmustine | 43 |

## RECIST response cohort (the pCR replacement)

Per-treatment `treatment_outcome`, restricted to RECIST-interpretable values only. Administrative states ('treatment ongoing', 'no measurable disease', 'persistent disease') are NOT counted as labels.

- **STRICT** (drug and outcome on the SAME treatment record) + diagnostic slide: **1565** patients, 3657 drug-outcome pairs
- loose (any drug on the case paired with any outcome on the case): 1759 - do NOT use this as the headline
- responders (CR or PR): 959
- non-responders (SD, PD or no response): 554
- mixed / contradictory records (excluded): 52
- distinct drugs: 170 (31 seen in >=20 patients)

| Drug | Patients |
|---|---:|
| cisplatin | 496 |
| fluorouracil | 312 |
| carboplatin | 256 |
| paclitaxel | 229 |
| temozolomide | 185 |
| cyclophosphamide | 179 |
| gemcitabine | 168 |
| etoposide | 131 |
| docetaxel | 123 |
| oxaliplatin | 113 |
| capecitabine | 97 |
| bleomycin | 86 |
| bevacizumab | 83 |
| doxorubicin hydrochloride | 74 |
| leucovorin calcium | 72 |
| leucovorin | 70 |
| doxorubicin | 67 |
| epirubicin | 51 |
| dacarbazine | 37 |
| gemcitabine hydrochloride | 35 |

## Per cancer type (OS cohort)

| Project | Usable patients | Distinct drugs |
|---|---:|---:|
| TCGA-BRCA | 753 | 63 |
| TCGA-UCEC | 502 | 28 |
| TCGA-GBM | 293 | 86 |
| TCGA-LGG | 268 | 42 |
| TCGA-STAD | 165 | 30 |
| TCGA-LUAD | 154 | 30 |
| TCGA-COAD | 152 | 20 |
| TCGA-SKCM | 148 | 59 |
| TCGA-CESC | 136 | 17 |
| TCGA-HNSC | 129 | 26 |
| TCGA-LUSC | 127 | 35 |
| TCGA-BLCA | 121 | 25 |
| TCGA-PAAD | 116 | 20 |
| TCGA-TGCT | 97 | 8 |
| TCGA-KIRC | 96 | 49 |
| TCGA-OV | 85 | 23 |
| TCGA-READ | 71 | 17 |
| TCGA-PRAD | 65 | 20 |
| TCGA-ACC | 56 | 13 |
| TCGA-UCS | 55 | 12 |
| TCGA-MESO | 54 | 28 |
| TCGA-LIHC | 52 | 20 |
| TCGA-THCA | 45 | 10 |
| TCGA-ESCA | 39 | 13 |
| TCGA-DLBC | 39 | 27 |
| TCGA-KIRP | 27 | 27 |
| TCGA-UVM | 16 | 10 |
| TCGA-THYM | 9 | 9 |
| TCGA-CHOL | 9 | 4 |
| TCGA-KICH | 9 | 8 |
| TCGA-PCPG | 6 | 7 |

## Caveats to carry into the paper

- OS is a patient-level outcome; the drug is one of many causes. Frame results as treatment-conditioned prediction, never as proof of drug efficacy.
- TCGA drug names are free text harmonized by the GDC; brand/generic and salt forms (e.g. 'doxorubicin' vs 'doxorubicin hydrochloride') appear separately and must be normalized before SMILES lookup.
- Combination therapy is common. Decide explicitly whether the drug branch takes one SMILES, a set, or a pooled regimen embedding.
- Diagnostic (FFPE) and Tissue (frozen) slides are counted separately. Frozen sections have much worse morphology; state which you used.
- Treatment start dates are frequently missing, so immortal-time bias cannot always be excluded. Report how many patients had a usable days_to_treatment_start.