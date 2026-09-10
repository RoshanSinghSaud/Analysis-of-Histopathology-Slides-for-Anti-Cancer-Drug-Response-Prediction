# TCGA drug + outcome audit (TCGA)

Generated 2026-08-26T04:51:31Z from the live GDC API.

## Funnel

| Step | Patients |
|---|---:|
| Cases in scope | 11428 |
| ...with >=1 NAMED therapeutic agent | 4986 |
| ...with a usable OS label | 8513 |
| ...with an explicit progression/recurrence label | 3320 |
| ...with a per-treatment treatment_outcome | 2595 |
| ...with >=1 diagnostic (FFPE H&E) slide | 9752 |

## The number that decides the project

| Cohort (drug AND outcome AND diagnostic slide) | Patients |
|---|---:|
| drug + overall survival + slide | **2771** |
| drug + progression/recurrence + slide | 1515 |
| drug + RECIST treatment outcome + slide (strict pairing) | **1564** |

## Drug variation in the OS cohort

- distinct drugs: **248**
- drugs seen in >=20 patients: 53
- drugs seen in >=50 patients: 24
- monotherapy patients: 1208
- combination patients: 1563

| Drug | Patients |
|---|---:|
| tamoxifen | 605 |
| cisplatin | 538 |
| carboplatin | 466 |
| temozolomide | 441 |
| fluorouracil | 375 |
| paclitaxel | 373 |
| gemcitabine | 245 |
| oxaliplatin | 208 |
| bevacizumab | 187 |
| etoposide | 168 |
| capecitabine | 142 |
| leucovorin | 142 |
| cyclophosphamide | 109 |
| docetaxel | 105 |
| doxorubicin | 96 |
| leucovorin calcium | 94 |
| bleomycin | 90 |
| vincristine | 76 |
| lomustine | 74 |
| irinotecan | 71 |
| gemcitabine hydrochloride | 63 |
| dexamethasone | 58 |
| mitotane | 56 |
| irinotecan hydrochloride | 53 |
| interferon | 44 |
| carmustine | 42 |
| sorafenib | 40 |
| doxorubicin hydrochloride | 40 |
| dacarbazine | 39 |
| prednisone | 38 |
| procarbazine | 38 |
| cetuximab | 37 |
| epirubicin | 33 |
| erlotinib hydrochloride | 32 |
| rituximab | 32 |
| vinorelbine | 31 |
| pemetrexed | 31 |
| pegylated liposomal doxorubicin hydrochloride | 31 |
| vinorelbine tartrate | 29 |
| topotecan | 29 |

## RECIST response cohort (the pCR replacement)

Per-treatment `treatment_outcome`, restricted to RECIST-interpretable values only. Administrative states ('treatment ongoing', 'no measurable disease', 'persistent disease') are NOT counted as labels.

- **STRICT** (drug and outcome on the SAME treatment record) + diagnostic slide: **1564** patients, 3657 drug-outcome pairs
- loose (any drug on the case paired with any outcome on the case): 1758 - do NOT use this as the headline
- responders (CR or PR): 958
- non-responders (SD, PD or no response): 554
- mixed / contradictory records (excluded): 52
- distinct drugs: 170 (31 seen in >=20 patients)

| Drug | Patients |
|---|---:|
| cisplatin | 495 |
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
| TCGA-UCEC | 502 | 28 |
| TCGA-LGG | 268 | 42 |
| TCGA-GBM | 244 | 82 |
| TCGA-STAD | 161 | 30 |
| TCGA-COAD | 152 | 20 |
| TCGA-SKCM | 134 | 58 |
| TCGA-LUSC | 127 | 35 |
| TCGA-PAAD | 116 | 20 |
| TCGA-HNSC | 99 | 25 |
| TCGA-TGCT | 97 | 8 |
| TCGA-BLCA | 94 | 23 |
| TCGA-OV | 85 | 23 |
| TCGA-READ | 71 | 17 |
| TCGA-KIRC | 61 | 40 |
| TCGA-ACC | 56 | 13 |
| TCGA-BRCA | 56 | 41 |
| TCGA-UCS | 55 | 12 |
| TCGA-MESO | 54 | 28 |
| TCGA-LUAD | 51 | 22 |
| TCGA-CESC | 49 | 10 |
| TCGA-THCA | 45 | 10 |
| TCGA-DLBC | 39 | 27 |
| TCGA-ESCA | 39 | 13 |
| TCGA-LIHC | 37 | 18 |
| TCGA-KIRP | 27 | 27 |
| TCGA-UVM | 16 | 10 |
| TCGA-CHOL | 9 | 4 |
| TCGA-THYM | 9 | 9 |
| TCGA-KICH | 9 | 8 |
| TCGA-PCPG | 6 | 7 |
| TCGA-PRAD | 3 | 7 |

## Caveats to carry into the paper

- OS is a patient-level outcome; the drug is one of many causes. Frame results as treatment-conditioned prediction, never as proof of drug efficacy.
- TCGA drug names are free text harmonized by the GDC; brand/generic and salt forms (e.g. 'doxorubicin' vs 'doxorubicin hydrochloride') appear separately and must be normalized before SMILES lookup.
- Combination therapy is common. Decide explicitly whether the drug branch takes one SMILES, a set, or a pooled regimen embedding.
- Diagnostic (FFPE) and Tissue (frozen) slides are counted separately. Frozen sections have much worse morphology; state which you used.
- Treatment start dates are frequently missing, so immortal-time bias cannot always be excluded. Report how many patients had a usable days_to_treatment_start.