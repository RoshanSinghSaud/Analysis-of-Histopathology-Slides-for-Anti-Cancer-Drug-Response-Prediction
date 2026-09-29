# Drug name -> structure resolution report

Generated 2026-09-19T12:32:13Z from PubChem PUG-REST. Every structure is fetched by CID; none is hardcoded.

## Summary

| | count |
|---|---:|
| distinct free-text drug strings in | 274 |
| resolved to a structure | 199 |
| **distinct molecules out** | **170** |
| strings collapsed onto another (salt/brand variants) | 29 |
| biologics (no small-molecule structure) | 55 |
| free-text placeholders, not a molecule | 12 |
| unresolved - MUST be handled explicitly | 8 |

## Name rewrites applied before lookup

Formulation wrappers removed so the molecule could be found. The structure still came from PubChem - only the search string was changed. Check each row.

| Original string | Looked up as |
|---|---|
| carmustine implant | carmustine |
| halichondrin b | eribulin |
| nanoparticle albumin-bound rapamycin | rapamycin |
| pegylated liposomal doxorubicin hydrochloride | doxorubicin hydrochloride |
| pv-10 | rose bengal |
| therapeutic hydrocortisone | hydrocortisone |

## Free-text placeholders - excluded from the cohort

These name no specific molecule, so no honest drug input exists. Their drug-outcome pairs are dropped rather than guessed at.

| String | RECIST pts | OS pts |
|---|---:|---:|
| amino acid injection | 1 | 1 |
| braf inhibitor | 3 | 3 |
| clinical trial | 3 | 27 |
| clinical trial agent | 15 | 25 |
| gonadotropin-releasing hormone analog | 1 | 4 |
| hormone therapy | 2 | 2 |
| mistletoe extract | 0 | 1 |
| ras inhibitor | 0 | 1 |
| taxane compound | 0 | 1 |
| thyroid extract | 0 | 1 |
| total androgen blockade | 0 | 1 |
| tyrosine kinase inhibitor | 2 | 2 |

## Variants collapsed onto one molecule

These are the strings that would otherwise have been fed to the model as separate, unrelated drugs.

| Canonical | CID | Formula | Collapsed strings | RECIST pts |
|---|---:|---|---|---:|
| paclitaxel | 36314 | C47H51NO14 | nab-paclitaxel; paclitaxel | 231 |
| gemcitabine | 60750 | C9H11F2N3O4 | gemcitabine; gemcitabine hydrochloride | 193 |
| doxorubicin | 31703 | C27H29NO11 | doxorubicin; doxorubicin hydrochloride; pegylated liposomal doxorubicin hydrochloride | 138 |
| leucovorin | 135403648 | C20H23N7O7 | leucovorin; leucovorin calcium | 132 |
| pemetrexed disodium | 135410875 | C20H21N5O6 | pemetrexed; pemetrexed disodium | 56 |
| vinorelbine | 5311497 | C45H54N4O8 | vinorelbine; vinorelbine tartrate | 39 |
| irinotecan | 60838 | C33H38N4O6 | irinotecan; irinotecan hydrochloride | 38 |
| tamoxifen | 2733526 | C26H29NO | tamoxifen; tamoxifen citrate | 27 |
| leuprolide acetate | 657181 | C59H84N16O12 | leuprolide; leuprolide acetate | 22 |
| sorafenib | 216239 | C21H16ClF3N4O3 | sorafenib; sorafenib tosylate | 21 |
| vincristine | 5978 | C46H56N4O10 | vincristine; vincristine sulfate | 17 |
| erlotinib hydrochloride | 176870 | C22H23N3O4 | erlotinib; erlotinib hydrochloride | 14 |
| carmustine | 2578 | C5H9Cl2N3O2 | carmustine; carmustine implant | 10 |
| procarbazine | 4915 | C12H19N3O | procarbazine; procarbazine hydrochloride | 10 |
| sunitinib | 5329102 | C22H27FN4O2 | sunitinib; sunitinib malate | 10 |
| goserelin acetate | 5311128 | C59H84N18O14 | goserelin; goserelin acetate | 7 |
| topotecan | 60700 | C23H23N3O5 | topotecan; topotecan hydrochloride | 5 |
| pazopanib | 10113978 | C21H23N7O2S | pazopanib; pazopanib hydrochloride | 4 |
| halichondrin b | 11354606 | C40H59NO11 | eribulin; eribulin mesylate; halichondrin b | 3 |
| imatinib mesylate | 5291 | C29H31N7O | imatinib; imatinib mesylate | 3 |
| pamidronate disodium | 73351 | C3H11NO7P2 | pamidronate disodium; pamidronic acid | 3 |
| levothyroxine sodium | 5819 | C15H11I4NO4 | levothyroxine; levothyroxine sodium | 1 |
| tegafur-gimeracil-oteracil potassium | 5386 | C8H9FN2O3 | tegafur; tegafur-gimeracil-oteracil potassium | 1 |
| triptorelin | 25074470 | C64H82N18O13 | triptorelin; triptorelin pamoate | 1 |
| clodronate disodium | 23724874 | CH4Cl2O6P2 | clodronate disodium; clodronic acid | 0 |
| enzastaurin | 176167 | C32H29N5O2 | enzastaurin; enzastaurin hydrochloride | 0 |
| sirolimus | 5284616 | C51H79NO13 | nanoparticle albumin-bound rapamycin; sirolimus | 0 |

## Biologics - no small-molecule structure exists

Antibodies and proteins have no meaningful SMILES. They must be excluded from the cohort or given a separate embedding - and that choice must be stated in the paper. They must never be silently zero-filled, which would turn them into without-SMILES controls inside the with-SMILES model.

| Drug | RECIST pts | OS pts |
|---|---:|---:|
| ae37 peptide/gm-csf vaccine | 0 | 2 |
| aldesleukin | 0 | 2 |
| amatuximab | 0 | 1 |
| anti-egfrviii immunotoxin mr1-1 | 1 | 1 |
| anti-vegf anticalin prs-050-peg40 | 1 | 1 |
| antineoplastic vaccine | 13 | 20 |
| astuprotimut-r | 1 | 1 |
| bcg solution | 10 | 36 |
| bevacizumab | 70 | 217 |
| cetuximab | 21 | 40 |
| chimeric monoclonal antibody 81c6 | 0 | 1 |
| cintredekin besudotox | 0 | 6 |
| conatumumab | 0 | 1 |
| daclizumab | 0 | 1 |
| dendritic cell vaccine | 0 | 4 |
| denosumab | 3 | 4 |
| filgrastim | 0 | 1 |
| interferon | 13 | 50 |
| interferon alfa-2b | 0 | 4 |
| iodine i 131 monoclonal antibody 81c6 | 2 | 3 |
| ipilimumab | 15 | 22 |
| light-emitting oncolytic vaccinia virus gl-onc1 | 1 | 1 |
| lontucirev | 1 | 1 |
| mage-a3 peptide vaccine | 0 | 5 |
| monoclonal antibody 81c6 | 0 | 7 |
| nivolumab | 2 | 2 |
| ofatumumab | 0 | 1 |
| ontuxizumab | 1 | 1 |
| panitumumab | 2 | 7 |
| pegfilgrastim | 7 | 13 |
| peginterferon alfa-2b | 0 | 2 |
| pelareorep | 1 | 1 |
| pembrolizumab | 3 | 7 |
| pep-3-klh conjugate vaccine | 0 | 2 |
| poly iclc | 0 | 1 |
| polysaccharide-k | 6 | 6 |
| recombinant erythropoietin | 1 | 1 |
| recombinant granulocyte-macrophage colony-stimulating factor | 0 | 1 |
| recombinant interferon alfa | 3 | 13 |
| recombinant interferon alfa-2a | 0 | 1 |
| recombinant interferon alfa-2b | 1 | 2 |
| recombinant interleukin-18 | 0 | 1 |
| recombinant interleukin-2 | 6 | 18 |
| recombinant luteinizing hormone | 0 | 1 |
| recombinant mage-3.1 antigen | 0 | 2 |
| recombinant prame protein plus as15 adjuvant gsk2302025a | 1 | 2 |
| rilotumumab | 0 | 1 |
| rituximab | 2 | 34 |
| sargramostim | 2 | 3 |
| talimogene laherparepvec | 1 | 1 |
| teprotumumab | 1 | 1 |
| tovetumab | 0 | 1 |
| trastuzumab | 18 | 75 |
| vitespen | 0 | 9 |
| ziv-aflibercept | 0 | 2 |

## Unresolved - blocking

| Drug | status | RECIST pts | OS pts |
|---|---|---:|---:|
| akt inhibitor mk2206 | unresolved_not_in_pubchem | 1 | 1 |
| anti-dr5 agonistic antibody ds-8273a | unresolved_not_in_pubchem | 1 | 1 |
| aurora kinase/vegfr2 inhibitor cyc116 | unresolved_not_in_pubchem | 0 | 1 |
| glutathione pegylated liposomal doxorubicin hydrochloride formulation 2b3-101 | unresolved_not_in_pubchem | 1 | 1 |
| muparfostat | unresolved_not_in_pubchem | 1 | 1 |
| paclitaxel poliglumex | unresolved_not_in_pubchem | 0 | 1 |
| pegdinetanib | unresolved_not_in_pubchem | 0 | 1 |
| porfimer sodium | unresolved_not_in_pubchem | 0 | 8 |

## Every resolved molecule

Check these by eye before training. A wrong-but-plausible structure is the failure mode that does not announce itself.

| Canonical | CID | PubChem title | Formula | MW | RECIST pts | OS pts | SMILES |
|---|---:|---|---|---:|---:|---:|---|
| cisplatin | 5702198 | azane;dichloroplatinum | H6Cl2N2Pt | 300.05 | 435 | 718 | `N.N.[Cl][Pt][Cl]` |
| fluorouracil | 3385 | 5-Fluorouracil | C4H3FN2O2 | 130.08 | 284 | 480 | `O=c1[nH]cc(F)c(=O)[nH]1` |
| carboplatin | 426756 | Platinum, diammine[1,1-cyclobutanedi(carboxylato-kO)(2-)]-, (SP-4-2)- | C6H12N2O4Pt | 371.25 | 240 | 561 | `O=C(O)C1(C(=O)O)CCC1.[NH2-].[NH2-].[Pt+2]` |
| paclitaxel | 36314 | Paclitaxel | C47H51NO14 | 853.92 | 231 | 637 | `CC(=O)O[C@H]1C(=O)[C@@]2(C)[C@H]([C@H](OC(=O)c3ccccc3)[C@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)c4ccccc4)c4ccccc4)C(C)=C1C3(C)C)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]2O` |
| gemcitabine | 60750 | Gemcitabine | C9H11F2N3O4 | 263.2 | 193 | 339 | `Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)C2(F)F)c(=O)n1` |
| cyclophosphamide | 2907 | Cyclophosphamide | C7H15Cl2N2O2P | 261.09 | 176 | 578 | `O=P1(N(CCCl)CCCl)NCCCO1` |
| doxorubicin | 31703 | Doxorubicin | C27H29NO11 | 543.53 | 138 | 515 | `COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1` |
| temozolomide | 5394 | Temozolomide | C6H6N6O2 | 194.15 | 138 | 492 | `Cn1nnc2c(C(N)=O)ncn2c1=O` |
| leucovorin | 135403648 | Leucovorin | C20H23N7O7 | 473.45 | 132 | 239 | `Nc1nc2c(c(=O)[nH]1)N(C=O)C(CNc1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1)CN2` |
| etoposide | 36462 | Etoposide | C29H32O13 | 588.56 | 130 | 181 | `COc1cc([C@@H]2c3cc4c(cc3[C@@H](O[C@@H]3O[C@@H]5CO[C@@H](C)O[C@H]5[C@H](O)[C@H]3O)[C@H]3COC(=O)[C@H]23)OCO4)cc(OC)c1O` |
| docetaxel | 148124 | Docetaxel | C43H53NO14 | 807.89 | 123 | 291 | `CC(=O)O[C@@]12CO[C@@H]1C[C@H](O)[C@@]1(C)C(=O)[C@H](O)C3=C(C)[C@@H](OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)c4ccccc4)C[C@@](O)([C@@H](OC(=O)c4ccccc4)[C@H]21)C3(C)C` |
| oxaliplatin | 9887053 | cis-oxalato-trans-l-1,2-diaminocyclohexaneplatinum(II) | C8H14N2O4Pt | 397.29 | 108 | 209 | `O=C(O)C(=O)O.[NH-][C@@H]1CCCC[C@H]1[NH-].[Pt+2]` |
| capecitabine | 60953 | Capecitabine | C15H22FN3O6 | 359.35 | 91 | 146 | `CCCCCOC(=O)Nc1nc(=O)n([C@@H]2O[C@H](C)[C@@H](O)[C@H]2O)cc1F` |
| bleomycin | 5360373 | Bleomycin | C55H84N17O21S3+ | 1415.58 | 85 | 91 | `Cc1c(N)nc([C@H](CC(N)=O)NC[C@H](N)C(N)=O)nc1C(=O)N[C@H](C(=O)N[C@H](C)[C@@H](O)[C@H](C)C(=O)N[C@H](C(=O)NCCc1nc(-c2nc(C(=O)NCCC[S+](C)C)cs2)cs1)[C@@H](C)O)[C@@H](OC1OC(CO)C(O)C(O)C1OC1OC(CO)C(O)C(OC(N)=O)C1O)c1cnc[nH]1` |
| pemetrexed disodium | 135410875 | Pemetrexed | C20H21N5O6 | 427.42 | 56 | 105 | `Nc1nc2[nH]cc(CCc3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc3)c2c(=O)[nH]1` |
| epirubicin | 41867 | Epirubicin | C27H29NO11 | 543.53 | 50 | 75 | `COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@@H](O)[C@H](C)O1` |
| vinorelbine | 5311497 | Vinorelbine | C45H54N4O8 | 778.95 | 39 | 74 | `CCC1=C[C@@H]2CN(C1)Cc1c([nH]c3ccccc13)[C@@](C(=O)OC)(c1cc3c(cc1OC)N(C)[C@H]1[C@@](O)(C(=O)OC)[C@H](OC(C)=O)[C@]4(CC)C=CCN5CC[C@]31[C@@H]54)C2` |
| irinotecan | 60838 | Irinotecan | C33H38N4O6 | 586.69 | 38 | 128 | `CCc1c2c(nc3ccc(OC(=O)N4CCC(N5CCCCC5)CC4)cc13)-c1cc3c(c(=O)n1C2)COC(=O)[C@]3(O)CC` |
| dacarbazine | 135398738 | Dacarbazine | C6H10N6O | 182.19 | 37 | 45 | `CN(C)/N=N/c1nc[nH]c1C(N)=O` |
| tamoxifen | 2733526 | Tamoxifen | C26H29NO | 371.52 | 27 | 837 | `CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1` |
| mitotane | 4211 | Mitotane | C14H10Cl4 | 320.05 | 26 | 56 | `Clc1ccc(C(c2ccccc2Cl)C(Cl)Cl)cc1` |
| leuprolide acetate | 657181 | Leuprolide | C59H84N16O12 | 1209.42 | 22 | 63 | `CCNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCN=C(N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CO)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H]1CCC(=O)N1` |
| lomustine | 3950 | Lomustine | C9H16ClN3O2 | 233.7 | 22 | 78 | `O=NN(CCCl)C(=O)NC1CCCCC1` |
| sorafenib | 216239 | Sorafenib | C21H16ClF3N4O3 | 464.83 | 21 | 60 | `CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1` |
| methotrexate | 126941 | Methotrexate | C20H22N8O5 | 454.45 | 20 | 49 | `CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1` |
| anastrozole | 2187 | Anastrozole | C17H19N5 | 293.37 | 19 | 247 | `CC(C)(C#N)c1cc(Cn2cncn2)cc(C(C)(C)C#N)c1` |
| ifosfamide | 3690 | Ifosfamide | C7H15Cl2N2O2P | 261.09 | 18 | 24 | `O=P1(NCCCl)OCCCN1CCCl` |
| bicalutamide | 2375 | Bicalutamide | C18H14F4N2O4S | 430.38 | 17 | 28 | `CC(O)(CS(=O)(=O)c1ccc(F)cc1)C(=O)Nc1ccc(C#N)c(C(F)(F)F)c1` |
| vinblastine | 13342 | Vinblastine | C46H58N4O9 | 810.99 | 17 | 23 | `CC[C@]1(O)C[C@@H]2CN(CCc3c([nH]c4ccccc34)[C@@](C(=O)OC)(c3cc4c(cc3OC)N(C)[C@H]3[C@@](O)(C(=O)OC)[C@H](OC(C)=O)[C@]5(CC)C=CCN6CC[C@]43[C@@H]65)C2)C1` |
| vincristine | 5978 | Vincristine | C46H56N4O10 | 824.97 | 17 | 81 | `CC[C@]1(O)C[C@H]2CN(CCc3c([nH]c4ccccc34)[C@@](C(=O)OC)(c3cc4c(cc3OC)N(C=O)[C@H]3[C@@](O)(C(=O)OC)[C@H](OC(C)=O)[C@]5(CC)C=CCN6CC[C@]43[C@@H]65)C2)C1` |
| erlotinib hydrochloride | 176870 | Erlotinib | C22H23N3O4 | 393.44 | 14 | 58 | `C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1` |
| carmustine | 2578 | Carmustine | C5H9Cl2N3O2 | 214.05 | 10 | 68 | `O=NN(CCCl)C(=O)NCCCl` |
| procarbazine | 4915 | Procarbazine | C12H19N3O | 221.3 | 10 | 39 | `CNNCc1ccc(C(=O)NC(C)C)cc1` |
| sunitinib | 5329102 | Sunitinib | C22H27FN4O2 | 398.48 | 10 | 46 | `CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\C(=O)Nc3ccc(F)cc32)c1C` |
| letrozole | 3902 | Letrozole | C17H11N5 | 285.31 | 8 | 76 | `N#Cc1ccc(C(c2ccc(C#N)cc2)n2cncn2)cc1` |
| fotemustine | 104799 | Fotemustine | C9H19ClN3O5P | 315.69 | 7 | 23 | `CCOP(=O)(OCC)C(C)NC(=O)N(CCCl)N=O` |
| goserelin acetate | 5311128 | Goserelin | C59H84N18O14 | 1269.43 | 7 | 22 | `CC(C)C[C@H](NC(=O)[C@@H](COC(C)(C)C)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CO)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H]1CCC(=O)N1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N1CCC[C@H]1C(=O)NNC(N)=O` |
| liothyronine sodium | 23666110 | Liothyronine Sodium | C15H12I3NO4 | 650.98 | 7 | 13 | `N[C@@H](Cc1cc(I)c(Oc2ccc(O)c(I)c2)c(I)c1)C(=O)O` |
| mitomycin | 5746 | Mitomycin C | C15H18N4O5 | 334.33 | 7 | 18 | `CO[C@@]12[C@H](COC(N)=O)C3=C(C(=O)C(C)=C(N)C3=O)N1C[C@@H]1N[C@@H]12` |
| dexamethasone | 5743 | Dexamethasone | C22H29FO5 | 392.47 | 6 | 59 | `C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO` |
| everolimus | 6442177 | Everolimus | C53H83NO14 | 958.24 | 6 | 15 | `CO[C@H]1C[C@@H]2CC[C@@H](C)[C@@](O)(O2)C(=O)C(=O)N2CCCC[C@H]2C(=O)O[C@H]([C@H](C)C[C@@H]2CC[C@@H](OCCO)[C@H](OC)C2)CC(=O)[C@H](C)/C=C(\C)[C@@H](O)[C@@H](OC)C(=O)[C@H](C)C[C@H](C)/C=C/C=C/C=C/1C` |
| dasatinib | 3062316 | Dasatinib | C22H26ClN7O2S | 488.02 | 5 | 6 | `Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1` |
| melphalan | 460612 | Melphalan | C13H18Cl2N2O2 | 305.2 | 5 | 5 | `N[C@@H](Cc1ccc(N(CCCl)CCCl)cc1)C(=O)O` |
| mesna | 23662354 | Mesna | C2H6O3S2 | 142.2 | 5 | 6 | `O=S(=O)(O)CCS` |
| topotecan | 60700 | Topotecan | C23H23N3O5 | 421.45 | 5 | 31 | `CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3c(CN(C)C)c(O)ccc3nc2-1` |
| zoledronic acid | 68740 | Zoledronic Acid | C5H10N2O7P2 | 272.09 | 5 | 28 | `O=P(O)(O)C(O)(Cn1ccnc1)P(=O)(O)O` |
| exemestane | 60198 | Exemestane | C20H24O2 | 296.41 | 4 | 45 | `C=C1C[C@@H]2[C@H](CC[C@]3(C)C(=O)CC[C@@H]23)[C@@]2(C)C=CC(=O)C=C12` |
| fulvestrant | 104741 | Fulvestrant | C32H47F5O3S | 606.78 | 4 | 9 | `C[C@]12CC[C@@H]3c4ccc(O)cc4C[C@@H](CCCCCCCCCS(=O)CCCC(F)(F)C(F)(F)F)[C@H]3[C@@H]1CC[C@@H]2O` |
| pazopanib | 10113978 | Pazopanib | C21H23N7O2S | 437.53 | 4 | 12 | `Cc1ccc(Nc2nccc(N(C)c3ccc4c(C)n(C)nc4c3)n2)cc1S(N)(=O)=O` |
| vemurafenib | 42611257 | Vemurafenib | C23H18ClF2N3O3S | 489.93 | 4 | 11 | `CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F` |
| axitinib | 6450551 | Axitinib | C22H18N4OS | 386.48 | 3 | 4 | `CNC(=O)c1ccccc1Sc1ccc2c(/C=C/c3ccccn3)n[nH]c2c1` |
| cabazitaxel | 9854073 | Cabazitaxel | C45H57NO14 | 835.94 | 3 | 3 | `CO[C@H]1C(=O)[C@]2(C)[C@@H](OC)C[C@H]3OC[C@@]3(OC(C)=O)[C@H]2[C@H](OC(=O)c2ccccc2)[C@]2(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)c3ccccc3)C(C)=C1C2(C)C` |
| dabrafenib | 44462760 | Dabrafenib | C23H20F3N5O2S2 | 519.57 | 3 | 5 | `CC(C)(C)c1nc(-c2cccc(NS(=O)(=O)c3c(F)cccc3F)c2F)c(-c2ccnc(N)n2)s1` |
| dactinomycin | 457193 | Dactinomycin | C62H86N12O16 | 1255.44 | 3 | 3 | `Cc1c2oc3c(C)ccc(C(=O)N[C@@H]4C(=O)N[C@H](C(C)C)C(=O)N5CCC[C@H]5C(=O)N(C)CC(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]4C)c3nc-2c(C(=O)N[C@@H]2C(=O)N[C@H](C(C)C)C(=O)N3CCC[C@H]3C(=O)N(C)CC(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]2C)c(N)c1=O` |
| didox | 3045 | N,3,4-Trihydroxybenzamide | C7H7NO4 | 169.14 | 3 | 4 | `O=C(NO)c1ccc(O)c(O)c1` |
| halichondrin b | 11354606 | Eribulin | C40H59NO11 | 729.91 | 3 | 4 | `C=C1C[C@@H]2CC[C@@]34C[C@H]5O[C@H]6[C@@H](O3)[C@H]3O[C@H](CC[C@@H]3O[C@H]6[C@H]5O4)CC(=O)C[C@@H]3[C@@H](OC)[C@@H](C[C@H](O)CN)O[C@H]3C[C@H]3O[C@@H](CC[C@@H]1O2)C[C@@H](C)C3=C` |
| imatinib mesylate | 5291 | Imatinib | C29H31N7O | 493.62 | 3 | 12 | `Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1` |
| pamidronate disodium | 73351 | Pamidronate Disodium Anhydrous | C3H11NO7P2 | 235.07 | 3 | 4 | `NCCC(O)(P(=O)(O)O)P(=O)(O)O` |
| temsirolimus | 6918289 | Temsirolimus | C56H87NO16 | 1030.3 | 3 | 22 | `CO[C@H]1C[C@@H]2CC[C@@H](C)[C@@](O)(O2)C(=O)C(=O)N2CCCC[C@H]2C(=O)O[C@H]([C@H](C)C[C@@H]2CC[C@@H](OC(=O)C(C)(CO)CO)[C@H](OC)C2)CC(=O)[C@H](C)/C=C(\C)[C@@H](O)[C@@H](OC)C(=O)[C@H](C)C[C@H](C)/C=C/C=C/C=C/1C` |
| thalidomide | 5426 | Thalidomide | C13H10N2O4 | 258.23 | 3 | 17 | `O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1` |
| cositecan | 148202 | Cositecan | C25H28N2O4Si | 448.6 | 2 | 2 | `CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2c-1nc1ccccc1c2CC[Si](C)(C)C` |
| doxifluridine | 18343 | Doxifluridine | C9H11FN2O5 | 246.19 | 2 | 8 | `C[C@H]1O[C@@H](n2cc(F)c(=O)[nH]c2=O)[C@H](O)[C@@H]1O` |
| lapatinib | 208908 | Lapatinib | C29H26ClFN4O4S | 581.07 | 2 | 4 | `CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1` |
| palonosetron hydrochloride | 6918303 | Palonosetron Hydrochloride | C19H24N2O | 296.41 | 2 | 2 | `O=C1c2cccc3c2[C@H](CCC3)CN1[C@@H]1CN2CCC1CC2` |
| selumetinib | 10127622 | Selumetinib | C17H15BrClFN4O3 | 457.69 | 2 | 3 | `Cn1cnc2c(F)c(Nc3ccc(Br)cc3Cl)c(C(=O)NOCCO)cc21` |
| tivantinib | 11494412 | Tivantinib | C23H19N3O2 | 369.42 | 2 | 2 | `O=C1NC(=O)[C@@H](c2cn3c4c(cccc24)CCC3)[C@@H]1c1c[nH]c2ccccc12` |
| trametinib | 11707110 | Trametinib | C26H23FIN5O4 | 615.4 | 2 | 2 | `CC(=O)Nc1cccc(-n2c(=O)n(C3CC3)c(=O)c3c(Nc4ccc(I)cc4F)n(C)c(=O)c(C)c32)c1` |
| veliparib | 11960529 | Veliparib | C13H16N4O | 244.3 | 2 | 3 | `C[C@]1(c2nc3c(C(N)=O)cccc3[nH]2)CCCN1` |
| vorinostat | 5311 | Vorinostat | C14H20N2O3 | 264.32 | 2 | 3 | `O=C(CCCCCCC(=O)Nc1ccccc1)NO` |
| apitolisib | 25254071 | Apitolisib | C23H30N8O3S | 498.61 | 1 | 1 | `Cc1c(CN2CCN(C(=O)[C@H](C)O)CC2)sc2c(N3CCOCC3)nc(-c3cnc(N)nc3)nc12` |
| cediranib | 9933475 | Cediranib | C25H27FN4O3 | 450.51 | 1 | 2 | `COc1cc2c(Oc3ccc4[nH]c(C)cc4c3F)ncnc2cc1OCCCN1CCCC1` |
| celecoxib | 2662 | Celecoxib | C17H14F3N3O2S | 381.38 | 1 | 9 | `Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1` |
| cilengitide | 176873 | Cilengitide | C27H40N8O7 | 588.67 | 1 | 6 | `CC(C)[C@H]1C(=O)N[C@@H](CCCN=C(N)N)C(=O)NCC(=O)N[C@@H](CC(=O)O)C(=O)N[C@H](Cc2ccccc2)C(=O)N1C` |
| cyclosporine | 5284373 | Cyclosporin A | C62H111N11O12 | 1202.63 | 1 | 1 | `C/C=C/C[C@@H](C)[C@@H](O)[C@H]1C(=O)N[C@@H](CC)C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N1C` |
| degarelix | 16136245 | Degarelix | C82H103ClN18O16 | 1632.29 | 1 | 2 | `CC(=O)N[C@H](Cc1ccc2ccccc2c1)C(=O)N[C@H](Cc1ccc(Cl)cc1)C(=O)N[C@H](Cc1cccnc1)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccc(NC(=O)[C@@H]2CC(=O)NC(=O)N2)cc1)C(=O)N[C@H](Cc1ccc(NC(N)=O)cc1)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCNC(C)C)C(=O)N1CCC[C@H]1C(=O)N[C@H](C)C(N)=O` |
| dexamethasone phosphate | 9400 | Dexamethasone Phosphate | C22H30FO8P | 472.45 | 1 | 1 | `C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)COP(=O)(O)O` |
| floxuridine | 5790 | Floxuridine | C9H11FN2O5 | 246.19 | 1 | 5 | `O=c1[nH]c(=O)n([C@H]2C[C@H](O)[C@@H](CO)O2)cc1F` |
| gefitinib | 123631 | Gefitinib | C22H24ClFN4O3 | 446.91 | 1 | 10 | `COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1` |
| hydroxyurea | 3657 | Hydroxyurea | CH4N2O2 | 76.05 | 1 | 13 | `NC(=O)NO` |
| isotretinoin | 5282379 | Isotretinoin | C20H28O2 | 300.44 | 1 | 11 | `CC1=C(/C=C/C(C)=C/C=C/C(C)=C\C(=O)O)C(C)(C)CCC1` |
| ixabepilone | 6445540 | Ixabepilone | C27H42N2O5S | 506.71 | 1 | 4 | `C/C(=C\c1csc(C)n1)[C@@H]1C[C@@H]2O[C@]2(C)CCC[C@H](C)[C@H](O)[C@@H](C)C(=O)C(C)(C)[C@@H](O)CC(=O)N1` |
| ketoconazole | 47576 | Ketoconazole | C26H28Cl2N4O4 | 531.44 | 1 | 2 | `CC(=O)N1CCN(c2ccc(OC[C@@H]3CO[C@@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1` |
| ladirubicin | 216232 | Ladirubicin | C29H31NO11S | 601.63 | 1 | 1 | `CC(=O)[C@]1(O)Cc2c(O)c3c(c(O)c2[C@@H](O[C@H]2C[C@H](N4CC4)[C@H](OS(C)(=O)=O)[C@H](C)O2)C1)C(=O)c1ccccc1C3=O` |
| leflunomide | 3899 | Leflunomide | C12H9F3N2O2 | 270.21 | 1 | 1 | `Cc1oncc1C(=O)Nc1ccc(C(F)(F)F)cc1` |
| levothyroxine sodium | 5819 | Thyroxine | C15H11I4NO4 | 776.87 | 1 | 42 | `N[C@@H](Cc1cc(I)c(Oc2cc(I)c(O)c(I)c2)c(I)c1)C(=O)O` |
| megestrol acetate | 11683 | Megestrol Acetate | C24H32O4 | 384.52 | 1 | 16 | `CC(=O)O[C@]1(C(C)=O)CC[C@H]2[C@@H]3C=C(C)C4=CC(=O)CC[C@]4(C)[C@H]3CC[C@@]21C` |
| methylprednisolone | 6741 | Methylprednisolone | C22H30O5 | 374.48 | 1 | 2 | `C[C@H]1C[C@@H]2[C@H]([C@@H](O)C[C@@]3(C)[C@H]2CC[C@]3(O)C(=O)CO)[C@@]2(C)C=CC(=O)C=C12` |
| mibefradil | 60663 | Mibefradil | C29H38FN3O3 | 495.64 | 1 | 1 | `COCC(=O)O[C@]1(CCN(C)CCCc2nc3ccccc3[nH]2)CCc2cc(F)ccc2[C@@H]1C(C)C` |
| mitoxantrone | 4212 | Mitoxantrone | C22H28N4O6 | 444.49 | 1 | 3 | `O=C1c2c(O)ccc(O)c2C(=O)c2c(NCCNCCO)ccc(NCCNCCO)c21` |
| nilutamide | 4493 | Nilutamide | C12H10F3N3O4 | 317.22 | 1 | 1 | `CC1(C)NC(=O)N(c2ccc([N+](=O)[O-])c(C(F)(F)F)c2)C1=O` |
| nimustine | 39214 | Nimustine | C9H13ClN6O2 | 272.7 | 1 | 1 | `Cc1ncc(CNC(=O)N(CCCl)N=O)c(N)n1` |
| o6-benzylguanine | 4578 | 6-(Phenylmethoxy)-9H-purin-2-amine | C12H11N5O | 241.25 | 1 | 6 | `Nc1nc(OCc2ccccc2)c2[nH]cnc2n1` |
| ondansetron hydrochloride | 68647 | Ondansetron Hydrochloride | C18H19N3O | 293.37 | 1 | 1 | `Cc1nccn1CC1CCc2c(c3ccccc3n2C)C1=O` |
| pasireotide | 9941444 | Pasireotide | C58H66N10O9 | 1047.23 | 1 | 1 | `NCCCC[C@@H]1NC(=O)[C@@H](Cc2c[nH]c3ccccc23)NC(=O)[C@H](c2ccccc2)NC(=O)[C@@H]2C[C@@H](OC(=O)NCCN)CN2C(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](Cc2ccc(OCc3ccccc3)cc2)NC1=O` |
| prednisone | 5865 | Prednisone | C21H26O5 | 358.43 | 1 | 40 | `C[C@]12C=CC(=O)C=C1CC[C@@H]1[C@@H]2C(=O)C[C@@]2(C)[C@H]1CC[C@]2(O)C(=O)CO` |
| pv-10 | 25473 | Rose Bengal | C20H4Cl4I4O5 | 973.67 | 1 | 1 | `O=C(O)c1c(Cl)c(Cl)c(Cl)c(Cl)c1-c1c2cc(I)c(=O)c(I)c-2oc2c(I)c(O)c(I)cc12` |
| ralimetinib mesylate | 11570805 | Ralimetinib Mesylate | C24H29FN6 | 420.54 | 1 | 1 | `CC(C)(C)Cn1c(N)nc2ccc(-c3[nH]c(C(C)(C)C)nc3-c3ccc(F)cc3)nc21` |
| regorafenib | 11167602 | Regorafenib | C21H15ClF4N4O3 | 482.82 | 1 | 1 | `CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)c(F)c2)ccn1` |
| resiquimod | 159603 | Resiquimod | C17H22N4O2 | 314.39 | 1 | 0 | `CCOCc1nc2c(N)nc3ccccc3c2n1CC(C)(C)O` |
| rigosertib | 6918736 | Rigosertib | C21H25NO8S | 451.5 | 1 | 1 | `COc1cc(OC)c(/C=C/S(=O)(=O)Cc2ccc(OC)c(NCC(=O)O)c2)c(OC)c1` |
| sonolisib | 9849735 | Sonolisib | C29H35NO8 | 525.6 | 1 | 1 | `C=CCN(/C=C1/C(=O)O[C@H](COC)[C@@]2(C)C1=C(O)C(=O)C1=C2[C@H](OC(C)=O)C[C@]2(C)C(=O)CC[C@@H]12)CC=C` |
| streptozocin | 29327 | Streptozocin | C8H15N3O7 | 265.22 | 1 | 2 | `CN(N=O)C(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O` |
| tegafur-gimeracil-oteracil potassium | 5386 | Tegafur | C8H9FN2O3 | 200.17 | 1 | 3 | `O=c1[nH]c(=O)n(C2CCCO2)cc1F` |
| teniposide | 452548 | Teniposide | C32H32O13S | 656.66 | 1 | 2 | `COc1cc([C@@H]2c3cc4c(cc3[C@@H](O[C@@H]3O[C@@H]5CO[C@@H](c6cccs6)O[C@H]5[C@H](O)[C@H]3O)[C@H]3COC(=O)[C@H]23)OCO4)cc(OC)c1O` |
| therapeutic hydrocortisone | 5754 | Cortisol | C21H30O5 | 362.47 | 1 | 1 | `C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2[C@@H](O)C[C@@]2(C)[C@H]1CC[C@]2(O)C(=O)CO` |
| toremifene citrate | 3005572 | Toremifene Citrate | C26H28ClNO | 405.97 | 1 | 3 | `CN(C)CCOc1ccc(/C(=C(/CCCl)c2ccccc2)c2ccccc2)cc1` |
| trabectedin | 108150 | Trabectedin | C39H43N3O11S | 761.85 | 1 | 1 | `COc1cc2c(cc1O)CCN[C@]21CS[C@@H]2c3c(OC(C)=O)c(C)c4c(c3[C@H](COC1=O)N1[C@@H]2[C@H]2c3c(cc(C)c(OC)c3O)C[C@@H]([C@@H]1O)N2C)OCO4` |
| triptorelin | 25074470 | Triptorelin | C64H82N18O13 | 1311.47 | 1 | 5 | `CC(C)C[C@H](NC(=O)[C@@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CO)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H]1CCC(=O)N1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N1CCC[C@H]1C(=O)NCC(N)=O` |
| vandetanib | 3081361 | Vandetanib | C22H24BrFN4O2 | 475.36 | 1 | 2 | `COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1` |
| vismodegib | 24776445 | Vismodegib | C19H14Cl2N2O3S | 421.31 | 1 | 1 | `CS(=O)(=O)c1ccc(C(=O)Nc2ccc(Cl)c(-c3ccccn3)c2)c(Cl)c1` |
| 2-methoxyestradiol | 66414 | 2-Methoxyestradiol | C19H26O3 | 302.41 | 0 | 1 | `COc1cc2c(cc1O)CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](O)CC[C@@H]12` |
| abiraterone | 132971 | Abiraterone | C24H31NO | 349.52 | 0 | 4 | `C[C@]12CC[C@H](O)CC1=CC[C@@H]1[C@@H]2CC[C@]2(C)C(c3cccnc3)=CC[C@@H]12` |
| aee788 | 10297043 | 6-(4-((4-Ethyl-1-piperazinyl)methyl)phenyl)-N-((1R)-1-phenylethyl)-7H-pyrrolo(2,3-d)pyrimidin-4-amine | C27H32N6 | 440.6 | 0 | 1 | `CCN1CCN(Cc2ccc(-c3cc4c(N[C@H](C)c5ccccc5)ncnc4[nH]3)cc2)CC1` |
| afatinib | 10184653 | Afatinib | C24H25ClFN5O3 | 485.95 | 0 | 1 | `CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1` |
| amifostine | 2141 | Amifostine | C5H15N2O3PS | 214.23 | 0 | 1 | `NCCCNCCSP(=O)(O)O` |
| aminocamptothecin | 72402 | 9-Aminocamptothecin | C20H17N3O4 | 363.37 | 0 | 1 | `CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3c(N)cccc3nc2-1` |
| arsenic trioxide | 14888 | Arsenic Trioxide | As2O3 | 197.84 | 0 | 2 | `[As+3].[As+3].[O-2].[O-2].[O-2]` |
| azathioprine | 2265 | Azathioprine | C9H7N7O2S | 277.27 | 0 | 2 | `Cn1cnc([N+](=O)[O-])c1Sc1ncnc2nc[nH]c12` |
| banoxantrone | 9955116 | Banoxantrone | C22H28N4O6 | 444.49 | 0 | 1 | `C[N+](C)([O-])CCNc1ccc(NCC[N+](C)(C)[O-])c2c1C(=O)c1c(O)ccc(O)c1C2=O` |
| belinostat | 6918638 | Belinostat | C15H14N2O4S | 318.35 | 0 | 1 | `O=C(/C=C/c1cccc(S(=O)(=O)Nc2ccccc2)c1)NO` |
| birinapant | 49836020 | Birinapant | C42H56F2N8O6 | 806.96 | 0 | 1 | `CC[C@H](NC(=O)[C@H](C)NC)C(=O)N1C[C@@H](O)C[C@H]1Cc1c(-c2[nH]c3cc(F)ccc3c2C[C@@H]2C[C@H](O)CN2C(=O)[C@H](CC)NC(=O)[C@H](C)NC)[nH]c2cc(F)ccc12` |
| bortezomib | 387447 | Bortezomib | C19H25BN4O4 | 384.25 | 0 | 4 | `CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)c1cnccn1)B(O)O` |
| broxuridine | 6035 | 5-Bromo-2'-deoxyuridine | C9H11BrN2O5 | 307.1 | 0 | 1 | `O=c1[nH]c(=O)n([C@H]2C[C@H](O)[C@@H](CO)O2)cc1Br` |
| cabozantinib s-malate | 25102846 | Cabozantinib S-malate | C28H24FN3O5 | 501.51 | 0 | 1 | `COc1cc2nccc(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3)c2cc1OC` |
| carboxyamidotriazole | 108144 | Carboxyamidotriazole | C17H12Cl3N5O2 | 424.68 | 0 | 1 | `NC(=O)c1nnn(Cc2cc(Cl)c(C(=O)c3ccc(Cl)cc3)c(Cl)c2)c1N` |
| chloroquine | 2719 | Chloroquine | C18H26ClN3 | 319.88 | 0 | 1 | `CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12` |
| clodronate disodium | 23724874 | Clodronate Disodium | CH4Cl2O6P2 | 244.89 | 0 | 5 | `O=P(O)(O)C(Cl)(Cl)P(=O)(O)O` |
| cytarabine | 6253 | Cytarabine | C9H13N3O5 | 243.22 | 0 | 3 | `Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@@H]2O)c(=O)n1` |
| defactinib | 25117126 | Defactinib | C20H21F3N8O3S | 510.5 | 0 | 1 | `CNC(=O)c1ccc(Nc2ncc(C(F)(F)F)c(NCc3nccnc3N(C)S(C)(=O)=O)n2)cc1` |
| diphencyprone | 65057 | Diphenylcyclopropenone | C15H10O | 206.24 | 0 | 1 | `O=c1c(-c2ccccc2)c1-c1ccccc1` |
| enzastaurin | 176167 | Enzastaurin | C32H29N5O2 | 515.62 | 0 | 2 | `Cn1cc(C2=C(c3cn(C4CCN(Cc5ccccn5)CC4)c4ccccc34)C(=O)NC2=O)c2ccccc21` |
| etoposide phosphate | 6918092 | Etoposide Phosphate | C29H33O16P | 668.54 | 0 | 1 | `COc1cc([C@@H]2c3cc4c(cc3[C@@H](O[C@@H]3O[C@@H]5CO[C@@H](C)O[C@H]5[C@H](O)[C@H]3O)[C@H]3COC(=O)[C@H]23)OCO4)cc(OC)c1OP(=O)(O)O` |
| fenretinide | 5288209 | Fenretinide | C26H33NO2 | 391.56 | 0 | 1 | `CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)Nc2ccc(O)cc2)C(C)(C)CCC1` |
| fludarabine | 657237 | Fludarabine | C10H12FN5O4 | 285.24 | 0 | 1 | `Nc1nc(F)nc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@@H]1O` |
| fosaprepitant | 135413538 | Fosaprepitant | C23H22F7N4O6P | 614.41 | 0 | 1 | `C[C@@H](O[C@H]1OCCN(Cc2nn(P(=O)(O)O)c(=O)[nH]2)[C@H]1c1ccc(F)cc1)c1cc(C(F)(F)F)cc(C(F)(F)F)c1` |
| gossypol | 3503 | Gossypol | C30H30O8 | 518.56 | 0 | 3 | `Cc1cc2c(C(C)C)c(O)c(O)c(C=O)c2c(O)c1-c1c(C)cc2c(C(C)C)c(O)c(O)c(C=O)c2c1O` |
| hpph | 148160 | 2-(1-Hexyloxyethyl)-2-devinyl pyropheophorbide-a | C39H48N4O4 | 636.84 | 0 | 3 | `CCCCCCOC(C)C1=C(C)C2=NC1=CC1=NC(=CC3=C(C)C4=C(O)CC(=C5NC(=C2)[C@@H](C)[C@@H]5CCC(=O)O)C4=N3)C(CC)=C1C` |
| hydroxychloroquine | 3652 | Hydroxychloroquine | C18H26ClN3O | 335.88 | 0 | 1 | `CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12` |
| ibandronate sodium | 23663991 | Ibandronate Sodium | C9H23NO7P2 | 319.23 | 0 | 1 | `CCCCCN(C)CCC(O)(P(=O)(O)O)P(=O)(O)O` |
| imiquimod | 57469 | Imiquimod | C14H16N4 | 240.31 | 0 | 1 | `CC(C)Cn1cnc2c(N)nc3ccccc3c21` |
| iniparib | 9796068 | Iniparib | C7H5IN2O3 | 292.03 | 0 | 2 | `NC(=O)c1ccc(I)c([N+](=O)[O-])c1` |
| irofulven | 148189 | (-)-Irofulven | C15H18O3 | 246.31 | 0 | 1 | `CC1=C(CO)C2=C(C)C3(CC3)[C@@](C)(O)C(=O)C2=C1` |
| lenalidomide | 216326 | Lenalidomide | C13H13N3O3 | 259.26 | 0 | 1 | `Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O` |
| levetiracetam | 5284583 | Levetiracetam | C8H14N2O2 | 170.21 | 0 | 5 | `CC[C@@H](C(N)=O)N1CCCC1=O` |
| lonafarnib | 148195 | Lonafarnib | C27H31Br2ClN4O2 | 638.83 | 0 | 3 | `NC(=O)N1CCC(CC(=O)N2CCC([C@H]3c4ncc(Br)cc4CCc4cc(Cl)cc(Br)c43)CC2)CC1` |
| marimastat | 119031 | Marimastat | C15H29N3O5 | 331.41 | 0 | 1 | `CNC(=O)[C@@H](NC(=O)[C@H](CC(C)C)[C@H](O)C(=O)NO)C(C)(C)C` |
| mercaptopurine | 667490 | 6-Mercaptopurine | C5H4N4S | 152.18 | 0 | 1 | `S=c1nc[nH]c2nc[nH]c12` |
| metformin | 4091 | Metformin | C4H11N5 | 129.17 | 0 | 1 | `CN(C)C(=N)N=C(N)N` |
| mivobulin isethionate | 182762 | Mivobulin isethionate | C17H19N5O2 | 325.37 | 0 | 4 | `CCOC(=O)Nc1cc2c(c(N)n1)N=C(c1ccccc1)[C@H](C)N2` |
| motesanib diphosphate | 16097729 | Motesanib Diphosphate | C22H23N5O | 373.46 | 0 | 1 | `CC1(C)CNc2cc(NC(=O)c3cccnc3NCc3ccncc3)ccc21` |
| motexafin gadolinium | 158385 | Motexafin gadolinium | C52H72GdN5O14 | 1148.42 | 0 | 1 | `CC(=O)[O-].CC(=O)[O-].CCc1c(CC)c2cc3nc(cnc4cc(OCCOCCOCCOC)c(OCCOCCOCCOC)cc4ncc4nc(cc1[n-]2)C(CCCO)=C4C)C(C)=C3CCCO.[Gd+3]` |
| sirolimus | 5284616 | Sirolimus | C51H79NO13 | 914.19 | 0 | 8 | `CO[C@H]1C[C@@H]2CC[C@@H](C)[C@@](O)(O2)C(=O)C(=O)N2CCCC[C@H]2C(=O)O[C@H]([C@H](C)C[C@@H]2CC[C@@H](O)[C@H](OC)C2)CC(=O)[C@H](C)/C=C(\C)[C@@H](O)[C@@H](OC)C(=O)[C@H](C)C[C@H](C)/C=C/C=C/C=C/1C` |
| nelipepimut-s | 9941306 | Nelipepimut-S | C50H78N10O11 | 995.23 | 0 | 3 | `CC[C@H](C)[C@H](NC(=O)[C@@H](N)CCCCN)C(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(C)C)C(=O)O` |
| octreotide acetate | 155804637 | Octreotide Acetate | C49H66N10O10S2 | 1019.26 | 0 | 1 | `C[C@@H](O)[C@@H]1NC(=O)[C@H](CCCCN)NC(=O)[C@@H](Cc2c[nH]c3ccccc23)NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@@H](NC(=O)[C@H](N)Cc2ccccc2)CSSC[C@@H](C(=O)N[C@H](CO)[C@@H](C)O)NC1=O` |
| oxcarbazepine | 34312 | Oxcarbazepine | C15H12N2O2 | 252.27 | 0 | 3 | `NC(=O)N1c2ccccc2CC(=O)c2ccccc21` |
| patupilone | 448013 | Epothilone B | C27H41NO6S | 507.69 | 0 | 1 | `C/C(=C\c1csc(C)n1)[C@@H]1C[C@@H]2O[C@]2(C)CCC[C@H](C)[C@H](O)[C@@H](C)C(=O)C(C)(C)[C@@H](O)CC(=O)O1` |
| perifosine | 148177 | Perifosine | C25H52NO4P | 461.67 | 0 | 1 | `CCCCCCCCCCCCCCCCCCOP(=O)([O-])OC1CC[N+](C)(C)CC1` |
| phenytoin sodium | 657302 | Phenytoin Sodium | C15H12N2O2 | 252.27 | 0 | 1 | `O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1` |
| plicamycin | 163659 | Mithramycin | C52H76O24 | 1085.16 | 0 | 1 | `CO[C@H](C(=O)[C@@H](O)[C@@H](C)O)[C@@H]1Cc2cc3cc(O[C@H]4C[C@@H](O[C@H]5C[C@@H](O)[C@H](O)[C@@H](C)O5)[C@H](O)[C@@H](C)O4)c(C)c(O)c3c(O)c2C(=O)[C@H]1O[C@H]1C[C@@H](O[C@H]2C[C@@H](O[C@H]3C[C@](C)(O)[C@H](O)[C@@H](C)O3)[C@@H](O)[C@@H](C)O2)[C@H](O)[C@@H](C)O1` |
| pyrazinamide | 1046 | Pyrazinamide | C5H5N3O | 123.11 | 0 | 1 | `NC(=O)c1cnccn1` |
| raltitrexed | 135400182 | Raltitrexed | C21H22N4O6S | 458.5 | 0 | 1 | `Cc1nc2ccc(CN(C)c3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)s3)cc2c(=O)[nH]1` |
| ridaforolimus | 11520894 | Ridaforolimus | C53H84NO14P | 990.22 | 0 | 1 | `CO[C@H]1C[C@@H]2CC[C@@H](C)[C@@](O)(O2)C(=O)C(=O)N2CCCC[C@H]2C(=O)O[C@H]([C@H](C)C[C@@H]2CC[C@@H](OP(C)(C)=O)[C@H](OC)C2)CC(=O)[C@H](C)/C=C(\C)[C@@H](O)[C@@H](OC)C(=O)[C@H](C)C[C@H](C)/C=C/C=C/C=C/1C` |
| suramin | 5361 | Suramin | C51H40N6O23S6 | 1297.3 | 0 | 1 | `Cc1ccc(C(=O)Nc2ccc(S(=O)(=O)O)c3cc(S(=O)(=O)O)cc(S(=O)(=O)O)c23)cc1NC(=O)c1cccc(NC(=O)Nc2cccc(C(=O)Nc3cc(C(=O)Nc4ccc(S(=O)(=O)O)c5cc(S(=O)(=O)O)cc(S(=O)(=O)O)c45)ccc3C)c2)c1` |
| tanespimycin | 6505803 | Tanespimycin | C31H43N3O8 | 585.7 | 0 | 1 | `C=CCNC1=C2C[C@@H](C)C[C@H](OC)[C@H](O)[C@@H](C)/C=C(\C)[C@H](OC(N)=O)[C@@H](OC)/C=C\C=C(/C)C(=O)NC(=CC1=O)C2=O` |
| tesetaxel | 6918574 | Tesetaxel | C46H60FN3O13 | 881.99 | 0 | 1 | `CC(=O)O[C@@]12CO[C@@H]1CC[C@@]1(C)[C@@H]3O[C@H](CN(C)C)O[C@@H]3C3=C(C)[C@@H](OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)c4ncccc4F)C[C@@](O)([C@@H](OC(=O)c4ccccc4)[C@@H]12)C3(C)C` |
| thioguanine | 2723601 | Thioguanine | C5H5N5S | 167.2 | 0 | 4 | `Nc1nc(=S)c2[nH]cnc2[nH]1` |
| thiotepa | 5453 | Thiotepa | C6H12N3PS | 189.22 | 0 | 1 | `S=P(N1CC1)(N1CC1)N1CC1` |
| tipifarnib | 159324 | Tipifarnib | C27H22Cl2N4O | 489.41 | 0 | 2 | `Cn1cncc1[C@@](N)(c1ccc(Cl)cc1)c1ccc2c(c1)c(-c1cccc(Cl)c1)cc(=O)n2C` |
| valproic acid | 3121 | Valproic Acid | C8H16O2 | 144.21 | 0 | 2 | `CCCC(CCC)C(=O)O` |
| vatalanib | 151194 | Vatalanib | C20H15ClN4 | 346.82 | 0 | 2 | `Clc1ccc(Nc2nnc(Cc3ccncc3)c3ccccc23)cc1` |