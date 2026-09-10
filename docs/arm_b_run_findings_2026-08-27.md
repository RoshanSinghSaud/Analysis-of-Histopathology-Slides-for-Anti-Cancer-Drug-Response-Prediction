# Arm B (TCGA-BRCA) first full run — result, failure analysis, and what changes

**Date:** 27 August 2026
**Run config hash:** `0b1ee333fd49` (in `output/eval/run_log.jsonl`)
**Script:** `scripts/08_evaluate_ablation.py`
**Pre-registration at time of run:** v2.4

Companion to `plain_language_log_2026-08-26.md`, which covers the four problems
found the previous day while building the harness.

---

## Definitions — read this first

### The two arms

The study runs on two nested cohorts with different jobs. They are not two
experiments; Arm B is a subset of Arm A, reported for different reasons.

**Arm A — the powered test. This is where the paper's actual claim is tested.**
The 10 largest usable TCGA projects: BRCA, LGG, STAD, TGCT, BLCA, SKCM, LUAD,
CESC, COAD, PAAD.

| | |
|---|---|
| cohort patients | 1,031 (650 responder / 381 non-responder) |
| **analysable** (drug has a molecular structure) | **981** (627 / 354) |
| drugs | 127 in cohort, **94** analysable |
| slides | 1,211 · about 1.25 TB |
| power | 0.99 at the smallest effect of interest, ΔAUROC = 0.05 |
| status | slides not yet downloaded — this is Gate 4 |

**Arm B — the clinical showcase. TCGA-BRCA only, a subset of Arm A.**
Its job is attention heatmaps, subtype and stage breakdowns, interpretability,
and the narrative a clinical reader follows.

| | |
|---|---|
| cohort patients | 199 (177 responder / 22 non-responder) |
| **analysable** | **195** (173 / 22) |
| drugs | 36 in cohort, **29** analysable |
| slides | 206 · 173.7 GB — already embedded |
| power | **12%** at ΔAUROC = 0.03 |
| status | embedded and run |

**Arm B is explicitly NOT a significance test.** At 12% power it would miss a
real effect seven times out of eight. Every Arm B number is reported
descriptively, with confidence intervals and without a p-value. No claim in the
paper rests on Arm B alone.

**Why not breast cancer alone?** Every label BRCA can supply is severely
imbalanced — 177 responders against 22 non-responders — and it offers only a
handful of drugs with enough patients, which starves the drug branch of the
variation it exists to exploit. Publishing a null from an underpowered cohort
means "we could not tell", not "there is no effect".

### The gates

The project is staged so that every expensive step sits behind a cheap check
that could have made it unnecessary.

| gate | cost | what it is | status |
|---|---|---|---|
| **Gate 0** | $0 | Derive the cohort, fix the labels, run the power analysis | done 26 Aug |
| **Gate 1** | pennies | **Measure** embedding speed: laptop vs rented GPU, before committing to either | done 25 Aug — the rented GPU was 19.6× faster (41 s vs 802 s per slide). Verdict: rent |
| **Gate 2** | ~$1.50 | Embed Arm B's 206 breast slides | done 25 Aug — 1 h 53 m, zero failures |
| **Gate 3** | $0 | Write the evaluation harness and run it on Arm B | harness written 26 Aug, run 27 Aug |
| **Gate 4** | ~$5–7 | **Download and embed the remaining 1,005 Arm A slides (~1.1 TB).** This is where the primary endpoint's data comes from | not started |
| **Gate 5** | $0 | Freeze the analysis, record the commit hash, write the discussion, submit | not started |

**Gate 1 is the one worth understanding.** It cost almost nothing and redirected
the entire plan away from roughly three days of laptop time. That is the standing
rule of this project: *every GPU dollar sits behind a free check that could have
made it wasted.*

**Gate 4 is the one that matters now.** Arm B is finished and its answer is
descriptive only. Until Gate 4 delivers Arm A's slides, the study's actual
question is unanswered.

---

---

## Summary in one paragraph

The first full cross-validated run of Arm B produced **AUROC ≈ 0.53 in every
arm**, against a chance level of 0.500. This is **not a null result and must
never be reported as one.** A follow-up capability check proved the training
loop is sound — it memorises 50 real patients within 5 epochs — so the failure
is not a bug. The mechanism is specific and now understood: the model overfits
within a handful of epochs, and early stopping cannot detect the moment because
it watches AUROC on an inner-validation set holding only 2–3 non-responders.
Arm B's descriptive result stands and is reportable. The primary endpoint was
never located here, and is unaffected. Two changes are owed to §10 before Arm A.

---

## 1. What was run

5 seeds × 5 folds × 4 arms = **100 model trainings**, 61.5 minutes on Apple MPS.
Analysable cohort: **195 patients** (173 responder / 22 non-responder),
528 (patient, drug) rows, 29 structurally-resolvable drugs
(see the biologic exclusion criterion, pre-registration §3 and §10).

| arm | mean AUROC | sd across seeds | per-seed |
|---|---:|---:|---|
| without_smiles | 0.5286 | 0.047 | 0.597, 0.496, 0.556, 0.483, 0.510 |
| with_smiles | 0.5282 | 0.058 | 0.619, 0.551, 0.499, 0.478, 0.493 |
| shuffled_drug | 0.5226 | 0.057 | 0.612, 0.544, 0.496, 0.468, 0.494 |
| onehot_drug | 0.5503 | 0.038 | 0.607, 0.571, 0.521, 0.524, 0.529 |

| contrast | ΔAUROC | sd | seeds significant |
|---|---:|---:|---:|
| with_smiles − without_smiles (primary) | **−0.0005** | 0.0424 | 0 / 5 |
| shuffled_drug − without_smiles | −0.0060 | 0.0405 | 0 / 5 |
| onehot_drug − without_smiles | +0.0217 | 0.0405 | 1 / 5 |
| with_smiles − shuffled_drug (capacity-matched) | +0.0055 | 0.0043 | 0 / 5 |

Baseline seed-1 95% CI: **[0.467, 0.714]** — contains 0.5.

---

## 2. Why this is NOT a null result

Three independent reasons, any one of which is sufficient:

1. **Every arm sits on chance.** 0.500 is the no-skill line. All four arms are
   within 0.03 of it, and **two of five baseline seeds fall below it.**
2. **The noise dwarfs the signal.** Seed-to-seed spread is 0.047. The primary Δ
   is 0.0005 — roughly one hundred times smaller than the run-to-run variation
   of the thing being measured.
3. **§8's precondition is not met.** The pre-registered null-result fallback is
   valid only when the analysis demonstrably *could* have detected an effect. A
   comparison between two models that both perform at chance demonstrates no
   such thing.

Stated plainly: there was no working model for the drug branch to improve upon.
The correct description is **"the measurement did not work in Arm B"**, not
"molecular structure adds nothing".

---

## 3. The diagnosis

The result above is compatible with two very different worlds, which demand
opposite responses:

- **(a)** the training loop is defective — gradients, loss, or bag pooling;
- **(b)** the loop is correct, and 195 patients with 22 non-responders carry no
  signal that generalises.

`--overfit-check` separates them. It trains the **baseline arm only** on ~50
patients with regularisation switched off and no early stopping, then scores it
on those same patients. It never computes a with-versus-without difference, so
it cannot influence the primary endpoint, and it reports performance on training
data, which is not an estimate of anything.

**Result on real BRCA bags — 50 patients (12 non-responder / 38 responder), 133 rows:**

| epoch | train loss | train AUROC |
|---:|---:|---:|
| 1 | 0.7361 | 0.929 |
| 5 | 0.0802 | **1.000** |
| 20 | 0.0011 | 1.000 |
| 60 | 0.0001 | 1.000 |

**Verdict: world (b).** The loop fits data effortlessly when signal is present.

### The sharper finding: real data memorises five times faster than synthetic

| | epoch 1 | reaches AUROC 1.000 |
|---|---:|---:|
| synthetic bags (control) | 0.645 | epoch 25 |
| **real BRCA bags** | **0.929** | **epoch 5** |

Real tissue was at 0.929 after a single pass. The reason is not model strength
but data richness: **every slide is visually distinctive, and UNI's features
make individual patients trivially identifiable.** With 4,000 richly-described
tiles per person, separating 50 people is easy. Learning what *responders* share
is an entirely different and much harder problem.

This is worth stating in the paper's limitations: representation quality that
helps discrimination between images can accelerate memorisation of individuals,
which at small n is a liability rather than an asset.

---

## 4. The failure mechanism, named

1. The model memorises its training patients within a handful of epochs.
2. Early stopping exists to catch that moment and halt.
3. But it watches **AUROC on ~23 patients holding 2–3 non-responders.**
4. With so few positives, AUROC takes only a handful of discrete values. The run
   log contains exact `1.000` and exact `0.500`, and best-across-40-epochs values
   as low as **0.143, 0.238, 0.276, 0.328** — meaning that in those folds no
   epoch ever produced a model ranking the positives above chance on validation.
5. Early stopping therefore selects whichever epoch won a coin flip.
6. Generalisation lands at chance.

Not a bug, and not bad luck: a design that requires more validation patients than
Arm B can supply. **Arm A largely self-corrects** — its inner-validation folds
will hold roughly 120 patients with ~43 non-responders instead of 23 with 2–3.

---

## 5. A watch item, resolved

Before the run, `with_smiles` and `shuffled_drug` producing identical fold scores
was flagged as a potential bug. They matched in 12 of 13 inspected folds.

**It is a symptom, not a defect.** The drug branch is gated off at initialisation
and opens only if it lowers validation loss. Nothing was learning, so it never
opened, so feeding the true molecule versus a random one barely changed the
model. Their contrast has sd 0.0043 — about ten times tighter than the others —
precisely because the two are very nearly the same model. Pooled out-of-fold
values did differ (0.5282 vs 0.5226), so they are not literally identical.

This is also a useful diagnostic to keep: **if those two arms ever converge
again, check whether anything is being learned at all.**

---

## 6. What Arm B contributes to the paper

Arm B was pre-registered as a descriptive showcase, never as a significance test
(§3, 12% power). Its finding is legitimate and reportable:

> In TCGA-BRCA alone, tissue morphology did not predict RECIST treatment response
> above chance (AUROC 0.53, 95% CI 0.47–0.71, n = 195, 22 non-responders),
> reported descriptively with confidence intervals and without a p-value, as
> pre-specified.

Accompanied by: the analysable-cohort counts and the biologic exclusion, the
tile-count distribution, the memorisation finding above, and the observation that
inner-validation model selection is unreliable at this cohort size.

**The primary endpoint is untouched.** It has always been located in Arm A.

---

## 7. Two changes owed to the deviations log — decided BEFORE Arm A data exists

Both were motivated exclusively by evidence from the **baseline arm and training
curves**. Neither was informed by any with-versus-without comparison. That is
what makes them legitimate corrections rather than post-hoc rescue.

1. **Early stopping keyed on validation LOSS, not validation AUROC.** Loss is
   continuous and distinguishes epoch 3 from epoch 30 even with few positive
   cases; AUROC on 2–3 positives cannot.

2. **Baseline regularisation strengthened.** A model that memorises 133 bags in
   5 epochs is under-regularised. **§6 explicitly permits this**: hyperparameters
   are selected once *on the without-SMILES model*, then frozen unchanged for the
   with-SMILES model. Tuning the baseline is what the plan prescribes. What
   remains forbidden: tuning after observing Δ, and giving the drug branch any
   tuning budget of its own.

---

## 8. What was deliberately NOT done

No hyperparameter was adjusted to move ΔAUROC. No architecture was swapped after
seeing the result. No arm was re-run and reported selectively. Every run
performed against real labels — including the 5-minute smoke test, which is
stamped `quick_smoke_test: true` — is recorded in `output/eval/run_log.jsonl`
with a configuration hash, per §8.

The value of the Arm B run lies precisely here: it was a dress rehearsal. This
failure surfaced on a free overnight laptop job rather than after a GPU rental
and 1.1 TB of downloads on Arm A.

---

## 9. Reproduction

```bash
conda activate histopath

# harness validation, no real labels touched
python scripts/08_evaluate_ablation.py --selftest

# data wiring only, no model built
python scripts/08_evaluate_ablation.py --tag brca --dry-run

# the run reported above
python scripts/08_evaluate_ablation.py --tag brca --seeds 5 --device mps

# the capability check
python scripts/08_evaluate_ablation.py --tag brca --overfit-check --device mps
```

## 10. Ordered work before Arm A

1. Log the two deviations in §10 of the pre-registration
2. Tune the baseline once, freeze it, record the frozen values
3. Batch the training loop — ~10× speed, and the only route to the 200-run
   permutation null the primary test now depends on
4. Replace the frozen-PCA drug encoder with a pretrained molecular encoder (§6)
5. Add a lazy bag loader — 981 patients will not fit in memory the way 195 do
6. Gate 4 — download and embed the remaining 1,005 slides

Items 1–4 require no new data and no GPU.
