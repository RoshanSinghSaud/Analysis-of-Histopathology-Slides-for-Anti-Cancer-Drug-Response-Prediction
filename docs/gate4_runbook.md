# Gate 4 runbook — fetch + embed the remaining 1,005 Arm A slides

Written 28 August 2026. Every figure below is either measured on this project or
read from the actual Arm A manifest — none is a guess unless it says so.

---

## The job

| | |
|---|---|
| slides remaining | **1,005** (1,211 in the manifest, 206 already done) |
| bytes to download | **1,105.6 GB** |
| slide size | mean 1.10 GB · median 1.07 GB · p90 2.07 GB · **max 4.76 GB** |
| features produced | ~16 GB (≈16 MB/slide at the 4,000-tile cap) |

**Arm A slides are bigger than Arm B's** — BRCA's median was 0.84 GB and its
largest 2.33 GB. Batch sizing carried over from Gate 2 does not transfer
unchanged, which is why the disk figures below were re-derived.

---

## What to rent

**RTX A5000 · Community Cloud · $0.27/hr · 24 GB VRAM · 9 vCPU · 25 GB RAM**

Why this and not something faster: the 41 s/slide embedding rate was **measured
on this exact card at the 4,000-tile cap** during Gate 1. Every time estimate
below rests on a real measurement rather than an extrapolation. ViT-L inference
needs about 6 GB, so 24 GB is ample and 48 GB is wasted money.

The RTX 3090 ($0.50/hr, 16 vCPU) is the tempting alternative — tile decoding is
CPU-bound, so 16 vCPU against 9 could cut the embedding half by up to ~1.7×.
That would be roughly 15 h at ~$7.60 against 20 h at ~$5.60. It is a
**projection, not a measurement**, and with ~$10.50 of credit left the cheaper
grounded option is the better risk. Do not rent an A100: at 8 vCPU it is
actively worse here than the A5000 and five times the price.

### Storage — two separate things, and the distinction matters

| | size | why |
|---|---|---|
| **Container disk** | **60 GB** | Holds one batch of raw `.svs`. Measured peak batch at `--batch-size 10` is **19.8 GB**; 60 GB leaves room for the writable layer, gdc-client temp files and a fat batch. ~$0.17 for the whole run at $0.10/GB/mo. |
| **Network volume** at `/workspace` | **40 GB** | Features (~16 GB), the HF cache holding UNI (1.3 GB), the repo. ~$0.08 for the run at $0.07/GB/mo — **but $2.80/month if you forget to delete it.** |

**Slides go on the local container disk, never the volume.** Embedding does
~4,000 random reads per `.svs`; over network-backed storage every one of those
pays network latency. Set it up as Gate 2 did:

```bash
mkdir -p /localdata/raw /localdata/wsi && ln -sfn /localdata data
```

Features and the repo stay on `/workspace` so they survive pod termination.

---

## Time and cost

| phase | figure | basis |
|---|---|---|
| embed | 1,005 × 41 s = **11.4 h** | measured, A5000, 4,000-tile cap |
| download | 1,105.6 GB at 300.7 Mbps = **8.4 h** | measured once, on one pod, one region |
| setup | ~0.5 h | `setup_pod.sh` |
| **total** | **~20–21 h** | the script fetches *then* embeds per batch, so these add rather than overlap |
| **cost** | **~$5.60** GPU + ~$0.25 storage | $0.27/hr |

Against ~$10.50 of remaining credit, with margin.

**The biggest uncertainty is download throughput.** 300.7 Mbps was seen once. If
it comes back at half that, add ~8 h and ~$2.30 — still inside budget, but worth
knowing before you commit. **Time batch 1 first** (step 5 below); fifteen minutes
of measurement beats twenty hours of hoping.

The GPU sits idle for the 8.4 h of downloading, costing about $2.30. Overlapping
the next batch's download with the current batch's embedding would roughly halve
the wall clock. That is a change to proven code immediately before a 20-hour run,
so it is **not** recommended now — noted as a real option for a future cohort.

---

## ⚠️ The pre-flight trap — read this one twice

`00_fetch_embed_batches.py` decides what is left to do from
**`output/fetch_state.json`**, and `output/` is **gitignored**.

A fresh `git clone` on a new pod therefore has no state file, so the script would
compute `todo` as all **1,211** slides and cheerfully re-download and re-embed the
206 BRCA slides that are already finished. That is ~3.6 wasted hours and about $1,
for nothing.

**This is why `output/fetch_state.json` is inside the step-3 tarball.** It cannot
be forgotten if it travels with the code. Still confirm with the dry run in step 4
that it says `remaining: 1005`, not 1211 — if it says 1211, the file did not land
where the script looks for it (`<repo>/output/fetch_state.json`).

---

## Order of operations

**1. Before renting anything**

- Confirm the UNI licence is still accepted at huggingface.co/MahmoodLab/UNI and
  you have `HF_TOKEN` to hand. `setup_pod.sh --check` fails loudly without it —
  but only once the pod is already billing.

**2. Create the pod**

RTX A5000, Community Cloud, RunPod PyTorch image, **container disk 60 GB**,
**network volume 40 GB mounted at `/workspace`**.

**3. Get the code onto the pod — NOT by cloning**

There is no remote repository, and that turns out to be the safe path anyway.
Of the six files the pod needs, **five are uncommitted**: `00_` and `01_` are
modified in the working tree, and `setup_pod.sh`, `requirements.txt` and
`gdc_manifest_arm_a.txt` are untracked. A `git clone` would deliver the *old*
`00_` and `01_` — the versions from before the 25 August pre-flight, missing the
per-slide RNG fix, the DataLoader/fp16 path and the portable manifest paths.

That is not a cosmetic difference. The old code seeds tile sampling from a
module-level global, so a slide's tiles depend on how many slides were drawn
before it in the same process. Arm A features built that way would not be
homogeneous with Arm B's, and the pre-registration's "seeded random sample"
claim in §3.5 could not be backed. Two Mac-embedded slides already had to be
deleted and re-done for exactly this reason.

**So the pod gets the working tree, not a commit.** On the Mac:

```bash
cd ~/Desktop/Research
tar -czf /tmp/gate4_pod_bundle.tgz \
  scripts/00_fetch_embed_batches.py scripts/01_preprocess_and_extract.py \
  scripts/cohort_labels.py scripts/setup_pod.sh requirements.txt \
  cohort/gdc_manifest_arm_a.txt cohort/cohort_arm_a.csv cohort/cohort_brca.csv \
  output/fetch_state.json
```

**187 KB.** `output/fetch_state.json` is bundled deliberately — it is the file
that stops the pod re-doing the 206 finished BRCA slides, and putting it in the
archive means it cannot be forgotten.

Transfer it with `runpodctl` (installed on the pod image already;
`brew install runpod/runpodctl/runpodctl` on the Mac):

```bash
runpodctl send /tmp/gate4_pod_bundle.tgz     # Mac — prints a one-time code
```
```bash
cd /workspace && mkdir -p Research && cd Research
runpodctl receive <code>                     # pod
tar -xzf gate4_pod_bundle.tgz
```

`scp -P <port> -i ~/.ssh/<key> /tmp/gate4_pod_bundle.tgz root@<ip>:/workspace/`
works equally well if you would rather use the SSH details RunPod shows.

Then, on the pod:

```bash
bash scripts/setup_pod.sh          # gdc-client, HF_HOME, CUDA/openslide checks
mkdir -p /localdata/raw /localdata/wsi && ln -sfn /localdata data
```

**When the repo does get created**, this step becomes a normal clone — but only
once `00_`, `01_`, `setup_pod.sh` and `requirements.txt` are actually committed.
Until then, cloning is the dangerous option and the tarball is the safe one.

**4. Verify before spending — costs nothing**

```bash
python scripts/00_fetch_embed_batches.py \
  --manifest cohort/gdc_manifest_arm_a.txt --batch-size 10 --dry-run
```

Expect exactly:

```
slides total    : 1211  (1279.3 GB)
already embedded: 206
remaining       : 1005  (1105.6 GB)
batch size      : 10 slides
batches to run  : 101
peak disk/batch : ~19.8 GB
delete raw .svs : yes, after verification
```

**Any other number for "remaining" means the state file did not arrive. Stop.**

**5. Time one batch before committing to twenty hours**

```bash
python scripts/00_fetch_embed_batches.py \
  --manifest cohort/gdc_manifest_arm_a.txt --batch-size 10 --max-batches 1
```

Batch 1 is 7.4 GB / 10 slides. Expect roughly 3–4 minutes of download and
7 minutes of embedding. Multiply by 101 and compare against the 20-hour estimate
above. If it is wildly worse, stop and diagnose rather than burning the credit.

**6. Run it**

```bash
nohup python scripts/00_fetch_embed_batches.py \
  --manifest cohort/gdc_manifest_arm_a.txt --batch-size 10 --quiet-gdc \
  > output/gate4.log 2>&1 &
```

Check on it with `tail -f output/gate4.log`.

---

## Gotchas that have already cost time on this project

1. **Do not STOP the pod mid-run — terminate it when finished.** Stopping wipes
   the container disk, taking the pip installs and `/localdata` with it. Saving
   $0.27/hr costs a re-run of `setup_pod.sh`.
2. **GDC ships a zip inside a zip.** `setup_pod.sh` already handles it.
3. **Delete the network volume when you are done.** A forgotten volume bills at
   $0.20/GB/mo when idle — double the running rate. This is the named cost risk,
   not the run itself.
4. A slide is deleted only after its features verify. Interruptions are safe;
   the run resumes from the state file.

---

## ⚠️ After the run — do NOT extract over the top

The pod produces a `feature_manifest.json` containing only the ~1,005 **new**
slides. The Mac's copy holds the 206 BRCA slides. Untarring the new features over
the existing directory would **replace** the manifest and orphan every BRCA
feature file — the store would still hold the `.pt` files but nothing would know
they exist.

The two manifests must be **merged**, not swapped. Before extracting anything:

- back up `output/features/feature_manifest.json` on the Mac
- merge the two JSON objects, then confirm the total is **1,211 entries** and that
  every entry still has `status: done` and the same provenance tuple
  (`cuda`, `fp16=True`, `max_tiles=4000`, `seed=42`)

Also remember from Gate 2: **the Cowork device bridge cannot overwrite or delete
files**, so a large archive must be extracted in a normal Terminal, not through
the bridge.

---

## Does the pharmacist's question block this run? No.

The open question — *will the medicine have affected the histopathology slide?* —
is a validity question, not a cohort question, and the cohort data already points
hard at an answer.

`days_to_treatment_start` is measured from diagnosis, and the diagnostic slide is
cut at diagnosis. Across the 2,448 Arm A treatment records:

| | |
|---|---|
| populated | 2,344 of 2,448 (95.8%) |
| median | **+85 days** — treatment starts about three months *after* diagnosis |
| range | −709 to +6,830 days |
| **negative — drug started BEFORE diagnosis** | **10 records (0.4%)** |

So for roughly 99.6% of records the drug was given after the slide was cut, and
cannot have altered it. The **10 negative records are the real exposure** and are
worth a pre-registered sensitivity analysis: repeat the primary analysis excluding
those patients. That rule should be fixed before any Arm A AUROC exists.

Note what does *not* answer this: `treatment_intent` is **blank for 2,147 of 2,448
records (87.7%)**, with only 296 marked Adjuvant. Do not lean on that field.

**And even if the answer changed the cohort, it would not change the download.**
The manifest is the 1,031-patient superset, taken before every exclusion. Any rule
your friend's answer produces removes patients from the *analysis*, leaving a few
unused slides on disk. Downloading is a superset operation — the same argument that
makes Sabin's pending SMILES review a non-blocker.

What his answer genuinely feeds is §9 (threats to validity) and the discussion.
Neither needs to exist before the slides are converted to numbers.

---

## What Gate 4 does not decide

Nothing in this run touches a label, a model or the primary comparison. It
converts slides into the same frozen feature representation Arm B already uses.
The analysis that follows is governed by the pre-registration as it stands at
v3.0 — hyperparameters frozen, the one permitted tuning pass spent.
