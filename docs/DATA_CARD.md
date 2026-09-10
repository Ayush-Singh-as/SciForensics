# Data Card

Provenance, licensing and split policy for every corpus SciForensics touches.

Written as part of **Stage B1**. The headline reason it exists is documented below: the prototype's
training set was **82% binary segmentation masks** and nobody knew, because nothing recorded what
had actually been loaded.

---

## The 670-vs-30,131 discovery

The legacy loader ([dataset.py:71](../src/global_matching/dataset.py#L71)) collected images with
`rglob` and then excluded masks with:

```python
self.fnames = [f for f in self.fnames if "mask" not in f.parts]
```

BBBC038 stores each sample as `<hash>/images/<hash>.png` beside `<hash>/masks/<n>.png` — one image
and *many* per-instance masks. The directory is **`masks`, plural**. `"mask" not in f.parts` tests
path components for the exact string `"mask"`, which never equals `"masks"`, so **the filter
excluded nothing**.

| | Files |
|---|---|
| Loaded by the prototype | **30,131** |
| Of which per-instance segmentation masks | **29,461** (97.8%) |
| Actual microscopy images | **670** |

The embedding model was therefore trained overwhelmingly on **white blobs on black backgrounds**
rather than on microscopy. Every similarity number the prototype ever reported was produced by that
model. This is also the most likely explanation for the inversion measured in
[PROJECT_BIBLE.md](../PROJECT_BIBLE.md) §8b, where a genuine manipulation embeds *further* from its
source (0.1877) than an unrelated image does (0.3781).

**The fix is structural, not a corrected string.** Substring matching over paths is the wrong tool
in both directions — too weak (misses `masks`, `mask_gt`, `ground_truth`) and too strong (drops a
legitimate figure from a paper about *masking*). Each dataset now declares a
[`Layout`](../src/sciforensics/global_match/data.py) and discovery asks the layout which files are
images, matching **path components exactly**, never as substrings.

`tests/test_data.py` reproduces the defective tree in miniature and asserts *both* halves: that the
legacy predicate excludes nothing, and that `discover()` returns only the images. Reproducing the
defect is deliberate — a regression cannot then pass by accident.

---

## Corpora

### BBBC038 — Broad Bioimage Benchmark Collection (Kaggle 2018 Data Science Bowl)

| | |
|---|---|
| Content | Microscopy images of cell nuclei, multiple stains and magnifications |
| Layout | `<hash>/images/<hash>.png` + `<hash>/masks/<n>.png` |
| Real images | ~670 |
| Licence | Public domain / CC0 (see Broad Institute terms) |
| Used for | **Training** (`train`/`val`), self-supervised via `manipulations.py` |
| Layout id | `bbbc038` |

Chosen because it is domain-relevant (real microscopy) and domain-*agnostic* enough that the model
does not overfit one imaging modality. Only the `images/` directory is ever read.

### Polimi Western Blots

| | |
|---|---|
| Content | Western blot images, with automatically- and manually-tampered variants |
| Subsets | `automatically_tampered`, `tampered_blots` (incl. GIMP- and DALL·E-tampered) |
| Ground truth | **Pixel-level masks** |
| Used for | `automatically_tampered` → training; `tampered_blots` → **localisation eval only** |
| Layout id | `polimi` |

The masks are free localisation ground truth and were **entirely unused** by the prototype. They
are the evaluation set for CMFD IoU/F1 in **B4**.

The DALL·E-tampered subset appears in the benchmark as a *robustness column* only. SciForensics
makes **no claim to detect AI-generated imagery** — that is a different problem and is out of scope.

### BioFors (ICCV'21)

| | |
|---|---|
| Content | Panel-organised figures from biomedical papers, with documented manipulations |
| Used for | **Evaluation only — never training** |
| Layout id | `biofors` |

**This is the benchmark for this exact task.** The prototype consumed it as unlabeled *training*
data, which forecloses using it to evaluate. It is now held out by a check in code, not by a
sentence in a README: `Layout.held_out=True` forces every file to `test`, and
`build_manifest(..., evaluation_only=False)` raises `DataError`.

BioFors is panel-organised, so it doubles as the evaluation set for panel splitting in **C1**.

---

## Split policy

Four splits, from [`data.Split`](../src/sciforensics/global_match/data.py):

| Split | Share | Purpose |
|---|---|---|
| `train` | 80% | Model fitting |
| `val` | 10% | Early stopping, hyperparameters |
| `calib` | 5% | **Calibrator fitting only** (B5) |
| `test` | 5% | Reported metrics |

`calib` is deliberately distinct from `test`: a calibrator fitted on the set it is then reported
against produces a reliability diagram that means nothing.

### Splits are content-addressed, not seeded

Membership is decided by hashing the **file's bytes** (`assign_split`), not by a seeded shuffle. A
seeded shuffle depends on the number and order of inputs, so adding a single file re-partitions the
whole corpus and silently moves existing files across the train/eval boundary.

Two properties follow, and both are tested:

1. **Stable under additions** — a file's split is a property of that file alone.
2. **Byte-identical duplicates share a split** — the same figure under two names cannot straddle
   the boundary. Duplicates are *recorded* in the manifest rather than dropped: in a figure corpus a
   repeat is itself a finding, and it must not be counted twice in a metric.

### Manifests

Frozen manifests live in `benchmarks/splits/*.json` and are **read back, never recomputed**, so any
published number traces to the exact files behind it. Each records the dataset, resolved layout,
per-split counts, `sha256 → relative path`, `sha256 → split`, and any duplicates.

```bash
sciforensics splits data/bbbc038 --dataset bbbc038
sciforensics splits data/biofors --dataset biofors --layout biofors   # forced to test
```

`check_no_leakage()` asserts no content hash appears on both sides. Cheap to run, catastrophic to
skip: one leaked figure turns a reported metric into a memorisation score.

---

## Known limitations

- **No dataset is downloaded in this repository.** `data/` is gitignored. Every path above is a
  convention the loader understands, not a directory that ships.
- **`models/weights.pth` was trained on the mask-contaminated set.** It is retained so the "found
  and fixed" comparison in B2 is measurable, but its numbers describe a model trained largely on
  the wrong modality. Do not present them as microscopy performance.
- **Precision is unmeasured.** `run_demo.py` paired each base only with its own manipulations —
  zero unrelated-pair controls — so no false-positive rate exists yet. **B6** is what closes this;
  no precision claim should be published before it.
- **The 670 figure is BBBC038's image count**, not a claim about how many are usable after quality
  filtering.
