# Model Card — SciForensics embedding model

Stage C4. Read this before quoting any number this tool produces.

**The headline you need first:** the shipped checkpoint was trained on a set that
was **82% binary segmentation masks** (bug 1), and it shows. On a measured pair, a
genuine manipulation embeds *further* from its source than an unrelated image
does. The global stage is the weakest component in the pipeline, and this card
exists to say so rather than to market it.

---

## What it is

| | |
|---|---|
| Task | Is this pair of scientific figure panels the same content under a transform? |
| Architecture | 4-block Siamese CNN → 128-d embedding, L1 distance |
| Parameters | **~8.8M**, 95% of them in one `fc1` 8192→1024 layer |
| Input | Greyscale, letterboxed to 128×128 (aspect-preserving) |
| Output | An L1 distance, mapped to `(0, 1)` by `σ((threshold − d) / temperature)` |
| Checkpoint | `models/weights.pth`, SHA-256 `befec5b8…` |

The README previously claimed "~1.2M parameters". The real figure is 8.8M;
`backbone.parameter_count()` now reports it from the model rather than from prose.

## What it is *not*

- **Not calibrated.** `ScanResult.calibrated` is `False` and every surface — CLI,
  PDF, web UI, `/v1/config` — says so in words. The score is a monotone ranking
  value. Reading 0.87 as "87% chance of manipulation" is wrong, and stage B5 is
  what would make it right.
- **Not a generated-image detector.** Polimi ships DALL·E-tampered blots, so they
  appear as a robustness column in the benchmark. Generative detection is a
  different problem and is not claimed.
- **Not the decision-maker.** Geometry overrides the embedding when they disagree,
  and the fused verdict follows geometry. The embedding's job is to decide whether
  the expensive local stage is worth running.

---

## Measured performance

From `sciforensics bench` over 19 cases (16 positives, 3 negative controls),
reproducible with `benchmarks/results.json`:

| Backend | Recall | FPR | Precision | Latency p50 |
|---|---|---|---|---|
| `orb+mutual_nn` (default) | **68.8%** | **0.0%** | **100.0%** | 0.30 s |
| `disk+lightglue` | 62.5% | 0.0% | 100.0% | 8.85 s |

Per-manipulation recall:

| Manipulation | `orb` | `disk` |
|---|---|---|
| `ex1_affine` (45° + 1.2× + mirror) | **0/3** | **0/3** |
| `ex2_degraded` (JPEG q25 + noise) | **1/3** | **1/3** |
| `ex3_copymove` | 3/3 | 3/3 |
| `ex4_exposure` | 3/3 | 3/3 |
| `ex5_blackout` | 3/3 | 3/3 |
| 180° rotation (`mountains`) | 1/1 | 0/1 |

**Three controls is not a false-positive rate.** A measured 0.0% over three pairs
is consistent with a true rate of several percent. It bounds the obvious failure
modes and nothing more. A publishable figure needs a real corpus — BioFors, held
out per [DATA_CARD.md](DATA_CARD.md).

---

## Known failure modes

Stated because a reviewer will find them anyway, and a tool that hides them is
worse than one that does not.

### 1. The embedding ranks a true positive below a negative control

Measured on the seeded fixtures:

| Pair | Similarity |
|---|---|
| `base_cell` vs its own JPEG-q25+noise copy (**true positive**) | **0.1877** |
| `base_cell` vs `base_cells_2` (**unrelated control**) | **0.3781** |

A real manipulation embeds *further away* than an unrelated image. No fusion
weight fixes this: any embedding weight large enough to promote the true positive
promotes the control with it. This is a property of the checkpoint, and the most
likely cause is bug 1 — the model largely learned white-blobs-on-black.

**Consequence:** do not use the global stage alone as a screen. Its value here is
triage, and `local_trigger_distance` is deliberately wider than the match
threshold so borderline pairs still reach geometry.

### 2. Reflection defeats the local stage

`ex1_affine` is 0/3 for **both** ORB and DISK. Neither descriptor is
mirror-invariant, so a mirrored panel yields too few correspondences to fit a
transform and geometry correctly abstains.

The flip *decode* is correct and unit-tested — a synthetic reflection recovers
`det = −1.440, flip=True, rotation −135°, scale 1.2000`, which was unreachable in
the prototype (bug 2). The gap is upstream, in matching, not in geometry.

**Mitigation that exists:** the manuscript scanner hashes all eight dihedral
orientations, so a mirrored or rotated reuse is not screened out before analysis.
**Mitigation that does not exist:** a mirror-invariant descriptor.

### 3. Heavy recompression suppresses keypoints

`ex2_degraded` is 1/3. JPEG q25 plus Gaussian noise removes the corner structure
ORB depends on. Swapping in DISK+LightGlue changed **nothing** — measured, not
assumed — which is what identifies this as an embedding problem rather than a
matcher problem.

Worse, the extra matches DISK finds are wrong. Against the known identity
transform:

| Pair | Backend | Matches | Correct | Precision |
|---|---|---|---|---|
| `base_cell → ex2_degraded` | orb | 4 | 2 | 50.0% |
| `base_cell → ex2_degraded` | **disk** | **25** | **0** | **0.0%** |

A harness counting matches would have called that a 6× improvement.

### 4. Sparse and small panels

- Panel splitting **refuses** below ~10% foreground occupancy, because no
  projection statistic separates "gap between panels" from "gap between cells".
  Those panels are reported with `confident=False`.
- CMFD's shipped defaults (`nn_ratio: 0.8`, `cluster.min_samples: 6`) miss small
  clones. An *exact* copy has near-identical descriptors, so Lowe's ratio
  `d1/d2 → 1` and the strict test rejects the very matches wanted (0.8 → 32
  self-matches; 0.9 → 258). A 128px clone on a 512×640 panel was recovered within
  a few pixels using `nn_ratio=0.9, min_samples=4`; a 40px clone on a 519×162
  strip produced 6 candidate clusters and 0 verified. Retuning is stage B4.

### 5. Resolution cap

Inputs are capped at `image.max_dimension` (2048) before analysis (bug 14).
Coordinates are mapped back to original pixels, and the downscale is recorded in
`ImageMeta.analysed_at` and surfaced as a warning — but a manipulation smaller
than the cap's sampling can be lost.

---

## Intended use

Automated **screening** to prioritise human review, in a research-integrity
workflow where a qualified reviewer has access to the original data.

**Out of scope:** any autonomous decision about misconduct. A finding is evidence
that two images share content under a transform. Duplicate imagery has legitimate
explanations — shared controls, tiled acquisition, properly attributed
republication. Absence of a finding is not proof of originality; see the failure
modes above for exactly when it means "we could not tell".

## Ethical considerations

A false accusation of research misconduct can end a career. That asymmetry is why
the pipeline is built to **abstain**: geometry returns verified, refuted, *or* a
named `RejectionReason`, and the report prints the reason in prose rather than a
bare "not verified". The prototype this replaced reported 121 phantom inliers and
"Strong evidence" on a pair whose smaller side held 12 keypoints.

## Reproducing

```bash
pip install -e ".[dev,report,api,match]"
python -m playwright install chromium          # PDF backend

sciforensics bench --set global_match.weights=models/weights.pth -b orb -b disk
#  -> benchmarks/RESULTS.md and results.json
```

Every figure in this card comes from that command or from the test suite. The
audit block in each report records the config fingerprint, git commit, weights
SHA-256, library versions and seed.
