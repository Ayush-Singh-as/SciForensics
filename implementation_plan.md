# SciForensics: zero → pro

## Context

SciForensics detects image plagiarism and manipulation in scientific publications. The current
implementation (2 commits, ~2,800 LOC) is a working three-stage prototype:

1. **Global** — a custom 4-block Siamese CNN → 128-d embedding, L1 distance, thresholded
   (`src/global_matching/model.py`, `pairwise_inference.py`)
2. **Local** — CLAHE/4× upscale → DoG ROI → ORB → BFMatcher + Lowe ratio
   (`src/local_matching/local_detector.py`)
3. **Geometry** — `cv2.findHomography` RANSAC + affine decode → rule-based 4-way verdict string
   (`forensic_scanner.py`), rendered as a 6-panel `cv2.putText` dashboard

It demonstrates the right *idea*, but it is a prototype in every dimension that matters for
credibility: no tests, no packaging, no evaluation metrics, no precision measurement, several
load-bearing correctness bugs, and 82% of the training set is binary segmentation masks. The
outputs I inspected contain visibly wrong numbers (`Estimated scale: 0.000`, `Flip detected: no`
on a vertically-flipped image, clipped verdict text, a dead all-blue heatmap).

**Goal:** rebuild this into something that survives scrutiny from a recruiter, a reviewer, *and* a
research-integrity team — staged, with each stage shipping something usable — and give it a real
web demo (FastAPI + Next.js) so the results are visible without cloning anything.

---

## Audit: what's actually wrong

Everything below was verified against the code and against the committed output images.

### Correctness bugs (silent wrong answers)

| # | Defect | Evidence |
|---|---|---|
| 1 | **82% of training data is binary masks.** `SimulatedDataset` filters `"mask" not in f.parts`, but BBBC038's directory is `masks` (plural). 29,461 of 30,131 loaded BBBC038 files are instance masks; the real image count is 670. | [dataset.py:72](src/global_matching/dataset.py:72) |
| 2 | **Flip detection is structurally impossible.** `estimateAffinePartial2D` returns a *similarity* transform `[[a,-b,tx],[b,a,ty]]`, so `det = a²+b² > 0` always → `flip_detected` can never be `True`. Confirmed: a vertically-flipped image reports "Flip detected: no". | [forensic_scanner.py:147](forensic_scanner.py:147), `_decode_affine` |
| 3 | **Mirrored content collapses the geometry estimate.** A similarity transform can't model a flip, so the fit degenerates. `base_cell_ex1_affine` (true: 45° + 1.2× + hflip) reports `rotation 15.4°, scale 0.048`; mountains reports `scale 0.000, translation 1635px`. | `outputs/demo/report_base_cell_ex1_affine.png` |
| 4 | **Degenerate many-to-one matching produces false verifications.** No cross-check / mutual-NN constraint. The DoG ROI filter's hard `< 8` fallback can leave one side with 12 keypoints and the other with 2,000 — mountains: "Keypoints: 2000 vs 12" yet **231 good matches and 121 RANSAC "inliers"**, all landing on ≤12 distinct points. Verdict: "Strong evidence". | [local_detector.py:144](src/local_matching/local_detector.py:144), `_filter_to_regions:224` |
| 5 | **Right-hand heatmap is mathematically dead.** `d = Σ|e_L − e_R|` ⇒ `∂d/∂e_R = −∂d/∂e_L`. The right branch's Grad-CAM weights are systematically negated, so `ReLU` zeroes the map. Every "Global Heatmap B" I looked at is uniform blue. | [pairwise_inference.py:94-98](src/global_matching/pairwise_inference.py:94) |
| 6 | **CMFD can only ever report one clone region.** A single global `estimateAffinePartial2D` over all self-matches; the `regions` list is misleading. The `qx < tx` direction-ordering hack also drops clones offset purely vertically. | [local_detector.py:363](src/local_matching/local_detector.py:363) |
| 7 | **Verdict text is clipped off the canvas.** Summary panel is a fixed 420px canvas; a full result writes its last wrapped line at baseline y=415. "scientific imagery" is cut in half in every report. | [forensic_scanner.py:212](forensic_scanner.py:212) |
| 8 | **Aspect ratio destroyed.** `cv2.resize(gray, (128,128))` is non-aspect-preserving, then the 8×8 heatmap is stretched back onto the original aspect — the overlay is geometrically inconsistent with the image. | [pairwise_inference.py:110](src/global_matching/pairwise_inference.py:110) |
| 9 | **Similarity score is capped at 0.731.** `σ(1−d)` maxes at `σ(1)`. Every reported "similarity" in `all_results.json` is ≤ 0.7234, including for identical content. | [model.py:82](src/global_matching/model.py:82) |
| 10 | **Threshold drift.** Trained decision boundary is `d=1.0`; `same_threshold` defaults to `1.1`; `--local-threshold` argparse default is `3.0`, `ScannerConfig` default is `2.0`, README says `2.0`. | `forensic_scanner.py:46,483` |
| 11 | Model reloaded from disk once per image in batch mode (`ForensicScanner` re-constructed inside the loop → 35MB `torch.load` × N). | [forensic_scanner.py:550](forensic_scanner.py:550) |
| 12 | `torch.load` without `weights_only=True` (arbitrary-code-execution surface, breaks on torch ≥2.6 defaults). | `pairwise_inference.py:53`, `test.py:42` |
| 13 | `enhance_for_orb`'s 4× scale is hardcoded as a bare `/4.0` in four places. Changing the scale silently corrupts every coordinate. | `local_detector.py:156,159,164` |

### Missing engineering

No `pyproject.toml`, no tests, no CI, no linting, no type checking, no `LICENSE`, no Dockerfile,
no config file (thresholds duplicated across dataclass defaults, argparse defaults and `run_demo.py`),
no logging, no seeding. `sys.path` is mutated in three places instead of using a package.
`models/weights.pth` (35MB) is committed directly into git history. `model.py` and `test.py` have
CRLF endings, producing a permanent phantom 168-line working-tree diff.

### Missing science

- **Precision is never measured.** `run_demo.py` only pairs each base with *its own* manipulations —
  there is not a single unrelated-pair control. The project cannot currently state a false-positive rate.
- **No metrics at all**: no ROC/AUC, PR, per-manipulation recall, localization IoU, or calibration.
- **BioFors is being consumed as unlabeled training data.** It is *the* benchmark for this exact task
  (ICCV'21) — using it for training forecloses using it for evaluation.
- **Polimi's ground-truth masks are unused.** `polimi_western_blots/{tampered_blots,automatically_tampered}`
  ship pixel masks (incl. GIMP- and DALL·E-tampered blots) — free localization ground truth, currently discarded.
- **Recall failure on realistic degradation.** ORB collapses under JPEG q25 + noise: signal-degradation
  pairs yield 6, 7 and 2 good matches → **0/3 detected**. `base_cells_1` affine also fails (4 matches).
- **The verdict is a hand-written if/else string**, not a calibrated confidence.

### Documentation drift

README documents `src/global_matching/grad_loc.py` (doesn't exist), `data/test/bbbc038` (doesn't exist,
so the documented eval command fails), "trained on BBBC038" (training now auto-discovers BioFors and
Polimi too), and "~1.2M parameters" (actual: **~8.8M**, 95% of them in the single `fc1` 8192→1024 layer).

---

## Target architecture

```
SciForensics/
├── pyproject.toml                  # packaging, deps, ruff/mypy/pytest config
├── src/sciforensics/
│   ├── config.py                   # pydantic-settings models, YAML-loadable
│   ├── types.py                    # Evidence, Verdict, ScanResult dataclasses
│   ├── io/
│   │   ├── images.py               # letterbox loading, SHA-256 hashing
│   │   ├── pdf.py                  # PyMuPDF figure extraction (Stage C)
│   │   └── panels.py               # multi-panel figure splitting (Stage C)
│   ├── global_match/
│   │   ├── backbone.py             # GAP head, aspect-preserving 224 input
│   │   ├── losses.py               # BCEWithLogits + learnable temperature
│   │   ├── data.py                 # ManipulationDataset, correct mask exclusion
│   │   ├── train.py / evaluate.py
│   │   └── embed.py                # inference + attribution (sign-bug fixed)
│   ├── local_match/
│   │   ├── keypoints.py            # ORB (CPU) | DISK/SuperPoint (GPU) behind one Protocol
│   │   ├── matcher.py             # mutual-NN + ratio; LightGlue via kornia
│   │   ├── geometry.py             # MAGSAC++, full-affine decode, degeneracy checks
│   │   └── copymove.py             # DBSCAN over (dx,dy,s,θ) → multi-region CMFD
│   ├── retrieval/
│   │   ├── hashing.py              # pHash/wavelet prefilter
│   │   └── index.py                # FAISS build/query for corpus-scale search
│   ├── fusion/
│   │   ├── features.py             # evidence → feature vector
│   │   └── calibrate.py            # logistic/isotonic → single probability
│   ├── report/
│   │   ├── templates/*.html.j2     # Jinja2
│   │   └── render.py               # HTML + WeasyPrint PDF, with audit trail
│   ├── api/                        # FastAPI app, routers, job queue
│   └── cli.py                      # Typer: compare | cmfd | scan | index | eval | report | serve
├── web/                            # Next.js 15 + Tailwind + shadcn/ui
├── benchmarks/                     # eval protocol, results tables, plots
├── tests/                          # pytest: unit + golden-file regression
├── configs/default.yaml
├── docker/ + docker-compose.yml
└── docs/                           # mkdocs-material, MODEL_CARD, DATA_CARD, tech report
```

Reuse, don't rewrite: `dog_regions`, `sobel_f`, `enhance_for_orb`, `_pts_to_bbox` and the
`draw_*_visualization` helpers in `local_detector.py` are sound and port over largely as-is.
The `ScannerConfig` / `GeometricVerificationResult` / `CopyMoveResult` dataclass shapes are a good
starting point for `types.py`.

---

## Plan

### Stage A — Correct, packaged, and visibly demoable

**A0. Repackage** — `pyproject.toml` (hatchling), move code to `src/sciforensics/`, delete all
`sys.path` mutation, Typer CLI, pydantic-settings config from `configs/default.yaml` so thresholds
live in exactly one place. `.gitattributes` (`* text=auto`, `*.py text eol=lf`) + normalize the CRLF
files to kill the phantom diff. `ruff` + `mypy` + `pre-commit`, `LICENSE` (MIT), structured logging,
global seeding. Weights move to a GitHub Release asset fetched by `sciforensics.weights.ensure()`
with SHA-256 verification. *Flagged decision:* purging the 35MB blob from existing history needs
`git filter-repo` (rewrites both commits) — I'll leave history intact unless you say otherwise.

**A1. Tests + CI** — `pytest` suite that runs on CPU in <60s using tiny synthetic fixtures:
per-bug regression tests (a known flip *must* report `flip=True`; a 12-vs-2000 keypoint pair *must not*
verify; the summary panel *must not* clip), property tests for geometry decode round-trips
(generate a known affine → recover rotation/scale/flip within tolerance), and golden-file tests on
report JSON. GitHub Actions: lint + mypy + pytest on 3.10/3.11/3.12, plus a Docker build job.

**A2. Fix bugs 1–13.** The substantive ones:
- Keypoint filter: exclude by resolved dataset layout, not substring matching.
- Geometry: `cv2.USAC_MAGSAC` for the homography; `estimateAffine2D` (full 6-DOF) for decomposition;
  decode rotation/scale/shear/flip by SVD of the linear part (`det < 0` ⇒ flip, now reachable).
  Add explicit degeneracy rejection: distinct-correspondence count, inlier spatial spread vs image
  diagonal, reprojection RMS, and a condition-number check on the homography.
- Matching: mutual nearest neighbour + ratio test + a minimum-keypoints-per-side floor, so one side
  can never collapse to a dozen points while the other keeps thousands.
- Attribution: fix the gradient-sign bug (use `|grad|`, computed symmetrically per branch), draw
  from a 16×16 feature map rather than 8×8, letterbox-correct the overlay, and *additionally* render
  the honest evidence overlay — the convex hull of verified inliers warped between images — which is
  what a reader should actually trust.
- Single source of truth for the enhancement scale, threaded through instead of `/4.0` literals.
- Load the model once; reuse the scanner across a batch.

**A3. Real reports** — replace `cv2.putText` with Jinja2 HTML → WeasyPrint PDF: no clipping, real
typography, an evidence table, thumbnails, and an audit block (input SHA-256s, tool version, config
hash, timestamp, git commit). Keep a compact PNG contact-sheet mode for README/social embedding.

**A4. Web demo (FastAPI + Next.js)** —
- API: `POST /v1/compare`, `POST /v1/cmfd`, `GET /v1/jobs/{id}` + WebSocket progress, `GET /v1/examples`.
  Returns structured evidence (not baked-in pixels) so the frontend can re-render.
- Frontend: dual-pane viewer with **synced pan/zoom**, SVG match lines drawn between panes,
  toggleable overlays (attribution heatmap / keypoints / verified region / clone mask), a calibrated
  confidence gauge with per-evidence contribution bars, **live threshold sliders that re-score from
  cached evidence without re-running the pipeline**, a curated example gallery (incl. the negative
  controls), and a "Download PDF report" button.
- Deploy: `docker-compose` for local; backend → HF Spaces (Docker) or Fly.io, frontend → Vercel.
  Record a GIF for the README.

*Stage A exit criteria:* `pip install -e .` → `pytest` green → `sciforensics compare a.png b.png`
produces a correct PDF → public demo URL works → README shows a GIF and a real before/after bug table.

### Stage B — Defensible science

**B1. Data hygiene** — Fix the loader (bug 1). Establish frozen splits with a manifest
(`benchmarks/splits/*.json`, content-hash based, no leakage): **train** on BBBC038 images + Polimi
`automatically_tampered`; **hold out BioFors entirely** as the untouched benchmark; hold out Polimi
`tampered_blots` (GIMP / DALL·E / cleanup) for localization eval. Write `DATA_CARD.md` documenting
provenance, licences and the exact 670-vs-29,461 mask discovery.

**B2. Retrain the embedding model** — Keep the architecture lineage (it's *your* model, and the
"I fixed and retrained it" story is worth more than a swap-in), but: replace the 8.4M-param `fc1`
with global average pooling (~8.8M → ~0.4M params, and it accepts any input size), L2-normalize
embeddings, replace the numerically-unstable `torch.log(σ(·))` with `BCEWithLogits` on a learnable
temperature/bias, add hard-negative mining, aspect-preserving letterbox to 224, AMP + cosine schedule
on the RTX 4060. Extend the augmentation set (already good — `JPEGCompression`, `GaussianNoise`,
`ColorShift` exist in `manipulations.py`) with crop-and-reinsert, local splice, gamma, rescale,
and print-scan simulation. Report against a **DINOv2 (timm) zero-shot baseline** so the comparison is honest.

**B3. Learned local matching** — Add DISK/SuperPoint + LightGlue via `kornia` behind the same
`KeypointDetector`/`Matcher` Protocol, ORB staying as the CPU fallback. This is the fix for the
0/3 signal-degradation failures. Benchmark ORB vs LightGlue vs LoFTR on the degradation sweep and
publish the table — the delta *is* the result.

**B4. Multi-region CMFD** — Mutual-NN self-matches → DBSCAN over `(dx, dy, log s, θ)` → per-cluster
MAGSAC → N regions with masks. Drop the `qx < tx` hack (it silently loses vertical clones).
Evaluate per-pixel against Polimi masks (IoU / F1).

**B5. Evidence fusion + calibration** — Features: embedding distance, mutual-NN match count, inlier
count/ratio, reprojection RMS, matched-area fraction, geometric-consistency flags, hash distance.
Fit logistic regression (+ isotonic) on a held-out calibration split → a single probability plus
per-evidence contributions. Report ROC-AUC, PR-AUC, Brier, ECE, and a reliability diagram. This
replaces the four hard-coded verdict strings with something a reader can act on.

**B6. Benchmark harness** — `sciforensics eval` producing versioned, reproducible artifacts:
ROC/PR curves, per-manipulation recall at fixed FPR, **precision measured against real negative
controls** (all cross-base pairs + unrelated imagery — the thing the current demo cannot do),
localization IoU, latency percentiles, and an ablation grid (ORB vs LightGlue, with/without DoG ROI,
with/without geometry gate). Results land in `benchmarks/RESULTS.md`, are surfaced on a `/benchmarks`
page in the web demo, and are regression-guarded in CI on a small subset.

*Stage B exit criteria:* one command reproduces every number in `RESULTS.md`; the README leads with a
real metrics table on held-out BioFors + Polimi; `MODEL_CARD.md` states known failure modes.

### Stage C — Deployable integrity tool

**C1. PDF → figures → panels** — PyMuPDF extraction plus gutter-projection panel splitting. This is
the actual workflow of Proofig/ImageTwin and turns the project from "compare two PNGs" into
"audit this manuscript". BioFors is panel-organized, so it doubles as the eval set here.

**C2. Corpus-scale retrieval** — pHash/wavelet prefilter → FAISS over embeddings.
`sciforensics index build <corpus>` / `search <figure>`. Benchmark recall@k and latency at 100k panels.

**C3. Production hardening** — Job queue with progress + cancellation, request size/rate limits,
input sanitization (decompression-bomb guards), Prometheus metrics, `/healthz`, structured JSON logs
with request IDs, ONNX/TorchScript export, a pinned Docker image, and an audit-trail-complete
signed PDF report.

**C4. Docs + write-up** — mkdocs-material site, quickstart, API reference, `CONTRIBUTING.md`,
`CHANGELOG.md`, two example notebooks, and a short arXiv-style tech report (`docs/report/`) covering
method, benchmark protocol and results.

---

## Sequencing

```
A0 ─┬─ A1 ─── A2 ─┬─ A3 ─── A4 ────────────────► public demo
    └─ B1 ─┬─ B2 ─┤
           └─ B3 ─┴─ B4 ─── B5 ─── B6 ──────────► benchmark + report
                                    └─ C1 ─ C2 ─ C3 ─ C4 ─► product
```

A0/A1 gate everything (nothing is safely changeable without tests). B1 can start in parallel with A1
since it touches only the data layer. B2 needs B1 (retraining on masks is pointless). B5 needs B3/B4
(fusion features come from the upgraded detectors). C depends on a stable Stage B pipeline.

Rough effort, assuming focused sessions: **A ≈ 40%, B ≈ 35%, C ≈ 25%** of total. Stage A alone gets
you a correct, tested, packaged tool with a live demo — that's the highest-leverage slice if you want
to stop early.

---

## Verification

**Per-phase, automated:**
```bash
pip install -e ".[dev]"
pytest -q                      # unit + regression + golden-file
ruff check . && mypy src       # lint + types
```

**Bug-by-bug, with the assets already in the repo** — each of these is currently wrong and must flip:
```bash
sciforensics compare inputs/mountains.jpg inputs/mountains_manipulated.jpg --report out/m.pdf
#   expect: flip=True, scale≈1.0, and either a sane homography or an explicit
#   "degenerate correspondences — rejected" instead of today's 121 phantom inliers
sciforensics compare inputs/base_cell.png inputs/base_cell_ex1_affine.png
#   expect: rotation≈45°, scale≈1.2, flip=True  (today: 15.4°, 0.048, flip=no)
sciforensics compare inputs/base_cell.png inputs/base_cells_2.png
#   negative control — expect low confidence, geometry unverified
sciforensics cmfd inputs/base_cells_1_ex3_copymove.png
#   expect the known 25%-side patch localized, and ≥1 region with a mask
python -c "from sciforensics.global_match.data import ManipulationDataset as D; \
           print(len(D('data/train/bbbc038')))"   # expect 670, not 30131
```

**Recall regression** — the three signal-degradation pairs (`*_ex2_degraded.jpg`) currently yield
2/6/7 matches and 0/3 detections. After B3 they must all be detected; this becomes a CI-guarded
assertion.

**Benchmark** — `sciforensics eval --config configs/bench.yaml` regenerates every figure and number in
`benchmarks/RESULTS.md` from the frozen splits; CI runs a subset and fails on metric regression.

**Web demo** — `docker compose up`, then drive the running app with the browser tools: upload an
example pair, confirm synced pan/zoom, toggle each overlay, move the threshold slider and confirm the
verdict re-scores without a backend round-trip, download the PDF and confirm the audit block is populated.
Playwright smoke test in CI.

---

## Explicitly out of scope

- **Training a pixel-level splicing-localization network** (TruFor/CAT-Net class). You chose
  "harden + learned matching". Polimi masks make this a natural follow-up — I'll leave the
  `report/` and `fusion/` interfaces mask-shaped so it can drop in later without rework.
- **Rewriting git history** to purge the 35MB weights blob (needs `git filter-repo`; say the word).
- **Any claim of detecting AI-*generated* imagery.** Polimi ships DALL·E-tampered blots, so it'll
  appear in the benchmark as a robustness column, but generative-detection is a different problem and
  I won't market it as solved.
