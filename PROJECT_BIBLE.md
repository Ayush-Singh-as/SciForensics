# SciForensics — Project Bible

> Single source of truth for **what this project is, what every file does, what changed, and why.**
> Read this first in any new session, before touching code.
>
> Last updated: 2026-09-07 · Branch `main` (parent repo `c:\ayush` on `master`) · HEAD `bcb29bc`

---

## 1. What this project is

**SciForensics detects image plagiarism and manipulation in scientific publications.**

Scientific fraud frequently reuses figures — the same western blot, cell micrograph, or
microscopy panel republished after cropping, rotation, flipping, rescaling, contrast
adjustment, or partial region duplication. Generic image-similarity tools miss these because
they are built for *retrieval*, not *forensics*: they answer "does this look similar?" when the
question is "is this the same source data, transformed to hide it?"

The system answers the forensic question with a **three-stage evidence pipeline**:

| Stage | Question | Method |
|---|---|---|
| **Global** | Are these plausibly the same content? | Siamese CNN → 128-d embedding → L1 distance |
| **Local** | Which *regions* correspond? | Enhance → DoG ROI → ORB keypoints → mutual-NN + ratio match |
| **Geometry** | Is the correspondence *physically consistent*? | MAGSAC++ homography + full 6-DOF affine decompose → rotation/scale/shear/**flip** |
| **Fusion** | How confident, and why? | Evidence features → logistic score → banded verdict + per-evidence contributions |

Plus **CMFD** (copy-move forgery detection): the same machinery run image-against-itself, with
DBSCAN clustering over offset space to find *multiple* independent cloned regions.

**The design commitment that matters:** the pipeline is allowed to say *"I don't know."*
Geometry can return `verified`, `refuted`, **or abstain with an explicit `RejectionReason`**.
A prototype that manufactures confident nonsense is worse than useless in a research-integrity
context, where a false accusation ends a career. Every stage can decline.

### Non-goals (deliberate, do not drift)

- **Not** an AI-generated-image detector. Polimi ships DALL·E-tampered blots, so they appear as a
  *robustness column* in the benchmark — generative detection is a different problem, not claimed solved.
- **Not** a pixel-level splicing-localization network (TruFor/CAT-Net class). Chosen path was
  "harden + learned matching". The `fusion/` and report interfaces are kept **mask-shaped** so such a
  model can drop in later without rework.
- Git history is **not** being rewritten to purge the 35 MB weights blob (needs `git filter-repo`).

---

## 2. Where the project stands right now

The repo was a **prototype**: 3 commits, ~2,800 LOC, working *idea*, but no tests, no packaging,
no metrics, and **13 load-bearing correctness bugs producing silently wrong answers**.

`implementation_plan.md` is the rebuild plan: Stage **A** (correct/packaged/demoable),
Stage **B** (defensible science), Stage **C** (deployable tool).

### Progress ledger

| Phase | Scope | State |
|---|---|---|
| **A0** Repackage | pyproject, `src/sciforensics/`, config, logging, seeding, weights fetch, LICENSE, `.gitattributes`, **CLI** | ✅ **done** |
| **A1** Tests + CI | **424 tests passing, 0 skipped**; ruff + mypy + pre-commit all green | ✅ **done, verified** |
| **A2** Fix bugs 1–13 | 13 of 14 fixed and **empirically verified against real images** | ✅ — only **bug 1** remains (needs B1's data loader) |
| **A3** Real reports | Jinja2 HTML → PDF (Chromium *or* WeasyPrint), raster overlays | ✅ **done** — **bug 7 closed and PDF verified**: 2-page PDF, extractable text, full disclaimer intact |
| **A4** Web demo | FastAPI + Next.js 16, docker-compose | ✅ **done** — API + UI build and run; end-to-end analysis verified via CLI |
| **B1** Data hygiene | Layout-aware discovery, frozen content-addressed splits, `DATA_CARD.md` | ✅ **done** — **bug 1 closed; all 14 bugs now fixed** |
| **B3** Learned matching | DISK + LightGlue via kornia, behind the existing Protocols | ✅ **implemented** — but **measured worse than ORB**; see §8c |
| **B6** Benchmark harness | `sciforensics bench`: recall, **FPR**, precision, correspondence precision, latency | ✅ **done** — first precision figure the project has ever had |
| **B2** Retrain | GAP head, BCEWithLogits, hard negatives | ⏳ **blocked**: no datasets downloaded (`data/` is gitignored and empty) |
| **B4** CMFD eval | Per-pixel IoU/F1 vs Polimi masks | ⏳ blocked on Polimi download; **defaults need retuning** (see §8b) |
| **B5** Calibration | Fit logistic/isotonic on the `calib` split | ⏳ blocked on B2/B4 |
| **C1–C4** Product | PDF→panels, FAISS retrieval, hardening, docs | ❌ **not started** |

**~14,500 lines** landed in commit `bcb29bc` ("Fixed some bugs") — the entire `src/sciforensics/`
package, the test suite, CI, and config.

### What landed on 2026-09-07 (uncommitted)

**A0's blocker.** `pyproject.toml` declared `sciforensics = "sciforensics.cli:app"` while
`cli.py` **did not exist** — the wheel-install CI job could not resolve the entry point and every
command in the plan's verification section was unrunnable. Written: `cli.py`, five commands
(`compare`, `cmfd`, `config`, `serve`, `version`).

**A3 — reports.** `report/` package: `overlays.py` (rasters), `render.py` (HTML/PDF/JSON),
and three Jinja/CSS templates. **Bug 7 is closed** — prose is flowed HTML with no fixed height
in the text path, so it cannot clip at any string length.

**A4 — web demo.** `api/app.py` (FastAPI) and `web/` (Next.js 16 + hand-written design system),
plus `docker/` and `docker-compose.yml`.

### Next up

**Stage B — defensible science**, starting with **B1 data hygiene** (fixes bug 1, the last
outstanding one: exclude BBBC038 masks by resolved layout, freeze content-hashed splits, hold
BioFors out entirely, write `DATA_CARD.md`). B1 gates B2's retrain. The highest-value item
overall is **B6's negative controls** — precision is still unmeasured, so no false-positive rate
exists yet.

### Environment (now fully installed)

Python **3.12.10**, package installed editable with `[dev,report,api]`. torch **2.14.0+cpu**,
opencv **5.0.0**, ruff, mypy, playwright+Chromium. `models/weights.pth` verifies against
`weights.EXPECTED_SHA256` (`befec5b8…`), so real analysis runs.

```bash
pytest -q                      # 424 passed, 0 skipped, ~26s
ruff check . && ruff format --check .
mypy                           # 43 files, clean
pre-commit run --all-files     # every hook passes
```

**PDF needs a backend.** WeasyPrint is installed but its native GTK libraries are not (winget and
choco both failed on permissions here), so **Chromium via playwright is the working backend** and
is now the *first* one tried. `python -m playwright install chromium` is the only extra step.

**Weights are not auto-resolved yet.** `RELEASE_URL` is `None` until the artifact is published, so
pass `--set global_match.weights=models/weights.pth` on the CLI for now.

---

## 3. The 14 bugs — the core of the work

Bugs 1–13 are from the plan's audit; **bug 14 was discovered during implementation.**
Marked ✅ where fixed in the shipped package, ⏳ where the owning module doesn't exist yet.
**All 14 are now fixed.**

| # | Bug | Why it silently lied | Fix | State |
|---|---|---|---|---|
| 1 | **82% of training data is binary masks.** Loader filtered `"mask" not in parts`, but BBBC038's dir is `masks` *(plural)*. 29,461 of 30,131 files were instance masks; real images: **670**. | Model largely trained on segmentation masks | `global_match/data.py`: each dataset declares a `Layout`; components matched **exactly**, never as substrings. Verified: 5,400-file synthetic tree (97.8% masks) → manifest of exactly **120 images** | ✅ |
| 2 | **Flip detection was mathematically unreachable.** `estimateAffinePartial2D` returns a *similarity* matrix `[[a,-b,tx],[b,a,ty]]`; `det = a²+b² > 0` **always**, yet the code tested `det < 0`. | "Flip detected: no" for *every* mirrored image, forever | `estimateAffine2D` (full 6-DOF), decompose by SVD/QR, `det < 0` now genuinely reachable | ✅ |
| 3 | **Mirrored content collapsed the geometry fit.** A similarity transform cannot represent a reflection, so estimation degenerated: true 45°+1.2×+hflip reported `rot 15.4°, scale 0.048`; mountains reported `scale 0.000`. | Garbage transform parameters | Same as bug 2 | ✅ |
| 4 | **Degenerate many-to-one matching → phantom verifications.** No cross-check. ROI filter's hard `<8` fallback allowed 12 keypoints vs 2,000 → **231 matches, 121 "inliers"** all landing on ≤12 distinct points. Verdict: *"Strong evidence."* | Confident false positives | Mutual-NN (**injective**) + ratio + per-side keypoint floor + degeneracy rejection (distinct count, spatial spread, reproj RMS, condition number) | ✅ |
| 5 | **Right-hand Grad-CAM was mathematically dead.** `d = Σ|e_L − e_R|` ⇒ `∂d/∂e_R = −∂d/∂e_L`. Signed gradients pooled then `ReLU`'d ⇒ the negative branch annihilated. Every "Heatmap B" was uniform blue. | Half the visual evidence was blank | Pool `abs` of the map, symmetric per branch; agent **measured** the asymmetry ratio (`relu`: 3–256×; `abs`/`signed`: exactly **1.0**) and kept `relu` selectable to reproduce the defect | ✅ |
| 6 | **CMFD could only ever report one region.** One global `estimateAffinePartial2D` over all self-matches; the `regions` list was a lie. The `qx < tx` ordering hack also dropped purely-vertical clones. | Multi-clone forgeries under-reported | DBSCAN over `(dx, dy, log s, θ)` → per-cluster MAGSAC → N regions with masks; `qx<tx` hack dropped for canonical orientation | ✅ |
| 7 | **Verdict text clipped off-canvas.** Fixed 420 px panel; last wrapped line at baseline y=415. "scientific imagery" cut in half in *every* report. | Conclusion literally unreadable | Flowed HTML/PDF, **no fixed height in the text path**; `test_report.py` asserts the full prose survives rather than measuring a pixel | ✅ |
| 8 | **Aspect ratio destroyed.** `resize(gray,(128,128))` non-aspect-preserving, then 8×8 heatmap stretched back onto original aspect. | Overlay geometrically inconsistent with image | Letterbox (aspect-preserving) + letterbox-correct overlay inversion | ✅ |
| 9 | **Similarity capped at 0.731.** `σ(1−d)` maxes at `σ(1)=0.7311`. *Identical* images scored 0.72; top 27% of range unreachable. | Scores meaningless | `σ((threshold−d)/temperature)` → 0.5 exactly on the boundary, →1 as d→0 | ✅ |
| 10 | **Threshold drift.** Trained boundary `d=1.0`; `same_threshold` default `1.1`; argparse `--local-threshold` `3.0`; `ScannerConfig` `2.0`; README `2.0`. | Four different answers for one number | Every threshold lives **only** in `configs/default.yaml` | ✅ |
| 11 | Model reloaded **per image** in batch mode (35 MB `torch.load` × N). | Gratuitously slow | `Pipeline` constructed once, expensive parts cached | ✅ |
| 12 | `torch.load` without `weights_only=True`. | Arbitrary-code-execution surface; breaks on torch ≥2.6 | `weights_only=True` | ✅ |
| 13 | **4× enhance scale hardcoded as bare `/4.0`** in six places, default defined elsewhere. | Changing scale silently corrupted every coordinate | Single config value threaded through; `rescale_box`/`rescale_affine`/`rescale_homography` | ✅ |
| 14 | **No resolution cap** *(found during implementation)*. Nothing bounded input before the 4× upscale: 3840×2400 → 15360×9600 (**147 MP**), >2 min per pair. | Pathological latency; DoS surface | `load_image` caps longest edge, records it in `ImageMeta`, maps all coords back to original frame | ✅ |

### Also fixed in A0

`sys.path` mutation in 3 places → real package · CRLF in `model.py`/`test.py` causing a permanent
phantom 168-line diff → `.gitattributes` normalization · no seeding → `seed_everything` ·
no logging → structured console/JSON · 35 MB weights in git → `weights.ensure()` fetches a Release
asset with SHA-256 verification · README drift (documented a nonexistent `grad_loc.py`, a
nonexistent `data/test/bbbc038`, and "~1.2M parameters" when the real figure is **~8.8M**, 95% of it
in one `fc1` 8192→1024 layer — hence the GAP-head rewrite in B2).

---

## 4. Every file, and what it does

### Root

| File | Purpose |
|---|---|
| `implementation_plan.md` | **The plan.** Audit + Stage A/B/C roadmap + verification commands. Untracked. |
| `PROJECT_BIBLE.md` | This file. |
| `pyproject.toml` | Hatchling build; deps + extras (`match`/`report`/`api`/`train`/`bench`/`dev`/`all`); ruff/mypy/pytest config. Declares the `sciforensics` script. Force-includes `configs/default.yaml` at `sciforensics/data/default.yaml` so a *wheel* can load config. **Ruff excludes the legacy prototype** so the audit's line references keep pointing at the code they describe. |
| `configs/default.yaml` | **The only place thresholds live** (bug 10's fix). 346 lines, heavily commented. |
| `.github/workflows/ci.yml` | 3 jobs: `gate` (pre-commit: lint+types+tests on 3.12) · `test` (matrix 3.10/3.11/3.12 + coverage) · `package` (build sdist/wheel, `twine check --strict`, install wheel into a clean env, **assert the packaged config resolves from site-packages**, re-run the suite against the artifact). |
| `.pre-commit-config.yaml` · `.gitattributes` · `.gitignore` · `LICENSE` | Commit gate · CRLF fix (`* text=auto`, `*.py text eol=lf`) · ignores · MIT. |
| `forensic_scanner.py`, `run_demo.py`, `generate_examples.py` | **Legacy prototype orchestrator/demos.** Superseded by `pipeline.py`. Kept (and ruff-excluded) until A0 closes so audit line refs stay valid. Delete together with their ruff exclusions. |
| `src/global_matching/`, `src/local_matching/` | **Legacy prototype.** `dataset.py` (bug 1), `model.py`, `pairwise_inference.py` (bugs 5, 8), `local_detector.py` (bugs 4, 6, 13), `train.py`, `test.py`, `manipulations.py`. Same deletion note. `manipulations.py`'s augmentations are **good** and get extended in B2. |
| `inputs/` | Test assets: `base_cell.png`, `base_cells_1.png`, `base_cells_2.png`, `mountains.jpg`, `mountains_manipulated.jpg`. |
| `models/weights.pth` | 35 MB checkpoint, committed into history. Being migrated to a Release asset. |
| `docs/assets/legacy/mountains_before.*` | **Preserved evidence** of the pre-fix wrong output — the "before" half of the README's before/after bug table. |
| `web/` | **A4 frontend.** Next.js 16 (App Router) + React 19, TypeScript strict, **zero UI dependencies**. `app/globals.css` is a hand-written design system (see §5.8); `lib/api.ts` types the API and holds `bandFor` (client-side re-banding); `components/Evidence.tsx` renders verdict/funnel/geometry/contributions; `components/Dropzone.tsx` mirrors the server's upload guards for fast feedback. `npm audit`: **0 vulnerabilities**. |
| `docker/`, `docker-compose.yml` | `api.Dockerfile` (CPU-only torch — the CUDA wheels are ~2.5 GB and a demo has no GPU; non-root; healthcheck on `/healthz`, which deliberately never touches the model so a slow first inference cannot mark it unhealthy) and `web.Dockerfile` (multi-stage). Weights are **not baked in** — fetched with SHA-256 verification into a cache volume. `docker compose config` validates. |

### `src/sciforensics/` — the real package

Nothing heavy is imported at package scope, so `import sciforensics` and `--help` stay fast.

| Module | Lines | Role |
|---|---|---|
| `__init__.py` | 21 | Version via `importlib.metadata`; deliberately no heavy imports. |
| `cli.py` | ~400 | **The console script** (`[project.scripts]` target). Commands: `compare`, `cmfd`, `config`, `version`. A deliberately thin adapter — parses args, loads config, builds **one** `Pipeline`, renders. Holds **no thresholds** and makes no forensic decisions. Imports `torch` lazily inside `_build_pipeline`, so `--help`/`version`/`config` work in an environment with no model stack (and degrade with an actionable message, not a traceback). Renders the bug-2/3 numbers (rotation/scale/**flip**), the bug-4 funnel (`raw → ratio → good` + distinct counts + a `not injective` flag), and the named `RejectionReason` rather than a bare "not verified". Prints an explicit *"monotone score, not a calibrated probability"* line while `calibrated=False`. **Exit codes: 0 ok, 1 handled error, 2 arg error** — the *verdict never affects exit status*, so CI can't be tempted to suppress findings. |
| `config.py` | 611 | Pydantic models for every knob + `load_config()`. Layering: `configs/default.yaml` → user file → `SCIFORENSICS_*` env → `dotted.key=value` overrides. Rejects unknown keys. Defines `ReportConfig`/`ApiConfig` **ahead of** the modules that will use them. |
| `types.py` | 435 | Serialisable evidence vocabulary: `Verdict`, `Stage`, **`RejectionReason`**, `ImageMeta`, `GlobalEvidence`, `KeypointEvidence`, `MatchEvidence`, `AffineDecomposition`, `GeometryEvidence`, `CopyMoveRegion/Evidence`, `EvidenceContribution`, `AuditBlock`, `ScanResult`, `CopyMoveResult`. Pydantic ⇒ free JSON for CLI/API. |
| `pipeline.py` | 542 | **Orchestrator.** `Pipeline.compare()` / `.copy_move()` → `PairAnalysis` / `CopyMoveAnalysis`. Holds expensive parts open (bug 11). Splits *serialisable* `result` from numpy/torch *working data* so a result serialises without deciding where overlays live. **Not thread-safe** (attribution's backward pass mutates `.grad`) — one instance per worker. |
| `runtime.py` | 206 | `resolve_device`, `seed_everything`, `setup_logging` (idempotent, console/JSON, attaches to the package logger only — never hijacks root). |
| `audit.py` | 164 | Provenance for reports: SHA-256 of inputs, tool version, git commit + dirty flag, UTC timestamp, dependency versions. |
| `weights.py` | 217 | `ensure()` — resolve/download checkpoint, **SHA-256 verify**, cache. Gets the 35 MB blob out of git. |
| `io/images.py` | 462 | `letterbox` (bug 8) / `squash` / `fit_for_embedding`; `load_image` with the **resolution cap** (bug 14). `Letterbox` inverts its own geometry so callers never re-derive it. |
| `global_match/backbone.py` | 545 | `EmbeddingNet` + `ConvBlock`; `FlattenHead` (legacy 8.8M) vs **`PoolHead`** (GAP, ~0.4M, any input size). `load_backbone` auto-detects head, remaps legacy keys, `weights_only=True` (bug 12). `similarity()` (bug 9), `parameter_count()` (README drift). |
| `global_match/embed.py` | 735 | `Embedder`, `preprocess`, **`grad_cam`** (bug 5 — `abs`/`signed`/`relu` modes, the last kept to reproduce the defect), `Attribution`, colourise/blend, `PairComparison`. |
| `local_match/keypoints.py` | 367 | `sobel_magnitude`, `enhance` (config-driven scale, bug 13), `dog_regions` (Otsu-thresholded), `_mask_from_boxes`, `OrbDetector`, `KeypointDetector` **Protocol** (the seam for DISK/SuperPoint in B3), `build_detector`. |
| `local_match/matcher.py` | 379 | `BruteForceMatcher` = ratio test **+ mutual-NN** ⇒ injective (bug 4). `crossCheck=False` deliberately (incompatible with `knnMatch`); mutual agreement enforced manually. `self_match` + `_g2nn_accept` for CMFD. `Matcher` Protocol (LightGlue seam). |
| `local_match/geometry.py` | 719 | **Bugs 2/3/4.** `estimate_affine` (full 6-DOF), `decompose_affine` (SVD/QR → rotation, scale, shear, **flip = det<0**), `compose_affine` (round-trip), MAGSAC++ homography, and the degeneracy battery: `distinct_count`, `_spread`, `_is_collinear`, `reprojection_errors`, `_condition_number`, `convex_hull`/`polygon_area`. Returns `Verification` — verified / refuted / **abstained with a reason**. Also the `rescale_*` family (bug 13). |
| `local_match/copymove.py` | 587 | **Bug 6.** `canonical_orientation`/`orient_correspondences` (replaces the `qx<tx` hack), `offset_features` → `cluster_offsets` (DBSCAN) → per-cluster MAGSAC → **N** `CopyMoveRegion`s with masks. |
| `fusion/features.py` | 591 | Evidence → named features via a declarative `Feature` table: embedding similarity, geometry verified/refuted/**abstained**, inlier strength/ratio, reproj tightness, matched area, transform-manipulated, **keypoint asymmetry**, **non-injective** (bug 4 as a *feature*), clone strength/area/tightness, cluster survival. |
| `fusion/rules.py` | 388 | Hand-set logistic weights → probability → `band()` → `Verdict`, **plus per-evidence `contributions`**. The honest interim: replaces the four hard-coded verdict strings; `ScanResult.calibrated=False` marks it as *not yet* calibrated. |
| `fusion/calibrate.py` | 332 | `LogisticCalibrator` + `Isotonic`, load/apply, schema-checked. **Scaffolded, not fitted** — awaits B5's calibration split. |
| `report/overlays.py` | ~330 | **A3.** The rasters a reader looks at, ordered by trustworthiness: `thumbnail` → `attribution_overlay` (dropped entirely when the map is degenerate — a missing overlay is more honest than a flat blue rectangle) → `keypoint_overlay` → `match_overlay` → the verified-inlier `_draw_hull`. **Rejections are drawn too**, per the `Verification` docstring's argument: 121 lines converging on a dozen points argues `DEGENERATE_CORRESPONDENCES` better than the sentence does. Every heavy import is under `TYPE_CHECKING`, so this draws with **only cv2+numpy** — no torch, no sklearn. |
| `report/render.py` | ~330 | **A3.** `render_pair` / `render_copy_move` → HTML + PDF + JSON into a self-contained directory (relative asset hrefs, so it zips or serves as-is). Jinja runs with **`StrictUndefined`**: a typo'd field must fail loudly, since Jinja's default would render a missing *measurement* as an empty string — indistinguishable from a measured zero. **PDF is optional**: WeasyPrint's native deps (Pango/cairo) are awkward on Windows and in slim images, so its absence is a warning plus the HTML, not a failure. |
| `report/templates/` | ~430 | `report.html.j2`, `copymove.html.j2`, `base.css`. Print-first CSS (block/table flow, not grid/flex — WeasyPrint renders those inconsistently), `@page` counters, `break-inside: avoid` on findings, `orphans/widows`, and `word-break` on SHA-256 digests (unbreakable strings would overflow the page box — bug 7's failure class in a new place). Document order *is* the argument: verdict → caveats → evidence → audit, so evidence never precedes the caveat qualifying it. |
| `api/app.py` | ~300 | **A4.** `create_app(cfg)` factory (not a module singleton, so tests inject config without loading a 35 MB checkpoint). Returns **structured evidence, never baked-in pixels** — that is what lets the UI re-threshold from cache. `Pipeline` is **lock-guarded** because it is not thread-safe (attribution's backward pass mutates `.grad`, so concurrent calls would interleave gradients between pairs). Uploads are hostile until proven otherwise: extension allow-list, **chunked** size cap, and `max_decoded_pixels` checked from the *header* before allocation. The asset route confines a client-supplied filename to the job directory. |
| `cli.py` | ~530 | Now five commands — `serve` added for A4, `--report`/`--format` for A3. |

### `tests/` — 337 tests

`conftest.py` (fixtures) · `helpers.py` (synthetic image/transform builders — tiny, CPU, <60 s, no
35 MB artifact needed since `Pipeline` accepts an injected `Embedder`).

`test_fusion.py` 67 · `test_embed.py` 39 · `test_backbone.py` 32 · `test_config.py` 31 ·
`test_copymove.py` 31 · `test_images.py` 26 · `test_geometry.py` 22 · `test_matcher.py` 21 ·
`test_keypoints.py` 20 · **`test_cli.py` 16** · `test_pipeline.py` 13 · **`test_api.py` 12** ·
**`test_report.py` 9**.

The three added this session are **torch-free by design** and are the only ones verified locally.
`test_api.py` is deliberately adversarial (traversal, decompression bombs, oversize, undecodable,
path disclosure) because the API accepts arbitrary bytes from anyone who can reach it.

`test_cli.py` is torch-free by design (drives args/config/errors only, so it runs in a bare
environment). Its load-bearing case is `test_each_command_builds`: Typer constructs click
parameters by **introspecting annotations at import time**, so an annotation it cannot interpret
is not a lint nit but an `AssertionError` that takes down *every* subcommand including `--help`.
Merely importing the module would not catch it — the command has to actually be built.

Style: **per-bug regression tests** (a known flip *must* report `flip=True`; a 12-vs-2000 keypoint
pair *must not* verify) + **property tests** (generate a known affine → recover rotation/scale/flip
within tolerance).

---

## 5. Design decisions worth not re-litigating

1. **Abstention is a first-class outcome.** `RejectionReason` exists so "degenerate
   correspondences — rejected" replaces 121 phantom inliers. In research integrity, a confident
   wrong answer is the expensive failure.
2. **Protocols at the ORB/BFMatcher seams.** `KeypointDetector` and `Matcher` are Protocols so
   B3 drops in DISK/SuperPoint/LightGlue with ORB as CPU fallback — *and* so the benchmark can
   report the delta. The Protocol **is** the experiment.
3. **Config has exactly one home.** Bug 10 was one number with four values. Nothing gets a
   threshold default in Python.
4. **Serialisable result split from working data.** `PairAnalysis.result` is pure evidence;
   numpy/torch live beside it. Lets the API return structured evidence and the frontend re-render
   (and re-threshold) **without re-running the pipeline**.
5. **Keep the model's lineage.** B2 fixes and retrains the existing architecture rather than
   swapping in a pretrained backbone — "I found the bugs and retrained it" is worth more than a
   drop-in, with a DINOv2 zero-shot baseline for honesty.
6. **The legacy prototype stays until A0 closes.** Audit line references (`local_detector.py:144`,
   `pairwise_inference.py:94`, `dataset.py:72`) must keep resolving. Reformatting or deleting early
   would make the audit describe a file that no longer exists.
7. **Comments explain *why*, with measurements.** The bug-5 docstring carries a measured asymmetry
   table. This is the house style — keep it.
8. **The UI is an instrument, not a landing page.** Reference points are Bloomberg terminals, DAW
   meters and lab equipment — *not* marketing pages, and explicitly not the default AI-generated
   look. Enforced concretely in `web/app/globals.css`: no purple/pink gradients, no glassmorphism,
   no hero section, no decorative shadows, `border-radius` capped at 6px, **data is the only
   saturated colour** (chrome is greyscale so the eye lands on evidence), **tabular numerals
   everywhere a number can change** so digits don't jitter as a score updates, one accent (amber)
   reserved for "needs a human", `image-rendering: pixelated` because interpolating a forensic
   image invents detail, and `prefers-reduced-motion` honoured. Hand-written CSS with real
   component classes rather than utility soup — and **zero UI dependencies**, so there is no
   component library's house style to fight.
9. **Evidence semantics avoid red/green.** Amber/teal in the UI, amber/teal in the PDF. These
   colours carry the verdict, and red-green is the most common colour-vision deficiency.
10. **Thresholds re-band client-side, never re-analyse.** `bandFor()` in `lib/api.ts`. This is not
    only a latency win: re-running the pipeline per slider tick would let the *evidence* shift
    while the user believes they are moving only a threshold.
11. **`calibrated` is load-bearing.** Every surface that shows a score consults it — CLI, PDF, UI,
    and `/v1/config`. It stays `False` until B5 fits a calibrator, and each surface says so in
    prose rather than in a tooltip.

---

## 6. Known risks / open threads

- **The 302 inherited tests have never been run on this machine.** Highest-priority unknown.
  `pip install -e ".[dev]"` was declined; `torch`/`pydantic-settings`/`scikit-learn`/`weasyprint`
  are absent. The 37 tests added in this session *are* verified passing.
- **The PDF path is unexercised.** WeasyPrint is not installed, so only the HTML fallback has run.
- **No end-to-end analysis has executed anywhere.** Every path through `Pipeline.compare` needs
  torch, so the CLI's result rendering, the report overlays' attribution branch and the API's
  analysis routes are verified only against synthetic/structural inputs.
- **Precision is still unmeasured.** `run_demo.py` pairs each base only with *its own*
  manipulations — **zero** unrelated-pair controls, so no false-positive rate exists yet. B6 fixes
  this; do not publish a precision claim before it.
- **Recall fails on realistic degradation.** JPEG q25 + noise ⇒ ORB yields 6/7/2 good matches ⇒
  **0/3 detected**. `base_cells_1` affine also fails (4 matches). This is B3's justification.
- **BioFors must stay held out.** It is *the* benchmark for this task (ICCV'21); the prototype
  consumed it as unlabeled *training* data. Training on it forecloses evaluating on it.
- **Polimi ground-truth masks are unused** — free localization ground truth currently discarded (B1/B4).
- `fusion/rules.py` weights are hand-set, not fitted. `ScanResult.calibrated` must stay `False` until B5.
- `models/weights.pth` still in git history pending an explicit decision on `git filter-repo`.

---

## 7. Verification commands (from the plan)

All of these depend on the CLI, which **now exists** — they are runnable once the stack is
installed. Each was **wrong** in the prototype and must flip:

```bash
pip install -e ".[dev]" && pytest -q && ruff check . && mypy src

sciforensics compare inputs/mountains.jpg inputs/mountains_manipulated.jpg --report out/m.pdf
#  expect flip=True, scale~1.0, and either a sane homography or an explicit
#  "degenerate correspondences - rejected"  (today: 121 phantom inliers)

sciforensics compare inputs/base_cell.png inputs/base_cell_ex1_affine.png
#  expect rotation~45deg, scale~1.2, flip=True   (today: 15.4deg, 0.048, flip=no)

sciforensics compare inputs/base_cell.png inputs/base_cells_2.png
#  negative control - expect low confidence, geometry unverified

sciforensics cmfd inputs/base_cells_1_ex3_copymove.png
#  expect the known 25%-side patch localized, >=1 region with a mask

python -c "from sciforensics.global_match.data import ManipulationDataset as D; print(len(D('data/train/bbbc038')))"
#  expect 670, not 30131   (bug 1)
```

Runnable **now**, with no install and no torch:

```bash
export PYTHONPATH=src
python -m pytest tests/test_cli.py tests/test_api.py tests/test_report.py -q   # 37 passed
python -m sciforensics.cli version
python -m sciforensics.cli config --set geometry.max_iters=5000 --json

# the web stack (API returns a clean 500 on analysis until torch is installed)
python -m sciforensics.cli serve --port 8000
cd web && npm install && npm run dev        # http://localhost:3000
```

**Stage A exit criteria:** `pip install -e .` → `pytest` green → `sciforensics compare a.png b.png`
produces a correct PDF → public demo URL → README shows a GIF and a real before/after bug table.

Remaining to close Stage A, all blocked on the same thing — **installing the stack**
(`torch`, `pydantic-settings`, `scikit-learn`, `weasyprint`):
1. Run the full 337-test suite and `ruff`/`mypy`.
2. Run the five bug-verification commands above and confirm each number flips.
3. Exercise the **PDF** path (WeasyPrint absent locally; only the HTML fallback is tested).
4. Deploy (backend → HF Spaces/Fly.io, frontend → Vercel) and record the README GIF.

---

## 8. Sequencing

```
A0 ─┬─ A1 ─── A2 ─┬─ A3 ─── A4 ────────────────► public demo
    └─ B1 ─┬─ B2 ─┤
           └─ B3 ─┴─ B4 ─── B5 ─── B6 ──────────► benchmark + report
                                    └─ C1 ─ C2 ─ C3 ─ C4 ─► product
```

A0/A1 gate everything (nothing is safely changeable without tests). B1 can run parallel to A1
(data layer only). B2 needs B1 (retraining on masks is pointless). B5 needs B3/B4 (fusion features
come from the upgraded detectors). C needs a stable Stage B.

Effort split: **A ≈ 40%, B ≈ 35%, C ≈ 25%**. Stage A alone yields a correct, tested, packaged tool
with a live demo — the highest-leverage slice to stop at.

---

## 8b. Stage A verification — measured results (2026-09-10)

Run with the real checkpoint. **These are the numbers, not expectations.**

| Case | Prototype | Now | Reading |
|---|---|---|---|
| `mountains` vs `mountains_manipulated` | `scale 0.000`, translation 1635 px, 121 phantom inliers, "Strong evidence" | `rot -180.00°`, **`scale 0.9999`**, RMS **0.46 px**, 377/397 inliers, **verified** | Bug 3 fixed. Ground truth checked by correlation: `rot180` (+0.066) beats both single-axis mirrors, and 180° flips *both* axes so `det=+1` — **`flip=no` is correct here and the plan's `flip=True` expectation was wrong about this pair.** |
| `base_cell` vs `base_cells_2` (**negative control**) | would report matches | **1 good match**, `too_few_matches`, `clean` | **Bug 4 fixed.** The exact 90-vs-2000 asymmetry that manufactured the phantom inliers now yields nothing, and the CLI prints `Sampling asymmetry 22.2x`. |
| Reflection decode (unit) | `flip` **unreachable** (`det = a²+b² > 0` always) | `det = -1.440`, **`flip=True`**, `rot -135.00°`, `scale 1.2000` | **Bug 2 fixed and reachable.** Verified at the geometry layer because ORB descriptors are not mirror-invariant, so no real *image* pair can demonstrate it — which is B3's remit. |
| `base_cells_2_ex3_copymove` (CMFD) | could only ever report **one** region | **1 region**: source `(60,44) 131×134`, clone `(380,299)`, `rot +0.1°`, `scale 1.002`, 42 inliers; funnel **6 clusters → 1 verified** | **Bug 6 fixed.** Generator copied a 128 px patch (64,51)→(384,307) — recovered within a few px. |
| Report PDF | verdict clipped mid-word on a 420 px canvas | **2-page PDF**, 7 images, extractable text, ends `"…absence of a finding is not proof of integrity."` | **Bug 7 closed.** The full trailing prose survives. |

**CMFD needs tuning per image, and this is a real finding.** The shipped `copy_move.nn_ratio: 0.8`
and `cluster.min_samples: 6` miss small clones: an *exact* copy has near-identical descriptors, so
Lowe's ratio `d1/d2 → 1` and the strict test rejects the very matches wanted (0.8 → 32 self-matches,
0.9 → 258). Detection needed `nn_ratio=0.9` **and** `min_samples=4`. It succeeded only on the
largest patch (128 px on a 512×640 panel); the 40 px clone on the 519×162 strip produced 6 candidate
clusters, **0 verified** — reported honestly as such. Retuning these defaults belongs to **B4**.

**Reproducibility bug found and fixed:** `generate_examples.py` added `np.random.normal` noise
**unseeded**, so every regeneration produced a different asset and any test asserting a similarity
against it broke on rebuild — which is exactly what happened to
`test_a_real_manipulation_ranks_below_a_negative_control`. Now seeded (`SEED = 1234`) and verified
byte-identical across runs. The test's hard-coded values were rebased and annotated as
seed-dependent; the *finding* they guard is seed-independent and still holds.

**The inversion that motivates Stage B**, now measured on the seeded asset:

| Pair | Embedding similarity |
|---|---|
| `base_cell` vs its own JPEG-q25+noise copy (**a true positive**) | **0.1877** |
| `base_cell` vs `base_cells_2` (**unrelated — a negative control**) | **0.3781** |

A real manipulation embeds **further away** than an unrelated image. No fusion weight fixes this:
any embedding weight large enough to promote the true positive also promotes the control. It is a
fact about the checkpoint, and it is precisely why **B1/B2 (retrain) and B3 (learned matching)**
exist.

---

## 8c. Stage B measured results (2026-09-10)

### B6 — the project's first precision figure

`sciforensics bench` over 19 cases (16 positives, 3 genuine negative controls):

| Backend | Recall | FPR | Precision | Latency p50 |
|---|---|---|---|---|
| `orb+mutual_nn` | **68.8%** | **0.0%** | **100.0%** | **0.30 s** |
| `disk+lightglue` | 62.5% | 0.0% | 100.0% | 8.85 s |

**Zero false positives on genuine unrelated pairs** — bug 4's fix holding under
measurement rather than assertion. `run_demo.py` could not produce any of these numbers: it
paired each base only with its own manipulations, so there was not one unrelated pair.

**Caveat, stated in the generated report too: 3 controls is not a rate.** A 0.0% FPR over three
pairs is consistent with a true rate of several percent. It bounds the obvious failure modes and
nothing more; a publishable figure needs a real corpus (BioFors, held out).

### B3 — implemented, and measured *worse* than ORB

The plan predicted learned matching would fix the 0/3 signal-degradation failures. **It does not.**

| Manipulation | `orb` | `disk` |
|---|---|---|
| `ex1_affine` (45°+1.2×+mirror) | 0/3 | 0/3 |
| `ex2_degraded` (JPEG q25+noise) | 1/3 | 1/3 |
| `ex3_copymove` | 3/3 | 3/3 |
| `ex4_exposure` | 3/3 | 3/3 |
| `ex5_blackout` | 3/3 | 3/3 |
| `known_pair` (mountains 180°) | **1/1** | **0/1** |

Identical everywhere except the one pair DISK *loses*, at **30× the latency**. So ORB remains the
default and DISK is a benchmark row, not a promotion.

**Why the match counts were misleading.** DISK produces *more* matches than ORB on two of three
degradation pairs — and measuring against the known transform shows the extra matches are junk:

| Pair | Backend | Matches | Correct | Precision |
|---|---|---|---|---|
| `base_cell → ex2_degraded` | orb | 4 | 2 | 50.0% |
| `base_cell → ex2_degraded` | **disk** | **25** | **0** | **0.0%** |
| `base_cells_1 → ex2_degraded` | orb | 111 | 88 | 79.3% |
| `base_cells_1 → ex2_degraded` | **disk** | **215** | **179** | **83.3%** |
| `base_cells_2 → ex2_degraded` | orb | 34 | 1 | 2.9% |
| `base_cells_2 → ex2_degraded` | **disk** | **5** | **0** | **0.0%** |

`ex2_degraded` applies **no geometric change**, so the true transform is the identity and
"correct" is checkable. A harness counting matches would have reported DISK as a 6× improvement on
`base_cell` while the pipeline got strictly worse. **This is why `bench` scores correspondences
against the transform, not the count.**

`ex1_affine` staying 0/3 is consistent with the Stage A finding that DISK is *also* not
mirror-invariant: 2,000 keypoints per side yielded 6 matches on a mirrored pair.

### A harness bug found by its own first run

The first run reported **FPR 10.0%, precision 90.9%**. Wrong — and wrong in the flattering
direction for the *controls*. `mountains`/`mountains_manipulated` share content but do not use the
`<base>_<suffix>` naming, so `discover_cases` paired them as unrelated and scored a **correct
detection as a false positive**. Corrected via `KNOWN_POSITIVES`, the true figures are 0.0% and
100.0%. A harness bug does not produce an obviously broken number, it produces a plausible wrong
one — hence `tests/test_benchmark.py` pins the classification rules with no model in the loop.

### Honest read on Stage B

What the numbers actually say is that **`ex1_affine` (0/3) and `ex2_degraded` (1/3) are not
matcher problems.** Swapping ORB for a state-of-the-art learned matcher changed neither. Both
failures are dominated by the **embedding**, which is the component trained on a set that was 82%
segmentation masks (bug 1) — and which scores a true positive at 0.1877 against 0.3781 for an
unrelated control. **B2 (retrain) is the critical path, not B3.**

---

## 9. Changelog

### 2026-09-07 (session 2) — A3 reports + A4 web demo

**A3 — reports.** Added `report/` (`overlays.py`, `render.py`, three templates). **Bug 7 closed:**
prose is flowed HTML with no fixed height in the text path, so it cannot clip at any length in any
language. `test_report.py` asserts the *property* (the full verdict, the rejection explanation and
the disclaimer all survive) rather than measuring a canvas. Wired `--report`/`--format` onto both
analysis commands rather than adding a `report` verb — a report is always *of* an analysis.

**A4 — web demo.** Added `api/app.py`, `web/` (Next.js 16 + React 19, TypeScript strict, zero UI
deps, hand-written design system), `docker/`, `docker-compose.yml`, `serve` on the CLI.

**Verified, not assumed:**
- 37 tests passing locally (16 CLI + 12 API + 9 report).
- Overlay rendering driven against real `inputs/mountains*.jpg` — both the rejected and verified
  paths produce non-blank output, and bug 14's cap engaged (2048 px).
- API serving `/healthz`, `/v1/config`, `/v1/examples`; CORS preflight correct for the frontend
  origin; a real `POST /v1/compare` returns a clean structured 500 (torch absent) in 0.42 s rather
  than hanging.
- `next build` succeeds; served HTML and the built stylesheet both contain the design tokens.
- `hatchling` wheel contains the Jinja templates and `data/default.yaml`.
- The `docker-compose.yml` env-override syntax was executed against `load_config` to confirm
  `SCIFORENSICS_API__CORS_ORIGINS` actually parses as documented.

**Security fix:** `next@15.1.6` ships CVE-2025-66478. Upgraded to **16.3.4**; `npm audit` now
reports 0 vulnerabilities.

**Two defects found in my own work while testing:**
1. `StrictUndefined` (correctly) rejected `overlays.matches` when the key was absent — but overlays
   are legitimately optional (attribution is dropped when degenerate, the local stage may not run).
   Switched the templates to `overlays.get(...)`; `test_report.py` covers it.
2. `report/overlays.py` imported `Attribution`/`CopyMoveDetection` eagerly for **annotation-only**
   use, dragging torch *and* scikit-learn into a module that is otherwise pure OpenCV — so drawing
   a box would have required the full model stack. Moved under `TYPE_CHECKING`.

Also corrected the CLI's `cmfd` region table, which read `transform.rotation_deg` — a field that
does not exist on `CopyMoveRegion` (they are flat: `rotation_deg`, `scale`, `flip`). It would have
printed `-` for every rotation. The defensive `_fmt_box` guessing that hid this is gone.

**Known gap:** WeasyPrint is not installed locally, so the **PDF path is unexercised** — only the
HTML it falls back to. `render.py` degrades with a warning by design, but do not claim PDF output
works until it has been run.

### 2026-09-07 (session 1) — bible created; CLI written, closing A0

**Analysis.** Read the full plan + all 33 package modules + 13 test modules + CI + config; wrote
this file. Established the progress ledger by static analysis (install declined ⇒ inherited suite
not run). Confirmed bugs 2,3,4,5,6,8,9,10,11,12,13,14 fixed in shipped code; 1 and 7 await their
modules. Noted the previous agent found and fixed an unlisted **bug 14** (no resolution cap).

**Found and fixed the A0 blocker.** `pyproject.toml` declared
`sciforensics = "sciforensics.cli:app"` but `cli.py` was absent — the wheel-install CI job could
not resolve the entry point and every command in the plan's verification section was unrunnable.

**Added `src/sciforensics/cli.py`** — `compare`, `cmfd`, `config`, `version`:
- Thin adapter by design: no thresholds (bug 10), no forensic decisions. Builds **one** `Pipeline`
  (bug 11) and seeds via `seed_everything`.
- `torch` imported lazily in `_build_pipeline`, so `--help`/`config`/`version` work with no model
  stack installed and a missing stack yields an actionable message, not a traceback.
- Rendering is deliberately shaped to make the fixed bugs *legible*: rotation/scale/**flip**
  (bugs 2/3), the `raw → ratio → good` funnel with distinct-count and a `not injective` flag
  (bug 4), the CMFD `clusters → verified regions` funnel (bug 6), and the named `RejectionReason`
  instead of a bare "not verified".
- Prints *"monotone score, not a calibrated probability"* whenever `calibrated=False`, so the CLI
  cannot repeat the legacy verdict strings' overclaiming.
- stdout = results, stderr = diagnostics, so `--json | jq` works with logs visible.
- Exit codes 0/1/2; **the verdict never affects exit status** (a build that fails on a finding
  incentivises suppressing findings).
- Did **not** stub the plan's `report`/`index`/`eval`/`serve` verbs — a subcommand that exists but
  does nothing is the same lie as a disabled detector reporting "no regions found".

**Added `tests/test_cli.py`** — 14 tests, torch-free, **verified passing locally in 0.42 s**
(the only tests actually run on this machine). Guards the entry point's existence and, critically,
that every command *builds* under Typer's import-time annotation introspection.

**Bug found while testing my own change:** a pyupgrade-style rewrite of `Optional[list[str]]` to
`list[str] | None` made Typer raise *"List types with complex sub-types are not currently
supported"* at import, breaking **all four** commands at once. Reverted to `Optional[list[str]]`
and added a **narrow, commented `UP045` per-file-ignore** in `pyproject.toml` rather than leaving
the lint gate red. `test_each_command_builds` now covers this class of failure.

### `bcb29bc` "Fixed some bugs" — Stage A0+A1+A2 (previous agent)

~14,500 lines: the whole `src/sciforensics/` package, 302 tests, 3-job CI, `configs/default.yaml`,
`pyproject.toml`, `LICENSE`, `.pre-commit-config.yaml`, `.gitattributes`, legacy-evidence assets.

### `ff1ff18` / `a6bd899` — prototype

Original three-stage prototype and README.
