# Robust Image Plagiarism and Manipulation Detection in Scientific Publications

## Overview

The integrity of scientific publications relies heavily on the authenticity of reported data, particularly imagery. Existing image similarity systems often fall short when identifying subtle, intentional manipulations common in academic plagiarism—such as cropping, rotation, flipping, scaling, contrast/brightness adjustments, and partial region reuse.

This project introduces a **multi-level forensic detection pipeline** that combines deep-learning-based global similarity measures with classical local feature matching and rigorous geometric verification. The system is capable of:

- **Manipulation-aware similarity scoring** under various image transformations (rotation, flip, crop, perspective warp, brightness/contrast changes).
- **Region-level duplication detection** (Copy-Move Forgery Detection / CMFD) using Difference-of-Gaussians (DoG) blob detection and ORB feature matching.
- **Geometric verification** of matched regions via RANSAC-based homography and affine estimation.
- **Automated forensic report generation** — a multi-panel visual dashboard including activation heatmaps, keypoint match visualizations, and a textual summary for each image pair.

---

## Table of Contents

1. [Architecture & Pipeline](#architecture--pipeline)
2. [Project Structure](#project-structure)
3. [Module Reference](#module-reference)
   - [Global Matching Pipeline](#1-global-matching-pipeline-srcglobal_matching)
   - [Local Matching Pipeline](#2-local-matching-pipeline-srclocal_matching)
   - [Forensic Scanner (Orchestrator)](#3-forensic-scanner-orchestrator-forensic_scannerpy)
4. [Data Classes & Configuration](#data-classes--configuration)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
   - [Running the Forensic Scanner](#running-the-forensic-scanner)
   - [Training the Siamese Network](#training-the-siamese-network)
   - [Evaluating the Trained Model](#evaluating-the-trained-model)
   - [Gradient Localization (Standalone)](#gradient-localization-standalone)
   - [Local ORB Matching (Standalone)](#local-orb-matching-standalone)
7. [CLI Reference](#cli-reference)
8. [Output Format & Forensic Report](#output-format--forensic-report)
9. [Dataset](#dataset)
10. [Technical Details](#technical-details)
11. [Dependencies](#dependencies)

---

## Architecture & Pipeline

The system operates in a **two-stage coarse-to-fine architecture** controlled by tunable distance thresholds:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE PAIR                           │
│                  (left_image, right_image)                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 1 — GLOBAL MATCHING (Deep Learning)            │
│                                                                     │
│  1. Load & preprocess both images (grayscale, resize 128×128,      │
│     normalize to [-1, 1])                                           │
│  2. Forward pass through Siamese CNN → 128-d embedding per image   │
│  3. Compute L1 distance between embeddings                         │
│  4. Compute similarity score = σ(1 − distance)                     │
│  5. Decision gates:                                                 │
│     • distance ≤ same_threshold  →  flag as "suspicious"           │
│     • distance ≤ local_trigger   →  proceed to Stage 2             │
│  6. If triggered: generate Grad-CAM-style activation heatmaps      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  (only if distance ≤ local_trigger)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 2 — LOCAL MATCHING (Classical CV)               │
│                                                                     │
│  1. Enhance images (4× upscale + CLAHE contrast enhancement)       │
│  2. Compute Sobel edge map → DoG blob detection → ROI boxes        │
│  3. Detect ORB keypoints & descriptors (up to max_features)        │
│  4. Filter keypoints to DoG ROI regions                             │
│  5. Brute-force Hamming matching + Lowe's ratio test               │
│  6. If ≥ min_matches good matches found → Stage 3                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│          STAGE 3 — GEOMETRIC VERIFICATION (RANSAC)                │
│                                                                     │
│  1. Estimate Homography via RANSAC (cv2.findHomography)            │
│  2. Compute inlier count & inlier ratio                            │
│  3. If inliers ≥ min_inliers AND ratio ≥ min_inlier_ratio         │
│     → "verified" manipulation                                      │
│  4. Estimate affine transform (cv2.estimateAffinePartial2D)        │
│     → extract rotation, scale, translation, flip detection          │
│  5. Compute bounding boxes around inlier keypoints in each image   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FORENSIC REPORT GENERATION                     │
│                                                                     │
│  6-panel dashboard (2 rows × 3 columns):                           │
│  ┌────────────┬────────────┬──────────────────┐                    │
│  │  Input A   │  Input B   │ Forensic Summary │                    │
│  │ (+ bbox)   │ (+ bbox)   │   (text panel)   │                    │
│  ├────────────┼────────────┼──────────────────┤                    │
│  │ Heatmap A  │ Heatmap B  │  Local Matches   │                    │
│  │ (Grad-CAM) │ (Grad-CAM) │  (ORB inliers)   │                    │
│  └────────────┴────────────┴──────────────────┘                    │
│                                                                     │
│  + JSON summary file per scan                                      │
│  + overall_summary.json for batch mode                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```text
SciForensics/
├── forensic_scanner.py            # Main CLI orchestrator — ties global + local pipelines together
├── generate_examples.py           # Script to generate synthetic malicious manipulation examples
├── run_demo.py                    # Script to auto-run the scanner on all base/generated examples
├── .gitignore                     # Git configuration to ignore large binaries and cache
├── README.md                      # This file
├── data/                          # Dataset root (ignored in VC)
│   ├── train/bbbc038/             # Raw .png training images
│   └── valid/bbbc038/             # Raw .png validation images
├── inputs/                        # Place suspect images here
│   ├── base_cell.png              # Reference base image
│   ├── base_cells_1.png           # Another reference base image
│   └── base_cells_2.png           # Another reference base image
├── outputs/                       # Generated forensic reports and JSON summaries appear here
│   └── .gitkeep                   # Placeholder for empty output dir
├── models/
│   ├── weights.pth                # Pre-trained Siamese CNN weights (~34 MB, ignored in VC)
│   └── .gitkeep                   # Placeholder for model directory
└── src/                           # Core library
    ├── global_matching/           # Deep learning Siamese pipeline
    │   ├── model.py               # CNN architecture + triplet loss + distance functions
    │   ├── dataset.py             # SimulatedDataset with online triplet generation
    │   ├── manipulations.py       # Custom augmentations (RandomText, RandomRect, RandomErase)
    │   ├── train.py               # Training loop with early stopping & TensorBoard logging
    │   ├── test.py                # Evaluation script — loss & accuracy on held-out data
    │   ├── pairwise_inference.py  # PairwiseEmbeddingModel — inference + heatmap generation
    │   └── grad_loc.py            # Standalone Grad-CAM gradient localization visualizer
    └── local_matching/            # Classical computer vision pipeline
        └── local_detector.py      # DoG blob detection + ORB matching + visualization
```

---

## Module Reference

### 1. Global Matching Pipeline (`src/global_matching/`)

#### `model.py` — Siamese CNN Architecture

The backbone is a custom 4-block convolutional network that projects a **128×128 grayscale** input into a **128-dimensional embedding** space:

```
Input (1×128×128)
   ↓
ConvLayer(1→16)   : BN → [LRN] → [1×1 NiN] → Conv3×3 → ReLU → Conv3×3 → ReLU → MaxPool2×2
ConvLayer(16→32)  : same block with optional NiN + LRN
ConvLayer(32→64)  : same block
ConvLayer(64→128) : same block
   ↓
[Conv1×1(128→128) + ReLU]     (Network-in-Network compression)
   ↓
Flatten (128×8×8 = 8192)
   ↓
FC(8192 → 1024) → ReLU
FC(1024 → 128)
   ↓
Output: 128-d embedding vector
```

**Key components:**
- **`ConvLayer`**: Modular block with optional BatchNorm, Local Response Normalization (LRN), and Network-in-Network (NiN) 1×1 convolutions.
- **`distance(a, b)`**: L1 (Manhattan) distance — `torch.sum(torch.abs(a - b), dim=-1)`.
- **`triplet_loss(anchor, same, diff)`**: Binary cross-entropy over sigmoid-mapped distances: `−E[log σ(1−d_same) + log(1−σ(1−d_diff))]`.
- **`triplet_acc(anchor, same, diff)`**: Fraction of correct predictions at the 0.5 sigmoid threshold.

#### `dataset.py` — SimulatedDataset (Online Triplet Mining)

Each `__getitem__` call produces a triplet `(anchor, same, diff)`:
- **Anchor**: The original image with basic transforms (grayscale, resize to 256, center-crop to 128, random flips, normalize).
- **Same (positive)**: The *same* image with aggressive synthetic manipulations:
  - `RandomPerspective(p=0.5)` — simulates camera angle changes
  - `RandomRotation(±20°)` — rotation tolerance
  - `RandomHorizontalFlip(p=0.5)` and `RandomVerticalFlip(p=0.5)` — mirror operations
  - `RandomText(p=0.5)` — overlays random ASCII strings (simulates watermarks/annotations)
  - `RandomRect(p=0.5)` — draws random rectangles (simulates figure borders/overlays)
  - `RandomErase(p=0.25)` — erases random circular regions (simulates occlusion)
  - `ColorJitter(brightness=0.2)` — brightness perturbation
- **Diff (negative)**: A *different* image from the dataset with the same manipulations.

#### `manipulations.py` — Custom Augmentation Transforms

Three PIL-based transforms that simulate real-world scientific image manipulations:

| Transform | Description | Probability |
|---|---|---|
| `RandomText` | Draws a random alphanumeric string (5–15 chars) at a random position | `p=0.5` |
| `RandomRect` | Draws an outlined rectangle with random dimensions and color | `p=0.5` |
| `RandomErase` | Erases a random circular region by filling with black | `p=0.25` |

#### `train.py` — Training Loop

```python
python src/global_matching/train.py --n_epochs 200 --bs 128 --lr 1e-4
```

| Argument | Default | Description |
|---|---|---|
| `--n_epochs` | `200` | Maximum training epochs |
| `--patience` | `50` | Early stopping patience (epochs without improvement) |
| `--bs` | `128` | Batch size |
| `--lr` | `1e-4` | Learning rate (Adam optimizer) |
| `--train_dir` | `data/train/bbbc038` | Training image directory |
| `--valid_dir` | `data/valid/bbbc038` | Validation image directory |

**Training features:**
- **Optimizer**: Adam with configurable learning rate.
- **Early stopping**: Halts if validation loss does not improve for `patience` epochs.
- **Checkpointing**: Saves `models/weights.pth` (best) and `models/checkpoint_{epoch}.pth` on each improvement.
- **TensorBoard**: Logs `loss/train`, `loss/valid`, and `acc/valid` scalars to `runs/`.
- **Auto device**: CUDA if available, else CPU.

#### `pairwise_inference.py` — PairwiseEmbeddingModel

The primary inference class used by `ForensicScanner`:

```python
model = PairwiseEmbeddingModel(
    weights_path="models/weights.pth",
    input_size=128,
    same_threshold=1.1,
    local_trigger_threshold=2.0,
)
result: GlobalSimilarityResult = model.compare("image_a.png", "image_b.png", generate_heatmaps=True)
```

**Returns a `GlobalSimilarityResult` dataclass:**
| Field | Type | Description |
|---|---|---|
| `distance` | `float` | L1 distance between the two 128-d embeddings |
| `similarity_score` | `float` | `σ(1 − distance)` — higher means more similar |
| `suspicious` | `bool` | `True` if distance ≤ `same_threshold` |
| `trigger_local` | `bool` | `True` if distance ≤ `local_trigger_threshold` |
| `left_heatmap` | `np.ndarray \| None` | Grad-CAM overlay on input A |
| `right_heatmap` | `np.ndarray \| None` | Grad-CAM overlay on input B |

**Heatmap generation** (`localize` method):
1. Forward pass through `model.features` to get final convolutional activations.
2. Retain gradients via `retain_grad()`.
3. Compute L1 distance between embeddings and backpropagate.
4. Global average-pool the gradients across spatial dimensions to get channel weights.
5. Multiply activations by channel weights, mean across channels, ReLU, normalize.
6. Resize heatmap to match original image dimensions and overlay using `JET` colormap (60% original + 40% heatmap).

#### `grad_loc.py` — Standalone Gradient Localization

A standalone `GradientLocalizaton` wrapper class that registers gradient hooks on the conv features for Grad-CAM-style visualization. The `impose()` function overlays heatmaps onto raw images:

```python
python src/global_matching/grad_loc.py --weights models/weights.pth --test_dir data/test/bbbc038
```

Outputs: `grad_anchor.png`, `raw_anchor.png`, `grad_same.png`, `raw_same.png`.

#### `test.py` — Model Evaluation

Evaluates trained model on a held-out test set, reporting average triplet loss and accuracy:

```python
python src/global_matching/test.py --weights models/weights.pth --test_dir data/test/bbbc038 --bs 128
```

Use `--display` flag to interactively visualize false positives, false negatives, and true positives via matplotlib.

---

### 2. Local Matching Pipeline (`src/local_matching/`)

#### `local_detector.py` — DoG + ORB Pairwise Matcher

This module implements the classical computer vision pipeline for region-level copy-move forgery detection.

**Processing steps:**

1. **Image enhancement** (`enhance_for_orb`):
   - 4× cubic upscale for better keypoint localization.
   - Gaussian blur (σ=0.8) to suppress noise.
   - CLAHE (Contrast Limited Adaptive Histogram Equalization, clipLimit=2.0, tileGridSize=8×8) for local contrast enhancement.

2. **Blob detection** (`dog_regions`):
   - Computes Sobel edge magnitude map.
   - Applies Difference-of-Gaussians (σ₁=1.0, σ₂=2.4) on the enhanced image.
   - Otsu thresholding + morphological opening to extract binary mask.
   - Finds contours → bounding boxes (up to 24 regions, sorted by area).
   - Falls back to scikit-image `blob_dog` if available.

3. **ORB feature extraction** (`match_pair`):
   - `cv2.ORB_create(nfeatures=2000, fastThreshold=5, edgeThreshold=15, patchSize=31, scaleFactor=1.2, nlevels=8)`.
   - Keypoints optionally filtered to DoG ROI regions (fallback to all keypoints if <8 remain after filtering).

4. **Brute-force matching**:
   - Hamming distance matcher (`cv2.BFMatcher(cv2.NORM_HAMMING)`).
   - k-NN matching (k=2) + Lowe's ratio test (default ratio = 0.8).

5. **Point coordinate rescaling**: All matched point coordinates are divided by the 4× enhancement scale to map back to original image space.

**Returns a `PairwiseMatchResult` dataclass:**
| Field | Type | Description |
|---|---|---|
| `left_work_bgr` / `right_work_bgr` | `np.ndarray` | Annotated images with DoG region boxes drawn |
| `left_keypoints` / `right_keypoints` | `list[cv2.KeyPoint]` | Detected keypoints (rescaled to 1× coordinates) |
| `good_matches` | `list[cv2.DMatch]` | ORB matches passing the ratio test |
| `left_points` / `right_points` | `np.ndarray \| None` | Matched point coordinates (N×1×2) |
| `dog_boxes_left` / `dog_boxes_right` | `list[tuple]` | DoG region bounding boxes (x, y, w, h) |
| `left_keypoint_count` / `right_keypoint_count` | `int` | Total keypoints detected per image |
| `good_match_count` | `int` | Number of good matches after ratio test |

**Standalone usage:**
```bash
python src/local_matching/local_detector.py image_a.png image_b.png --output matches.png
```

---

### 3. Forensic Scanner (Orchestrator) — `forensic_scanner.py`

The top-level CLI application that orchestrates the full pipeline. It operates in two modes:

#### Single-Pair Mode
Compare two specific images:
```bash
python forensic_scanner.py inputs/image_A.png inputs/image_B.png
```

#### Batch / Dataset Mode
Compare a base image against every `.jpg` and `.png` in a directory:
```bash
python forensic_scanner.py inputs/krishna.png inputs/ --output outputs/
```

**Pipeline flow:**
1. Load and read both images via OpenCV.
2. Run `PairwiseEmbeddingModel.compare()` → `GlobalSimilarityResult`.
3. If `trigger_local` is `True`:
   - Run `local_detector.match_pair()` → `PairwiseMatchResult`.
   - Run `ForensicScanner._verify_geometry()` → `GeometricVerificationResult`.
4. Build 6-panel forensic report image and save to `outputs/`.
5. Generate JSON summary per image pair.
6. In batch mode: generate `overall_summary.json` aggregating all results.

**Decision logic:**
| Condition | Decision |
|---|---|
| Geometry verified **AND** globally suspicious | `"Strong evidence of reused or manipulated scientific imagery"` |
| Geometry verified only | `"Potential partial reuse verified geometrically"` |
| Globally suspicious, geometry inconclusive | `"Globally similar pair; local verification inconclusive"` |
| Neither | `"No strong evidence of manipulated reuse"` |

---

## Data Classes & Configuration

### `ScannerConfig`

Central configuration dataclass for the forensic scanner with all tunable parameters:

```python
@dataclass
class ScannerConfig:
    weights_path: Path                  # Path to Siamese CNN weights (.pth file)
    output_path: Path                   # Output path for the report image
    same_threshold: float = 1.1         # L1 distance ≤ this → "suspicious" flag
    local_trigger_threshold: float = 2.0  # L1 distance ≤ this → trigger local ORB stage
    nn_ratio: float = 0.8              # Lowe's ratio test threshold
    max_features: int = 2000           # Max ORB keypoints to detect per image
    min_matches: int = 8               # Min good ORB matches to proceed to geometry stage
    min_inliers: int = 8               # Min RANSAC inliers for positive geometric verification
    min_inlier_ratio: float = 0.18     # Min inlier/total ratio for positive verification
    ransac_reproj_threshold: float = 4.0  # RANSAC reprojection error tolerance (pixels)
```

### `GeometricVerificationResult`

Returned by the geometric verification stage:

```python
@dataclass
class GeometricVerificationResult:
    verified: bool                          # True if manipulation is geometrically confirmed
    homography: Optional[np.ndarray]        # 3×3 homography matrix (or None)
    inlier_mask: Optional[np.ndarray]       # Boolean mask over good_matches
    inlier_count: int                       # Number of RANSAC inliers
    inlier_ratio: float                     # inlier_count / good_match_count
    left_bbox: Optional[tuple[int,int,int,int]]   # Bounding box (x,y,w,h) of inliers in left image
    right_bbox: Optional[tuple[int,int,int,int]]  # Bounding box (x,y,w,h) of inliers in right image
    rotation_deg: Optional[float]           # Estimated rotation angle (degrees)
    scale: Optional[float]                  # Estimated scale factor
    translation: Optional[tuple[float,float]]  # Estimated (tx, ty) translation
    flip_detected: Optional[bool]           # True if determinant of affine is negative → mirror flip
```

**Affine decomposition** (`_decode_affine`):
- Extracts rotation, scale, translation, and flip from the estimated 2×3 affine matrix.
- Rotation = `atan2(a₁₀, a₀₀)` in degrees.
- Scale = `‖column₀‖₂` of the linear part.
- Flip = `det(linear) < 0`.

---

## Installation & Setup

### Prerequisites
- **Python 3.8+**
- **CUDA** (optional, for GPU-accelerated inference and training)

### Install Dependencies

```bash
pip install torch torchvision numpy opencv-python Pillow matplotlib tensorboard
```

### Verify Installation

```bash
python -c "import torch; import cv2; print(f'PyTorch {torch.__version__}, OpenCV {cv2.__version__}')"
```

### Pre-trained Weights
The repository includes pre-trained weights at `models/weights.pth` (~34 MB). No additional download is needed for inference.

---

## Usage Guide

### Running the Forensic Scanner

#### Quick Start — Single Pair
```bash
python forensic_scanner.py inputs/krishna.png inputs/Lord_Krishna.jpg --output outputs/report.png
```

#### Intra-Image CMFD (Copy-Move Detection) — Single Image
Scan a single image for duplicated or cloned regions:
```bash
python forensic_scanner.py inputs/krishna.png --cmfd
```

#### Batch Scan — One Base Image vs. a Directory
```bash
python forensic_scanner.py inputs/base_cell.png inputs/ --output outputs/
```
This will:
1. Find all `.jpg` and `.png` files in `inputs/`.
2. Compare each against `base_cell.png`.
3. Save a `report_<name>.png` and `summary_<name>.json` for each pair.
4. Save an aggregated `overall_summary.json`.

#### Generating Demo Manipulations
The project includes a script to synthesize simulated plagiarism attacks natively using the base images in `inputs/`:
```bash
python generate_examples.py
```
This generates 5 different malicious manipulations (Affine scaling/rotation, copy-move forgery, signal degradation, blackout masking, and extreme exposure shifts) for each base image.

#### Running the Full Demo Suite
If you want to run the scanner exhaustively on all permutations of the base and generated demo files, use the `run_demo.py` orchestrator:
```bash
python run_demo.py
```
This script acts as an automated wrapper for `forensic_scanner.py`. It loops over all input combinations, saves detailed individual reports to `outputs/demo/`, and stitches them together into massive consolidated visual forensics dashboards.

#### Example JSON Output
```json
{
  "global_distance": 2.136157,
  "similarity_score": 0.243027,
  "triggered_local_stage": true,
  "good_matches": 98,
  "homography_verified": true,
  "inlier_count": 88,
  "inlier_ratio": 0.897959,
  "decision": "Potential partial reuse verified geometrically",
  "report_path": "C:\\...\\outputs\\report_Lord_Krishna.png"
}
```

---

### Training the Siamese Network

If you wish to re-train the global similarity model instead of using the provided `weights.pth`:

#### 1. Prepare the Data
The network is trained on the **Kaggle 2018 Data Science Bowl** (BBBC038) microscopy image dataset. Download and flatten `.png` images into:
```
data/train/bbbc038/*.png
data/valid/bbbc038/*.png
```

#### 2. Run Training
```bash
python src/global_matching/train.py --n_epochs 200 --bs 128 --lr 1e-4
```

#### 3. Monitor with TensorBoard
```bash
tensorboard --logdir runs
```
Navigate to `http://localhost:6006` to view `loss/train`, `loss/valid`, and `acc/valid` curves in real-time.

---

### Evaluating the Trained Model

```bash
python src/global_matching/test.py --weights models/weights.pth --test_dir data/test/bbbc038 --bs 128
```

Add the `--display` flag to visually inspect true positives, false positives, and false negatives:
```bash
python src/global_matching/test.py --weights models/weights.pth --test_dir data/test/bbbc038 --display
```

---

### Gradient Localization (Standalone)

Generate Grad-CAM heatmap overlays independently of the full scanner:
```bash
python src/global_matching/grad_loc.py --weights models/weights.pth --test_dir data/test/bbbc038
```
Outputs four images: `grad_anchor.png`, `raw_anchor.png`, `grad_same.png`, `raw_same.png`.

---

### Local ORB Matching (Standalone)

Run the local feature matching independently:
```bash
python src/local_matching/local_detector.py image_a.png image_b.png --output orb_matches.png
```

---

## CLI Reference

### `forensic_scanner.py`

| Argument | Type | Default | Description |
|---|---|---|---|
| `left_image` | positional | `inputs/krishna.png` | Path to the base/reference image |
| `right_image` | positional | `inputs/` | Path to comparison image or directory of images |
| `--cmfd` | flag | `False` | Run intra-image copy-move forgery detection on left_image instead of pairwise comparison |
| `--output` | str | `outputs/` | Output path (file for single pair, directory for batch) |
| `--weights` | str | `models/weights.pth` | Path to pre-trained PyTorch weights |
| `--same-threshold` | float | `1.1` | L1 distance threshold — below this the pair is flagged as "suspicious" |
| `--local-threshold` | float | `3.0` | L1 distance threshold — below this triggers local ORB + geometry stage |
| `--ratio` | float | `0.8` | Lowe's nearest-neighbor ratio test threshold for ORB matching |
| `--max-features` | int | `2000` | Maximum number of ORB keypoints to detect per image |
| `--min-matches` | int | `8` | Minimum good ORB matches required to attempt geometric verification |
| `--min-inliers` | int | `8` | Minimum RANSAC inliers required for positive geometric verification |
| `--min-inlier-ratio` | float | `0.18` | Minimum inlier-to-match ratio for positive verification |
| `--ransac-threshold` | float | `4.0` | RANSAC reprojection error threshold in pixels |

---

## Output Format & Forensic Report

### Visual Dashboard

Each scan generates a **1620×936 pixel** forensic report image (3 columns × 2 rows, each panel 540×468):

| | Column 1 | Column 2 | Column 3 |
|---|---|---|---|
| **Top Row** | **Input A** — original image with verified region bounding box (orange) | **Input B** — comparison image with matched region bounding box (orange) | **Forensic Summary** — text panel with all numeric results and final decision |
| **Bottom Row** | **Global Heatmap A** — Grad-CAM activation overlay on image A | **Global Heatmap B** — Grad-CAM activation overlay on image B | **Local Matches** — ORB keypoint matches visualization (green lines = inliers) |

### JSON Summary Fields

| Field | Type | Description |
|---|---|---|
| `global_distance` | float | L1 distance between 128-d embeddings |
| `similarity_score` | float | Sigmoid-mapped similarity ∈ [0, 1] |
| `triggered_local_stage` | bool | Whether the local ORB stage was triggered |
| `good_matches` | int | Number of ORB matches passing ratio test |
| `homography_verified` | bool | Whether geometric verification passed |
| `inlier_count` | int | RANSAC inlier count |
| `inlier_ratio` | float | Fraction of good matches that are inliers |
| `decision` | string | Final human-readable verdict |
| `report_path` | string | Absolute path to the generated report image |

---

## Dataset

The model is trained on the **Broad Bioimage Benchmark Collection BBBC038** dataset (Kaggle 2018 Data Science Bowl), containing microscopy images of cell nuclei. This domain-agnostic dataset is chosen because:

1. Images contain rich, complex textures suitable for learning manipulation-invariant features.
2. The dataset is freely available and well-established.
3. Microscopy images share structural properties with many scientific figure types (plots, scans, micrographs).

**Note**: The model generalizes beyond microscopy — the synthetic manipulation pipeline ensures the learned embedding space is robust to transformations commonly seen across all types of scientific imagery.

---

## Technical Details

### Threshold Tuning Guide

| Parameter | Low Value Effect | High Value Effect | Recommendation |
|---|---|---|---|
| `same-threshold` | More conservative — only near-duplicates flagged | More aggressive — catches distant manipulations but may false-positive | Start at `1.1`; lower to `0.8` for strict near-duplicate detection |
| `local-threshold` | Fewer images enter local stage (faster, more precise) | More images enter local stage (slower, higher recall) | Start at `2.0`–`3.0`; raise to `5.0` for exhaustive scanning |
| `ratio` | Stricter matching — fewer but more reliable matches | Looser matching — more matches but more noise | `0.75`–`0.85` is typical |
| `min-inlier-ratio` | Easier to verify — risk of false positives | Harder to verify — misses partial reuse | `0.15`–`0.25` works for most cases |

### Model Capacity

| Metric | Value |
|---|---|
| Input size | 128 × 128 × 1 (grayscale) |
| Embedding dimension | 128 |
| Total parameters | ~1.2M |
| Weights file size | ~34 MB |
| Inference time (CPU) | ~50–100 ms per pair |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥1.9 | Neural network backbone, training, inference |
| `torchvision` | ≥0.10 | Image transforms, data augmentation pipeline |
| `numpy` | ≥1.19 | Array operations, geometric computations |
| `opencv-python` | ≥4.5 | ORB, RANSAC, image I/O, heatmap overlay, report rendering |
| `Pillow` | ≥8.0 | PIL-based custom augmentations (text, rect, erase) |
| `matplotlib` | ≥3.3 | Visualization in test/evaluation scripts |
| `tensorboard` | ≥2.4 | Training monitoring (loss/accuracy curves) |