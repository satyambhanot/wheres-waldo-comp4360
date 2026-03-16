# Where's Waldo? - CNN-Based Character Detection in Cluttered Scenes

**Course:** COMP 4360 Machine Learning, University of Manitoba
**Team:** Rudy Kissel, Satyam Bhanot, Gabriel Barnabe
**Instructor:** Dr. Christopher Henry
**Date:** March 2026

This project adapts a strawberry detection system (Lamb & Chuah, IEEE Big Data 2018) to the domain of locating Waldo in "Where's Waldo?" puzzle illustrations. We replace the original 3-layer CNN with a MobileNet V3 Large backbone, extend the SSD detection heads from 4 to 6 levels, and introduce dual HSV masking with Sobel edge detection to exploit Waldo's distinctive red-and-white stripe pattern. The model processes high-resolution puzzle images as overlapping tiles at 64x64, 128x128, and 256x256 resolutions, predicting per-tile bounding boxes for Waldo's location.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/satyambhanot/wheres-waldo-comp4360.git
cd wheres-waldo-comp4360

# 2. Set up the environment
make setup

# 3. Place raw images in data/raw/ (see Data Setup below)

# 4. Generate tiles and train
make data
make train

# 5. Evaluate on test set
make evaluate
```

> **Note:** Raw puzzle images must be placed manually in `data/raw/images/` and `data/raw/annotations/` before running. These files are not included in the repository due to size and copyright. See [Data Setup](#data-setup) for instructions.

## Prerequisites

- **Python 3.11+**
- **macOS with Apple Silicon** (M1/M2) for MPS acceleration, OR **Linux/Windows with NVIDIA GPU** (CUDA)
- ~2 GB disk space for Python dependencies
- ~200 MB for processed tiles
- Training takes 2-4 hours on an M2 MacBook (200 epochs at 256px). Adjust expectations for slower hardware.

## Project Structure

```
wheres-waldo-comp4360/
├── src/
│   ├── __init__.py                [CUSTOM]  Package init
│   ├── config.py                  [CUSTOM]  Central configuration (paths, hyperparams)
│   ├── preprocess.py              [CUSTOM]  Tiling pipeline, HSV masking, Sobel edges
│   ├── dataset.py                 [CUSTOM]  PyTorch Dataset and DataLoader
│   ├── model.py                   [CUSTOM]  WaldoSSD (MobileNet V3 + SSD heads)
│   ├── train.py                   [CUSTOM]  Training loop, SSD loss, hard negative mining
│   ├── evaluate.py                [CUSTOM]  Metrics computation + visual result examples
│   └── ablation.py                [CUSTOM]  Experiment runner for all 4 ablation studies
├── data/
│   ├── data.json                             Waldo grid locations (reference)
│   ├── raw/
│   │   ├── images/                           19 original Waldo illustrations (.jpg)
│   │   └── annotations/                      Pascal VOC XML bounding box annotations
│   └── processed/
│       ├── splits.json                        Train/val/test split assignments
│       ├── 64/                                ~15,884 tiles at 64x64
│       ├── 128/                               ~3,789 tiles at 128x128
│       └── 256/                               ~844 tiles at 256x256
├── results/
│   ├── checkpoints/                           Saved model weights (.pth)
│   ├── plots/                                 Visual TP/FP/FN examples
│   └── ablations/                             Per-experiment JSON results
├── notebooks/                                 Jupyter notebooks (exploration)
├── scripts/
│   ├── check.sh                  [CUSTOM]   8-point code quality checker
│   ├── commit.sh                 [CUSTOM]   Safe commit wrapper
│   └── run_pipeline.sh           [CUSTOM]   Full pipeline runner
├── Makefile                      [CUSTOM]   Project command entry point
├── requirements.txt                          Pinned Python dependencies
├── pyproject.toml                            Tool configuration (black, isort, mypy)
├── CODING_STANDARDS.md           [CUSTOM]   Project coding conventions
├── .github/CODEOWNERS                        Default reviewer
└── .gitignore
```

**File origin key:**
- `[CUSTOM]` - Entirely written by our team for this project
- `[ADAPTED]` - Based on an existing implementation with significant modifications
- `[ORIGINAL]` - Unmodified from original source

All `src/` modules are `[CUSTOM]` - written from scratch by our team.

**Data sources:**
- Puzzle images: [Hey-Waldo](https://github.com/vc1492a/Hey-Waldo) by vc1492a
- Bounding box annotations: [TheresWaldo](https://github.com/bckenstler/TheresWaldo) by bckenstler

## Data Setup

Raw images are not included in the repository (too large, copyright restrictions). To obtain them:

1. **Clone the Hey-Waldo repository** (puzzle images):
   ```bash
   git clone https://github.com/vc1492a/Hey-Waldo.git /tmp/hey-waldo
   ```

2. **Clone the TheresWaldo repository** (bounding box annotations):
   ```bash
   git clone https://github.com/bckenstler/TheresWaldo.git /tmp/theres-waldo
   ```

3. **Copy images** to the project:
   ```bash
   cp /tmp/hey-waldo/images/*.jpg data/raw/images/
   ```

4. **Copy annotations** to the project:
   ```bash
   cp /tmp/theres-waldo/annotations/*.xml data/raw/annotations/
   ```

5. **Generate tiles** at all resolutions:
   ```bash
   make data
   ```

The tiling pipeline produces overlapping tiles (15% overlap) at three resolutions:

| Resolution | Total Tiles | Positive (Waldo) | Negative (Background) | Positive Rate |
|------------|-------------|-------------------|----------------------|---------------|
| 256x256    | ~844        | ~24               | ~820                 | ~2.8%         |
| 128x128    | ~3,789      | ~30               | ~3,759               | ~0.8%         |
| 64x64      | ~15,884     | ~32               | ~15,852              | ~0.2%         |

Splits are by original image (not by tile) to prevent data leakage:
- **Train:** 11 images (525 tiles at 256px)
- **Val:** 3 images (144 tiles at 256px)
- **Test:** 5 images (175 tiles at 256px)

## Training

### Run Baseline Training

```bash
make train
```

This trains the default configuration: 256px resolution, color mode, Sobel ON, 6 SSD heads, 200 epochs.

### What to Expect

- **Epochs 1-20:** Loss decreases rapidly from ~300 to ~50. AP remains near 0.
- **Epochs 20-50:** Loss stabilizes. First non-zero AP values may appear.
- **Epochs 50-200:** Gradual AP improvement as the model learns to distinguish Waldo from hard negatives.

### Checkpoints

- **Best model:** `results/checkpoints/best_model.pth` (saved when val AP improves)
- **Periodic:** `results/checkpoints/checkpoint_epoch_XXX.pth` (every 20 epochs)

Each checkpoint stores model weights, optimizer state, scheduler state, validation AP, and the config used.

### Custom Training

```python
from src.train import train

best_ap = train(config_overrides={
    "resolution": 128,
    "color_mode": "grayscale",
    "use_sobel": False,
    "num_heads": 4,
    "num_epochs": 100,
})
```

## Evaluation

### Run Evaluation

```bash
make evaluate
```

This loads the best checkpoint, runs inference on the test set, and produces:

- **Metrics:** AP, precision, recall, F1, detection counts
- **Visual examples:** Annotated images saved to `results/plots/`
  - `tp_*.jpg` - True positives (green GT box + red prediction box)
  - `fp_*.jpg` - False positives (red prediction box, no matching GT)
  - `fn_*.jpg` - False negatives (green GT box, model missed)
- **Summary JSON:** `results/plots/examples_summary.json`

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **AP** | PASCAL VOC 11-point interpolated Average Precision |
| **Precision** | TP / (TP + FP) at the configured confidence threshold |
| **Recall** | TP / (TP + FN) at the configured confidence threshold |
| **F1** | Harmonic mean of precision and recall |

## Ablation Studies

We run 4 ablation studies, each isolating a single variable against the baseline (256px, color, Sobel ON, 6 heads):

### Run All Experiments

```bash
make ablation
```

### Run a Single Experiment

```bash
make ablation-single EXP=abl2_res128
```

### Experiment Keys

| Key | Description | Variable Changed |
|-----|-------------|-----------------|
| `abl1_sobel_on` | Sobel ON (baseline) | - |
| `abl1_sobel_off` | Sobel OFF | 4ch to 3ch input |
| `abl2_res256` | 256px (baseline) | - |
| `abl2_res128` | 128px | Tile resolution |
| `abl2_res64` | 64px | Tile resolution |
| `abl3_color` | Color (baseline) | - |
| `abl3_grayscale` | Grayscale | Color mode |
| `abl3_bw` | Black & White | Color mode |
| `abl4_6heads` | 6 heads (baseline) | - |
| `abl4_4heads` | 4 heads | Detection levels |
| `abl4_pruned` | 50% pruned extras | Channel reduction |

Results are saved as individual JSON files in `results/ablations/` and a combined `all_results.json`.

Estimated time: 2-4 hours per experiment, ~18-30 hours total. The runner supports resumability - restart after a crash and it skips completed experiments.

## Code Quality

### Check Code

```bash
make check    # read-only checks
make fix      # auto-fix formatting, then check
```

### 8 Automated Checks

1. **black** - Code formatting (100 char line length)
2. **isort** - Import ordering (black-compatible profile)
3. **flake8** - Linting (unused imports, undefined variables)
4. **mypy** - Type checking (all functions have type hints)
5. **Docstring coverage** - Every public function/class has a Google-style docstring
6. **Type hint coverage** - Every parameter and return value is annotated
7. **No print statements** - All output uses the `logging` module
8. **Magic number detection** - Warns about hardcoded numbers outside `config.py`

All code in this repository passes all 8 checks.

## Model Architecture

```
Input (B, 3or4, H, H)
    |
    +-- MobileNetV3 Large Backbone (ImageNet pretrained)
    |   +-- Stage 1: features[0:4]   -> 24ch
    |   +-- Stage 2: features[4:7]   -> 40ch
    |   +-- Stage 3: features[7:13]  -> 112ch
    |   +-- Stage 4: features[13:17] -> 960ch
    |
    +-- Extra Block 1: 960 -> 256 -> 512ch (stride 2)
    +-- Extra Block 2: 512 -> 128 -> 256ch (stride 2)

Each level -> Classification Head (3 anchors x 2 classes)
           -> Regression Head    (3 anchors x 4 offsets)
```

- **Input:** 256x256 tiles (or 128/64 for ablation), 3ch RGB or 4ch RGB+Sobel
- **Anchors:** 3 per location (aspect ratios 1.0, 0.5, 2.0), 16,380 total at 256px
- **Output:** Per-anchor class logits + bounding box offsets
- **Parameters:** ~5.07M (all trainable)
- **Loss:** SSD multi-box loss with hard negative mining (3:1 neg:pos ratio)

## Adaptation from Original Paper

**Source paper:** N. Lamb and S. Chuah, "A Strawberry Detection System Using Convolutional Neural Networks," *2018 IEEE International Conference on Big Data*, 2018.

### What We Kept
- SSD-style single-shot detection approach
- Tile-based processing of large images
- Multi-scale feature extraction with detection heads at each level

### What We Changed
| Component | Original (Strawberry) | Ours (Waldo) |
|-----------|----------------------|--------------|
| Backbone | Custom 3-layer CNN | MobileNet V3 Large (pretrained) |
| Detection heads | 4 levels | 6 levels (+ 4-head ablation) |
| Input source | Field video frames | Illustrated puzzle scans |
| Preprocessing | Single color mask | Dual HSV (red+white) + Sobel edges + density filter |
| Input channels | 3 (RGB) | 3 (RGB) or 4 (RGB + Sobel edge) |
| Anchors | Fixed scales | Linearly scaled per level, 3 aspect ratios |
| Training device | CUDA GPU | Apple MPS (M2) with CPU fallback |

## One-Command Reproduction

For a complete end-to-end run (TA usage):

```bash
# Full run (200 epochs, ~3 hours)
./scripts/run_pipeline.sh

# Quick test (10 epochs, ~15 minutes)
./scripts/run_pipeline.sh --quick
```

## References

[1] N. Lamb and S. Chuah, "A Strawberry Detection System Using Convolutional Neural Networks," *2018 IEEE International Conference on Big Data (Big Data)*, Seattle, WA, USA, 2018, pp. 2515-2520.

[2] vc1492a, "Hey-Waldo," GitHub repository. [Online]. Available: https://github.com/vc1492a/Hey-Waldo

[3] bckenstler, "TheresWaldo," GitHub repository. [Online]. Available: https://github.com/bckenstler/TheresWaldo

[4] A. Howard et al., "Searching for MobileNetV3," *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 1314-1324.

[5] W. Liu et al., "SSD: Single Shot MultiBox Detector," *European Conference on Computer Vision (ECCV)*, 2016, pp. 21-37.

## License

This is a university course project for COMP 4360 at the University of Manitoba. Not intended for redistribution. Where's Waldo images are copyrighted by Martin Handford / Walker Books.
