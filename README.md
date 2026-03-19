# Where's Waldo?
# CNN-Based Character Detection in Cluttered Scenes

**Course:** COMP 4360 Machine Learning, University of Manitoba
**Team:** Rudy Kissel, Satyam Bhanot, Gabriel Barnabe
**Instructor:** Dr. Christopher Henry
**Date:** January 2026 to March 2026

This project takes a strawberry detection system (Lamb & Chuah, IEEE Big Data 2018) and repurposes it for finding Waldo in "Where's Waldo?" puzzle illustrations. The original paper used a simple 3-layer CNN — we swapped that out for a MobileNet V3 Large backbone, bumped the SSD detection heads from 4 to 6 levels, and added dual HSV colour masking with Sobel edge detection to pick up on Waldo's red-and-white stripes. The pipeline slices full puzzle images into overlapping tiles at 64, 128, and 256px, then predicts bounding boxes for Waldo on each tile.

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/satyambhanot/wheres-waldo-comp4360.git
cd wheres-waldo-comp4360

# 2. Set up Python venv + dependencies
make setup

# 3. Train the baseline model
make train

# 4. Evaluate on the test set
make evaluate

# 5. Generate report figures
make plots
```

> Raw images, annotations, and pre-generated tiles are all included in the repo. No extra data downloads needed.

## Prerequisites

- **Python 3.11+**
- **macOS with Apple Silicon** (M1/M2) for MPS acceleration, OR **NVIDIA GPU** (CUDA) on Linux/Windows
- ~2 GB disk space for dependencies
- Training runs about 2-4 hours on an M2 MacBook at 200 epochs. Could be faster or slower depending on your hardware.

## Project Structure

```
wheres-waldo-comp4360/
├── src/
│   ├── __init__.py              Package init
│   ├── config.py                Central config (paths, hyperparams, device)
│   ├── preprocess.py            Tiling pipeline, HSV masking, Sobel edges
│   ├── dataset.py               PyTorch Dataset and DataLoader
│   ├── model.py                 WaldoSSD (MobileNet V3 + SSD heads)
│   ├── train.py                 Training loop, SSD loss, hard neg mining
│   ├── evaluate.py              Test metrics + visual detection examples
│   ├── ablation.py              Ablation experiment runner (11 configs)
│   └── plots.py                 Report figure generation from saved JSON
├── data/
│   ├── data.json                Waldo grid locations (reference)
│   ├── raw/
│   │   ├── images/              19 original Waldo puzzle images
│   │   └── annotations/         Pascal VOC XML bounding boxes
│   └── processed/
│       ├── splits.json          Train/val/test split assignments
│       ├── dataset_manifest.json  Tile counts per split per resolution
│       ├── 64/                  Tiles at 64x64 + cached Sobel + density
│       ├── 128/                 Tiles at 128x128
│       └── 256/                 Tiles at 256x256
├── results/
│   ├── checkpoints/             Model weights (.pth files)
│   ├── training/                Per-run training history JSON
│   ├── metrics/                 Test metrics + PR curve data
│   ├── plots/                   All generated figures (PNG)
│   └── ablations/               Per-experiment JSON + PR curves
├── scripts/
│   ├── check.sh                 8-point code quality checker
│   ├── commit.sh                Git commit wrapper
│   └── run_pipeline.sh          Full end-to-end pipeline
├── Makefile                     Main entry point for all commands
├── requirements.txt             Pinned dependencies
├── pyproject.toml               black/isort/mypy config
├── CODING_STANDARDS.md          Team coding conventions
└── .gitignore
```

All `src/` modules were written from scratch by our team.

**Data sources:**
- Puzzle images from [Hey-Waldo](https://github.com/vc1492a/Hey-Waldo) by vc1492a
- Bounding box annotations from [TheresWaldo](https://github.com/bckenstler/TheresWaldo) by bckenstler

## Data Setup

Everything is already in the repo — no downloads needed.

- **Raw images:** 19 puzzle illustrations in `data/raw/images/`
- **Annotations:** Pascal VOC XML files in `data/raw/annotations/`
- **Pre-generated tiles:** Ready in `data/processed/`

If you want to regenerate tiles from scratch:
```bash
make clean-data
make data
```

The tiling pipeline slides a window across each image with 15% overlap at three resolutions:

| Resolution | Total Tiles | Waldo Tiles | Background | Positive Rate |
|------------|-------------|-------------|------------|---------------|
| 256x256    | ~844        | ~24         | ~820       | ~2.8%         |
| 128x128    | ~3,789      | ~30         | ~3,759     | ~0.8%         |
| 64x64      | ~15,884     | ~32         | ~15,852    | ~0.2%         |

Splits are done at the **image level** (not tile level) to avoid data leakage:
- **Train:** 11 images (~525 tiles at 256px)
- **Val:** 3 images (~144 tiles at 256px)
- **Test:** 5 images (~175 tiles at 256px)

Detailed tile counts per split and resolution are saved to `data/processed/dataset_manifest.json` when you run the tiling pipeline.

## Training

```bash
make train
```

Trains the default configuration: 256px tiles, full colour, Sobel ON, 6 SSD heads, up to 200 epochs.

### Training Features

- **LR warmup:** Linear ramp from ~0 to full LR over the first 5 epochs, then cosine annealing decay
- **Early stopping:** Stops if validation AP doesn't improve for 6 consecutive checks (checked every 5 epochs)
- **Resume support:** Pass `resume_from="path/to/checkpoint.pth"` to pick up where you left off
- **Training history:** Per-epoch loss (total/cls/reg), LR, and val AP saved to `results/training/`

### What to Expect

- **Epochs 1-20:** Loss drops fast. AP still near 0.
- **Epochs 20-50:** Loss levels off. Might start seeing non-zero AP.
- **Epochs 50-200:** AP gradually improves. Early stopping may kick in if the model plateaus.

### Checkpoints

- `results/checkpoints/best_model.pth` — saved whenever val AP improves (includes full config, optimizer + scheduler state)
- `results/checkpoints/best_model_epoch_XXX.pth` — periodic saves every 20 epochs

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

```bash
make evaluate
```

Loads the best checkpoint, runs on the test set, and saves:

- **`results/metrics/test_metrics.json`** — AP, precision, recall, F1, TP/FP/FN counts
- **`results/metrics/pr_curve.json`** — full precision-recall curve data (for plotting)
- **`results/plots/tp_*.jpg`** — annotated true positive examples (green=GT, red=prediction)
- **`results/plots/fp_*.jpg`** — false positive examples
- **`results/plots/fn_*.jpg`** — false negatives the model missed
- **`results/plots/examples_summary.json`** — machine-readable index of the above

### Metrics

| Metric | What it measures |
|--------|-----------------|
| **AP** | PASCAL VOC 11-point interpolated Average Precision |
| **Precision** | TP / (TP + FP) — how many detections were correct |
| **Recall** | TP / (TP + FN) — how many Waldos were found |
| **F1** | Harmonic mean of precision and recall |

## Report Figures

```bash
make plots
```

Reads all saved JSON results and generates:

| Figure | File | What it shows |
|--------|------|---------------|
| Training curves | `*_training_curves.png` | Loss, val AP, and LR over epochs |
| PR curve | `baseline_pr_curve.png` | Precision vs recall with 11-pt interpolation |
| Confidence histogram | `baseline_confidence_hist.png` | TP vs FP score distributions |
| Dataset summary | `dataset_summary.png` | Tile counts per split/resolution |
| Ablation comparisons | `ablation{1-4}_comparison.png` | Grouped bar charts (AP/prec/rec/F1) |
| PR overlays | `ablation{1-4}_pr_overlay.png` | PR curves overlaid per ablation group |
| Efficiency plot | `efficiency_comparison.png` | Params vs inference time, coloured by AP |

All plots go to `results/plots/`.

## Ablation Studies

Four ablation studies, each changing one variable against the baseline (256px, colour, Sobel ON, 6 heads):

```bash
make ablation           # run all 11 conditions
make ablation-single EXP=abl2_res128   # run just one
```

### Experiment Keys

| Key | What it tests | Changed variable |
|-----|--------------|-----------------|
| `abl1_sobel_on` | Sobel ON (baseline) | — |
| `abl1_sobel_off` | Sobel OFF | 4ch -> 3ch input |
| `abl2_res256` | 256px (baseline) | — |
| `abl2_res128` | 128px | Tile resolution |
| `abl2_res64` | 64px | Tile resolution |
| `abl3_color` | Color (baseline) | — |
| `abl3_grayscale` | Grayscale | Colour mode |
| `abl3_bw` | Black & White | Colour mode |
| `abl4_6heads` | 6 heads (baseline) | — |
| `abl4_4heads` | 4 heads | Detection levels |
| `abl4_pruned` | 50% pruned extras | Channel width |

### What Gets Saved

Each experiment produces:
- `results/ablations/{key}.json` — metrics (AP, precision, recall, F1, TP/FP/FN), efficiency stats (param count, inference ms, training time), best epoch
- `results/ablations/{key}_pr_curve.json` — full PR curve data
- `results/checkpoints/{key}_best.pth` — model weights
- `results/training/{key}_best_history.json` — per-epoch training history

After all experiments finish, `results/ablations/all_results.json` has everything combined for easy comparison.

Estimated time: 2-4 hours per experiment, 18-30 hours total. The runner is resumable — if it crashes, restart and it'll skip finished experiments.

## Code Quality

```bash
make check    # read-only
make fix      # auto-fix formatting, then check
```

8 automated checks run before committing:

1. **black** — formatting (100 char lines)
2. **isort** — import ordering
3. **flake8** — linting
4. **mypy** — type checking
5. **Docstring coverage** — public functions must have docstrings
6. **Type hints** — all params and returns annotated
7. **No print()** — everything goes through `logging`
8. **Magic numbers** — warns if hardcoded numbers aren't in `config.py`

## Model Architecture

```
Input (B, 3or4, H, H)
    |
    +-- MobileNetV3 Large (ImageNet pretrained)
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

- **Input:** 256x256 tiles, 3ch (RGB) or 4ch (RGB + Sobel edge map)
- **Anchors:** 3 per spatial location (aspect ratios 1.0, 0.5, 2.0)
- **Parameters:** ~5.07M, all trainable
- **Loss:** SSD multi-box loss with hard negative mining (3:1 neg:pos ratio)
- **Optimizer:** Adam with linear warmup (5 epochs) + cosine annealing

## Adaptation from Original Paper

**Source:** N. Lamb and S. Chuah, "A Strawberry Detection System Using Convolutional Neural Networks," IEEE Big Data 2018.

### What we kept from the original
- SSD-style single-shot detection
- Tile-based processing of large images
- Multi-scale feature extraction

### What we changed

| Component | Original (Strawberry) | Ours (Waldo) |
|-----------|----------------------|--------------|
| Backbone | 3-layer CNN | MobileNet V3 Large (pretrained) |
| Detection heads | 4 levels | 6 levels (+ 4-head ablation) |
| Input | Field video frames | Illustrated puzzle scans |
| Preprocessing | Single colour mask | Dual HSV (red+white) + Sobel edges + density filter |
| Input channels | 3 (RGB) | 3 or 4 (RGB + Sobel) |
| Anchors | Fixed scales | Linear scaling per level, 3 aspect ratios |
| Training | CUDA GPU | Apple MPS (M2) with CPU fallback |
| Regularization | None mentioned | Early stopping + LR warmup + grad clipping |

## One-Command Reproduction

```bash
# Full run — data, train, evaluate, ablation, plots
make reproduce

# Or use the pipeline script
./scripts/run_pipeline.sh              # full (200 epochs)
./scripts/run_pipeline.sh --quick      # quick test (10 epochs)
```

## References

[1] N. Lamb and S. Chuah, "A Strawberry Detection System Using Convolutional Neural Networks," *2018 IEEE International Conference on Big Data*, Seattle, WA, 2018, pp. 2515-2520.

[2] vc1492a, "Hey-Waldo," GitHub. https://github.com/vc1492a/Hey-Waldo

[3] bckenstler, "TheresWaldo," GitHub. https://github.com/bckenstler/TheresWaldo

[4] A. Howard et al., "Searching for MobileNetV3," *ICCV*, 2019, pp. 1314-1324.

[5] W. Liu et al., "SSD: Single Shot MultiBox Detector," *ECCV*, 2016, pp. 21-37.

## License

University course project for COMP 4360, University of Manitoba. Not for redistribution. Where's Waldo images are copyrighted by Martin Handford / Walker Books.
