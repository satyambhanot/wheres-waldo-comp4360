# Where's Waldo Object Detection - COMP 4360

An SSD (Single Shot MultiBox Detector) pipeline for automatically locating Waldo in "Where's Waldo?" puzzle illustrations. Adapted from a strawberry detection architecture to the Waldo domain as part of a university machine learning course.

## Project Overview

This project builds an end-to-end object detection system that takes a high-resolution Waldo puzzle illustration and predicts bounding boxes around Waldo. The pipeline covers data preprocessing, multi-scale tiling, model architecture, training with hard negative mining, and evaluation with PASCAL VOC Average Precision.

### Key Design Decisions

- **Tiling approach**: Original illustrations are too large for direct input, so we tile them into 64x64, 128x128, and 256x256 patches with 15% overlap to prevent splitting Waldo at tile boundaries.
- **HSV + Sobel preprocessing**: Exploits Waldo's red-and-white striped pattern via dual HSV masking and Sobel edge detection, added as an optional 4th input channel.
- **Hard negative mining**: The dataset is severely imbalanced (~3% positive tiles at 256px), so the loss function selects the hardest background examples to prevent the model from trivially predicting "no Waldo" everywhere.
- **Ablation-ready**: The training pipeline accepts config overrides so experiments can systematically vary resolution, color mode, Sobel usage, and number of detection heads.

## Architecture

```
Input (B, 3or4, H, H)
    │
    ├── MobileNetV3 Large Backbone (pretrained on ImageNet)
    │   ├── Stage 1: features[0:4]   → 24ch   (H/2  × H/2)
    │   ├── Stage 2: features[4:7]   → 40ch   (H/4  × H/4)
    │   ├── Stage 3: features[7:13]  → 112ch  (H/8  × H/8)
    │   └── Stage 4: features[13:17] → 960ch  (H/16 × H/16)
    │
    ├── Extra Block 1: 960 → 256 → 512ch  (H/32 × H/32)
    └── Extra Block 2: 512 → 128 → 256ch  (H/64 × H/64)

Each level → Classification Head (3 anchors × 2 classes)
           → Regression Head    (3 anchors × 4 offsets)
```

| Input Size | Feature Map Sizes | Total Anchors | Parameters |
|------------|-------------------|---------------|------------|
| 64×64      | 16, 8, 4, 2, 1, 1 | 1,026 | 5.07M |
| 128×128    | 32, 16, 8, 4, 2, 1 | 4,095 | 5.07M |
| 256×256    | 64, 32, 16, 8, 4, 2 | 16,380 | 5.07M |

## Project Structure

```
wheres-waldo-comp4360/
├── src/
│   ├── __init__.py
│   ├── config.py          # Central configuration (paths, hyperparameters, constants)
│   ├── preprocess.py      # Tiling pipeline, HSV masking, Sobel edge detection
│   ├── dataset.py         # PyTorch Dataset and DataLoader with color modes
│   ├── model.py           # WaldoSSD architecture with MobileNetV3 backbone
│   └── train.py           # Training loop, SSD loss, anchor matching, AP evaluation
├── data/
│   ├── data.json          # Waldo locations per image at 128px grid
│   ├── raw/
│   │   ├── images/        # 19 original Waldo illustrations (.jpg)
│   │   └── annotations/   # Pascal VOC XML bounding box annotations
│   └── processed/
│       ├── splits.json    # Train/val/test split assignments
│       ├── 64/            # ~15,884 tiles at 64×64
│       ├── 128/           # ~3,789 tiles at 128×128
│       └── 256/           # ~844 tiles at 256×256
├── results/
│   ├── checkpoints/       # Saved model weights (.pth)
│   └── plots/             # Evaluation visualizations
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/
│   ├── check.sh           # Code quality checker (black, isort, flake8, mypy, ...)
│   └── commit.sh          # Commit helper
├── requirements.txt       # Pinned Python dependencies
├── pyproject.toml         # Tool configuration (black, isort, mypy)
├── CODING_STANDARDS.md    # Project coding conventions
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.11+
- macOS with Apple Silicon (M1/M2) for MPS acceleration, or any system with CPU/CUDA

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd wheres-waldo-comp4360

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Place the 19 original Waldo illustrations in `data/raw/images/` as `.jpg` files and their Pascal VOC annotations in `data/raw/annotations/` as `.xml` files. Then run the tiling pipeline:

```bash
python -m src.preprocess
```

This generates overlapping tiles at all three resolutions (64, 128, 256) and creates the train/val/test splits.

## Usage

### Training

Run the full training pipeline (200 epochs by default):

```bash
python -m src.train
```

The `__main__` block runs a quick 3-epoch sanity check. For a full training run, call `train()` directly or modify the config overrides:

```python
from src.train import train

# Default full training
best_ap = train()

# Ablation example: grayscale, no Sobel, 4 detection heads
best_ap = train(config_overrides={
    "color_mode": "grayscale",
    "use_sobel": False,
    "num_heads": 4,
    "num_epochs": 100,
})
```

### Sanity Checks

Each module has a `__main__` block for standalone verification:

```bash
python -m src.preprocess   # Run tiling pipeline
python -m src.dataset      # Verify Dataset and DataLoader shapes
python -m src.model        # Verify model forward pass and anchor generation
python -m src.train        # 3-epoch training test
```

## Training Details

### Dataset Splits

Splits are done by original image (not by tile) to prevent data leakage:

| Split | Images | Tiles (256px) |
|-------|--------|---------------|
| Train | 11 | 525 (16 positive, 509 negative) |
| Val   | 3  | 144 (6 positive, 138 negative) |
| Test  | 5  | 175 |

### Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Batch size | 32 | `config.BATCH_SIZE` |
| Learning rate | 1e-3 | `config.LEARNING_RATE` |
| Weight decay | 1e-4 | `config.WEIGHT_DECAY` |
| Epochs | 200 | `config.NUM_EPOCHS` |
| LR scheduler | Cosine annealing | `config.LR_SCHEDULER` |
| Neg:pos ratio | 3:1 | `config.NEG_POS_RATIO` |
| IoU threshold | 0.5 | `config.IOU_THRESHOLD` |
| NMS threshold | 0.45 | `config.NMS_THRESHOLD` |
| Confidence threshold | 0.5 | `config.CONFIDENCE_THRESHOLD` |
| Gradient clip norm | 5.0 | `train._GRAD_CLIP_MAX_NORM` |

### SSD Loss Function

The loss combines two components, both normalized by the number of positive anchors:

1. **Classification loss**: Cross-entropy over positive anchors + hard-mined negatives (top 3× negatives sorted by loss descending)
2. **Localization loss**: Smooth L1 over positive anchors only, with offsets encoded as `[dx, dy, dw, dh]` relative to matched anchors

### Preprocessing Pipeline

1. **HSV masking**: Dual-range red detection (hue 0-10 and 170-180) + white detection to isolate Waldo's stripes
2. **Sobel edges**: Gradient magnitude computed only within masked regions
3. **Density filtering**: Tiles with edge density below 5% are discarded as pure background
4. **Color modes**: Full color (RGB), grayscale (replicated 3ch), or binary black/white

### Ablation Variables

The architecture supports systematic ablation studies across:

| Variable | Options |
|----------|---------|
| Resolution | 64, 128, 256 |
| Color mode | color, grayscale, bw |
| Sobel channel | on (4ch) / off (3ch) |
| Detection heads | 6 (full) / 4 (coarse only) |

## Code Quality

The project enforces strict code quality standards via `scripts/check.sh`:

```bash
# Check without modifying files
./scripts/check.sh

# Auto-fix formatting issues
./scripts/check.sh --fix
```

### Checks Performed

1. **black** - Code formatting (100 char line length)
2. **isort** - Import ordering (black-compatible profile)
3. **flake8** - Linting (unused imports, undefined variables)
4. **mypy** - Type checking (all functions must have type hints)
5. **Docstring coverage** - Every public function and class requires a Google-style docstring
6. **Type hint coverage** - Every public function parameter and return must be annotated
7. **No print statements** - All output uses the `logging` module
8. **Magic number detection** - Warns about hardcoded numbers that should be in `config.py`

## Tech Stack

| Component | Library |
|-----------|---------|
| Deep learning | PyTorch 2.10, torchvision 0.25 |
| Backbone | MobileNet V3 Large (ImageNet pretrained) |
| Image processing | OpenCV (headless) |
| Compute device | Apple MPS (M2), CPU fallback |
| Formatting | black, isort |
| Linting | flake8, mypy |
| Visualization | matplotlib |
| Metrics | scikit-learn |

## License

University coursework - COMP 4360.
