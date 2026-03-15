import os
from pathlib import Path
import torch

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
ANNOTATIONS = DATA_RAW / 'annotations'
RESULTS = PROJECT_ROOT / 'results'
CHECKPOINTS = RESULTS / 'checkpoints'

# === Dataset ===
RESOLUTIONS = [64, 128, 256]
DEFAULT_RESOLUTION = 256
TILE_OVERLAP = 0.15
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
COLOR_MODES = ['color', 'grayscale', 'bw']
DEFAULT_COLOR_MODE = 'color'

# === Preprocessing ===
HSV_RED_LOW1 = (0, 70, 50)
HSV_RED_HIGH1 = (10, 255, 255)
HSV_RED_LOW2 = (170, 70, 50)
HSV_RED_HIGH2 = (180, 255, 255)
HSV_WHITE_LOW = (0, 0, 200)
HSV_WHITE_HIGH = (180, 30, 255)
DENSITY_THRESHOLD = 0.05

# === Model ===
NUM_CLASSES = 2
NUM_SSD_HEADS = 6
BACKBONE = 'mobilenet_v3_large'
PRETRAINED = True

# === Training ===
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
LR_SCHEDULER = 'cosine'
NEG_POS_RATIO = 3

# === Evaluation ===
IOU_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5

# === Device ===
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
