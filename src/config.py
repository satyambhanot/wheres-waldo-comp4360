"""Central configuration for the Where's Waldo detection project.

All hyperparameters, file paths, and constants are defined here.
Every other module imports from this file — never hardcode values elsewhere.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# === Paths ===
PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
ANNOTATIONS: Path = DATA_RAW / "annotations"
RESULTS: Path = PROJECT_ROOT / "results"
CHECKPOINTS: Path = RESULTS / "checkpoints"

# === Dataset ===
RESOLUTIONS: list[int] = [64, 128, 256]
DEFAULT_RESOLUTION: int = 256
TILE_OVERLAP: float = 0.15
TRAIN_SPLIT: float = 0.6
VAL_SPLIT: float = 0.2
TEST_SPLIT: float = 0.2
COLOR_MODES: list[str] = ["color", "grayscale", "bw"]
DEFAULT_COLOR_MODE: str = "color"

# === Preprocessing ===
HSV_RED_LOW1: tuple[int, int, int] = (0, 70, 50)
HSV_RED_HIGH1: tuple[int, int, int] = (10, 255, 255)
HSV_RED_LOW2: tuple[int, int, int] = (170, 70, 50)
HSV_RED_HIGH2: tuple[int, int, int] = (180, 255, 255)
HSV_WHITE_LOW: tuple[int, int, int] = (0, 0, 200)
HSV_WHITE_HIGH: tuple[int, int, int] = (180, 30, 255)
DENSITY_THRESHOLD: float = 0.05

# === Model ===
NUM_CLASSES: int = 2
NUM_SSD_HEADS: int = 6
BACKBONE: str = "mobilenet_v3_large"
PRETRAINED: bool = True

# === Training ===
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 200
LR_SCHEDULER: str = "cosine"
NEG_POS_RATIO: int = 3

# === Evaluation ===
IOU_THRESHOLD: float = 0.5
NMS_THRESHOLD: float = 0.45
CONFIDENCE_THRESHOLD: float = 0.5

# === Device ===
DEVICE: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)
