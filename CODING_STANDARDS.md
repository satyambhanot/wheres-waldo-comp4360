# Coding Standards — Where's Waldo (COMP 4360)

## Tools (auto-enforced)
- **Formatter**: `black` (line length 100)
- **Import sorting**: `isort` (black-compatible profile)
- **Linter**: `flake8`
- **Type checking**: `mypy`
- **Editor**: VS Code with format-on-save enabled

## Run Before Every Commit
```bash
black src/
isort src/
flake8 src/
mypy src/
```

## Naming Conventions

| Element         | Convention           | Example                          |
|-----------------|----------------------|----------------------------------|
| Files/modules   | `snake_case.py`      | `preprocess.py`, `ssd_loss.py`   |
| Functions       | `snake_case`         | `tile_image()`, `compute_iou()`  |
| Classes         | `PascalCase`         | `WaldoDataset`, `WaldoSSD`       |
| Constants       | `UPPER_SNAKE_CASE`   | `BATCH_SIZE`, `DEVICE`           |
| Variables       | `snake_case`         | `train_loader`, `best_val_ap`    |
| Private methods | `_leading_underscore` | `_load_boxes()`, `_augment()`    |

## Type Hints — Required on ALL Functions
```python
# YES
def compute_iou(box_a: list[float], box_b: list[float]) -> float:

# YES — use numpy/torch types
def preprocess_tile(tile: np.ndarray, use_sobel: bool = True) -> Optional[np.ndarray]:

# YES — complex return types
def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:

# NO — never omit types
def compute_iou(box_a, box_b):
```

## Docstrings — Required on ALL Functions and Classes

Use Google-style docstrings:
```python
def tile_image(
    image_path: Path,
    annotation_path: Path,
    output_dir: Path,
    resolution: int,
) -> tuple[int, int]:
    """Tile a high-res Waldo image into fixed-size tiles with annotations.

    Takes a single original Waldo illustration and its Pascal VOC bounding
    box annotation, then generates overlapping tiles at the specified
    resolution. Each tile gets a YOLO-format label file.

    Args:
        image_path: Path to the original .jpg image.
        annotation_path: Path to the Pascal VOC .xml annotation.
        output_dir: Directory to save tiles and labels.
        resolution: Tile size in pixels (64, 128, or 256).

    Returns:
        Tuple of (positive_tile_count, negative_tile_count).

    Raises:
        FileNotFoundError: If image or annotation file doesn't exist.
        ValueError: If resolution is not in [64, 128, 256].
    """
```

### Class Docstrings
```python
class WaldoDataset(Dataset):
    """PyTorch Dataset for loading Waldo detection tiles.

    Loads pre-tiled images at a specified resolution, applies optional
    preprocessing (Sobel edge detection, color mode conversion), and
    returns (image_tensor, target_dict) pairs for SSD training.

    Attributes:
        resolution: Tile size in pixels.
        color_mode: One of 'color', 'grayscale', 'bw'.
        use_sobel: Whether to include Sobel edge channel.
        tiles: List of tile file paths in this split.
    """
```

## Logging — No Print Statements
```python
import logging

logger = logging.getLogger(__name__)

# YES
logger.info("Training epoch %d/%d, loss=%.4f", epoch, num_epochs, loss)
logger.warning("No positive tiles found for image %s", image_path.name)
logger.error("Failed to load annotation: %s", annotation_path)
logger.debug("Tile %s: IoU=%.3f, label=%d", tile_name, iou, label)

# NO
print(f"Training epoch {epoch}")
print("WARNING: no positive tiles")
```

### Logging Setup (in each script's `__main__` block)
```python
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
```

## Error Handling

### Validate inputs early with clear messages
```python
def tile_image(image_path: Path, resolution: int) -> None:
    """..."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Invalid resolution {resolution}, must be one of {RESOLUTIONS}")
```

### Use specific exception types
```python
# YES
raise ValueError("Resolution must be 64, 128, or 256")
raise FileNotFoundError(f"Missing annotation: {path}")

# NO
raise Exception("something went wrong")
```

### Log and re-raise in processing loops
```python
for image_path in image_paths:
    try:
        process_image(image_path)
    except Exception:
        logger.exception("Failed to process %s", image_path.name)
        raise
```

## File Structure

Every Python file follows this order:
```python
"""Module docstring — one-line summary.

Longer description if needed. Explains what this module does
and how it fits into the pipeline.
"""

# Standard library imports
import logging
import os
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import torch

# Local imports
from src.config import DEVICE, RESOLUTIONS

logger = logging.getLogger(__name__)
```

## Constants

ALL configurable values live in `src/config.py`. Never hardcode numbers in other files:
```python
# YES
from src.config import BATCH_SIZE, DENSITY_THRESHOLD

# NO
batch_size = 32  # magic number in training script
```

## Git Commits

Format: `<type>: <short description>`

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

Examples:
- `feat: add tiling pipeline with overlap support`
- `fix: correct anchor matching IoU calculation`
- `refactor: extract HSV masking into separate function`
- `docs: add docstrings to WaldoDataset`
- `test: add sanity check for DataLoader output shapes`
- `chore: update requirements.txt`
