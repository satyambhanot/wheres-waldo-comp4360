"""PyTorch Dataset and DataLoader for the tiled Waldo images.

Loads tiles from disk, applies colour mode conversion and optional
Sobel edge stacking, and serves (image, target) pairs for training.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import (
    BATCH_SIZE,
    COLOR_MODES,
    DATA_PROCESSED,
    DEFAULT_COLOR_MODE,
    DEFAULT_RESOLUTION,
    DENSITY_THRESHOLD,
    RESOLUTIONS,
)
from src.preprocess import dual_hsv_mask, edge_density, sobel_edges

logger = logging.getLogger(__name__)


class WaldoDataset(Dataset):
    """Loads Waldo tiles from disk and serves them to the model.

    Each tile comes back as an image tensor plus a dictionary of
    bounding-box targets.  Optional knobs let you toggle colour
    modes, Sobel edge features, and data augmentation.

    Attributes:
        resolution: Tile size in pixels (64, 128, or 256).
        color_mode: One of 'color', 'grayscale', 'bw'.
        use_sobel: Whether to include Sobel edge channel.
        augment: Whether to apply data augmentation.
        tiles: List of tile file paths in this split.
    """

    def __init__(
        self,
        split: str = "train",
        resolution: int = DEFAULT_RESOLUTION,
        color_mode: str = DEFAULT_COLOR_MODE,
        use_sobel: bool = True,
        use_density_filter: bool = True,
        augment: bool = False,
    ) -> None:
        """Set up the dataset for a specific split and configuration.

        Args:
            split: One of 'train', 'val', or 'test'.
            resolution: Tile size in pixels (64, 128, or 256).
            color_mode: One of 'color', 'grayscale', 'bw'.
            use_sobel: Whether to add Sobel magnitude as 4th channel.
            use_density_filter: Whether to throw away boring background tiles.
            augment: Whether to apply random augmentation (training only).

        Raises:
            FileNotFoundError: If splits.json or tile directory doesn't exist.
            ValueError: If split, resolution, or color_mode is invalid.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{split}', must be 'train', 'val', or 'test'")
        if resolution not in RESOLUTIONS:
            raise ValueError(f"Invalid resolution {resolution}, must be one of {RESOLUTIONS}")
        if color_mode not in COLOR_MODES:
            raise ValueError(f"Invalid color_mode '{color_mode}', must be one of {COLOR_MODES}")

        self.resolution = resolution
        self.color_mode = color_mode
        self.use_sobel = use_sobel
        self.augment = augment

        # Figure out which original images belong to this split
        splits_path = DATA_PROCESSED / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"splits.json not found at {splits_path}. Run tiling pipeline first."
            )

        with open(splits_path) as f:
            splits = json.load(f)

        allowed_originals = set(splits[split])

        # Gather every tile that came from one of this split's images
        tile_dir = DATA_PROCESSED / str(resolution)
        if not tile_dir.exists():
            raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

        all_tiles = sorted(tile_dir.glob("*.jpg"))
        split_tiles = [
            tp for tp in all_tiles if tp.stem.rsplit("_tile_", 1)[0] in allowed_originals
        ]

        # Optionally drop tiles that are mostly empty background
        if use_density_filter:
            density_path = tile_dir / "density.json"
            self.tiles = self._apply_density_filter(split_tiles, density_path)
        else:
            self.tiles = split_tiles

        # Quick tally for the log message
        waldo_count = sum(1 for tp in self.tiles if self._has_labels(tp))
        background_count = len(self.tiles) - waldo_count

        logger.info(
            "%s set (%dpx, %s, sobel=%s): %d tiles (%d waldo, %d background)",
            split,
            resolution,
            color_mode,
            use_sobel,
            len(self.tiles),
            waldo_count,
            background_count,
        )

    def _apply_density_filter(
        self, tile_paths: list[Path], density_path: Path
    ) -> list[Path]:
        """Throw away tiles that are mostly blank background.

        We always keep tiles that contain Waldo, no matter how low
        their edge density is.  For background tiles, we only keep the
        ones that have enough going on to be worth training on.

        Uses pre-computed density scores from the tiling pipeline when
        available; falls back to computing on the fly if the cache is
        missing.

        Args:
            tile_paths: List of tile image paths to filter.
            density_path: Path to the density.json cache file.

        Returns:
            Filtered list of tile paths.
        """
        # Load cached density scores if available
        density_cache: dict[str, float] = {}
        if density_path.exists():
            with open(density_path) as f:
                density_cache = json.load(f)

        use_cache = len(density_cache) > 0
        if not use_cache:
            logger.warning("No density cache at %s, computing on the fly", density_path)

        filtered: list[Path] = []

        for tp in tile_paths:
            # Always keep tiles that contain Waldo
            if self._has_labels(tp):
                filtered.append(tp)
                continue

            if use_cache:
                density = density_cache.get(tp.stem, 0.0)
            else:
                img = cv2.imread(str(tp))
                if img is None:
                    logger.warning("Failed to read tile for density check: %s", tp.name)
                    continue
                mask = dual_hsv_mask(img)
                sobel = sobel_edges(img, mask)
                density = edge_density(sobel)

            if density >= DENSITY_THRESHOLD:
                filtered.append(tp)

        logger.info(
            "Density filter: %d -> %d tiles (%.0f%% removed%s)",
            len(tile_paths),
            len(filtered),
            (1 - len(filtered) / max(1, len(tile_paths))) * 100,
            ", cached" if use_cache else "",
        )
        return filtered

    def _has_labels(self, tile_path: Path) -> bool:
        """Does this tile actually contain Waldo?

        Args:
            tile_path: Path to the tile .jpg file.

        Returns:
            True if the matching .txt label file is non-empty.
        """
        label_path = tile_path.with_suffix(".txt")
        if not label_path.exists():
            return False
        return label_path.stat().st_size > 0

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Grab one tile and its labels, ready for the model.

        Args:
            idx: Index into the tile list.

        Returns:
            Tuple of (image_tensor, target_dict) where:
                - image_tensor: (C, H, W) float32, C=3 (RGB) or C=4 (RGB+Sobel).
                - target_dict: {'boxes': (N, 4) float32, 'labels': (N,) int64}.
        """
        tile_path = self.tiles[idx]
        raw_img = cv2.imread(str(tile_path))
        if raw_img is None:
            raise ValueError(f"Failed to read tile: {tile_path}")
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # Read Waldo bounding boxes (if any) from the label file
        boxes = self._load_boxes(tile_path)

        # Compute Sobel BEFORE switching colour modes, because the
        # HSV mask needs real colour information to work properly.
        sobel_channel: Optional[np.ndarray] = None
        if self.use_sobel:
            bgr_for_sobel = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = dual_hsv_mask(bgr_for_sobel)
            sobel_out = sobel_edges(bgr_for_sobel, mask)
            sobel_channel = sobel_out.astype(np.float32) / 255.0

        # Random flips / jitter for training variety
        if self.augment:
            img, boxes, sobel_channel = self._augment(img, boxes, sobel_channel)

        # Switch to greyscale / B&W if requested
        img = self._apply_color_mode(img)

        # Normalise pixels to 0-1 and rearrange to channels-first
        img_float = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1)))

        # Append the edge map as a 4th channel when Sobel is enabled
        if self.use_sobel and sobel_channel is not None:
            sobel_tensor = torch.from_numpy(sobel_channel).unsqueeze(0)
            img_tensor = torch.cat([img_tensor, sobel_tensor], dim=0)

        # Package the labels the way PyTorch detection models expect
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones(len(boxes), dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.long)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        return img_tensor, target

    def _load_boxes(self, tile_path: Path) -> list[list[float]]:
        """Read the YOLO label file and convert to pixel-coordinate boxes.

        The label file stores normalised centre+size; we convert back
        to the corner format [xmin, ymin, xmax, ymax] in pixels.

        Args:
            tile_path: Path to the tile .jpg file. The label file sits
                right next to it with a .txt extension.

        Returns:
            List of boxes as [xmin, ymin, xmax, ymax] in pixel coords.
        """
        label_path = tile_path.with_suffix(".txt")
        boxes: list[list[float]] = []

        if not label_path.exists():
            return boxes

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                centre_x = float(parts[1])
                centre_y = float(parts[2])
                box_w = float(parts[3])
                box_h = float(parts[4])
                size = self.resolution
                xmin = (centre_x - box_w / 2.0) * size
                ymin = (centre_y - box_h / 2.0) * size
                xmax = (centre_x + box_w / 2.0) * size
                ymax = (centre_y + box_h / 2.0) * size
                boxes.append([xmin, ymin, xmax, ymax])

        return boxes

    def _apply_color_mode(self, img: np.ndarray) -> np.ndarray:
        """Switch to greyscale or black-and-white if requested.

        The backbone always expects three channels, so single-channel
        results get copied across all three slots.

        Args:
            img: RGB image array (H, W, 3).

        Returns:
            Processed image (H, W, 3) in the configured color mode.
        """
        if self.color_mode == "color":
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.color_mode == "grayscale":
            return np.stack([gray, gray, gray], axis=-1)

        if self.color_mode == "bw":
            _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            return np.stack([bw, bw, bw], axis=-1)

        logger.warning("Unknown color mode '%s', falling back to color", self.color_mode)
        return img

    def _augment(
        self,
        img: np.ndarray,
        boxes: list[list[float]],
        sobel_channel: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list[list[float]], Optional[np.ndarray]]:
        """Add a bit of randomness so the model doesn't memorise tiles.

        Right now we randomly flip horizontally and jitter the colour.
        Both the boxes and the Sobel map are adjusted to match.

        Args:
            img: RGB image array (H, W, 3).
            boxes: List of [xmin, ymin, xmax, ymax] in pixel coords.
            sobel_channel: Optional Sobel edge map (H, W) float32.

        Returns:
            Tuple of (augmented_image, augmented_boxes, augmented_sobel).
        """
        h, w = img.shape[:2]

        # Coin flip: mirror the tile left-to-right
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
            boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]
            if sobel_channel is not None:
                sobel_channel = np.fliplr(sobel_channel).copy()

        # Coin flip: slightly tweak saturation and brightness
        if self.color_mode == "color" and np.random.random() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
            hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img, boxes, sobel_channel


# ---- DataLoader helpers ----


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Stack images into a batch while keeping targets as a list.

    PyTorch's default collate chokes because different tiles have
    different numbers of bounding boxes.  We just stack the images
    and leave the target dicts in a plain list.

    Args:
        batch: List of (image_tensor, target_dict) tuples from Dataset.

    Returns:
        Tuple of (batched_images, list_of_targets) where:
            - batched_images: (B, C, H, W) float32 tensor.
            - list_of_targets: List of B target dicts.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def get_dataloader(
    split: str = "train",
    resolution: int = DEFAULT_RESOLUTION,
    color_mode: str = DEFAULT_COLOR_MODE,
    use_sobel: bool = True,
    use_density_filter: bool = True,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> DataLoader:
    """One-liner to build a ready-to-go DataLoader for any split.

    Training gets shuffled and augmented; validation and test do not.

    Args:
        split: One of 'train', 'val', or 'test'.
        resolution: Tile size in pixels.
        color_mode: One of 'color', 'grayscale', 'bw'.
        use_sobel: Whether to add Sobel edge channel.
        use_density_filter: Whether to skip low-density tiles.
        batch_size: Number of tiles per batch.
        num_workers: Number of parallel data loading workers.
            Use 0 on macOS/MPS to avoid multiprocessing issues.

    Returns:
        Configured DataLoader ready for training or evaluation.
    """
    is_train = split == "train"

    dataset = WaldoDataset(
        split=split,
        resolution=resolution,
        color_mode=color_mode,
        use_sobel=use_sobel,
        use_density_filter=use_density_filter,
        augment=is_train,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # MPS doesn't benefit from pin_memory
    )

    logger.info(
        "Created %s DataLoader: %d tiles, batch_size=%d, shuffle=%s",
        split,
        len(dataset),
        batch_size,
        is_train,
    )

    return loader


# ---- Main ----


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Running Dataset sanity checks...")

    # --- Test each split ---
    for split_name in ("train", "val", "test"):
        loader = get_dataloader(
            split=split_name,
            resolution=DEFAULT_RESOLUTION,
            use_sobel=True,
            use_density_filter=False,  # Skip density filter for speed
            batch_size=2,
        )

        # Grab one batch
        images, targets = next(iter(loader))

        logger.info(
            "  %s batch: images=%s, dtype=%s",
            split_name,
            list(images.shape),
            images.dtype,
        )

        for i, tgt in enumerate(targets):
            logger.info(
                "    sample %d: boxes=%s, labels=%s",
                i,
                list(tgt["boxes"].shape),
                list(tgt["labels"].shape),
            )

    # --- Test color modes ---
    for mode in ("color", "grayscale", "bw"):
        ds = WaldoDataset(
            split="train",
            color_mode=mode,
            use_sobel=False,
            use_density_filter=False,
        )
        img, tgt = ds[0]
        logger.info("  color_mode=%s, sobel=False: shape=%s", mode, list(img.shape))

    # --- Test without Sobel ---
    ds_no_sobel = WaldoDataset(
        split="train",
        use_sobel=False,
        use_density_filter=False,
    )
    img_ns, _ = ds_no_sobel[0]
    logger.info(
        "  sobel=False: shape=%s (should be [3, %d, %d])",
        list(img_ns.shape),
        DEFAULT_RESOLUTION,
        DEFAULT_RESOLUTION,
    )

    # --- Test with Sobel ---
    ds_sobel = WaldoDataset(
        split="train",
        use_sobel=True,
        use_density_filter=False,
    )
    img_s, _ = ds_sobel[0]
    logger.info(
        "  sobel=True: shape=%s (should be [4, %d, %d])",
        list(img_s.shape),
        DEFAULT_RESOLUTION,
        DEFAULT_RESOLUTION,
    )

    logger.info("All sanity checks passed!")
