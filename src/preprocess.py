"""Tiling pipeline and preprocessing for detection training data.

Takes 19 original Where's Waldo illustrations with Pascal VOC
bounding box annotations and generates overlapping tiles at
64/128/256 resolutions with per-tile YOLO-format labels. Also
provides HSV masking, Sobel edge detection, and density filtering
for the preprocessing pipeline used during training.
"""

import json
import logging
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import (
    ANNOTATIONS,
    DATA_PROCESSED,
    DATA_RAW,
    DENSITY_THRESHOLD,
    HSV_RED_HIGH1,
    HSV_RED_HIGH2,
    HSV_RED_LOW1,
    HSV_RED_LOW2,
    HSV_WHITE_HIGH,
    HSV_WHITE_LOW,
    OVERLAP_THRESHOLD,
    RANDOM_SEED,
    RESOLUTIONS,
    SOBEL_ACTIVE_THRESHOLD,
    TILE_OVERLAP,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ANNOTATION PARSING
# =============================================================================


def parse_voc_annotation(xml_path: Path) -> dict:
    """Parse a Pascal VOC XML annotation file.

    Extracts image dimensions and all object bounding boxes from a
    standard Pascal VOC format XML file.

    Args:
        xml_path: Path to the .xml annotation file.

    Returns:
        Dictionary with keys:
            - 'filename': Original image filename (str).
            - 'width': Image width in pixels (int).
            - 'height': Image height in pixels (int).
            - 'boxes': List of [xmin, ymin, xmax, ymax] pixel coords.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        ValueError: If the XML is missing required elements.
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"Annotation not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- Extract image metadata ---
    filename_elem = root.find("filename")
    size_elem = root.find("size")
    if filename_elem is None or size_elem is None:
        raise ValueError(f"Malformed VOC XML (missing filename or size): {xml_path}")

    width = int(size_elem.findtext("width", "0"))
    height = int(size_elem.findtext("height", "0"))
    if width == 0 or height == 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}: {width}x{height}")

    # --- Extract bounding boxes ---
    boxes: list[list[int]] = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            logger.warning("Object without bndbox in %s, skipping", xml_path.name)
            continue

        xmin = int(float(bndbox.findtext("xmin", "0")))
        ymin = int(float(bndbox.findtext("ymin", "0")))
        xmax = int(float(bndbox.findtext("xmax", "0")))
        ymax = int(float(bndbox.findtext("ymax", "0")))
        boxes.append([xmin, ymin, xmax, ymax])

    logger.debug(
        "Parsed %s: %dx%d, %d boxes",
        xml_path.name,
        width,
        height,
        len(boxes),
    )

    return {
        "filename": filename_elem.text or "",
        "width": width,
        "height": height,
        "boxes": boxes,
    }


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================


def compute_iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Both boxes are in [xmin, ymin, xmax, ymax] pixel coordinate format.
    Returns 0.0 if there is no overlap.

    Args:
        box_a: First bounding box [xmin, ymin, xmax, ymax].
        box_b: Second bounding box [xmin, ymin, xmax, ymax].

    Returns:
        IoU value in range [0.0, 1.0].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_overlap_ratio(waldo_box: list[int], tile_box: list[int]) -> float:
    """Compute what fraction of the Waldo box is inside the tile.

    This is different from IoU — it measures how much of Waldo is
    visible in this tile, which is more relevant for detection labels
    than symmetric IoU. A tile that contains 80% of Waldo should be
    labeled positive even if the tile is much larger than Waldo.

    Args:
        waldo_box: Waldo bounding box [xmin, ymin, xmax, ymax].
        tile_box: Tile bounding box [xmin, ymin, xmax, ymax].

    Returns:
        Fraction of Waldo's area that overlaps with the tile [0.0, 1.0].
    """
    x1 = max(waldo_box[0], tile_box[0])
    y1 = max(waldo_box[1], tile_box[1])
    x2 = min(waldo_box[2], tile_box[2])
    y2 = min(waldo_box[3], tile_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    waldo_area = (waldo_box[2] - waldo_box[0]) * (waldo_box[3] - waldo_box[1])

    if waldo_area == 0:
        return 0.0

    return intersection / waldo_area


def clip_box_to_tile(
    waldo_box: list[int],
    tile_box: list[int],
) -> list[int]:
    """Clip a Waldo bounding box to tile boundaries and make tile-relative.

    Converts absolute pixel coordinates to coordinates relative to the
    tile's top-left corner. Clips any portion that extends outside the tile.

    Args:
        waldo_box: Waldo box in absolute coords [xmin, ymin, xmax, ymax].
        tile_box: Tile box in absolute coords [xmin, ymin, xmax, ymax].

    Returns:
        Clipped box relative to tile origin [xmin, ymin, xmax, ymax].
    """
    rel_xmin = max(waldo_box[0], tile_box[0]) - tile_box[0]
    rel_ymin = max(waldo_box[1], tile_box[1]) - tile_box[1]
    rel_xmax = min(waldo_box[2], tile_box[2]) - tile_box[0]
    rel_ymax = min(waldo_box[3], tile_box[3]) - tile_box[1]

    return [rel_xmin, rel_ymin, rel_xmax, rel_ymax]


def format_yolo_line(class_id: int, box: list[int], resolution: int) -> str:
    """Convert a pixel-coordinate box to YOLO annotation format.

    YOLO format: class_id center_x center_y width height
    All values normalized to [0, 1] relative to tile dimensions.

    Args:
        class_id: Object class (0 for Waldo).
        box: Box in tile-relative pixel coords [xmin, ymin, xmax, ymax].
        resolution: Tile size in pixels (used for normalization).

    Returns:
        YOLO-format annotation string.
    """
    cx = (box[0] + box[2]) / 2.0 / resolution
    cy = (box[1] + box[3]) / 2.0 / resolution
    bw = (box[2] - box[0]) / resolution
    bh = (box[3] - box[1]) / resolution

    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# =============================================================================
# TILING PIPELINE
# =============================================================================


def tile_single_image(
    image_path: Path,
    annotation_path: Path,
    output_dir: Path,
    resolution: int,
) -> tuple[int, int]:
    """Tile a single high-res Waldo image into fixed-size detection tiles.

    Slides a window of size (resolution x resolution) across the original
    image with configurable overlap. Each tile is saved as a .jpg file,
    and its YOLO-format annotation is saved as a matching .txt file.
    Negative tiles (no Waldo) get empty .txt files.

    Args:
        image_path: Path to the original .jpg image.
        annotation_path: Path to the Pascal VOC .xml annotation.
        output_dir: Directory to save tiles and labels.
        resolution: Tile size in pixels (64, 128, or 256).

    Returns:
        Tuple of (positive_tile_count, negative_tile_count).

    Raises:
        FileNotFoundError: If image or annotation doesn't exist.
        ValueError: If resolution is not in the allowed set.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Invalid resolution {resolution}, must be one of {RESOLUTIONS}")

    # --- Load image and annotation ---
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    annotation = parse_voc_annotation(annotation_path)
    waldo_boxes = annotation["boxes"]
    img_h, img_w = img.shape[:2]

    # Overlap prevents Waldo from being split at tile boundaries
    stride = int(resolution * (1.0 - TILE_OVERLAP))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = image_path.stem
    tile_id = 0
    pos_count = 0
    neg_count = 0

    # --- Slide window across image ---
    for y in range(0, img_h - resolution + 1, stride):
        for x in range(0, img_w - resolution + 1, stride):
            tile = img[y : y + resolution, x : x + resolution]
            tile_box = [x, y, x + resolution, y + resolution]

            # --- Check each Waldo box for overlap with this tile ---
            label_lines: list[str] = []
            for waldo_box in waldo_boxes:
                overlap = compute_overlap_ratio(waldo_box, tile_box)

                if overlap >= OVERLAP_THRESHOLD:
                    clipped = clip_box_to_tile(waldo_box, tile_box)
                    yolo_line = format_yolo_line(0, clipped, resolution)
                    label_lines.append(yolo_line)

            # --- Save tile image and label ---
            tile_name = f"{base_name}_tile_{tile_id:05d}"
            cv2.imwrite(str(output_dir / f"{tile_name}.jpg"), tile)

            with open(output_dir / f"{tile_name}.txt", "w") as f:
                f.write("\n".join(label_lines))

            if label_lines:
                pos_count += 1
            else:
                neg_count += 1

            tile_id += 1

    logger.info(
        "Image %s @ %dpx: %d tiles (%d positive, %d negative)",
        base_name,
        resolution,
        tile_id,
        pos_count,
        neg_count,
    )

    return pos_count, neg_count


# =============================================================================
# TRAIN / VAL / TEST SPLIT
# =============================================================================


def create_splits(processed_dir: Path) -> dict[str, list[str]]:
    """Create reproducible train/val/test splits by original image.

    Splits are done by original image name (not by tile) to prevent
    data leakage — tiles from the same source image must all be in
    the same split.

    Args:
        processed_dir: Path to data/processed/ directory. The splits
            file is saved here as splits.json.

    Returns:
        Dictionary mapping split name to list of original image stems.
        Example: {'train': ['1','3','5',...], 'val': ['2','7'], 'test': ['4','9']}.
    """
    # --- Collect unique original image names from 256px tiles ---
    tile_dir = processed_dir / str(RESOLUTIONS[-1])
    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}. Run tiling pipeline first.")

    originals = sorted({f.stem.rsplit("_tile_", 1)[0] for f in tile_dir.glob("*.jpg")})

    if not originals:
        raise ValueError(f"No tiles found in {tile_dir}")

    # --- Shuffle with fixed seed for reproducibility ---
    random.seed(RANDOM_SEED)
    random.shuffle(originals)

    n = len(originals)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    splits = {
        "train": originals[:n_train],
        "val": originals[n_train : n_train + n_val],
        "test": originals[n_train + n_val :],
    }

    # --- Save to JSON ---
    splits_path = processed_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    for split_name, names in splits.items():
        # Count tiles belonging to this split across all resolutions
        tile_count = sum(
            1
            for tile_file in tile_dir.glob("*.jpg")
            if tile_file.stem.rsplit("_tile_", 1)[0] in set(names)
        )
        logger.info(
            "Split '%s': %d originals -> %d tiles",
            split_name,
            len(names),
            tile_count,
        )

    logger.info("Splits saved to %s", splits_path)
    return splits


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================


def run_tiling_pipeline(
    resolutions: Optional[list[int]] = None,
) -> dict[int, tuple[int, int]]:
    """Run the full tiling pipeline across all images and resolutions.

    Processes every original image in data/raw/images/ with its
    corresponding annotation, generating tiles at each specified
    resolution. Then creates train/val/test splits.

    Args:
        resolutions: List of tile sizes to generate. Defaults to
            config.RESOLUTIONS ([64, 128, 256]).

    Returns:
        Dictionary mapping resolution to (total_positive, total_negative)
        tile counts.

    Raises:
        FileNotFoundError: If no images or annotations are found.
    """
    if resolutions is None:
        resolutions = RESOLUTIONS

    images_dir = DATA_RAW / "images"
    annotations_dir = ANNOTATIONS

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {images_dir}")

    logger.info("Found %d original images", len(image_paths))
    logger.info("Tiling at resolutions: %s", resolutions)
    logger.info("Tile overlap: %.0f%%", TILE_OVERLAP * 100)

    results: dict[int, tuple[int, int]] = {}

    for resolution in resolutions:
        output_dir = DATA_PROCESSED / str(resolution)
        total_pos = 0
        total_neg = 0

        for image_path in image_paths:
            annotation_path = annotations_dir / f"{image_path.stem}.xml"

            if not annotation_path.exists():
                logger.warning(
                    "No annotation for %s, skipping",
                    image_path.name,
                )
                continue

            try:
                pos, neg = tile_single_image(image_path, annotation_path, output_dir, resolution)
                total_pos += pos
                total_neg += neg
            except Exception:
                logger.exception("Failed to tile %s", image_path.name)
                raise

        pos_ratio = total_pos / max(1, total_pos + total_neg) * 100
        logger.info(
            "Resolution %d: %d positive, %d negative (%.1f%% positive)",
            resolution,
            total_pos,
            total_neg,
            pos_ratio,
        )
        results[resolution] = (total_pos, total_neg)

    # --- Create splits after tiling ---
    logger.info("Creating train/val/test splits...")
    create_splits(DATA_PROCESSED)

    return results


# =============================================================================
# HSV MASKING AND SOBEL EDGE PREPROCESSING
# =============================================================================


def dual_hsv_mask(tile: np.ndarray) -> np.ndarray:
    """Create a binary mask highlighting red and white regions.

    Targets Waldo's distinctive red-and-white striped pattern using
    dual HSV ranges. Red requires two ranges because it wraps around
    in HSV space (hue 0-10 and 170-180).

    Args:
        tile: BGR image tile as numpy array (H, W, 3).

    Returns:
        Binary mask (H, W) where 255 = red or white region.
    """
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

    # --- Red mask (two ranges because red wraps in HSV) ---
    red_low = cv2.inRange(hsv, np.array(HSV_RED_LOW1), np.array(HSV_RED_HIGH1))
    red_high = cv2.inRange(hsv, np.array(HSV_RED_LOW2), np.array(HSV_RED_HIGH2))
    red_mask = cv2.bitwise_or(red_low, red_high)

    # --- White mask (high value, low saturation) ---
    white_mask = cv2.inRange(hsv, np.array(HSV_WHITE_LOW), np.array(HSV_WHITE_HIGH))

    combined = cv2.bitwise_or(red_mask, white_mask)
    return combined


def sobel_edges(tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply Sobel edge detection to masked regions of a tile.

    Computes gradient magnitude using Sobel operators in X and Y
    directions, but only within the HSV-masked region. This highlights
    Waldo's stripe edges while ignoring background texture.

    Args:
        tile: BGR image tile as numpy array (H, W, 3).
        mask: Binary mask from dual_hsv_mask (H, W).

    Returns:
        Edge magnitude image (H, W) normalized to [0, 255] uint8.
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

    # Apply mask so edges are only detected in red/white regions
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    sobel_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize to uint8 range
    max_val = magnitude.max()
    if max_val > 0:
        result: np.ndarray = (magnitude / max_val * 255).astype(np.uint8)
    else:
        result = magnitude.astype(np.uint8)

    return result


def edge_density(sobel_output: np.ndarray) -> float:
    """Compute fraction of pixels with significant edge energy.

    Used as a quick filter to discard tiles that are pure background
    (no interesting edges). Tiles below DENSITY_THRESHOLD are skipped
    during training to reduce false positives and computation.

    Args:
        sobel_output: Edge magnitude image from sobel_edges (H, W).

    Returns:
        Fraction of pixels above the activity threshold [0.0, 1.0].
    """
    active_pixels = int(np.sum(sobel_output > SOBEL_ACTIVE_THRESHOLD))
    total_pixels = sobel_output.shape[0] * sobel_output.shape[1]

    if total_pixels == 0:
        return 0.0

    return float(active_pixels / total_pixels)


def preprocess_tile(
    tile: np.ndarray,
    use_sobel: bool = True,
) -> Optional[np.ndarray]:
    """Run the full preprocessing pipeline on a single tile.

    Applies dual HSV masking and Sobel edge detection, then checks
    edge density. If density is below threshold, returns None to
    indicate this tile should be skipped. Otherwise returns the
    processed image array.

    Args:
        tile: BGR image tile as numpy array (H, W, 3).
        use_sobel: If True, stack Sobel magnitude as 4th channel.
            If False, return normalized RGB only.

    Returns:
        Processed image as float32 array with shape (3, H, W) or
        (4, H, W) if use_sobel=True. Returns None if the tile
        fails the density filter.
    """
    mask = dual_hsv_mask(tile)
    sobel = sobel_edges(tile, mask)
    density = edge_density(sobel)

    if density < DENSITY_THRESHOLD:
        return None

    # --- Convert BGR to RGB and normalize to [0, 1] ---
    rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))  # (H,W,3) -> (3,H,W)

    if use_sobel:
        sobel_norm = sobel.astype(np.float32) / 255.0
        sobel_channel = np.expand_dims(sobel_norm, axis=0)  # (1,H,W)
        return np.concatenate([rgb, sobel_channel], axis=0)  # (4,H,W)

    return rgb  # (3,H,W)


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_tiling_pipeline()
