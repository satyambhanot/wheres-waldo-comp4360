"""Tiling pipeline for the Where's Waldo detection project.

Slices each of the 19 original puzzle images into overlapping tiles
at 64/128/256 px with matching YOLO label files. Also provides HSV
colour masking and Sobel edge detection used during training.
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


# ---- Annotation parsing ----


def parse_voc_annotation(xml_path: Path) -> dict:
    """Read a Pascal VOC XML file and pull out the bounding boxes.

    Each annotation file tells us how big the image is and where
    Waldo (or any other labelled object) sits inside it.

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

    # Grab the basic image info (name, width, height)
    filename_elem = root.find("filename")
    size_elem = root.find("size")
    if filename_elem is None or size_elem is None:
        raise ValueError(f"Malformed VOC XML (missing filename or size): {xml_path}")

    width = int(size_elem.findtext("width", "0"))
    height = int(size_elem.findtext("height", "0"))
    if width == 0 or height == 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}: {width}x{height}")

    # Now collect every bounding box in the file
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


# ---- Geometry helpers ----


def compute_overlap_ratio(waldo_box: list[int], tile_box: list[int]) -> float:
    """What fraction of Waldo's area falls inside this tile?

    Not the same as IoU — this only cares about how much of Waldo
    is visible, regardless of how big the tile is.

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
    """Clip Waldo's box to the tile boundary and convert to tile-local coords.

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


def format_yolo_line(class_id: int, box: list[int], tile_size: int) -> str:
    """Turn a pixel-coordinate box into a YOLO annotation line.

    YOLO wants: class_id  centre_x  centre_y  width  height
    with every value normalised to [0, 1] relative to the tile.

    Args:
        class_id: Object class (0 for Waldo).
        box: Box in tile-relative pixel coords [xmin, ymin, xmax, ymax].
        tile_size: Tile size in pixels (used for normalization).

    Returns:
        YOLO-format annotation string.
    """
    centre_x = (box[0] + box[2]) / 2.0 / tile_size
    centre_y = (box[1] + box[3]) / 2.0 / tile_size
    box_width = (box[2] - box[0]) / tile_size
    box_height = (box[3] - box[1]) / tile_size

    return f"{class_id} {centre_x:.6f} {centre_y:.6f} {box_width:.6f} {box_height:.6f}"


# ---- Tiling ----


def tile_single_image(
    image_path: Path,
    annotation_path: Path,
    output_dir: Path,
    resolution: int,
) -> tuple[int, int, dict[str, float]]:
    """Slice one full-size image into square tiles with YOLO labels.

    Slides a window across the image with some overlap so Waldo
    doesn't get cut in half at tile boundaries.

    Args:
        image_path: Path to the original .jpg image.
        annotation_path: Path to the Pascal VOC .xml annotation.
        output_dir: Directory to save tiles and labels.
        resolution: Tile size in pixels (64, 128, or 256).

    Returns:
        Tuple of (positive_tile_count, negative_tile_count, density_scores)
        where density_scores maps tile names to their edge density values.

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

    # Load the full-size image and its Waldo locations
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    annotation = parse_voc_annotation(annotation_path)
    waldo_boxes = annotation["boxes"]
    img_h, img_w = img.shape[:2]

    # A little overlap between neighbouring tiles keeps Waldo whole
    stride = int(resolution * (1.0 - TILE_OVERLAP))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = image_path.stem
    tile_number = 0
    waldo_tiles = 0
    background_tiles = 0
    density_scores: dict[str, float] = {}

    # Walk the sliding window across the whole image
    for y in range(0, img_h - resolution + 1, stride):
        for x in range(0, img_w - resolution + 1, stride):
            tile = img[y : y + resolution, x : x + resolution]
            tile_box = [x, y, x + resolution, y + resolution]

            # Does this tile contain (enough of) Waldo?
            label_lines: list[str] = []
            for waldo_box in waldo_boxes:
                overlap = compute_overlap_ratio(waldo_box, tile_box)

                if overlap >= OVERLAP_THRESHOLD:
                    clipped = clip_box_to_tile(waldo_box, tile_box)
                    yolo_line = format_yolo_line(0, clipped, resolution)
                    label_lines.append(yolo_line)

            # Write the tile image and its label file to disk
            tile_name = f"{image_name}_tile_{tile_number:05d}"
            cv2.imwrite(str(output_dir / f"{tile_name}.jpg"), tile)

            with open(output_dir / f"{tile_name}.txt", "w") as f:
                f.write("\n".join(label_lines))

            # Pre-compute Sobel edges and density so the dataset can
            # load the cached Sobel channel and filter by density
            # without re-reading/recomputing from the tile image.
            mask = dual_hsv_mask(tile)
            sobel = sobel_edges(tile, mask)
            density = edge_density(sobel)
            density_scores[tile_name] = density
            cv2.imwrite(str(output_dir / f"{tile_name}_sobel.png"), sobel)

            if label_lines:
                waldo_tiles += 1
            else:
                background_tiles += 1

            tile_number += 1

    logger.info(
        "Image %s @ %dpx: %d tiles (%d with Waldo, %d background)",
        image_name,
        resolution,
        tile_number,
        waldo_tiles,
        background_tiles,
    )

    return waldo_tiles, background_tiles, density_scores


# ---- Train/val/test split ----


def create_splits(processed_dir: Path) -> dict[str, list[str]]:
    """Divide the original images into train, val, and test groups.

    We split at the *image* level, not the tile level, so tiles from
    the same puzzle picture never leak across splits.

    Args:
        processed_dir: Path to data/processed/ directory. The splits
            file is saved here as splits.json.

    Returns:
        Dictionary mapping split name to list of original image stems.
        Example: {'train': ['1','3','5',...], 'val': ['2','7'], 'test': ['4','9']}.
    """
    # Figure out which original images we have by looking at tile names
    tile_dir = processed_dir / str(RESOLUTIONS[-1])
    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}. Run tiling pipeline first.")

    originals = sorted({f.stem.rsplit("_tile_", 1)[0] for f in tile_dir.glob("*.jpg")})

    if not originals:
        raise ValueError(f"No tiles found in {tile_dir}")

    # Shuffle with a fixed seed so the split is the same every time
    random.seed(RANDOM_SEED)
    random.shuffle(originals)

    total_images = len(originals)
    num_train = int(total_images * TRAIN_SPLIT)
    num_val = int(total_images * VAL_SPLIT)

    splits = {
        "train": originals[:num_train],
        "val": originals[num_train : num_train + num_val],
        "test": originals[num_train + num_val :],
    }

    # Persist the split so every module uses the same one
    splits_path = processed_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    for split_name, names in splits.items():
        # Count tiles belonging to this split across all resolutions
        names_set = set(names)
        tile_count = sum(
            1
            for tile_file in tile_dir.glob("*.jpg")
            if tile_file.stem.rsplit("_tile_", 1)[0] in names_set
        )
        logger.info(
            "Split '%s': %d originals -> %d tiles",
            split_name,
            len(names),
            tile_count,
        )

    logger.info("Splits saved to %s", splits_path)
    return splits


# ---- Pipeline runner ----


def run_tiling_pipeline(
    resolutions: Optional[list[int]] = None,
) -> dict[int, tuple[int, int]]:
    """Tile every image at every resolution, then create the data splits.

    This is the main entry point: point it at data/raw/images/ and it
    will produce all the tiles we need for training.

    Args:
        resolutions: List of tile sizes to generate. Defaults to
            config.RESOLUTIONS ([64, 128, 256]).

    Returns:
        Dictionary mapping resolution to (waldo_tiles, background_tiles)
        counts.

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
        total_waldo = 0
        total_background = 0
        all_density_scores: dict[str, float] = {}

        for image_path in image_paths:
            annotation_path = annotations_dir / f"{image_path.stem}.xml"

            if not annotation_path.exists():
                logger.warning(
                    "No annotation for %s, skipping",
                    image_path.name,
                )
                continue

            try:
                waldo, background, density_scores = tile_single_image(
                    image_path, annotation_path, output_dir, resolution
                )
                total_waldo += waldo
                total_background += background
                all_density_scores.update(density_scores)
            except Exception:
                logger.exception("Failed to tile %s", image_path.name)
                raise

        # Save density scores so the dataset can filter tiles without
        # re-reading every image from disk.
        density_path = output_dir / "density.json"
        with open(density_path, "w") as f:
            json.dump(all_density_scores, f)
        logger.info("Saved density scores for %d tiles to %s", len(all_density_scores), density_path)

        waldo_percent = total_waldo / max(1, total_waldo + total_background) * 100
        logger.info(
            "Resolution %d: %d with Waldo, %d background (%.1f%% Waldo)",
            resolution,
            total_waldo,
            total_background,
            waldo_percent,
        )
        results[resolution] = (total_waldo, total_background)

    # Now that all tiles exist, decide which images go to which split
    logger.info("Creating train/val/test splits...")
    create_splits(DATA_PROCESSED)

    return results


# ---- HSV masking and Sobel edges ----


def dual_hsv_mask(tile: np.ndarray) -> np.ndarray:
    """Create a binary mask of red and white pixels (Waldo's colours).

    Red wraps around in HSV space, so we need two ranges.

    Args:
        tile: BGR image tile as numpy array (H, W, 3).

    Returns:
        Binary mask (H, W) where 255 = red or white region.
    """
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

    # Red sits at both ends of the hue wheel, so we need two ranges
    red_low_range = cv2.inRange(hsv, np.array(HSV_RED_LOW1), np.array(HSV_RED_HIGH1))
    red_high_range = cv2.inRange(hsv, np.array(HSV_RED_LOW2), np.array(HSV_RED_HIGH2))
    red_mask = cv2.bitwise_or(red_low_range, red_high_range)

    # White = bright pixels with very little colour saturation
    white_mask = cv2.inRange(hsv, np.array(HSV_WHITE_LOW), np.array(HSV_WHITE_HIGH))

    combined = cv2.bitwise_or(red_mask, white_mask)
    return combined


def sobel_edges(tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sobel edge detection restricted to the masked (red/white) area.

    Computes gradient magnitude in X and Y, but only on pixels that
    passed the HSV mask. This highlights stripe edges on Waldo's shirt.

    Args:
        tile: BGR image tile as numpy array (H, W, 3).
        mask: Binary mask from dual_hsv_mask (H, W).

    Returns:
        Edge magnitude image (H, W) normalized to [0, 255] uint8.
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

    # Zero-out everything outside the colour mask before edge detection
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    gradient_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Scale to 0-255 so it fits in a single image channel
    max_val = magnitude.max()
    if max_val > 0:
        normalized = (magnitude / max_val * 255).astype(np.uint8)
    else:
        normalized = magnitude.astype(np.uint8)

    return normalized


def edge_density(sobel_output: np.ndarray) -> float:
    """Fraction of pixels with significant edge energy.

    Low-density tiles are mostly empty background and can be skipped.

    Args:
        sobel_output: Edge magnitude image from sobel_edges (H, W).

    Returns:
        Fraction of pixels above the activity threshold [0.0, 1.0].
    """
    strong_pixels = int(np.sum(sobel_output > SOBEL_ACTIVE_THRESHOLD))
    total_pixels = sobel_output.shape[0] * sobel_output.shape[1]

    if total_pixels == 0:
        return 0.0

    return float(strong_pixels / total_pixels)


def preprocess_tile(
    tile: np.ndarray,
    use_sobel: bool = True,
) -> Optional[np.ndarray]:
    """Apply HSV masking + Sobel to one tile. Returns None if below density threshold.

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

    # Flip BGR to RGB and scale pixel values to 0-1
    rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))  # channels-first for PyTorch

    if use_sobel:
        # Tack the edge map on as a 4th channel alongside RGB
        sobel_normalised = sobel.astype(np.float32) / 255.0
        sobel_channel = np.expand_dims(sobel_normalised, axis=0)
        return np.concatenate([rgb, sobel_channel], axis=0)  # (4,H,W)

    return rgb


# ---- Main ----


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_tiling_pipeline()
