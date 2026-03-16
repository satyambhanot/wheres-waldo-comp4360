"""Detailed evaluation module for WaldoSSD with metrics and visual results.

Computes Average Precision (PASCAL VOC 11-point), precision, recall, F1,
and generates annotated images of TP/FP/FN examples for the final report.
Also provides a quick AP function importable by the training loop.

Usage:
    python -m src.evaluate     # evaluate on test set
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS,
    CONFIDENCE_THRESHOLD,
    DEFAULT_COLOR_MODE,
    DEFAULT_RESOLUTION,
    DEVICE,
    IOU_THRESHOLD,
    NMS_THRESHOLD,
    NUM_CLASSES,
    NUM_SSD_HEADS,
    RESULTS,
)
from src.dataset import get_dataloader
from src.model import WaldoSSD, decode_predictions

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Number of recall points for PASCAL VOC 11-point AP interpolation
_AP_RECALL_POINTS: int = 11

# Number of visual examples to save per category (TP, FP, FN)
_NUM_EXAMPLES_PER_CATEGORY: int = 5

# Drawing colors in BGR for cv2
_COLOR_GREEN_BGR: tuple[int, int, int] = (0, 255, 0)
_COLOR_RED_BGR: tuple[int, int, int] = (0, 0, 255)
_COLOR_WHITE_BGR: tuple[int, int, int] = (255, 255, 255)

# Box drawing thickness in pixels
_BOX_THICKNESS: int = 2

# Font scale for cv2.putText
_FONT_SCALE: float = 0.5


# =============================================================================
# DETECTION MATCHING
# =============================================================================


def match_detections_to_gt(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[list[tuple[float, bool]], list[int]]:
    """Match predicted detections to ground truth boxes for a single image.

    Each GT box can be matched to at most one detection (highest IoU first).
    Detections that don't match any GT are false positives. Unmatched GT
    boxes contribute to false negatives.

    Args:
        pred_boxes: Predicted boxes (D, 4) in [xmin, ymin, xmax, ymax].
        pred_scores: Confidence scores (D,) for each prediction.
        gt_boxes: Ground truth boxes (G, 4) in [xmin, ymin, xmax, ymax].
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        Tuple of (detections, unmatched_gt_indices) where:
            - detections: List of (confidence, is_true_positive) tuples.
            - unmatched_gt_indices: List of GT box indices that were missed.
    """
    detections: list[tuple[float, bool]] = []
    num_gt = gt_boxes.size(0) if gt_boxes.numel() > 0 else 0

    if pred_boxes.numel() == 0:
        return detections, list(range(num_gt))

    if num_gt == 0:
        for d_idx in range(pred_boxes.size(0)):
            detections.append((pred_scores[d_idx].item(), False))
        return detections, []

    gt_matched = torch.zeros(num_gt, dtype=torch.bool)
    iou_matrix = box_iou(pred_boxes, gt_boxes)

    for d_idx in range(pred_boxes.size(0)):
        best_iou, best_gt = iou_matrix[d_idx].max(dim=0)

        if best_iou >= iou_threshold and not gt_matched[best_gt]:
            detections.append((pred_scores[d_idx].item(), True))
            gt_matched[best_gt] = True
        else:
            detections.append((pred_scores[d_idx].item(), False))

    unmatched_gt = [i for i in range(num_gt) if not gt_matched[i]]
    return detections, unmatched_gt


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    nms_threshold: float = NMS_THRESHOLD,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter detections by confidence and apply Non-Maximum Suppression.

    Args:
        boxes: Predicted boxes (A, 4) in [xmin, ymin, xmax, ymax].
        scores: Confidence scores (A,) for Waldo class.
        nms_threshold: IoU threshold for NMS suppression.
        confidence_threshold: Minimum score to keep a detection.

    Returns:
        Tuple of (filtered_boxes, filtered_scores) after NMS.
        Both may be empty tensors if no detections survive.
    """
    keep_mask = scores > confidence_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]

    if scores.numel() == 0:
        return boxes, scores

    keep_nms = nms(boxes, scores, nms_threshold)
    return boxes[keep_nms], scores[keep_nms]


# =============================================================================
# AVERAGE PRECISION COMPUTATION
# =============================================================================


def _compute_voc_ap(
    detections: list[tuple[float, bool]],
    total_gt: int,
) -> float:
    """Compute PASCAL VOC 11-point interpolated Average Precision.

    Sorts detections by confidence, computes the precision-recall curve,
    then interpolates precision at 11 evenly spaced recall points.

    Args:
        detections: List of (confidence, is_true_positive) tuples.
        total_gt: Total number of ground truth boxes in the dataset.

    Returns:
        AP value in [0.0, 1.0]. Returns 0.0 if no GT boxes exist.
    """
    if total_gt == 0:
        return 0.0

    detections.sort(key=lambda x: x[0], reverse=True)

    tp_cumsum = 0
    fp_cumsum = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for _, is_tp in detections:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt
        precisions.append(precision)
        recalls.append(recall)

    # 11-point interpolation: for each recall threshold, take max precision
    # at recall >= threshold (right-side maximum envelope)
    ap = 0.0
    for r_thresh in [i / (_AP_RECALL_POINTS - 1) for i in range(_AP_RECALL_POINTS)]:
        max_prec = 0.0
        for p, r in zip(precisions, recalls):
            if r >= r_thresh:
                max_prec = max(max_prec, p)
        ap += max_prec

    return ap / _AP_RECALL_POINTS


# =============================================================================
# DETAILED EVALUATION
# =============================================================================


def evaluate_detailed(
    model: WaldoSSD,
    loader: DataLoader,
    anchors: torch.Tensor,
    resolution: int,
    device: torch.device,
) -> dict[str, float]:
    """Run full evaluation with AP, precision, recall, and F1 metrics.

    Processes every batch in the loader, collects all detections and GT
    matches, then computes comprehensive metrics for the final report.

    Args:
        model: Trained WaldoSSD model.
        loader: DataLoader for the evaluation split.
        anchors: Pre-computed anchors (num_anchors, 4) in [cx, cy, w, h].
        resolution: Image resolution in pixels for box denormalization.
        device: Device to run inference on.

    Returns:
        Dict with keys: 'ap', 'precision', 'recall', 'f1',
        'num_detections', 'num_gt', 'num_tp', 'num_fp', 'num_fn'.
    """
    model.eval()

    all_detections: list[tuple[float, bool]] = []
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            cls_preds, loc_preds = model(images)

            cls_probs = F.softmax(cls_preds, dim=-1)
            decoded_boxes = decode_predictions(loc_preds, anchors)

            for b in range(images.size(0)):
                gt_boxes_px = targets[b]["boxes"]
                num_gt_this = gt_boxes_px.size(0)
                total_gt += num_gt_this

                # Waldo confidence (class 1)
                scores = cls_probs[b, :, 1]
                boxes = decoded_boxes[b]

                boxes_filtered, scores_filtered = apply_nms(boxes, scores)

                # Convert to pixel coords for GT matching
                boxes_px = boxes_filtered * resolution

                gt_on_device = gt_boxes_px.to(device).float()
                dets, unmatched = match_detections_to_gt(boxes_px, scores_filtered, gt_on_device)

                all_detections.extend(dets)
                batch_tp = sum(1 for _, is_tp in dets if is_tp)
                batch_fp = sum(1 for _, is_tp in dets if not is_tp)
                total_tp += batch_tp
                total_fp += batch_fp
                total_fn += len(unmatched)

    ap = _compute_voc_ap(all_detections, total_gt)

    num_detections = total_tp + total_fp
    precision = total_tp / max(1, num_detections)
    recall = total_tp / max(1, total_gt)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    model.train()

    return {
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_detections": float(num_detections),
        "num_gt": float(total_gt),
        "num_tp": float(total_tp),
        "num_fp": float(total_fp),
        "num_fn": float(total_fn),
    }


# =============================================================================
# EVALUATION SUMMARY
# =============================================================================


def print_evaluation_summary(metrics: dict[str, float]) -> None:
    """Log a formatted summary table of evaluation metrics.

    Args:
        metrics: Dict from evaluate_detailed with AP, precision, recall, etc.
    """
    logger.info("=" * 50)
    logger.info("  EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info("  Average Precision (AP):  %.4f", metrics["ap"])
    logger.info("  Precision:               %.4f", metrics["precision"])
    logger.info("  Recall:                  %.4f", metrics["recall"])
    logger.info("  F1 Score:                %.4f", metrics["f1"])
    logger.info("-" * 50)
    logger.info("  Total GT boxes:          %d", int(metrics["num_gt"]))
    logger.info("  Total detections:        %d", int(metrics["num_detections"]))
    logger.info("  True Positives:          %d", int(metrics["num_tp"]))
    logger.info("  False Positives:         %d", int(metrics["num_fp"]))
    logger.info("  False Negatives:         %d", int(metrics["num_fn"]))
    logger.info("=" * 50)


# =============================================================================
# VISUAL RESULT EXAMPLES
# =============================================================================


def _tensor_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to BGR uint8 for cv2 drawing.

    Only uses the first 3 channels (RGB), ignoring the Sobel channel
    if present in 4-channel input.

    Args:
        image_tensor: Image tensor (C, H, W) in [0, 1] float, C=3 or 4.

    Returns:
        BGR uint8 numpy array (H, W, 3) suitable for cv2 operations.
    """
    # Take only RGB channels, drop Sobel if present
    rgb = image_tensor[:3].cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    bgr: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def _draw_box(
    img: np.ndarray,
    box: list[float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw a bounding box with label on an image in-place.

    Args:
        img: BGR uint8 image to draw on (modified in-place).
        box: Box coords [xmin, ymin, xmax, ymax] in pixels.
        color: BGR color tuple for the rectangle and text.
        label: Text label to display above the box.
    """
    x1, y1 = int(box[0]), int(box[1])
    x2, y2 = int(box[2]), int(box[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, _BOX_THICKNESS)
    cv2.putText(
        img,
        label,
        (x1, max(y1 - _BOX_THICKNESS, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        _FONT_SCALE,
        color,
        1,
    )


def _draw_category_label(
    img: np.ndarray,
    category: str,
) -> None:
    """Draw the category label (TP/FP/FN) in the top-left corner.

    Args:
        img: BGR uint8 image to draw on (modified in-place).
        category: Category string ("TP", "FP", or "FN").
    """
    cv2.putText(
        img,
        category,
        (3, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        _FONT_SCALE,
        _COLOR_WHITE_BGR,
        1,
    )


def save_detection_examples(
    model: WaldoSSD,
    loader: DataLoader,
    anchors: torch.Tensor,
    resolution: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Save annotated images of TP, FP, and FN detection examples.

    Runs inference on the entire dataset and collects the most
    illustrative examples of each outcome category. Saves annotated
    images with bounding boxes and a JSON summary for the final report.

    Args:
        model: Trained WaldoSSD model.
        loader: DataLoader for the evaluation split.
        anchors: Pre-computed anchors (num_anchors, 4) in [cx, cy, w, h].
        resolution: Image resolution in pixels.
        device: Device to run inference on.
        output_dir: Directory to save example images and summary JSON.
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect examples: (confidence, image_tensor, pred_box_px, gt_boxes_px, category)
    tp_examples: list[tuple[float, torch.Tensor, list[float], torch.Tensor]] = []
    fp_examples: list[tuple[float, torch.Tensor, list[float], torch.Tensor]] = []
    fn_examples: list[tuple[torch.Tensor, list[float]]] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            cls_preds, loc_preds = model(images)

            cls_probs = F.softmax(cls_preds, dim=-1)
            decoded_boxes = decode_predictions(loc_preds, anchors)

            for b in range(images.size(0)):
                gt_boxes_px = targets[b]["boxes"]
                img_tensor = images[b]

                scores = cls_probs[b, :, 1]
                boxes = decoded_boxes[b]

                boxes_filtered, scores_filtered = apply_nms(boxes, scores)
                boxes_px = boxes_filtered * resolution

                gt_on_device = gt_boxes_px.to(device).float()

                # Reuse match_detections_to_gt for consistent logic
                dets, unmatched_gt = match_detections_to_gt(boxes_px, scores_filtered, gt_on_device)

                # Classify each detection as TP or FP
                for d_idx, (conf, is_tp) in enumerate(dets):
                    pred_box_list = boxes_px[d_idx].cpu().tolist()
                    if is_tp:
                        tp_examples.append((conf, img_tensor, pred_box_list, gt_boxes_px))
                    else:
                        fp_examples.append((conf, img_tensor, pred_box_list, gt_boxes_px))

                # Unmatched GT boxes are false negatives
                for g_idx in unmatched_gt:
                    fn_examples.append((img_tensor, gt_on_device[g_idx].cpu().tolist()))

    # --- Sort and select top examples ---
    # TPs: highest confidence correct detections
    tp_examples.sort(key=lambda x: x[0], reverse=True)
    # FPs: highest confidence wrong detections (most interesting errors)
    fp_examples.sort(key=lambda x: x[0], reverse=True)

    summary: dict[str, list[dict[str, object]]] = {"tp": [], "fp": [], "fn": []}

    # --- Save True Positive examples ---
    for i, (conf, img_t, pred_box, gt_boxes) in enumerate(tp_examples[:_NUM_EXAMPLES_PER_CATEGORY]):
        img_bgr = _tensor_to_bgr(img_t)
        for gt_box in gt_boxes.tolist():
            _draw_box(img_bgr, gt_box, _COLOR_GREEN_BGR, "GT")
        _draw_box(img_bgr, pred_box, _COLOR_RED_BGR, f"{conf:.2f}")
        _draw_category_label(img_bgr, "TP")

        filename = f"tp_{i + 1:02d}.jpg"
        cv2.imwrite(str(output_dir / filename), img_bgr)
        summary["tp"].append({"file": filename, "confidence": conf, "pred_box": pred_box})

    # --- Save False Positive examples ---
    for i, (conf, img_t, pred_box, gt_boxes) in enumerate(fp_examples[:_NUM_EXAMPLES_PER_CATEGORY]):
        img_bgr = _tensor_to_bgr(img_t)
        for gt_box in gt_boxes.tolist():
            _draw_box(img_bgr, gt_box, _COLOR_GREEN_BGR, "GT")
        _draw_box(img_bgr, pred_box, _COLOR_RED_BGR, f"{conf:.2f}")
        _draw_category_label(img_bgr, "FP")

        filename = f"fp_{i + 1:02d}.jpg"
        cv2.imwrite(str(output_dir / filename), img_bgr)
        summary["fp"].append({"file": filename, "confidence": conf, "pred_box": pred_box})

    # --- Save False Negative examples ---
    for i, (img_t, gt_box) in enumerate(fn_examples[:_NUM_EXAMPLES_PER_CATEGORY]):
        img_bgr = _tensor_to_bgr(img_t)
        _draw_box(img_bgr, gt_box, _COLOR_GREEN_BGR, "GT")
        _draw_category_label(img_bgr, "FN - MISSED")

        filename = f"fn_{i + 1:02d}.jpg"
        cv2.imwrite(str(output_dir / filename), img_bgr)
        summary["fn"].append({"file": filename, "gt_box": gt_box})

    # --- Save summary JSON ---
    summary_path = output_dir / "examples_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Saved detection examples: %d TP, %d FP, %d FN to %s",
        min(len(tp_examples), _NUM_EXAMPLES_PER_CATEGORY),
        min(len(fp_examples), _NUM_EXAMPLES_PER_CATEGORY),
        min(len(fn_examples), _NUM_EXAMPLES_PER_CATEGORY),
        output_dir,
    )


# =============================================================================
# MAIN - TEST SET EVALUATION
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    resolution = DEFAULT_RESOLUTION
    use_sobel = True
    in_channels = 4 if use_sobel else 3

    # --- Load or create model ---
    model = WaldoSSD(
        num_classes=NUM_CLASSES,
        num_heads=NUM_SSD_HEADS,
        in_channels=in_channels,
    )

    checkpoint_path = CHECKPOINTS / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)", checkpoint_path, checkpoint["epoch"])
    else:
        logger.warning(
            "No checkpoint found at %s - running with random weights (results meaningless)",
            checkpoint_path,
        )

    # Generate anchors before moving model to device
    anchors = model.get_anchors(resolution)
    model = model.to(DEVICE)
    anchors = anchors.to(DEVICE)

    # --- Test set DataLoader ---
    test_loader = get_dataloader(
        split="test",
        resolution=resolution,
        color_mode=DEFAULT_COLOR_MODE,
        use_sobel=use_sobel,
        use_density_filter=False,
        batch_size=BATCH_SIZE,
    )

    # --- Run detailed evaluation ---
    logger.info("Running detailed evaluation on test set (%dpx)...", resolution)
    metrics = evaluate_detailed(model, test_loader, anchors, resolution, DEVICE)
    print_evaluation_summary(metrics)

    # --- Save visual examples ---
    plots_dir = RESULTS / "plots"
    logger.info("Saving detection examples to %s...", plots_dir)
    save_detection_examples(model, test_loader, anchors, resolution, DEVICE, plots_dir)

    logger.info("Evaluation complete.")
