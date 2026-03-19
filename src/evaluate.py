"""Scores the model's performance and saves visual examples.

After training we want to know: how well does the model actually find
Waldo?  This module computes the numbers (AP, precision, recall, F1)
and also saves annotated images showing true positives, false positives,
and the Waldos the model missed.

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


# ---- Constants ----

# PASCAL VOC uses 11 evenly spaced recall thresholds for AP
_AP_RECALL_POINTS: int = 11

# How many example images to save for each category (TP / FP / FN)
_EXAMPLES_TO_SAVE: int = 5

# Colours for drawing boxes (BGR because OpenCV)
_GREEN: tuple[int, int, int] = (0, 255, 0)
_RED: tuple[int, int, int] = (0, 0, 255)
_WHITE: tuple[int, int, int] = (255, 255, 255)

# Visual settings for the annotated images
_LINE_THICKNESS: int = 2
_LABEL_FONT_SCALE: float = 0.5


# ---- Detection matching ----


def match_detections_to_gt(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[list[tuple[float, bool]], list[int]]:
    """For one image, figure out which predictions are right and which aren't.

    Each GT box can only be claimed by one prediction (first come,
    highest IoU served).  Leftover predictions are false positives,
    and unclaimed GT boxes are false negatives.

    Args:
        pred_boxes: Predicted boxes (D, 4) in [xmin, ymin, xmax, ymax].
        pred_scores: Confidence scores (D,) for each prediction.
        gt_boxes: Ground truth boxes (G, 4) in [xmin, ymin, xmax, ymax].
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        Tuple of (detections, missed_gt_indices) where:
            - detections: List of (confidence, is_correct) tuples.
            - missed_gt_indices: GT indices the model failed to find.
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

    # Process highest-confidence detections first so they get first
    # claim on GT boxes (standard PASCAL VOC matching protocol).
    num_dets = pred_boxes.size(0)
    sorted_indices = pred_scores.argsort(descending=True)
    is_correct = [False] * num_dets

    for rank_pos in range(num_dets):
        d_idx = sorted_indices[rank_pos].item()
        best_iou, best_gt = iou_matrix[d_idx].max(dim=0)

        if best_iou >= iou_threshold and not gt_matched[best_gt]:
            is_correct[d_idx] = True
            gt_matched[best_gt] = True

    # Return results in original input order so callers can index
    # into pred_boxes by position.
    for d_idx in range(num_dets):
        detections.append((pred_scores[d_idx].item(), is_correct[d_idx]))

    unmatched_gt = [i for i in range(num_gt) if not gt_matched[i]]
    return detections, unmatched_gt


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    nms_threshold: float = NMS_THRESHOLD,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Throw away low-confidence guesses and remove duplicate detections.

    Args:
        boxes: Predicted boxes (A, 4) in [xmin, ymin, xmax, ymax].
        scores: Confidence scores (A,) for Waldo class.
        nms_threshold: IoU threshold for NMS suppression.
        confidence_threshold: Minimum score to keep a detection.

    Returns:
        Tuple of (surviving_boxes, surviving_scores) after filtering.
    """
    keep_mask = scores > confidence_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]

    if scores.numel() == 0:
        return boxes, scores

    keep_nms = nms(boxes, scores, nms_threshold)
    return boxes[keep_nms], scores[keep_nms]


# ---- Average Precision ----


def _compute_voc_ap(
    detections: list[tuple[float, bool]],
    total_gt: int,
) -> float:
    """Compute the PASCAL VOC 11-point interpolated Average Precision.

    We sort all detections by confidence, walk through them to build
    a precision-recall curve, and average the best precision at 11
    evenly spaced recall levels.

    Args:
        detections: List of (confidence, is_true_positive) tuples.
        total_gt: Total number of ground truth boxes in the dataset.

    Returns:
        AP value in [0.0, 1.0]. Returns 0.0 if no GT boxes exist.
    """
    pr_curve = compute_pr_curve(detections, total_gt)
    return pr_curve["ap"]


def compute_pr_curve(
    detections: list[tuple[float, bool]],
    total_gt: int,
) -> dict:
    """Build the full precision-recall curve and compute AP.

    Args:
        detections: List of (confidence, is_true_positive) tuples.
        total_gt: Total number of ground truth boxes in the dataset.

    Returns:
        Dict with keys 'ap', 'precisions', 'recalls', 'confidences',
        and 'interpolated_precisions' (11-point).
    """
    if total_gt == 0:
        return {
            "ap": 0.0,
            "precisions": [],
            "recalls": [],
            "confidences": [],
            "interpolated_precisions": [0.0] * _AP_RECALL_POINTS,
        }

    detections.sort(key=lambda x: x[0], reverse=True)

    running_tp = 0
    running_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    confidences: list[float] = []

    for conf, is_correct in detections:
        if is_correct:
            running_tp += 1
        else:
            running_fp += 1

        precision = running_tp / (running_tp + running_fp)
        recall = running_tp / total_gt
        precisions.append(precision)
        recalls.append(recall)
        confidences.append(conf)

    # Build the precision envelope: at each point, the best precision
    # at that recall or higher. Single reverse pass is O(n).
    n = len(precisions)
    max_precision_at_or_after = [0.0] * n
    max_precision_at_or_after[n - 1] = precisions[n - 1]
    for i in range(n - 2, -1, -1):
        max_precision_at_or_after[i] = max(precisions[i], max_precision_at_or_after[i + 1])

    # At each of the 11 recall thresholds, binary-search for the first
    # detection at that recall level and read the envelope value.
    ap = 0.0
    interpolated: list[float] = []
    for recall_threshold in [i / (_AP_RECALL_POINTS - 1) for i in range(_AP_RECALL_POINTS)]:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if recalls[mid] < recall_threshold:
                lo = mid + 1
            else:
                hi = mid
        prec_at_thresh = max_precision_at_or_after[lo] if lo < n else 0.0
        interpolated.append(prec_at_thresh)
        ap += prec_at_thresh

    return {
        "ap": ap / _AP_RECALL_POINTS,
        "precisions": precisions,
        "recalls": recalls,
        "confidences": confidences,
        "interpolated_precisions": interpolated,
    }


# ---- Detailed evaluation ----


def evaluate_detailed(
    model: WaldoSSD,
    loader: DataLoader,
    anchors: torch.Tensor,
    resolution: int,
    device: torch.device,
) -> dict[str, float]:
    """Run the model on every tile and compute AP, precision, recall, F1.

    This is the "real" evaluation — used for the final report, not
    the quick AP check during training.

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

                # How confident is the model that each anchor is Waldo?
                scores = cls_probs[b, :, 1]
                boxes = decoded_boxes[b]

                boxes_filtered, scores_filtered = apply_nms(boxes, scores)

                # Scale back to pixels so we can compare with GT
                boxes_px = boxes_filtered * resolution

                gt_on_device = gt_boxes_px.to(device).float()
                dets, unmatched = match_detections_to_gt(boxes_px, scores_filtered, gt_on_device)

                all_detections.extend(dets)
                batch_tp = sum(1 for _, correct in dets if correct)
                batch_fp = sum(1 for _, correct in dets if not correct)
                total_tp += batch_tp
                total_fp += batch_fp
                total_fn += len(unmatched)

    pr_curve = compute_pr_curve(all_detections, total_gt)

    num_detections = total_tp + total_fp
    precision = total_tp / max(1, num_detections)
    recall = total_tp / max(1, total_gt)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    model.train()

    return {
        "ap": pr_curve["ap"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_detections": float(num_detections),
        "num_gt": float(total_gt),
        "num_tp": float(total_tp),
        "num_fp": float(total_fp),
        "num_fn": float(total_fn),
        "pr_curve": {
            "precisions": pr_curve["precisions"],
            "recalls": pr_curve["recalls"],
            "confidences": pr_curve["confidences"],
            "interpolated_precisions": pr_curve["interpolated_precisions"],
        },
    }


# ---- Summary printing ----


def print_evaluation_summary(metrics: dict[str, float]) -> None:
    """Print a nicely formatted results table to the log.

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


# ---- Visual examples ----


def _tensor_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    """Turn a model tensor back into a normal image OpenCV can draw on.

    Drops the Sobel channel (if present) and converts from the
    normalised RGB tensor format to the BGR uint8 that cv2 expects.

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
    """Draw a labelled rectangle on the image (modifies it in place).

    Args:
        img: BGR uint8 image to draw on.
        box: Box coords [xmin, ymin, xmax, ymax] in pixels.
        color: BGR colour for the rectangle and text.
        label: Short text to show just above the box.
    """
    x1, y1 = int(box[0]), int(box[1])
    x2, y2 = int(box[2]), int(box[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, _LINE_THICKNESS)
    cv2.putText(
        img,
        label,
        (x1, max(y1 - _LINE_THICKNESS, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        _LABEL_FONT_SCALE,
        color,
        1,
    )


def _draw_category_label(
    img: np.ndarray,
    category: str,
) -> None:
    """Stamp "TP", "FP", or "FN" in the top-left corner of the image.

    Args:
        img: BGR uint8 image to draw on (modified in-place).
        category: Category string ("TP", "FP", or "FN").
    """
    cv2.putText(
        img,
        category,
        (3, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        _LABEL_FONT_SCALE,
        _WHITE,
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
    """Save the most interesting detection results as annotated images.

    We collect the highest-confidence correct detections (TP), the
    highest-confidence mistakes (FP), and the missed Waldos (FN),
    draw boxes on them, and save them for the report.

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

    # We'll gather examples in three buckets
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

                # Use the same matching logic we use everywhere else
                dets, unmatched_gt = match_detections_to_gt(boxes_px, scores_filtered, gt_on_device)

                # Sort each detection into the right bucket
                for d_idx, (conf, is_correct) in enumerate(dets):
                    pred_box_list = boxes_px[d_idx].cpu().tolist()
                    if is_correct:
                        tp_examples.append((conf, img_tensor, pred_box_list, gt_boxes_px))
                    else:
                        fp_examples.append((conf, img_tensor, pred_box_list, gt_boxes_px))

                # GT boxes nobody found are false negatives
                for g_idx in unmatched_gt:
                    fn_examples.append((img_tensor, gt_on_device[g_idx].cpu().tolist()))

    # Sort by confidence so we save the most interesting examples
    tp_examples.sort(key=lambda x: x[0], reverse=True)
    fp_examples.sort(key=lambda x: x[0], reverse=True)

    summary: dict[str, list[dict[str, object]]] = {"tp": [], "fp": [], "fn": []}

    # Save the best True Positive examples
    for i, (conf, img_t, pred_box, gt_boxes) in enumerate(tp_examples[:_EXAMPLES_TO_SAVE]):
        img_bgr = _tensor_to_bgr(img_t)
        for gt_box in gt_boxes.tolist():
            _draw_box(img_bgr, gt_box, _GREEN, "GT")
        _draw_box(img_bgr, pred_box, _RED, f"{conf:.2f}")
        _draw_category_label(img_bgr, "TP")

        filename = f"tp_{i + 1:02d}.jpg"
        cv2.imwrite(str(output_dir / filename), img_bgr)
        summary["tp"].append({"file": filename, "confidence": conf, "pred_box": pred_box})

    # Save the worst False Positive mistakes
    for i, (conf, img_t, pred_box, gt_boxes) in enumerate(fp_examples[:_EXAMPLES_TO_SAVE]):
        img_bgr = _tensor_to_bgr(img_t)
        for gt_box in gt_boxes.tolist():
            _draw_box(img_bgr, gt_box, _GREEN, "GT")
        _draw_box(img_bgr, pred_box, _RED, f"{conf:.2f}")
        _draw_category_label(img_bgr, "FP")

        filename = f"fp_{i + 1:02d}.jpg"
        cv2.imwrite(str(output_dir / filename), img_bgr)
        summary["fp"].append({"file": filename, "confidence": conf, "pred_box": pred_box})

    # Save examples of Waldos the model missed entirely
    for i, (img_t, gt_box) in enumerate(fn_examples[:_EXAMPLES_TO_SAVE]):
        img_bgr = _tensor_to_bgr(img_t)
        _draw_box(img_bgr, gt_box, _GREEN, "GT")
        _draw_category_label(img_bgr, "FN - MISSED")

        filename = f"fn_{i + 1:02d}.jpg"
        cv2.imwrite(str(output_dir / filename), img_bgr)
        summary["fn"].append({"file": filename, "gt_box": gt_box})

    # Write a machine-readable summary alongside the images
    summary_path = output_dir / "examples_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Saved detection examples: %d TP, %d FP, %d FN to %s",
        min(len(tp_examples), _EXAMPLES_TO_SAVE),
        min(len(fp_examples), _EXAMPLES_TO_SAVE),
        min(len(fn_examples), _EXAMPLES_TO_SAVE),
        output_dir,
    )


# ---- Main ----


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    resolution = DEFAULT_RESOLUTION
    use_sobel = True
    in_channels = 4 if use_sobel else 3

    # Build the model and load the best checkpoint (if one exists)
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

    # Anchors first (needs a CPU forward pass), then move to device
    anchors = model.get_anchors(resolution)
    model = model.to(DEVICE)
    anchors = anchors.to(DEVICE)

    # Load the test tiles
    test_loader = get_dataloader(
        split="test",
        resolution=resolution,
        color_mode=DEFAULT_COLOR_MODE,
        use_sobel=use_sobel,
        use_density_filter=False,
        batch_size=BATCH_SIZE,
    )

    # Compute all the metrics
    logger.info("Running detailed evaluation on test set (%dpx)...", resolution)
    metrics = evaluate_detailed(model, test_loader, anchors, resolution, DEVICE)
    print_evaluation_summary(metrics)

    # Save metrics to JSON for the report
    metrics_dir = RESULTS / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Separate the large PR curve arrays from the summary metrics
    pr_curve_data = metrics.pop("pr_curve", {})

    metrics_path = metrics_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved test metrics to %s", metrics_path)

    pr_path = metrics_dir / "pr_curve.json"
    with open(pr_path, "w") as f:
        json.dump(pr_curve_data, f, indent=2)
    logger.info("Saved PR curve data to %s", pr_path)

    # Save annotated example images for the report
    plots_dir = RESULTS / "plots"
    logger.info("Saving detection examples to %s...", plots_dir)
    save_detection_examples(model, test_loader, anchors, resolution, DEVICE, plots_dir)

    logger.info("Evaluation complete.")
