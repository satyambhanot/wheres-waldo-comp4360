"""Training pipeline for WaldoSSD with SSD loss, hard negative mining, and AP evaluation.

Implements the complete training loop including anchor matching, multi-box
loss with hard negative mining, cosine LR scheduling, gradient clipping,
periodic validation with PASCAL VOC 11-point AP, and checkpoint saving.
Designed for ablation experiments via config_overrides.

Usage:
    python -m src.train        # 3-epoch sanity check
"""

import logging
import time
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    LEARNING_RATE,
    NEG_POS_RATIO,
    NMS_THRESHOLD,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_SSD_HEADS,
    WEIGHT_DECAY,
)
from src.dataset import get_dataloader
from src.model import WaldoSSD, decode_predictions

logger = logging.getLogger(__name__)


# =============================================================================
# BOX FORMAT CONVERSION UTILITIES
# =============================================================================

# IoU matching threshold for anchor-to-GT assignment
_IOU_MATCH_THRESHOLD: float = 0.5

# Gradient clipping norm to stabilize early training
_GRAD_CLIP_MAX_NORM: float = 5.0

# Validation frequency in epochs
_VAL_EVERY_N_EPOCHS: int = 5

# Checkpoint save frequency in epochs
_CHECKPOINT_EVERY_N_EPOCHS: int = 20

# Number of recall points for PASCAL VOC 11-point AP interpolation
_AP_RECALL_POINTS: int = 11


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from center format to corner format.

    Args:
        boxes: Tensor of shape (..., 4) in [cx, cy, w, h] format.

    Returns:
        Tensor of shape (..., 4) in [xmin, ymin, xmax, ymax] format.
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from corner format to center format.

    Args:
        boxes: Tensor of shape (..., 4) in [xmin, ymin, xmax, ymax] format.

    Returns:
        Tensor of shape (..., 4) in [cx, cy, w, h] format.
    """
    xmin, ymin, xmax, ymax = (
        boxes[..., 0],
        boxes[..., 1],
        boxes[..., 2],
        boxes[..., 3],
    )
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return torch.stack([cx, cy, w, h], dim=-1)


# =============================================================================
# ANCHOR MATCHING
# =============================================================================


def match_anchors_to_targets(
    anchors: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ground truth boxes to anchor boxes for SSD training.

    For each image in the batch, assigns each anchor to either a GT box
    (positive) or background (negative). Uses a two-pass strategy:
    first ensures every GT has its best anchor, then adds any anchor
    with IoU >= 0.5 to any GT box.

    Args:
        anchors: Anchor boxes of shape (num_anchors, 4) in normalized
            [cx, cy, w, h] format.
        targets: List of B target dicts, each with 'boxes' (N, 4) in
            pixel [xmin, ymin, xmax, ymax] and 'labels' (N,).
        resolution: Image resolution in pixels, for normalizing GT boxes.

    Returns:
        Tuple of (cls_targets, reg_targets) where:
            - cls_targets: (B, num_anchors) int64 class labels
              (0=background, 1=Waldo).
            - reg_targets: (B, num_anchors, 4) float32 regression offsets
              [dx, dy, dw, dh] relative to matched anchors.
    """
    batch_size = len(targets)
    num_anchors = anchors.size(0)
    device = anchors.device

    cls_targets = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
    reg_targets = torch.zeros(batch_size, num_anchors, 4, dtype=torch.float32, device=device)

    # Pre-convert anchors to xyxy for IoU computation
    anchors_xyxy = cxcywh_to_xyxy(anchors)

    for i, target in enumerate(targets):
        gt_boxes_px = target["boxes"]

        if gt_boxes_px.numel() == 0:
            continue

        # Normalize GT boxes from pixel coords to [0, 1]
        gt_boxes_xyxy = gt_boxes_px.to(device).float() / resolution
        gt_boxes_cxcywh = xyxy_to_cxcywh(gt_boxes_xyxy)

        # IoU matrix: (num_anchors, num_gt)
        iou_matrix = box_iou(anchors_xyxy, gt_boxes_xyxy)

        # --- Pass 1: best anchor for each GT (guarantees every GT gets a match) ---
        best_anchor_per_gt = iou_matrix.argmax(dim=0)
        for gt_idx in range(gt_boxes_xyxy.size(0)):
            anchor_idx = best_anchor_per_gt[gt_idx]
            cls_targets[i, anchor_idx] = 1
            reg_targets[i, anchor_idx] = _encode_offsets(
                anchors[anchor_idx], gt_boxes_cxcywh[gt_idx]
            )

        # --- Pass 2: any anchor with IoU >= threshold gets its best GT ---
        best_iou_per_anchor, best_gt_per_anchor = iou_matrix.max(dim=1)
        positive_mask = best_iou_per_anchor >= _IOU_MATCH_THRESHOLD

        positive_indices = positive_mask.nonzero(as_tuple=True)[0]
        for anchor_idx in positive_indices:
            gt_idx = best_gt_per_anchor[anchor_idx]
            cls_targets[i, anchor_idx] = 1
            reg_targets[i, anchor_idx] = _encode_offsets(
                anchors[anchor_idx], gt_boxes_cxcywh[gt_idx]
            )

    return cls_targets, reg_targets


def _encode_offsets(
    anchor: torch.Tensor,
    gt_box: torch.Tensor,
) -> torch.Tensor:
    """Compute regression offsets from an anchor to a ground truth box.

    Both inputs are in normalized [cx, cy, w, h] format. The encoding
    normalizes center offsets by anchor size and uses log-scale for
    width/height, which stabilizes training for boxes of varying sizes.

    Args:
        anchor: Single anchor box (4,) in [cx, cy, w, h].
        gt_box: Single GT box (4,) in [cx, cy, w, h].

    Returns:
        Offset tensor (4,) as [dx, dy, dw, dh].
    """
    dx = (gt_box[0] - anchor[0]) / anchor[2]
    dy = (gt_box[1] - anchor[1]) / anchor[3]
    # Clamp to prevent log(0) from degenerate GT boxes
    dw = torch.log(gt_box[2] / anchor[2].clamp(min=1e-6))
    dh = torch.log(gt_box[3] / anchor[3].clamp(min=1e-6))
    return torch.stack([dx, dy, dw, dh])


# =============================================================================
# SSD LOSS WITH HARD NEGATIVE MINING
# =============================================================================


class SSDLoss(nn.Module):
    """SSD multi-box loss with hard negative mining.

    Combines classification (cross-entropy) and localization (smooth L1)
    losses. Hard negative mining selects the most confidently wrong
    background anchors to balance the severe class imbalance in Waldo
    detection (~3% positive tiles).

    Attributes:
        neg_pos_ratio: Number of hard negatives per positive anchor.
        num_classes: Number of object classes (including background).
    """

    def __init__(
        self,
        neg_pos_ratio: int = NEG_POS_RATIO,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Initialize SSDLoss.

        Args:
            neg_pos_ratio: Ratio of hard negatives to positives per image.
                Higher values include more background examples but slow
                convergence on positives.
            num_classes: Number of object classes (background + Waldo = 2).
        """
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.num_classes = num_classes

    def forward(
        self,
        cls_preds: torch.Tensor,
        loc_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute SSD loss with hard negative mining.

        Args:
            cls_preds: Classification logits (B, num_anchors, num_classes).
            loc_preds: Regression offsets (B, num_anchors, 4).
            cls_targets: Ground truth class labels (B, num_anchors).
            reg_targets: Ground truth regression offsets (B, num_anchors, 4).

        Returns:
            Tuple of (total_loss, cls_loss_value, reg_loss_value) where
            total_loss is the differentiable loss tensor and the other
            two are detached float scalars for logging.
        """
        batch_size = cls_preds.size(0)
        num_anchors = cls_preds.size(1)

        # --- Localization loss: only on positive anchors ---
        pos_mask = cls_targets > 0  # (B, num_anchors)
        num_positives = pos_mask.sum().item()

        if num_positives == 0:
            # No positives in batch — return zero loss to avoid div-by-zero
            zero = torch.tensor(0.0, device=cls_preds.device, requires_grad=True)
            return zero, 0.0, 0.0

        # Expand mask for 4 regression coordinates
        pos_mask_loc = pos_mask.unsqueeze(2).expand_as(loc_preds)
        reg_loss = F.smooth_l1_loss(
            loc_preds[pos_mask_loc].view(-1, 4),
            reg_targets[pos_mask_loc].view(-1, 4),
            reduction="sum",
        )

        # --- Classification loss: positives + hard negatives ---
        # Compute per-anchor classification loss for mining
        cls_loss_all = F.cross_entropy(
            cls_preds.view(-1, self.num_classes),
            cls_targets.view(-1),
            reduction="none",
        ).view(batch_size, num_anchors)

        # Zero out positive anchor losses — we'll add them back below
        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[pos_mask] = 0.0

        # --- Hard negative mining: keep the worst background predictions ---
        # Sort on CPU because MPS torch.sort can produce inconsistent results
        cls_loss_neg_cpu = cls_loss_neg.cpu()
        _, neg_sort_idx = cls_loss_neg_cpu.sort(dim=1, descending=True)
        neg_sort_idx = neg_sort_idx.to(cls_preds.device)

        # Determine how many hard negatives per image
        num_pos_per_image = pos_mask.sum(dim=1)
        num_neg_per_image = (num_pos_per_image * self.neg_pos_ratio).clamp(min=1)

        # Build mask for selected hard negatives
        neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
        for b in range(batch_size):
            n_neg = int(num_neg_per_image[b].item())
            # Select top-n hardest negative anchor indices for this image
            hard_neg_indices = neg_sort_idx[b, :n_neg]
            neg_mask[b, hard_neg_indices] = True

        # Combined loss over positives + hard negatives
        selected_mask = pos_mask | neg_mask
        cls_loss = cls_loss_all[selected_mask].sum()

        # Normalize by number of positive anchors across the batch
        num_pos_float = float(num_positives)
        total_loss = (cls_loss + reg_loss) / num_pos_float

        return total_loss, cls_loss.item() / num_pos_float, reg_loss.item() / num_pos_float


# =============================================================================
# EVALUATION — AVERAGE PRECISION
# =============================================================================


def evaluate(
    model: WaldoSSD,
    loader: DataLoader,
    anchors: torch.Tensor,
    resolution: int,
    device: torch.device,
) -> float:
    """Compute Average Precision on a dataset split.

    Runs the model in eval mode, decodes predictions, applies NMS,
    and computes PASCAL VOC 11-point interpolated AP.

    Args:
        model: Trained WaldoSSD model.
        loader: DataLoader for the evaluation split.
        anchors: Pre-computed anchors (num_anchors, 4) in [cx, cy, w, h].
        resolution: Image resolution in pixels for box denormalization.
        device: Device to run evaluation on.

    Returns:
        Average Precision as a float in [0.0, 1.0].
    """
    model.eval()

    # Collect all detections and GT counts for AP computation
    all_detections: list[tuple[float, bool]] = []
    total_gt_boxes = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            cls_preds, loc_preds = model(images)

            # Softmax for confidence scores
            cls_probs = F.softmax(cls_preds, dim=-1)
            decoded_boxes = decode_predictions(loc_preds, anchors)

            for b in range(images.size(0)):
                gt_boxes_px = targets[b]["boxes"]
                total_gt_boxes += gt_boxes_px.size(0)

                # Waldo confidence (class 1)
                scores = cls_probs[b, :, 1]
                boxes = decoded_boxes[b]

                # Filter by confidence threshold
                keep_mask = scores > CONFIDENCE_THRESHOLD
                scores = scores[keep_mask]
                boxes = boxes[keep_mask]

                if scores.numel() == 0:
                    continue

                # NMS to remove duplicate detections
                keep_nms = nms(boxes, scores, NMS_THRESHOLD)
                scores = scores[keep_nms]
                boxes = boxes[keep_nms]

                # Convert to pixel coords for matching against GT
                boxes_px = boxes * resolution

                # Match detections to GT boxes
                gt_matched = torch.zeros(gt_boxes_px.size(0), dtype=torch.bool)

                if gt_boxes_px.numel() > 0:
                    gt_on_device = gt_boxes_px.to(device).float()
                    det_iou = box_iou(boxes_px, gt_on_device)

                    for d_idx in range(boxes_px.size(0)):
                        if det_iou.size(1) == 0:
                            all_detections.append((scores[d_idx].item(), False))
                            continue

                        best_iou, best_gt = det_iou[d_idx].max(dim=0)

                        if best_iou >= IOU_THRESHOLD and not gt_matched[best_gt]:
                            all_detections.append((scores[d_idx].item(), True))
                            gt_matched[best_gt] = True
                        else:
                            all_detections.append((scores[d_idx].item(), False))
                else:
                    # All detections are false positives
                    for d_idx in range(scores.size(0)):
                        all_detections.append((scores[d_idx].item(), False))

    ap = _compute_voc_ap(all_detections, total_gt_boxes)

    model.train()
    return ap


def _compute_voc_ap(
    detections: list[tuple[float, bool]],
    total_gt: int,
) -> float:
    """Compute PASCAL VOC 11-point interpolated Average Precision.

    Sorts detections by confidence, computes precision/recall curve,
    then interpolates precision at 11 evenly spaced recall points.

    Args:
        detections: List of (confidence, is_true_positive) tuples.
        total_gt: Total number of ground truth boxes in the dataset.

    Returns:
        AP value in [0.0, 1.0]. Returns 0.0 if no GT boxes exist.
    """
    if total_gt == 0:
        return 0.0

    # Sort by confidence descending
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
    # at recall >= threshold (right-side maximum)
    ap = 0.0
    for r_thresh in [i / (_AP_RECALL_POINTS - 1) for i in range(_AP_RECALL_POINTS)]:
        max_prec = 0.0
        for p, r in zip(precisions, recalls):
            if r >= r_thresh:
                max_prec = max(max_prec, p)
        ap += max_prec

    return ap / _AP_RECALL_POINTS


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train(
    config_overrides: Union[dict[str, Any], None] = None,
) -> float:
    """Train the WaldoSSD model with the full pipeline.

    Sets up model, optimizer, scheduler, data loaders, and runs the
    training loop with periodic validation and checkpointing. Accepts
    config overrides for ablation experiments.

    Args:
        config_overrides: Optional dict to override default config values.
            Supported keys: 'resolution', 'color_mode', 'use_sobel',
            'num_heads', 'batch_size', 'learning_rate', 'num_epochs',
            'use_density_filter'.

    Returns:
        Best validation AP achieved during training.
    """
    overrides = config_overrides or {}

    # --- Resolve configuration ---
    resolution: int = int(overrides.get("resolution", DEFAULT_RESOLUTION))
    color_mode: str = str(overrides.get("color_mode", DEFAULT_COLOR_MODE))
    use_sobel: bool = bool(overrides.get("use_sobel", True))
    num_heads: int = int(overrides.get("num_heads", NUM_SSD_HEADS))
    batch_size: int = int(overrides.get("batch_size", BATCH_SIZE))
    learning_rate: float = float(overrides.get("learning_rate", LEARNING_RATE))
    num_epochs: int = int(overrides.get("num_epochs", NUM_EPOCHS))
    use_density_filter: bool = bool(overrides.get("use_density_filter", True))

    in_channels = 4 if use_sobel else 3

    logger.info("=== Training Configuration ===")
    logger.info(
        "resolution=%d, color=%s, sobel=%s, heads=%d, epochs=%d",
        resolution,
        color_mode,
        use_sobel,
        num_heads,
        num_epochs,
    )
    logger.info("batch_size=%d, lr=%.1e, device=%s", batch_size, learning_rate, DEVICE)

    # --- Model ---
    model = WaldoSSD(
        num_classes=NUM_CLASSES,
        num_heads=num_heads,
        in_channels=in_channels,
    )
    # Generate anchors on CPU before moving model to GPU/MPS
    # (get_feature_map_sizes runs a dummy forward pass that needs model on CPU)
    anchors = model.get_anchors(resolution)

    model = model.to(DEVICE)
    anchors = anchors.to(DEVICE)
    logger.info("Generated %d anchors for %dpx input", anchors.size(0), resolution)

    # --- Data loaders ---
    train_loader = get_dataloader(
        split="train",
        resolution=resolution,
        color_mode=color_mode,
        use_sobel=use_sobel,
        use_density_filter=use_density_filter,
        batch_size=batch_size,
    )
    val_loader = get_dataloader(
        split="val",
        resolution=resolution,
        color_mode=color_mode,
        use_sobel=use_sobel,
        use_density_filter=False,
        batch_size=batch_size,
    )

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- Loss ---
    criterion = SSDLoss(neg_pos_ratio=NEG_POS_RATIO, num_classes=NUM_CLASSES)

    # --- Checkpoint directory ---
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    best_val_ap: float = 0.0
    start_time = time.time()

    # --- Training loop ---
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = images.to(DEVICE)

            # Forward
            cls_preds, loc_preds = model(images)

            # Match anchors to GT
            cls_targets, reg_targets = match_anchors_to_targets(anchors, targets, resolution)

            # Compute loss
            total_loss, cls_loss_val, reg_loss_val = criterion(
                cls_preds, loc_preds, cls_targets, reg_targets
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_MAX_NORM)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss_val
            epoch_reg_loss += reg_loss_val
            num_batches += 1

        scheduler.step()

        # --- Epoch logging ---
        avg_loss = epoch_loss / max(1, num_batches)
        avg_cls = epoch_cls_loss / max(1, num_batches)
        avg_reg = epoch_reg_loss / max(1, num_batches)
        elapsed = time.time() - start_time

        logger.info(
            "Epoch %d/%d — loss=%.4f (cls=%.4f, reg=%.4f) lr=%.2e [%.0fs]",
            epoch,
            num_epochs,
            avg_loss,
            avg_cls,
            avg_reg,
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

        # --- Periodic validation ---
        if epoch % _VAL_EVERY_N_EPOCHS == 0 or epoch == num_epochs:
            val_ap = evaluate(model, val_loader, anchors, resolution, DEVICE)
            logger.info("  Val AP = %.4f (best = %.4f)", val_ap, best_val_ap)

            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_path = CHECKPOINTS / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_ap": val_ap,
                        "config": {
                            "resolution": resolution,
                            "color_mode": color_mode,
                            "use_sobel": use_sobel,
                            "num_heads": num_heads,
                            "in_channels": in_channels,
                        },
                    },
                    best_path,
                )
                logger.info("  Saved new best model to %s", best_path)

        # --- Periodic checkpoint ---
        if epoch % _CHECKPOINT_EVERY_N_EPOCHS == 0:
            ckpt_path = CHECKPOINTS / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_ap": best_val_ap,
                },
                ckpt_path,
            )
            logger.info("  Saved checkpoint to %s", ckpt_path)

    total_time = time.time() - start_time
    logger.info(
        "Training complete: %d epochs in %.0fs, best val AP = %.4f",
        num_epochs,
        total_time,
        best_val_ap,
    )

    return best_val_ap


# =============================================================================
# MAIN — QUICK SANITY CHECK
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Running 3-epoch training sanity check...")

    best_ap = train(
        config_overrides={
            "num_epochs": 3,
            "resolution": 256,
            "use_sobel": True,
            "use_density_filter": False,
        }
    )

    logger.info("Sanity check complete — best val AP = %.4f", best_ap)
