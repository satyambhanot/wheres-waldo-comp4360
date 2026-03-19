"""Trains the Waldo detector from scratch (or fine-tunes it).

Handles the full loop: match anchors to ground truth, compute the SSD
loss with hard negative mining, update weights, and periodically check
validation AP.  Pass config_overrides to swap in ablation settings.

Usage:
    python -m src.train        # 3-epoch sanity check
"""

import logging
import time
from pathlib import Path
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS,
    DEFAULT_COLOR_MODE,
    DEFAULT_RESOLUTION,
    DEVICE,
    LEARNING_RATE,
    NEG_POS_RATIO,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_SSD_HEADS,
    WEIGHT_DECAY,
)
from src.dataset import get_dataloader
from src.evaluate import _compute_voc_ap, apply_nms, match_detections_to_gt
from src.model import WaldoSSD, decode_predictions

logger = logging.getLogger(__name__)


# ---- Constants ----

# An anchor needs at least this much overlap with a GT box to be "positive"
_IOU_MATCH_THRESHOLD: float = 0.5

# Cap gradient norms to keep early training stable
_GRAD_CLIP_MAX_NORM: float = 5.0

# How often to check validation performance (every N epochs)
_VAL_EVERY_N_EPOCHS: int = 5

# How often to save a periodic checkpoint (every N epochs)
_CHECKPOINT_EVERY_N_EPOCHS: int = 20

# Stop training if val AP doesn't improve for this many validation checks
_EARLY_STOP_PATIENCE: int = 6

# Linearly ramp up LR from near-zero over this many epochs
_WARMUP_EPOCHS: int = 5


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Centre+size  ->  corner format.

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
    """Corner format  ->  centre+size.

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


# ---- Anchor matching ----


def match_anchors_to_targets(
    anchors: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decide which anchors are "positive" (responsible for detecting Waldo).

    Two-pass approach:
      1. Every GT box gets its single best anchor (so nothing is missed).
      2. Any remaining anchor with IoU >= 0.5 also becomes positive.

    Args:
        anchors: Anchor boxes (num_anchors, 4) in normalised [cx, cy, w, h].
        targets: List of B target dicts, each with 'boxes' (N, 4) in
            pixel [xmin, ymin, xmax, ymax] and 'labels' (N,).
        resolution: Image resolution in pixels, for normalising GT boxes.

    Returns:
        Tuple of (class_labels, box_offsets) where:
            - class_labels: (B, num_anchors) — 0 = background, 1 = Waldo.
            - box_offsets: (B, num_anchors, 4) — regression targets.
    """
    batch_size = len(targets)
    num_anchors = anchors.size(0)
    device = anchors.device

    cls_targets = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
    reg_targets = torch.zeros(batch_size, num_anchors, 4, dtype=torch.float32, device=device)

    anchors_xyxy = cxcywh_to_xyxy(anchors)

    for i, target in enumerate(targets):
        gt_boxes_px = target["boxes"]

        if gt_boxes_px.numel() == 0:
            continue

        # Scale GT from pixels to [0, 1] to match anchor coordinates
        gt_boxes_xyxy = gt_boxes_px.to(device).float() / resolution
        gt_boxes_cxcywh = xyxy_to_cxcywh(gt_boxes_xyxy)

        iou_matrix = box_iou(anchors_xyxy, gt_boxes_xyxy)

        # Pass 1: guarantee every GT box has at least one positive anchor
        best_anchor_per_gt = iou_matrix.argmax(dim=0)
        for gt_idx in range(gt_boxes_xyxy.size(0)):
            anchor_idx = best_anchor_per_gt[gt_idx]
            cls_targets[i, anchor_idx] = 1
            reg_targets[i, anchor_idx] = _encode_offsets(
                anchors[anchor_idx], gt_boxes_cxcywh[gt_idx]
            )

        # Pass 2: also promote any anchor that overlaps enough with a GT
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
    """How far does the anchor need to shift to match the ground truth?

    Centre offsets are divided by anchor size so small and large
    anchors learn on the same scale.  Width/height use log-space.

    Args:
        anchor: Single anchor box (4,) in [cx, cy, w, h].
        gt_box: Single GT box (4,) in [cx, cy, w, h].

    Returns:
        Offset tensor (4,) as [dx, dy, dw, dh].
    """
    # Tiny clamp to avoid division-by-zero or log(0)
    anchor_w = anchor[2].clamp(min=1e-6)
    anchor_h = anchor[3].clamp(min=1e-6)

    dx = (gt_box[0] - anchor[0]) / anchor_w
    dy = (gt_box[1] - anchor[1]) / anchor_h
    dw = torch.log(gt_box[2].clamp(min=1e-6) / anchor_w)
    dh = torch.log(gt_box[3].clamp(min=1e-6) / anchor_h)
    return torch.stack([dx, dy, dw, dh])


# ---- SSD loss ----


class SSDLoss(nn.Module):
    """Classification + localisation loss, balanced by hard negative mining.

    Waldo tiles are hugely imbalanced (~3 % positive), so we can't
    just average over all anchors.  Instead we keep all positives and
    cherry-pick the hardest negatives — the background anchors the
    model was most confused about.

    Attributes:
        neg_pos_ratio: Number of hard negatives per positive anchor.
        num_classes: Number of object classes (including background).
    """

    def __init__(
        self,
        neg_pos_ratio: int = NEG_POS_RATIO,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Set up the loss function.

        Args:
            neg_pos_ratio: How many hard negatives to keep for each positive.
                Higher = more background examples but slower to learn Waldo.
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
        """Compute the combined classification + box-regression loss.

        Args:
            cls_preds: Classification logits (B, num_anchors, num_classes).
            loc_preds: Regression offsets (B, num_anchors, 4).
            cls_targets: Ground truth class labels (B, num_anchors).
            reg_targets: Ground truth regression offsets (B, num_anchors, 4).

        Returns:
            (total_loss, cls_loss_float, reg_loss_float) — the first is
            differentiable for backprop, the others are for logging.
        """
        batch_size = cls_preds.size(0)
        num_anchors = cls_preds.size(1)

        # Which anchors actually have a Waldo to learn from?
        positive_mask = cls_targets > 0
        num_positives = positive_mask.sum().item()

        if num_positives == 0:
            # Nothing to learn this batch — zero loss avoids div-by-zero
            zero = torch.tensor(0.0, device=cls_preds.device, requires_grad=True)
            return zero, 0.0, 0.0

        # Box regression only matters for positive anchors
        pos_mask_expanded = positive_mask.unsqueeze(2).expand_as(loc_preds)
        reg_loss = F.smooth_l1_loss(
            loc_preds[pos_mask_expanded].view(-1, 4),
            reg_targets[pos_mask_expanded].view(-1, 4),
            reduction="sum",
        )

        # Classification loss: we need it per-anchor first for mining
        cls_loss_all = F.cross_entropy(
            cls_preds.view(-1, self.num_classes),
            cls_targets.view(-1),
            reduction="none",
        ).view(batch_size, num_anchors)

        # Temporarily blank out positives so we can rank negatives
        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[positive_mask] = 0.0

        # Hard negative mining: keep only the background anchors the
        # model was most wrong about (sort on CPU — MPS sort is flaky)
        cls_loss_neg_cpu = cls_loss_neg.cpu()
        _, neg_sort_idx = cls_loss_neg_cpu.sort(dim=1, descending=True)
        neg_sort_idx = neg_sort_idx.to(cls_preds.device)

        # For each image, keep neg_pos_ratio negatives per positive
        num_pos_per_image = positive_mask.sum(dim=1)
        num_neg_per_image = (num_pos_per_image * self.neg_pos_ratio).clamp(min=1)

        hard_neg_mask = torch.zeros_like(positive_mask, dtype=torch.bool)
        for b in range(batch_size):
            num_hard = int(num_neg_per_image[b].item())
            hardest_indices = neg_sort_idx[b, :num_hard]
            hard_neg_mask[b, hardest_indices] = True

        # Final loss uses only positives + the hardest negatives
        selected_mask = positive_mask | hard_neg_mask
        cls_loss = cls_loss_all[selected_mask].sum()

        # Normalise by the number of positive anchors in the batch
        num_pos_float = float(num_positives)
        total_loss = (cls_loss + reg_loss) / num_pos_float

        return total_loss, cls_loss.item() / num_pos_float, reg_loss.item() / num_pos_float


# ---- Validation ----


def evaluate(
    model: WaldoSSD,
    loader: DataLoader,
    anchors: torch.Tensor,
    resolution: int,
    device: torch.device,
) -> float:
    """Fast AP check to see how training is going.

    Uses the same NMS / matching / AP helpers as the full evaluator
    but skips the visual-example saving.

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

    all_scored_detections: list[tuple[float, bool]] = []
    total_gt_boxes = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            cls_preds, loc_preds = model(images)

            cls_probs = F.softmax(cls_preds, dim=-1)
            decoded_boxes = decode_predictions(loc_preds, anchors)

            for b in range(images.size(0)):
                gt_boxes_px = targets[b]["boxes"]
                total_gt_boxes += gt_boxes_px.size(0)

                scores = cls_probs[b, :, 1]
                boxes = decoded_boxes[b]

                boxes_filtered, scores_filtered = apply_nms(boxes, scores)
                boxes_px = boxes_filtered * resolution

                gt_on_device = gt_boxes_px.to(device).float()
                dets, _ = match_detections_to_gt(boxes_px, scores_filtered, gt_on_device)
                all_scored_detections.extend(dets)

    ap = _compute_voc_ap(all_scored_detections, total_gt_boxes)

    model.train()
    return ap


# ---- Training loop ----


def train(
    config_overrides: dict[str, Any] | None = None,
    checkpoint_prefix: str = "best_model",
    resume_from: str | None = None,
) -> float:
    """Train WaldoSSD end-to-end and return the best validation AP.

    Everything is wired up here: model, optimiser, data loaders,
    learning-rate schedule, checkpointing.  Pass config_overrides to
    swap in different settings for ablation experiments.

    Args:
        config_overrides: Optional dict to override default config values.
            Supported keys: 'resolution', 'color_mode', 'use_sobel',
            'num_heads', 'batch_size', 'learning_rate', 'num_epochs',
            'use_density_filter'.
        checkpoint_prefix: Filename prefix for the best-model checkpoint.
            Each caller can supply a unique prefix to avoid overwriting
            other experiments' checkpoints.
        resume_from: Path to a checkpoint file to resume training from.
            Restores model weights, optimizer state, scheduler state,
            and best val AP, then continues from the saved epoch.

    Returns:
        Best validation AP achieved during training.
    """
    overrides = config_overrides or {}

    # Pull out the settings we'll actually use
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

    # Build the model and its anchors
    model = WaldoSSD(
        num_classes=NUM_CLASSES,
        num_heads=num_heads,
        in_channels=in_channels,
    )
    # Anchors must be generated on CPU first (dummy forward pass)
    anchors = model.get_anchors(resolution)

    model = model.to(DEVICE)
    anchors = anchors.to(DEVICE)
    logger.info("Generated %d anchors for %dpx input", anchors.size(0), resolution)

    # Set up the data pipelines
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

    # Optimiser with linear warmup then cosine decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=_WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - _WARMUP_EPOCHS)
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[_WARMUP_EPOCHS],
    )

    # Loss function with hard negative mining
    criterion = SSDLoss(neg_pos_ratio=NEG_POS_RATIO, num_classes=NUM_CLASSES)

    # Make sure the checkpoint folder exists
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    best_val_ap: float = 0.0
    epochs_without_improvement: int = 0
    start_epoch: int = 1

    # Resume from a previous checkpoint if requested
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_ap = ckpt.get("val_ap", 0.0)
        start_epoch = ckpt["epoch"] + 1
        logger.info(
            "Resumed from %s (epoch %d, val_ap=%.4f)",
            resume_path,
            ckpt["epoch"],
            best_val_ap,
        )

    start_time = time.time()

    # Go!
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = images.to(DEVICE)
            cls_preds, loc_preds = model(images)
            cls_targets, reg_targets = match_anchors_to_targets(anchors, targets, resolution)
            total_loss, cls_loss_val, reg_loss_val = criterion(
                cls_preds, loc_preds, cls_targets, reg_targets
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_MAX_NORM)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss_val
            epoch_reg_loss += reg_loss_val
            num_batches += 1

        scheduler.step()

        # Log this epoch's numbers
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

        # Check validation AP every few epochs
        if epoch % _VAL_EVERY_N_EPOCHS == 0 or epoch == num_epochs:
            val_ap = evaluate(model, val_loader, anchors, resolution, DEVICE)
            logger.info("  Val AP = %.4f (best = %.4f)", val_ap, best_val_ap)

            if val_ap > best_val_ap:
                best_val_ap = val_ap
                epochs_without_improvement = 0
                best_path = CHECKPOINTS / f"{checkpoint_prefix}.pth"
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
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= _EARLY_STOP_PATIENCE:
                    logger.info(
                        "  Early stopping: no improvement for %d validation checks",
                        _EARLY_STOP_PATIENCE,
                    )
                    break

        # Save a checkpoint every so often, just in case
        if epoch % _CHECKPOINT_EVERY_N_EPOCHS == 0:
            ckpt_path = CHECKPOINTS / f"{checkpoint_prefix}_epoch_{epoch:03d}.pth"
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


# ---- Main ----


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
