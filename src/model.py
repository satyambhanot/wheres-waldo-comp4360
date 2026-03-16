"""SSD (Single Shot MultiBox Detector) with MobileNet V3 Large backbone.

Adapted from a strawberry detection architecture to the Where's Waldo
domain. The backbone is split into 4 stages producing multi-scale
feature maps, followed by 2 extra convolutional blocks for a total of
6 detection levels. Supports both 3-channel (RGB) and 4-channel
(RGB + Sobel edge) input for the preprocessing ablation study.

Usage:
    python -m src.model        # run sanity checks
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

from src.config import NUM_CLASSES, NUM_SSD_HEADS, PRETRAINED, RESOLUTIONS

logger = logging.getLogger(__name__)


# === Constants ===

# Channel counts at the output of each MobileNetV3 backbone stage
_BACKBONE_CHANNELS: list[int] = [24, 40, 112, 960]

# Slice boundaries to split MobileNetV3 .features into 4 stages
_STAGE_SLICES: list[tuple[int, int]] = [(0, 4), (4, 7), (7, 13), (13, 17)]

# Extra block output channel counts (after backbone)
_EXTRA1_OUT: int = 512
_EXTRA2_OUT: int = 256

# All 6 feature map channel counts in detection order
_ALL_CHANNELS_6: list[int] = _BACKBONE_CHANNELS + [_EXTRA1_OUT, _EXTRA2_OUT]

# Anchors per spatial location — one per aspect ratio
_ASPECT_RATIOS: list[float] = [1.0, 0.5, 2.0]
_NUM_ANCHORS_PER_LOCATION: int = len(_ASPECT_RATIOS)


# =============================================================================
# ANCHOR GENERATION
# =============================================================================


def generate_anchors(
    feature_map_sizes: list[tuple[int, int]],
    image_size: int,
) -> torch.Tensor:
    """Generate SSD anchor boxes for all feature map levels.

    Creates a set of default anchor boxes at each spatial location of
    every feature map. Anchors are scaled proportionally so that earlier
    (finer) feature maps detect smaller objects and later (coarser)
    maps detect larger ones.

    Args:
        feature_map_sizes: List of (height, width) tuples, one per
            detection level, ordered from finest to coarsest.
        image_size: Input image size in pixels (assumes square input).

    Returns:
        Tensor of shape (total_anchors, 4) in [cx, cy, w, h] format,
        all values normalized to [0, 1].
    """
    anchors: list[torch.Tensor] = []
    num_levels = len(feature_map_sizes)

    for level, (fmap_h, fmap_w) in enumerate(feature_map_sizes):
        # Linearly spaced scales so deeper levels cover larger objects
        scale = (level + 1) / (num_levels + 1)

        for row in range(fmap_h):
            for col in range(fmap_w):
                # Center of this anchor cell, normalized to [0, 1]
                cx = (col + 0.5) / fmap_w
                cy = (row + 0.5) / fmap_h

                for ratio in _ASPECT_RATIOS:
                    w = scale * math.sqrt(ratio)
                    h = scale / math.sqrt(ratio)
                    anchors.append(torch.tensor([cx, cy, w, h]))

    return torch.stack(anchors, dim=0)


# =============================================================================
# DECODE PREDICTIONS
# =============================================================================


def decode_predictions(
    loc_preds: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Convert SSD offset predictions to absolute bounding boxes.

    Applies the predicted (dx, dy, dw, dh) offsets to the anchor boxes
    to obtain absolute coordinates suitable for NMS and evaluation.

    Args:
        loc_preds: Predicted offsets of shape (batch, num_anchors, 4).
            Format: [dx, dy, dw, dh] where dx/dy are center offsets
            scaled by anchor size, and dw/dh are log-scale adjustments.
        anchors: Anchor boxes of shape (num_anchors, 4) in
            [cx, cy, w, h] normalized format.

    Returns:
        Decoded boxes of shape (batch, num_anchors, 4) in
        [xmin, ymin, xmax, ymax] format, normalized to [0, 1].
    """
    # anchors: (A, 4) -> (1, A, 4) for broadcasting with batch dim
    a_cx = anchors[:, 0].unsqueeze(0)
    a_cy = anchors[:, 1].unsqueeze(0)
    a_w = anchors[:, 2].unsqueeze(0)
    a_h = anchors[:, 3].unsqueeze(0)

    pred_cx = a_cx + loc_preds[:, :, 0] * a_w
    pred_cy = a_cy + loc_preds[:, :, 1] * a_h
    pred_w = a_w * torch.exp(loc_preds[:, :, 2])
    pred_h = a_h * torch.exp(loc_preds[:, :, 3])

    # Convert (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
    xmin = pred_cx - pred_w / 2
    ymin = pred_cy - pred_h / 2
    xmax = pred_cx + pred_w / 2
    ymax = pred_cy + pred_h / 2

    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


# =============================================================================
# EXTRA CONVOLUTIONAL BLOCKS
# =============================================================================


def _make_extra_block(
    in_channels: int,
    mid_channels: int,
    out_channels: int,
) -> nn.Sequential:
    """Build one extra SSD feature extraction block.

    Each block consists of a 1x1 bottleneck conv followed by a 3x3
    strided conv that halves the spatial resolution. Both layers use
    BatchNorm and ReLU.

    Args:
        in_channels: Number of input channels.
        mid_channels: Number of channels after the 1x1 bottleneck.
        out_channels: Number of output channels after the 3x3 conv.

    Returns:
        Sequential block that reduces spatial dims by 2x.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


# =============================================================================
# SSD DETECTION HEADS
# =============================================================================


def _make_head(
    in_channels: int,
    num_outputs_per_anchor: int,
) -> nn.Conv2d:
    """Create a single SSD prediction head (classification or regression).

    Uses Xavier initialization for stable early training — default
    PyTorch init can cause large initial losses with SSD.

    Args:
        in_channels: Number of input feature channels.
        num_outputs_per_anchor: Number of outputs per anchor (num_classes
            for classification, 4 for regression).

    Returns:
        Conv2d layer mapping features to per-anchor predictions.
    """
    num_outputs = _NUM_ANCHORS_PER_LOCATION * num_outputs_per_anchor
    conv = nn.Conv2d(in_channels, num_outputs, kernel_size=3, padding=1)
    nn.init.xavier_uniform_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


# =============================================================================
# WALDO SSD MODEL
# =============================================================================


class WaldoSSD(nn.Module):
    """SSD object detector with MobileNet V3 Large backbone for Waldo detection.

    The architecture extracts multi-scale features from 4 backbone stages
    plus 2 extra convolutional blocks, giving 6 detection levels total.
    Each level applies a classification head (is this Waldo?) and a
    regression head (where exactly is Waldo?) with 3 anchors per
    spatial location.

    A 4-head ablation mode drops the two finest-resolution levels
    (stages 1–2) to study the effect of small-scale features.

    Attributes:
        num_classes: Number of object classes (including background).
        num_heads: Number of active detection levels (4 or 6).
        in_channels: Number of input image channels (3 or 4).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_heads: int = NUM_SSD_HEADS,
        in_channels: int = 3,
        pretrained: bool = PRETRAINED,
    ) -> None:
        """Initialize WaldoSSD.

        Args:
            num_classes: Number of object classes (background + Waldo = 2).
            num_heads: Number of SSD detection levels. Use 6 for the full
                model or 4 for the ablation study (drops finest 2 levels).
            in_channels: Number of input channels. Use 3 for RGB or 4 for
                RGB + Sobel edge channel.
            pretrained: Whether to load ImageNet-pretrained backbone weights.

        Raises:
            ValueError: If num_heads is not 4 or 6.
        """
        super().__init__()

        if num_heads not in (4, 6):
            raise ValueError(f"num_heads must be 4 or 6, got {num_heads}")

        self.num_classes = num_classes
        self.num_heads = num_heads
        self.in_channels = in_channels

        # --- Backbone ---
        weights = "DEFAULT" if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)

        # Modify first conv if using 4-channel input
        if in_channels != 3:
            self._adapt_first_conv(backbone, in_channels)

        # Split backbone .features into 4 stages for multi-scale extraction
        self.stage1 = nn.Sequential(*list(backbone.features[_STAGE_SLICES[0][0]:_STAGE_SLICES[0][1]]))
        self.stage2 = nn.Sequential(*list(backbone.features[_STAGE_SLICES[1][0]:_STAGE_SLICES[1][1]]))
        self.stage3 = nn.Sequential(*list(backbone.features[_STAGE_SLICES[2][0]:_STAGE_SLICES[2][1]]))
        self.stage4 = nn.Sequential(*list(backbone.features[_STAGE_SLICES[3][0]:_STAGE_SLICES[3][1]]))

        # --- Extra layers for additional detection scales ---
        self.extra1 = _make_extra_block(_BACKBONE_CHANNELS[3], 256, _EXTRA1_OUT)
        self.extra2 = _make_extra_block(_EXTRA1_OUT, 128, _EXTRA2_OUT)

        # --- Detection heads ---
        # In 4-head mode, skip the 2 finest levels (stage1, stage2)
        if num_heads == 6:
            active_channels = _ALL_CHANNELS_6
        else:
            active_channels = _ALL_CHANNELS_6[2:]  # stage3 onward

        self.cls_heads = nn.ModuleList([
            _make_head(ch, num_classes) for ch in active_channels
        ])
        self.reg_heads = nn.ModuleList([
            _make_head(ch, 4) for ch in active_channels
        ])

        logger.info(
            "WaldoSSD initialized: %d classes, %d heads, %d-ch input, pretrained=%s",
            num_classes,
            num_heads,
            in_channels,
            pretrained,
        )

    def _adapt_first_conv(
        self,
        backbone: nn.Module,
        in_channels: int,
    ) -> None:
        """Modify the backbone's first conv layer for non-3-channel input.

        Copies pretrained RGB weights into the first 3 channels and
        initializes extra channels as the mean of the RGB weights.
        This lets the Sobel channel start with a reasonable filter
        rather than random noise.

        Args:
            backbone: The MobileNetV3 model whose first conv to modify.
            in_channels: Desired number of input channels.
        """
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            # Copy pretrained RGB weights
            new_conv.weight[:, :3] = old_conv.weight
            # Extra channels get the mean of RGB — better than random
            rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = rgb_mean

            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        backbone.features[0][0] = new_conv

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass through the SSD.

        Extracts multi-scale features from the backbone and extra layers,
        then applies classification and regression heads at each level.
        Predictions from all levels are concatenated into flat tensors.

        Args:
            x: Input image batch of shape (B, C, H, W) where C is
                3 (RGB) or 4 (RGB + Sobel).

        Returns:
            Tuple of (cls_preds, loc_preds) where:
                - cls_preds: (B, total_anchors, num_classes) classification
                  logits (NOT softmaxed).
                - loc_preds: (B, total_anchors, 4) bounding box offsets
                  as [dx, dy, dw, dh].
        """
        # --- Backbone feature extraction ---
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        # --- Extra layers ---
        e1 = self.extra1(s4)
        e2 = self.extra2(e1)

        # --- Collect feature maps for active heads ---
        if self.num_heads == 6:
            feature_maps = [s1, s2, s3, s4, e1, e2]
        else:
            feature_maps = [s3, s4, e1, e2]

        # --- Apply detection heads ---
        cls_preds: list[torch.Tensor] = []
        loc_preds: list[torch.Tensor] = []
        batch_size = x.size(0)

        for fmap, cls_head, reg_head in zip(
            feature_maps, self.cls_heads, self.reg_heads
        ):
            # cls: (B, anchors_per_loc * num_classes, H, W)
            cls = cls_head(fmap)
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(batch_size, -1, self.num_classes)
            cls_preds.append(cls)

            # loc: (B, anchors_per_loc * 4, H, W)
            loc = reg_head(fmap)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(batch_size, -1, 4)
            loc_preds.append(loc)

        # Concatenate predictions from all levels
        cls_out = torch.cat(cls_preds, dim=1)
        loc_out = torch.cat(loc_preds, dim=1)

        return cls_out, loc_out

    def get_feature_map_sizes(
        self,
        image_size: int,
    ) -> list[tuple[int, int]]:
        """Compute feature map spatial dimensions for a given input size.

        Runs a dummy forward pass to determine the exact output sizes
        at each detection level. This accounts for any rounding in
        stride/padding arithmetic.

        Args:
            image_size: Input image size in pixels (assumes square).

        Returns:
            List of (height, width) tuples, one per active detection level.
        """
        self.eval()
        dummy = torch.zeros(1, self.in_channels, image_size, image_size)
        sizes: list[tuple[int, int]] = []

        with torch.no_grad():
            s1 = self.stage1(dummy)
            s2 = self.stage2(s1)
            s3 = self.stage3(s2)
            s4 = self.stage4(s3)
            e1 = self.extra1(s4)
            e2 = self.extra2(e1)

            if self.num_heads == 6:
                feature_maps = [s1, s2, s3, s4, e1, e2]
            else:
                feature_maps = [s3, s4, e1, e2]

            for fmap in feature_maps:
                sizes.append((fmap.size(2), fmap.size(3)))

        return sizes

    def get_anchors(
        self,
        image_size: int,
    ) -> torch.Tensor:
        """Generate anchors matching the model's output dimensions.

        Convenience method that chains get_feature_map_sizes and
        generate_anchors so callers don't need to handle the plumbing.

        Args:
            image_size: Input image size in pixels (assumes square).

        Returns:
            Tensor of shape (total_anchors, 4) in [cx, cy, w, h] format,
            normalized to [0, 1].
        """
        fmap_sizes = self.get_feature_map_sizes(image_size)
        return generate_anchors(fmap_sizes, image_size)


# =============================================================================
# MAIN — SANITY CHECKS
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Running WaldoSSD sanity checks...")

    # --- Test 1: 3-channel forward pass ---
    model_3ch = WaldoSSD(num_classes=NUM_CLASSES, num_heads=6, in_channels=3)
    x_rgb = torch.randn(2, 3, 256, 256)
    cls_out, loc_out = model_3ch(x_rgb)
    anchors_256 = model_3ch.get_anchors(256)

    assert cls_out.size(0) == 2, "Batch size mismatch"
    assert cls_out.size(1) == anchors_256.size(0), (
        f"Anchor count mismatch: cls={cls_out.size(1)}, anchors={anchors_256.size(0)}"
    )
    assert cls_out.size(2) == NUM_CLASSES, "Class dim mismatch"
    assert loc_out.size(2) == 4, "Loc dim mismatch"
    logger.info(
        "  3ch RGB (2,3,256,256): cls=%s, loc=%s, anchors=%d",
        list(cls_out.shape),
        list(loc_out.shape),
        anchors_256.size(0),
    )

    # --- Test 2: 4-channel forward pass ---
    model_4ch = WaldoSSD(num_classes=NUM_CLASSES, num_heads=6, in_channels=4)
    x_sobel = torch.randn(2, 4, 256, 256)
    cls_4ch, loc_4ch = model_4ch(x_sobel)
    anchors_4ch = model_4ch.get_anchors(256)

    assert cls_4ch.size(1) == anchors_4ch.size(0), "4ch anchor mismatch"
    logger.info(
        "  4ch RGB+Sobel (2,4,256,256): cls=%s, loc=%s",
        list(cls_4ch.shape),
        list(loc_4ch.shape),
    )

    # --- Test 3: 4-head ablation ---
    model_4h = WaldoSSD(num_classes=NUM_CLASSES, num_heads=4, in_channels=3)
    cls_4h, loc_4h = model_4h(x_rgb)
    anchors_4h = model_4h.get_anchors(256)

    assert cls_4h.size(1) == anchors_4h.size(0), "4-head anchor mismatch"
    assert cls_4h.size(1) < cls_out.size(1), "4-head should have fewer anchors"
    logger.info(
        "  4-head ablation: cls=%s, loc=%s, anchors=%d (vs 6-head=%d)",
        list(cls_4h.shape),
        list(loc_4h.shape),
        anchors_4h.size(0),
        anchors_256.size(0),
    )

    # --- Test 4: All resolutions ---
    for res in RESOLUTIONS:
        fmap_sizes = model_3ch.get_feature_map_sizes(res)
        anchors = model_3ch.get_anchors(res)
        x_res = torch.randn(1, 3, res, res)
        cls_r, loc_r = model_3ch(x_res)

        assert cls_r.size(1) == anchors.size(0), (
            f"Resolution {res}: cls={cls_r.size(1)} != anchors={anchors.size(0)}"
        )
        logger.info(
            "  %dpx: fmap_sizes=%s, anchors=%d",
            res,
            fmap_sizes,
            anchors.size(0),
        )

    # --- Test 5: Decode predictions ---
    decoded = decode_predictions(loc_out, anchors_256)
    assert decoded.shape == (2, anchors_256.size(0), 4), (
        f"Decode shape mismatch: {decoded.shape}"
    )
    logger.info("  decode_predictions: %s", list(decoded.shape))

    # --- Test 6: Feature map sizes ---
    for res in RESOLUTIONS:
        sizes = model_3ch.get_feature_map_sizes(res)
        logger.info("  Feature maps @ %dpx: %s", res, sizes)

    # --- Test 7: Parameter count ---
    total_params = sum(p.numel() for p in model_3ch.parameters())
    trainable_params = sum(p.numel() for p in model_3ch.parameters() if p.requires_grad)
    logger.info(
        "  Parameters: %.2fM total, %.2fM trainable",
        total_params / 1e6,
        trainable_params / 1e6,
    )

    logger.info("All sanity checks passed!")
