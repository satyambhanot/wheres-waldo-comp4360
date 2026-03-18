"""The neural network that actually finds Waldo.

We use MobileNet V3 Large as a lightweight backbone and bolt SSD
detection heads onto six feature-map levels so the model can spot
Waldo at different scales.  Supports 3-channel RGB or 4-channel
RGB + Sobel edge input.

Usage:
    python -m src.model        # run sanity checks
"""

import logging
import math

import torch
import torch.nn as nn
import torchvision.models as models

from src.config import NUM_CLASSES, NUM_SSD_HEADS, PRETRAINED, RESOLUTIONS

logger = logging.getLogger(__name__)


# --- Architecture constants ---

# How many channels come out of each backbone stage
_BACKBONE_CHANNELS: list[int] = [24, 40, 112, 960]

# Which layers of MobileNetV3.features form each stage
_STAGE_SLICES: list[tuple[int, int]] = [(0, 4), (4, 7), (7, 13), (13, 17)]

# Channel counts for the two extra blocks we add after the backbone
_EXTRA_BLOCK_1_CHANNELS: int = 512
_EXTRA_BLOCK_2_CHANNELS: int = 256

# All six feature-map channel counts, in detection order
_ALL_CHANNELS_6: list[int] = _BACKBONE_CHANNELS + [_EXTRA_BLOCK_1_CHANNELS, _EXTRA_BLOCK_2_CHANNELS]

# We try three anchor shapes at every spatial location
_ASPECT_RATIOS: list[float] = [1.0, 0.5, 2.0]
_ANCHORS_PER_CELL: int = len(_ASPECT_RATIOS)


# ---------------------------------------------------------------------------
# Anchor box generation
# ---------------------------------------------------------------------------


def generate_anchors(
    feature_map_sizes: list[tuple[int, int]],
    image_size: int,
) -> torch.Tensor:
    """Place default anchor boxes at every position on every feature map.

    Earlier (larger) feature maps get small anchors for catching tiny
    objects; later (smaller) maps get bigger anchors.  Three aspect
    ratios at each location give the detector some shape flexibility.

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
        # Deeper levels get larger anchors (linear spacing)
        scale = (level + 1) / (num_levels + 1)

        for row in range(fmap_h):
            for col in range(fmap_w):
                # Centre of this grid cell, normalised to [0, 1]
                cx = (col + 0.5) / fmap_w
                cy = (row + 0.5) / fmap_h

                for ratio in _ASPECT_RATIOS:
                    # Cap at 1.0 so anchors don't poke out of the image
                    w = min(scale * math.sqrt(ratio), 1.0)
                    h = min(scale / math.sqrt(ratio), 1.0)
                    anchors.append(torch.tensor([cx, cy, w, h]))

    return torch.stack(anchors, dim=0)


# ---------------------------------------------------------------------------
# Turning predictions back into real boxes
# ---------------------------------------------------------------------------


def decode_predictions(
    loc_preds: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Apply the model's offset predictions to the anchors to get real boxes.

    The model outputs small adjustments (dx, dy, dw, dh) relative to
    each anchor.  This function applies those tweaks and converts the
    result to corner-format [xmin, ymin, xmax, ymax].

    Args:
        loc_preds: Predicted offsets of shape (batch, num_anchors, 4).
        anchors: Anchor boxes of shape (num_anchors, 4) in
            [cx, cy, w, h] normalized format.

    Returns:
        Decoded boxes of shape (batch, num_anchors, 4) in
        [xmin, ymin, xmax, ymax] format, normalized to [0, 1].
    """
    # Unsqueeze anchors so they broadcast across the batch dimension
    a_cx = anchors[:, 0].unsqueeze(0)
    a_cy = anchors[:, 1].unsqueeze(0)
    a_w = anchors[:, 2].unsqueeze(0)
    a_h = anchors[:, 3].unsqueeze(0)

    pred_cx = a_cx + loc_preds[:, :, 0] * a_w
    pred_cy = a_cy + loc_preds[:, :, 1] * a_h
    pred_w = a_w * torch.exp(loc_preds[:, :, 2])
    pred_h = a_h * torch.exp(loc_preds[:, :, 3])

    # Switch from centre+size to corner format
    xmin = pred_cx - pred_w / 2
    ymin = pred_cy - pred_h / 2
    xmax = pred_cx + pred_w / 2
    ymax = pred_cy + pred_h / 2

    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


# ---------------------------------------------------------------------------
# Extra convolutional blocks (appended after the backbone)
# ---------------------------------------------------------------------------


def _make_extra_block(
    in_channels: int,
    mid_channels: int,
    out_channels: int,
) -> nn.Sequential:
    """Build one extra conv block that halves the spatial resolution.

    A 1x1 bottleneck squeezes channels, then a 3x3 strided conv
    expands them again while cutting the feature map in half.

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


# ---------------------------------------------------------------------------
# Detection heads (one conv each for classification and box regression)
# ---------------------------------------------------------------------------


def _make_head(
    in_channels: int,
    num_outputs_per_anchor: int,
) -> nn.Conv2d:
    """Create a single prediction head (either "is it Waldo?" or "where?").

    Xavier init keeps the initial loss reasonable; PyTorch's default
    would produce wild predictions on the first forward pass.

    Args:
        in_channels: Number of input feature channels.
        num_outputs_per_anchor: Number of outputs per anchor (num_classes
            for classification, 4 for regression).

    Returns:
        Conv2d layer mapping features to per-anchor predictions.
    """
    num_outputs = _ANCHORS_PER_CELL * num_outputs_per_anchor
    conv = nn.Conv2d(in_channels, num_outputs, kernel_size=3, padding=1)
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


# ---------------------------------------------------------------------------
# The main model
# ---------------------------------------------------------------------------


class WaldoSSD(nn.Module):
    """Our Waldo detector: MobileNet V3 backbone + SSD detection heads.

    Six feature-map levels look for Waldo at different scales — from
    fine details on the early maps to coarse shapes on the later ones.
    Each level asks two questions at every anchor: "is this Waldo?"
    (classification) and "where exactly is he?" (box regression).

    A 4-head ablation mode drops the two finest levels to see whether
    those small-scale features actually help.

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
        """Wire up the backbone, extra blocks, and detection heads.

        Args:
            num_classes: Number of object classes (background + Waldo = 2).
            num_heads: Detection levels — 6 for full model, 4 to skip the
                two finest scales.
            in_channels: 3 for plain RGB, 4 when Sobel edges are included.
            pretrained: Load ImageNet weights into the backbone?

        Raises:
            ValueError: If num_heads is not 4 or 6.
        """
        super().__init__()

        if num_heads not in (4, 6):
            raise ValueError(f"num_heads must be 4 or 6, got {num_heads}")

        self.num_classes = num_classes
        self.num_heads = num_heads
        self.in_channels = in_channels

        # Load the MobileNetV3 backbone (optionally with ImageNet weights)
        weights = "DEFAULT" if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)

        # If we're feeding in 4 channels (RGB + Sobel), widen the first conv
        if in_channels != 3:
            self._adapt_first_conv(backbone, in_channels)

        # Carve the backbone into four stages for multi-scale features
        self.stage1 = nn.Sequential(
            *list(backbone.features[_STAGE_SLICES[0][0] : _STAGE_SLICES[0][1]])
        )
        self.stage2 = nn.Sequential(
            *list(backbone.features[_STAGE_SLICES[1][0] : _STAGE_SLICES[1][1]])
        )
        self.stage3 = nn.Sequential(
            *list(backbone.features[_STAGE_SLICES[2][0] : _STAGE_SLICES[2][1]])
        )
        self.stage4 = nn.Sequential(
            *list(backbone.features[_STAGE_SLICES[3][0] : _STAGE_SLICES[3][1]])
        )

        # Two extra blocks give us coarser feature maps for large-scale detection
        self.extra1 = _make_extra_block(_BACKBONE_CHANNELS[3], 256, _EXTRA_BLOCK_1_CHANNELS)
        self.extra2 = _make_extra_block(_EXTRA_BLOCK_1_CHANNELS, 128, _EXTRA_BLOCK_2_CHANNELS)

        # Build one classification + one regression head per active level
        # (in 4-head mode we simply skip the two finest-resolution stages)
        if num_heads == 6:
            active_channels = _ALL_CHANNELS_6
        else:
            active_channels = _ALL_CHANNELS_6[2:]  # stage3 onward

        self.cls_heads = nn.ModuleList([_make_head(ch, num_classes) for ch in active_channels])
        self.reg_heads = nn.ModuleList([_make_head(ch, 4) for ch in active_channels])

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
        """Widen the first conv to accept extra input channels.

        We keep the pretrained RGB weights intact and initialise the
        new channel(s) as the average of the RGB filters — a much
        better starting point than random noise.

        Args:
            backbone: The MobileNetV3 model whose first conv to modify.
            in_channels: Desired number of input channels.
        """
        # The very first layer is a ConvBNActivation block; dig into it
        first_block = backbone.features[0]  # type: ignore[index]
        old_conv = first_block[0]
        assert isinstance(old_conv, nn.Conv2d)

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,  # type: ignore[arg-type]
            stride=old_conv.stride,  # type: ignore[arg-type]
            padding=old_conv.padding,  # type: ignore[arg-type]
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            # Carry over the learned RGB filters
            new_conv.weight[:, :3] = old_conv.weight
            # Initialise extra channels with the RGB average
            rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = rgb_mean

            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        first_block[0] = new_conv  # type: ignore[assignment]

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Push a batch of images through the network.

        Each detection level produces per-anchor class scores and box
        offsets; we concatenate everything into two big tensors.

        Args:
            x: Input image batch of shape (B, C, H, W) where C is
                3 (RGB) or 4 (RGB + Sobel).

        Returns:
            Tuple of (class_logits, box_offsets) where:
                - class_logits: (B, total_anchors, num_classes).
                - box_offsets: (B, total_anchors, 4) as [dx, dy, dw, dh].
        """
        # Run the image through four backbone stages
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        # Push through the two extra blocks for coarser features
        e1 = self.extra1(s4)
        e2 = self.extra2(e1)

        # Pick the feature maps we're actually using
        if self.num_heads == 6:
            feature_maps = [s1, s2, s3, s4, e1, e2]
        else:
            feature_maps = [s3, s4, e1, e2]

        # Run every active head and reshape to (B, anchors, ...)
        cls_preds: list[torch.Tensor] = []
        loc_preds: list[torch.Tensor] = []
        batch_size = x.size(0)

        for fmap, cls_head, reg_head in zip(feature_maps, self.cls_heads, self.reg_heads):
            cls = cls_head(fmap)
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(batch_size, -1, self.num_classes)
            cls_preds.append(cls)

            loc = reg_head(fmap)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(batch_size, -1, 4)
            loc_preds.append(loc)

        # Flatten across all levels into one big prediction tensor
        cls_out = torch.cat(cls_preds, dim=1)
        loc_out = torch.cat(loc_preds, dim=1)

        return cls_out, loc_out

    def get_feature_map_sizes(
        self,
        image_size: int,
    ) -> list[tuple[int, int]]:
        """Figure out how big each feature map is for a given input size.

        We just run a dummy image through and measure, which is the
        only reliable way to account for rounding in stride/padding.

        Args:
            image_size: Input image size in pixels (assumes square).

        Returns:
            List of (height, width) tuples, one per active detection level.
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        dummy = torch.zeros(1, self.in_channels, image_size, image_size, device=device)
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

        # Put the model back in whatever mode it was in before
        if was_training:
            self.train()

        return sizes

    def get_anchors(
        self,
        image_size: int,
    ) -> torch.Tensor:
        """Shortcut: generate the right anchors for this model and input size.

        Args:
            image_size: Input image size in pixels (assumes square).

        Returns:
            Tensor of shape (total_anchors, 4) in [cx, cy, w, h] format,
            normalized to [0, 1].
        """
        fmap_sizes = self.get_feature_map_sizes(image_size)
        return generate_anchors(fmap_sizes, image_size)


# ---------------------------------------------------------------------------
# Quick sanity checks when run directly
# ---------------------------------------------------------------------------


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
    assert cls_out.size(1) == anchors_256.size(
        0
    ), f"Anchor count mismatch: cls={cls_out.size(1)}, anchors={anchors_256.size(0)}"
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
    # Use eval mode because 64px produces 1x1 feature maps where
    # BatchNorm requires >1 element in training mode.
    model_3ch.eval()
    for res in RESOLUTIONS:
        fmap_sizes = model_3ch.get_feature_map_sizes(res)
        anchors = model_3ch.get_anchors(res)
        x_res = torch.randn(1, 3, res, res)
        cls_r, loc_r = model_3ch(x_res)

        assert cls_r.size(1) == anchors.size(
            0
        ), f"Resolution {res}: cls={cls_r.size(1)} != anchors={anchors.size(0)}"
        logger.info(
            "  %dpx: fmap_sizes=%s, anchors=%d",
            res,
            fmap_sizes,
            anchors.size(0),
        )

    # --- Test 5: Decode predictions ---
    decoded = decode_predictions(loc_out, anchors_256)
    assert decoded.shape == (2, anchors_256.size(0), 4), f"Decode shape mismatch: {decoded.shape}"
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
