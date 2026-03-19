"""Runs all our "what if we changed X?" experiments.

Four ablation studies test how Sobel edges, tile resolution, colour
mode, and model size each affect detection quality.  Every experiment
is reproducible (seeded), resumable (skips finished runs), and logs
its metrics to JSON.

Usage:
    python -m src.ablation                  # run all experiments
    python -m src.ablation abl2_res128      # run a single experiment
"""

import copy
import json
import logging
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS,
    DEFAULT_COLOR_MODE,
    DEFAULT_RESOLUTION,
    DEVICE,
    NUM_CLASSES,
    NUM_SSD_HEADS,
    RESULTS,
)
from src.dataset import get_dataloader
from src.evaluate import evaluate_detailed
from src.model import WaldoSSD, _make_extra_block, _make_head
from src.train import train

logger = logging.getLogger(__name__)


# ---- Constants ----

_SEED: int = 42

# For timing: warm up the GPU, then time this many passes
_WARMUP_RUNS: int = 10
_TIMED_RUNS: int = 100

# How much to shrink the extra blocks in the pruning experiment
_CHANNEL_PRUNE_FRACTION: float = 0.5

# The "default" experiment — other studies that match it skip retraining
_BASELINE_KEY: str = "abl2_res256"
_BASELINE_ALIASES: list[str] = ["abl1_sobel_on", "abl3_color", "abl4_6heads"]

# Where to save per-experiment JSON files
_ABLATION_DIR: Path = RESULTS / "ablations"


# ---- Experiment definitions ----

EXPERIMENTS: dict[str, dict[str, Any]] = {
    # --- Ablation 1: Does the Sobel edge channel help? ---
    "abl1_sobel_on": {
        "name": "Ablation 1 - Sobel ON (baseline)",
        "config": {
            "resolution": DEFAULT_RESOLUTION,
            "color_mode": DEFAULT_COLOR_MODE,
            "use_sobel": True,
            "num_heads": NUM_SSD_HEADS,
        },
    },
    "abl1_sobel_off": {
        "name": "Ablation 1 - Sobel OFF",
        "config": {
            "resolution": DEFAULT_RESOLUTION,
            "color_mode": DEFAULT_COLOR_MODE,
            "use_sobel": False,
            "num_heads": NUM_SSD_HEADS,
        },
    },
    # --- Ablation 2: Does tile size matter? ---
    "abl2_res256": {
        "name": "Ablation 2 - 256px (baseline)",
        "config": {
            "resolution": 256,
            "color_mode": DEFAULT_COLOR_MODE,
            "use_sobel": True,
            "num_heads": NUM_SSD_HEADS,
        },
    },
    "abl2_res128": {
        "name": "Ablation 2 - 128px",
        "config": {
            "resolution": 128,
            "color_mode": DEFAULT_COLOR_MODE,
            "use_sobel": True,
            "num_heads": NUM_SSD_HEADS,
        },
    },
    "abl2_res64": {
        "name": "Ablation 2 - 64px",
        "config": {
            "resolution": 64,
            "color_mode": DEFAULT_COLOR_MODE,
            "use_sobel": True,
            "num_heads": NUM_SSD_HEADS,
        },
    },
    # --- Ablation 3: How important is colour? ---
    "abl3_color": {
        "name": "Ablation 3 - Color (baseline)",
        "config": {
            "resolution": DEFAULT_RESOLUTION,
            "color_mode": "color",
            "use_sobel": True,
            "num_heads": NUM_SSD_HEADS,
        },
    },
    "abl3_grayscale": {
        "name": "Ablation 3 - Grayscale",
        "config": {
            "resolution": DEFAULT_RESOLUTION,
            "color_mode": "grayscale",
            "use_sobel": True,
            "num_heads": NUM_SSD_HEADS,
        },
    },
    "abl3_bw": {
        "name": "Ablation 3 - Black & White",
        "config": {
            "resolution": DEFAULT_RESOLUTION,
            "color_mode": "bw",
            "use_sobel": True,
            "num_heads": NUM_SSD_HEADS,
        },
    },
    # --- Ablation 4: Can we shrink the model without losing quality? ---
    "abl4_6heads": {
        "name": "Ablation 4 - 6 heads (baseline)",
        "config": {
            "resolution": DEFAULT_RESOLUTION,
            "color_mode": DEFAULT_COLOR_MODE,
            "use_sobel": True,
            "num_heads": NUM_SSD_HEADS,
        },
    },
    "abl4_4heads": {
        "name": "Ablation 4 - 4 heads",
        "config": {
            "resolution": DEFAULT_RESOLUTION,
            "color_mode": DEFAULT_COLOR_MODE,
            "use_sobel": True,
            "num_heads": 4,
        },
    },
    "abl4_pruned": {
        "name": "Ablation 4 - 50% pruned extras",
        "config": {
            "resolution": DEFAULT_RESOLUTION,
            "color_mode": DEFAULT_COLOR_MODE,
            "use_sobel": True,
            "num_heads": NUM_SSD_HEADS,
            "filter_prune_ratio": _CHANNEL_PRUNE_FRACTION,
        },
    },
}

# Run resolution first (validates the whole pipeline), then the rest.
# The baseline trains once; aliases just copy its checkpoint.
_EXECUTION_ORDER: list[str] = [
    "abl2_res256",
    "abl2_res128",
    "abl2_res64",
    "abl1_sobel_on",
    "abl1_sobel_off",
    "abl3_color",
    "abl3_grayscale",
    "abl3_bw",
    "abl4_6heads",
    "abl4_4heads",
    "abl4_pruned",
]


# ---- Reproducibility ----


def set_seed(seed: int = _SEED) -> None:
    """Fix all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic behaviour on NVIDIA GPUs
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


# ---- Pruned model ----


def create_pruned_model(
    num_classes: int = NUM_CLASSES,
    in_channels: int = 3,
    prune_ratio: float = _CHANNEL_PRUNE_FRACTION,
) -> WaldoSSD:
    """Build a WaldoSSD with fewer channels in the extra blocks.

    The backbone stays full-size (pretrained weights are valuable),
    but we thin out the extra convolutional layers to see how much
    model capacity we actually need.

    Args:
        num_classes: Number of object classes (background + Waldo = 2).
        in_channels: Number of input channels (3 for RGB, 4 for RGB+Sobel).
        prune_ratio: Fraction of channels to chop (0.5 = half the channels).

    Returns:
        WaldoSSD model with smaller extra layers and matching heads.
    """
    model = WaldoSSD(
        num_classes=num_classes,
        num_heads=NUM_SSD_HEADS,
        in_channels=in_channels,
    )

    # Figure out how many channels survive the pruning
    extra1_out = int(512 * (1.0 - prune_ratio))
    extra2_out = int(256 * (1.0 - prune_ratio))

    # Swap in the thinner blocks
    model.extra1 = _make_extra_block(960, int(256 * (1.0 - prune_ratio)), extra1_out)
    model.extra2 = _make_extra_block(extra1_out, int(128 * (1.0 - prune_ratio)), extra2_out)

    # Detection heads need to match the new channel widths
    all_channels = [24, 40, 112, 960, extra1_out, extra2_out]
    model.cls_heads = nn.ModuleList([_make_head(ch, num_classes) for ch in all_channels])
    model.reg_heads = nn.ModuleList([_make_head(ch, 4) for ch in all_channels])

    logger.info(
        "Created pruned model: extra1=%dch, extra2=%dch (%.0f%% reduction)",
        extra1_out,
        extra2_out,
        prune_ratio * 100,
    )

    return model


# ---- Inference timing ----


def measure_inference_time(
    model: WaldoSSD,
    resolution: int,
    device: torch.device,
) -> float:
    """How many milliseconds does one forward pass take?

    Runs warmup passes first, then times with proper synchronisation.

    Args:
        model: WaldoSSD model to benchmark.
        resolution: Input resolution in pixels (square).
        device: Device to run inference on.

    Returns:
        Average milliseconds per forward pass.
    """
    model.eval()
    dummy = torch.randn(1, model.in_channels, resolution, resolution, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(_WARMUP_RUNS):
            model(dummy)

    # Make sure all async ops are done before timing
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(_TIMED_RUNS):
            model(dummy)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms / _TIMED_RUNS


# ---- Single experiment ----


def run_experiment(
    exp_key: str,
    exp_def: dict[str, Any],
) -> dict[str, Any]:
    """Train, evaluate, and benchmark one ablation configuration.

    If the result JSON already exists on disk, we skip everything
    (resumability).  Otherwise we train the model, test it, time
    its inference, and write the results out.

    Args:
        exp_key: Unique experiment identifier (e.g., 'abl2_res128').
        exp_def: Experiment definition dict with 'name' and 'config'.

    Returns:
        Result dict with metrics, efficiency measurements, and metadata.
    """
    _ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    result_path = _ABLATION_DIR / f"{exp_key}.json"

    # Already done? Skip it.
    if result_path.exists():
        with open(result_path) as f:
            existing = json.load(f)
        logger.info("Skipping %s - result already exists at %s", exp_key, result_path)
        cached: dict[str, Any] = existing
        return cached

    logger.info("=" * 60)
    logger.info("Starting experiment: %s (%s)", exp_key, exp_def["name"])
    logger.info("=" * 60)

    config = exp_def["config"]
    set_seed(_SEED)

    resolution: int = config.get("resolution", DEFAULT_RESOLUTION)
    color_mode: str = config.get("color_mode", DEFAULT_COLOR_MODE)
    use_sobel: bool = config.get("use_sobel", True)
    num_heads: int = config.get("num_heads", NUM_SSD_HEADS)
    prune_ratio: float = config.get("filter_prune_ratio", 0.0)
    in_channels = 4 if use_sobel else 3

    is_pruned = prune_ratio > 0.0

    # --- Train the model ---
    exp_ckpt_prefix = f"{exp_key}_best"
    exp_ckpt_path = CHECKPOINTS / f"{exp_ckpt_prefix}.pth"
    train_start = time.time()

    if is_pruned:
        # The pruned model has custom layer sizes, so the standard
        # train() function won't work — we handle it ourselves.
        model = create_pruned_model(
            num_classes=NUM_CLASSES,
            in_channels=in_channels,
            prune_ratio=prune_ratio,
        )
        anchors = model.get_anchors(resolution)
        model = model.to(DEVICE)
        anchors = anchors.to(DEVICE)

        best_val_ap, best_epoch = _train_pruned_model(model, anchors, config)
    else:
        best_val_ap = train(
            config_overrides=config,
            checkpoint_prefix=exp_ckpt_prefix,
        )

        # Load the best checkpoint back in for evaluation
        model = WaldoSSD(
            num_classes=NUM_CLASSES,
            num_heads=num_heads,
            in_channels=in_channels,
        )
        best_epoch = 0
        if exp_ckpt_path.exists():
            checkpoint = torch.load(exp_ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            best_epoch = checkpoint.get("epoch", 0)

    training_time_min = (time.time() - train_start) / 60.0

    # Anchors on CPU first, then ship everything to the device
    anchors = model.get_anchors(resolution)
    model = model.to(DEVICE)
    anchors = anchors.to(DEVICE)

    # --- See how well it does on the test set ---
    test_loader = get_dataloader(
        split="test",
        resolution=resolution,
        color_mode=color_mode,
        use_sobel=use_sobel,
        use_density_filter=False,
        batch_size=BATCH_SIZE,
    )

    metrics = evaluate_detailed(model, test_loader, anchors, resolution, DEVICE)

    # --- How fast is it? ---
    inference_ms = measure_inference_time(model, resolution, DEVICE)

    # --- How big is it? ---
    param_count = sum(p.numel() for p in model.parameters())

    # --- Save PR curve data separately (too large for summary JSON) ---
    pr_curve_data = metrics.pop("pr_curve", {})
    pr_path = _ABLATION_DIR / f"{exp_key}_pr_curve.json"
    with open(pr_path, "w") as f:
        json.dump(pr_curve_data, f)
    logger.info("Saved PR curve to %s", pr_path)

    # --- Bundle everything into a result dict ---
    result: dict[str, Any] = {
        "experiment_key": exp_key,
        "experiment_name": exp_def["name"],
        "config": config,
        "metrics": {
            "test_ap": metrics["ap"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "num_tp": metrics["num_tp"],
            "num_fp": metrics["num_fp"],
            "num_fn": metrics["num_fn"],
            "num_gt": metrics["num_gt"],
            "best_val_ap": best_val_ap,
            "best_epoch": best_epoch,
            "test_tiles": len(test_loader.dataset),
        },
        "efficiency": {
            "param_count": param_count,
            "inference_ms_per_tile": round(inference_ms, 2),
            "training_time_minutes": round(training_time_min, 1),
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": _SEED,
        },
    }

    # Write the results to disk
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved result to %s", result_path)

    # Save the pruned model checkpoint (non-pruned models are saved by train())
    if is_pruned:
        torch.save({"model_state_dict": model.state_dict()}, exp_ckpt_path)
    logger.info("Checkpoint at %s", exp_ckpt_path)

    return result


def _train_pruned_model(
    model: WaldoSSD,
    anchors: torch.Tensor,
    config: dict[str, Any],
) -> tuple[float, int]:
    """Train the pruned model (its custom layers mean we can't use train()).

    Args:
        model: Pre-constructed pruned WaldoSSD already on DEVICE.
        anchors: Pre-computed anchors already on DEVICE.
        config: Experiment config dict with resolution, color_mode, etc.

    Returns:
        Tuple of (best_val_ap, best_epoch).
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from src.config import LEARNING_RATE, NEG_POS_RATIO, NUM_EPOCHS, WEIGHT_DECAY
    from src.train import (
        _GRAD_CLIP_MAX_NORM,
        _VAL_EVERY_N_EPOCHS,
        SSDLoss,
        evaluate,
        match_anchors_to_targets,
    )

    resolution: int = config.get("resolution", DEFAULT_RESOLUTION)
    color_mode: str = config.get("color_mode", DEFAULT_COLOR_MODE)
    use_sobel: bool = config.get("use_sobel", True)

    train_loader = get_dataloader(
        split="train",
        resolution=resolution,
        color_mode=color_mode,
        use_sobel=use_sobel,
        use_density_filter=True,
        batch_size=BATCH_SIZE,
    )
    val_loader = get_dataloader(
        split="val",
        resolution=resolution,
        color_mode=color_mode,
        use_sobel=use_sobel,
        use_density_filter=False,
        batch_size=BATCH_SIZE,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = SSDLoss(neg_pos_ratio=NEG_POS_RATIO, num_classes=NUM_CLASSES)

    best_val_ap: float = 0.0
    best_epoch: int = 0
    best_state: dict[str, object] = {}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = images.to(DEVICE)
            cls_preds, loc_preds = model(images)
            cls_targets, reg_targets = match_anchors_to_targets(anchors, targets, resolution)
            total_loss, _, _ = criterion(cls_preds, loc_preds, cls_targets, reg_targets)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_MAX_NORM)
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, num_batches)

        if epoch % _VAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            val_ap = evaluate(model, val_loader, anchors, resolution, DEVICE)
            logger.info(
                "Pruned epoch %d/%d - loss=%.4f, val_ap=%.4f",
                epoch,
                NUM_EPOCHS,
                avg_loss,
                val_ap,
            )
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)  # type: ignore[arg-type]

    return best_val_ap, best_epoch


# ---- Run all ----


def run_all() -> dict[str, Any]:
    """Train the baseline, then run every variant and compare.

    Identical configs get their results copied from the baseline
    instead of retraining, saving a lot of time.

    Returns:
        Dict mapping experiment keys to their result dicts.
    """
    _ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}

    unique_experiments = len(_EXECUTION_ORDER) - len(_BASELINE_ALIASES)
    logger.info(
        "Running %d experiments (%d unique + %d baseline aliases)",
        len(_EXECUTION_ORDER),
        unique_experiments,
        len(_BASELINE_ALIASES),
    )

    for exp_key in _EXECUTION_ORDER:
        # If this experiment is identical to the baseline, just copy results
        if exp_key in _BASELINE_ALIASES:
            baseline_path = _ABLATION_DIR / f"{_BASELINE_KEY}.json"
            alias_path = _ABLATION_DIR / f"{exp_key}.json"

            if baseline_path.exists() and not alias_path.exists():
                # Clone the baseline result, just rename it
                with open(baseline_path) as f:
                    baseline_result = json.load(f)

                alias_result = baseline_result.copy()
                alias_result["experiment_key"] = exp_key
                alias_result["experiment_name"] = EXPERIMENTS[exp_key]["name"]
                alias_result["metadata"] = baseline_result.get("metadata", {}).copy()
                alias_result["metadata"]["copied_from"] = _BASELINE_KEY

                with open(alias_path, "w") as f:
                    json.dump(alias_result, f, indent=2)

                # Same for the checkpoint file
                src_ckpt = CHECKPOINTS / f"{_BASELINE_KEY}_best.pth"
                dst_ckpt = CHECKPOINTS / f"{exp_key}_best.pth"
                if src_ckpt.exists() and not dst_ckpt.exists():
                    shutil.copy2(src_ckpt, dst_ckpt)

                logger.info(
                    "Copied baseline result for %s from %s",
                    exp_key,
                    _BASELINE_KEY,
                )
                all_results[exp_key] = alias_result
                continue

        result = run_experiment(exp_key, EXPERIMENTS[exp_key])
        all_results[exp_key] = result

    # Write one big JSON with everything for easy comparison
    combined_path = _ABLATION_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved combined results to %s", combined_path)

    print_results_summary(all_results)
    return all_results


# ---- Results summary ----


def print_results_summary(all_results: dict[str, Any]) -> None:
    """Print a side-by-side comparison of every experiment to the log.

    Grouped by ablation study so you can see the effect at a glance.

    Args:
        all_results: Dict mapping experiment keys to result dicts.
    """
    logger.info("")
    logger.info("=" * 90)
    logger.info("  ABLATION RESULTS SUMMARY")
    logger.info("=" * 90)
    logger.info(
        "  %-28s %6s %6s %6s %6s %8s %6s %6s",
        "Experiment",
        "AP",
        "Prec",
        "Rec",
        "F1",
        "Params",
        "ms",
        "min",
    )
    logger.info("-" * 90)

    ablation_groups = [
        ("Ablation 1 - Sobel", ["abl1_sobel_on", "abl1_sobel_off"]),
        ("Ablation 2 - Resolution", ["abl2_res256", "abl2_res128", "abl2_res64"]),
        ("Ablation 3 - Color", ["abl3_color", "abl3_grayscale", "abl3_bw"]),
        ("Ablation 4 - Model Size", ["abl4_6heads", "abl4_4heads", "abl4_pruned"]),
    ]

    for group_name, keys in ablation_groups:
        logger.info("  --- %s ---", group_name)
        for key in keys:
            if key not in all_results:
                logger.info("  %-28s  (not run)", key)
                continue

            r = all_results[key]
            m = r.get("metrics", {})
            e = r.get("efficiency", {})

            logger.info(
                "  %-28s %6.4f %6.4f %6.4f %6.4f %7dk %5.1f %5.1f",
                r.get("experiment_name", key)[:28],
                m.get("test_ap", 0.0),
                m.get("test_precision", 0.0),
                m.get("test_recall", 0.0),
                m.get("test_f1", 0.0),
                e.get("param_count", 0) / 1000,
                e.get("inference_ms_per_tile", 0.0),
                e.get("training_time_minutes", 0.0),
            )
        logger.info("")

    logger.info("=" * 90)


# ---- Main ----


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) > 1:
        exp_key = sys.argv[1]
        if exp_key not in EXPERIMENTS:
            logger.error(
                "Unknown experiment key '%s'. Available: %s",
                exp_key,
                ", ".join(EXPERIMENTS.keys()),
            )
            sys.exit(1)

        logger.info("Running single experiment: %s", exp_key)
        run_experiment(exp_key, EXPERIMENTS[exp_key])
    else:
        logger.info("Running all %d ablation experiments...", len(_EXECUTION_ORDER))
        run_all()
