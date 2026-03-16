"""Ablation experiment runner for systematic WaldoSSD evaluation.

Executes 4 ablation studies (Sobel bypass, resolution, color mode, model
size) with full reproducibility, JSON result logging, and resumability.
Each experiment trains a model variant, evaluates on the test set, and
records metrics plus efficiency measurements.

Usage:
    python -m src.ablation                  # run all experiments
    python -m src.ablation abl2_res128      # run a single experiment
"""

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


# =============================================================================
# CONSTANTS
# =============================================================================

_SEED: int = 42

# Inference timing parameters
_WARMUP_PASSES: int = 10
_TIMED_PASSES: int = 100

# Pruning ratio for ablation 4
_PRUNE_RATIO: float = 0.5

# The baseline config key that other studies share
_BASELINE_KEY: str = "abl2_res256"

# Experiment keys that are identical to baseline (avoid retraining)
_BASELINE_ALIASES: list[str] = ["abl1_sobel_on", "abl3_color", "abl4_6heads"]

# Output directory for ablation results
_ABLATION_DIR: Path = RESULTS / "ablations"


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

EXPERIMENTS: dict[str, dict[str, Any]] = {
    # --- Ablation 1: Sobel Edge Bypass ---
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
    # --- Ablation 2: Image Resolution ---
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
    # --- Ablation 3: Color Information ---
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
    # --- Ablation 4: Model Size ---
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
            "filter_prune_ratio": _PRUNE_RATIO,
        },
    },
}

# Execution order: resolution first (validates pipeline), then Sobel,
# color, model size. Baseline trains once via abl2_res256, aliases copy it.
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


# =============================================================================
# REPRODUCIBILITY
# =============================================================================


def set_seed(seed: int = _SEED) -> None:
    """Set random seeds for full reproducibility across runs.

    Configures Python, NumPy, and PyTorch RNGs, plus disables cuDNN
    non-deterministic algorithms. This guarantees identical results
    when rerunning the same experiment.

    Args:
        seed: Random seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Disable non-deterministic cuDNN algorithms
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


# =============================================================================
# PRUNED MODEL CREATION
# =============================================================================


def create_pruned_model(
    num_classes: int = NUM_CLASSES,
    in_channels: int = 3,
    prune_ratio: float = _PRUNE_RATIO,
) -> WaldoSSD:
    """Create a WaldoSSD with pruned extra convolutional layers.

    Reduces the channel count in extra1 and extra2 blocks by the prune
    ratio, producing a smaller model for the model-size ablation study.
    The backbone remains untouched since it uses pretrained weights.

    Args:
        num_classes: Number of object classes (background + Waldo = 2).
        in_channels: Number of input channels (3 for RGB, 4 for RGB+Sobel).
        prune_ratio: Fraction of channels to remove (0.5 = 50% reduction).

    Returns:
        WaldoSSD model with pruned extra layers and updated detection heads.
    """
    model = WaldoSSD(
        num_classes=num_classes,
        num_heads=NUM_SSD_HEADS,
        in_channels=in_channels,
    )

    # Compute pruned channel counts (round to nearest int)
    extra1_out = int(512 * (1.0 - prune_ratio))
    extra2_out = int(256 * (1.0 - prune_ratio))

    # Replace extra layers with smaller versions
    model.extra1 = _make_extra_block(960, int(256 * (1.0 - prune_ratio)), extra1_out)
    model.extra2 = _make_extra_block(extra1_out, int(128 * (1.0 - prune_ratio)), extra2_out)

    # Rebuild detection heads to match new channel counts
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


# =============================================================================
# INFERENCE TIMING
# =============================================================================


def measure_inference_time(
    model: WaldoSSD,
    resolution: int,
    device: torch.device,
) -> float:
    """Measure average inference time per tile in milliseconds.

    Runs warmup passes to stabilize GPU state, then times a fixed
    number of forward passes with proper device synchronization.

    Args:
        model: WaldoSSD model to benchmark.
        resolution: Input resolution in pixels (square).
        device: Device to run inference on.

    Returns:
        Average milliseconds per forward pass.
    """
    model.eval()
    dummy = torch.randn(1, model.in_channels, resolution, resolution, device=device)

    # Warmup to stabilize GPU clock and caches
    with torch.no_grad():
        for _ in range(_WARMUP_PASSES):
            model(dummy)

    # Synchronize before timing
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(_TIMED_PASSES):
            model(dummy)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms / _TIMED_PASSES


# =============================================================================
# SINGLE EXPERIMENT RUNNER
# =============================================================================


def run_experiment(
    exp_key: str,
    exp_def: dict[str, Any],
) -> dict[str, Any]:
    """Train and evaluate a single ablation experiment.

    Checks for existing results (resumability), trains the model, runs
    test-set evaluation, measures inference time, and saves everything
    to a JSON file.

    Args:
        exp_key: Unique experiment identifier (e.g., 'abl2_res128').
        exp_def: Experiment definition dict with 'name' and 'config'.

    Returns:
        Result dict with metrics, efficiency measurements, and metadata.
    """
    _ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    result_path = _ABLATION_DIR / f"{exp_key}.json"

    # --- Resumability: skip if result already exists ---
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

    # --- Train ---
    train_start = time.time()

    if is_pruned:
        # Pruned model needs special construction, then standard training
        # Pass config without the prune key so train() uses standard setup
        train_config = {k: v for k, v in config.items() if k != "filter_prune_ratio"}
        best_val_ap = train(config_overrides=train_config)
    else:
        best_val_ap = train(config_overrides=config)

    training_time_min = (time.time() - train_start) / 60.0

    # --- Load best model for evaluation ---
    best_ckpt_path = CHECKPOINTS / "best_model.pth"

    if is_pruned:
        model = create_pruned_model(
            num_classes=NUM_CLASSES,
            in_channels=in_channels,
            prune_ratio=prune_ratio,
        )
        # Re-train the pruned model properly
        set_seed(_SEED)
        # For pruned, we need to train from scratch with the pruned architecture
        # The initial train() call above used standard architecture, so retrain
        logger.info("Retraining pruned model variant...")
        train_start = time.time()

        # Create pruned model, generate anchors, then move to device
        pruned_model = create_pruned_model(
            num_classes=NUM_CLASSES,
            in_channels=in_channels,
            prune_ratio=prune_ratio,
        )
        anchors = pruned_model.get_anchors(resolution)
        pruned_model = pruned_model.to(DEVICE)
        anchors = anchors.to(DEVICE)

        # Run custom training loop for pruned model
        best_val_ap = _train_pruned_model(pruned_model, anchors, config)
        training_time_min = (time.time() - train_start) / 60.0
        model = pruned_model
    else:
        model = WaldoSSD(
            num_classes=NUM_CLASSES,
            num_heads=num_heads,
            in_channels=in_channels,
        )
        if best_ckpt_path.exists():
            checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])

    # Generate anchors on CPU, then move everything to device
    anchors = model.get_anchors(resolution)
    model = model.to(DEVICE)
    anchors = anchors.to(DEVICE)

    # --- Test set evaluation ---
    test_loader = get_dataloader(
        split="test",
        resolution=resolution,
        color_mode=color_mode,
        use_sobel=use_sobel,
        use_density_filter=False,
        batch_size=BATCH_SIZE,
    )

    metrics = evaluate_detailed(model, test_loader, anchors, resolution, DEVICE)

    # --- Inference timing ---
    inference_ms = measure_inference_time(model, resolution, DEVICE)

    # --- Parameter count ---
    param_count = sum(p.numel() for p in model.parameters())

    # --- Compile result ---
    result: dict[str, Any] = {
        "experiment_key": exp_key,
        "experiment_name": exp_def["name"],
        "config": config,
        "metrics": {
            "test_ap": metrics["ap"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "best_val_ap": best_val_ap,
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

    # --- Save result JSON ---
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved result to %s", result_path)

    # --- Save experiment-specific checkpoint ---
    exp_ckpt_path = CHECKPOINTS / f"{exp_key}_best.pth"
    if best_ckpt_path.exists() and not is_pruned:
        shutil.copy2(best_ckpt_path, exp_ckpt_path)
    elif is_pruned:
        torch.save({"model_state_dict": model.state_dict()}, exp_ckpt_path)
    logger.info("Saved checkpoint to %s", exp_ckpt_path)

    return result


def _train_pruned_model(
    model: WaldoSSD,
    anchors: torch.Tensor,
    config: dict[str, Any],
) -> float:
    """Train a pruned WaldoSSD model using the standard training loop internals.

    Since the pruned model has different layer dimensions, we can't use
    train() directly (it constructs its own model). Instead, we replicate
    the essential training logic here with the pre-built pruned model.

    Args:
        model: Pre-constructed pruned WaldoSSD already on DEVICE.
        anchors: Pre-computed anchors already on DEVICE.
        config: Experiment config dict with resolution, color_mode, etc.

    Returns:
        Best validation AP achieved during training.
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

    return best_val_ap


# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================


def run_all() -> dict[str, Any]:
    """Execute all ablation experiments in recommended order.

    Trains the baseline once, copies results for identical experiment
    configs, and runs each unique variant. Saves a combined results
    file for easy comparison.

    Returns:
        Dict mapping experiment keys to their result dicts.
    """
    _ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}

    total_unique = len(_EXECUTION_ORDER) - len(_BASELINE_ALIASES)
    logger.info(
        "Running %d experiments (%d unique + %d baseline aliases)",
        len(_EXECUTION_ORDER),
        total_unique,
        len(_BASELINE_ALIASES),
    )

    for exp_key in _EXECUTION_ORDER:
        # Check if this is a baseline alias that can be copied
        if exp_key in _BASELINE_ALIASES:
            baseline_path = _ABLATION_DIR / f"{_BASELINE_KEY}.json"
            alias_path = _ABLATION_DIR / f"{exp_key}.json"

            if baseline_path.exists() and not alias_path.exists():
                # Copy baseline result with updated key and name
                with open(baseline_path) as f:
                    baseline_result = json.load(f)

                alias_result = baseline_result.copy()
                alias_result["experiment_key"] = exp_key
                alias_result["experiment_name"] = EXPERIMENTS[exp_key]["name"]
                alias_result["metadata"] = baseline_result.get("metadata", {}).copy()
                alias_result["metadata"]["copied_from"] = _BASELINE_KEY

                with open(alias_path, "w") as f:
                    json.dump(alias_result, f, indent=2)

                # Copy checkpoint too
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

    # --- Save combined results ---
    combined_path = _ABLATION_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved combined results to %s", combined_path)

    print_results_summary(all_results)
    return all_results


# =============================================================================
# RESULTS SUMMARY
# =============================================================================


def print_results_summary(all_results: dict[str, Any]) -> None:
    """Log a formatted comparison table of all experiment results.

    Groups results by ablation study for easy side-by-side comparison
    in terminal output.

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

    # Group by ablation study prefix
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


# =============================================================================
# MAIN
# =============================================================================


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
