"""Plot generation from saved experiment results."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import RESULTS

logger = logging.getLogger(__name__)

_PLOTS_DIR: Path = RESULTS / "plots"

# Consistent colours for ablation groups
_COLORS: list[str] = [
    "#2196F3",
    "#FF5722",
    "#4CAF50",
    "#9C27B0",
    "#FF9800",
    "#00BCD4",
]


# --- Training curves ---


def plot_training_curves(history_path: Path, output_dir: Path) -> None:
    """Plot loss, val AP, and LR schedule from a training history JSON."""
    with open(history_path) as f:
        data = json.load(f)

    epochs_data = data["epochs"]
    if not epochs_data:
        logger.warning("No epoch data in %s, skipping", history_path)
        return

    epochs = [e["epoch"] for e in epochs_data]
    losses = [e["loss"] for e in epochs_data]
    cls_losses = [e["cls_loss"] for e in epochs_data]
    reg_losses = [e["reg_loss"] for e in epochs_data]
    lrs = [e["lr"] for e in epochs_data]
    val_epochs = [e["epoch"] for e in epochs_data if "val_ap" in e]
    val_aps = [e["val_ap"] for e in epochs_data if "val_ap" in e]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(epochs, losses, label="Total", linewidth=1.5)
    ax.plot(epochs, cls_losses, label="Classification", linewidth=1, alpha=0.7)
    ax.plot(epochs, reg_losses, label="Regression", linewidth=1, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if val_epochs:
        ax.plot(val_epochs, val_aps, "o-", color="#4CAF50", markersize=3)
        ax.axhline(y=data["best_val_ap"], color="red", linestyle="--",
                    linewidth=0.8, label=f"Best={data['best_val_ap']:.4f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.set_title("Validation AP")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, lrs, color="#FF9800")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    stem = history_path.stem.replace("_history", "")
    out_path = output_dir / f"{stem}_training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves to %s", out_path)


# ---- Precision-Recall curve ----


def plot_pr_curve(pr_path: Path, output_dir: Path, title: str = "Precision-Recall Curve") -> None:
    """Plot precision-recall curve from saved PR data."""
    with open(pr_path) as f:
        pr_data = json.load(f)

    recalls = pr_data.get("recalls", [])
    precisions = pr_data.get("precisions", [])
    interp = pr_data.get("interpolated_precisions", [])

    if not recalls:
        logger.warning("No PR data in %s, skipping", pr_path)
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, color="#2196F3", linewidth=1, alpha=0.5, label="Raw")

    recall_points = [i / 10.0 for i in range(11)]
    if interp:
        ax.step(recall_points, interp, where="post", color="#FF5722",
                linewidth=2, label="11-pt interpolated")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    stem = pr_path.stem.replace("_pr_curve", "").replace("pr_curve", "baseline")
    out_path = output_dir / f"{stem}_pr_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curve to %s", out_path)


# ---- Ablation comparison bar charts ----


def plot_ablation_comparison(all_results_path: Path, output_dir: Path) -> None:
    """Grouped bar charts comparing ablation experiments."""
    with open(all_results_path) as f:
        all_results = json.load(f)

    ablation_groups = [
        ("Ablation 1 - Sobel Edge Channel", ["abl1_sobel_on", "abl1_sobel_off"]),
        ("Ablation 2 - Tile Resolution", ["abl2_res256", "abl2_res128", "abl2_res64"]),
        ("Ablation 3 - Color Mode", ["abl3_color", "abl3_grayscale", "abl3_bw"]),
        ("Ablation 4 - Model Size", ["abl4_6heads", "abl4_4heads", "abl4_pruned"]),
    ]

    metric_keys = ["test_ap", "test_precision", "test_recall", "test_f1"]
    metric_labels = ["AP", "Precision", "Recall", "F1"]

    for group_idx, (group_name, keys) in enumerate(ablation_groups):
        present_keys = [k for k in keys if k in all_results]
        if not present_keys:
            continue

        names = [all_results[k]["experiment_name"].split(" - ", 1)[-1] for k in present_keys]
        values = {
            ml: [all_results[k]["metrics"].get(mk, 0.0) for k in present_keys]
            for mk, ml in zip(metric_keys, metric_labels)
        }

        fig, ax = plt.subplots(figsize=(max(6, len(present_keys) * 2), 5))
        x = np.arange(len(present_keys))
        width = 0.18
        offsets = np.arange(len(metric_labels)) - (len(metric_labels) - 1) / 2

        for i, label in enumerate(metric_labels):
            bars = ax.bar(x + offsets[i] * width, values[label], width,
                          label=label, color=_COLORS[i], alpha=0.85)
            for bar, val in zip(bars, values[label]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylabel("Score")
        ax.set_title(group_name)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        out_path = output_dir / f"ablation{group_idx + 1}_comparison.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s to %s", group_name, out_path)

    _plot_efficiency_comparison(all_results, output_dir)


def _plot_efficiency_comparison(all_results: dict, output_dir: Path) -> None:
    """Scatter plot of model size vs inference speed, colored by AP."""
    # Deduplicate baseline aliases
    seen_configs: set[str] = set()
    keys_to_plot: list[str] = []
    for k, v in all_results.items():
        config_sig = json.dumps(v.get("config", {}), sort_keys=True)
        if config_sig not in seen_configs:
            seen_configs.add(config_sig)
            keys_to_plot.append(k)

    if len(keys_to_plot) < 2:
        return

    params = [all_results[k]["efficiency"]["param_count"] / 1e6 for k in keys_to_plot]
    ms = [all_results[k]["efficiency"]["inference_ms_per_tile"] for k in keys_to_plot]
    aps = [all_results[k]["metrics"]["test_ap"] for k in keys_to_plot]
    names = [all_results[k]["experiment_name"].split(" - ", 1)[-1] for k in keys_to_plot]

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(params, ms, c=aps, cmap="RdYlGn", s=100,
                         edgecolors="black", linewidths=0.5, vmin=0, vmax=1)

    for i, name in enumerate(names):
        ax.annotate(name, (params[i], ms[i]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Test AP")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Inference Time (ms/tile)")
    ax.set_title("Efficiency vs Quality")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "efficiency_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved efficiency comparison to %s", out_path)


# ---- Overlay PR curves from multiple ablation experiments ----


def plot_ablation_pr_overlay(ablation_dir: Path, output_dir: Path) -> None:
    """Overlay PR curves from all ablation experiments on one plot per group."""
    ablation_groups = [
        ("Ablation 1 - Sobel", ["abl1_sobel_on", "abl1_sobel_off"]),
        ("Ablation 2 - Resolution", ["abl2_res256", "abl2_res128", "abl2_res64"]),
        ("Ablation 3 - Color", ["abl3_color", "abl3_grayscale", "abl3_bw"]),
        ("Ablation 4 - Model Size", ["abl4_6heads", "abl4_4heads", "abl4_pruned"]),
    ]

    for group_idx, (group_name, keys) in enumerate(ablation_groups):
        fig, ax = plt.subplots(figsize=(6, 5))
        has_data = False

        for i, key in enumerate(keys):
            pr_path = ablation_dir / f"{key}_pr_curve.json"
            if not pr_path.exists():
                continue

            with open(pr_path) as f:
                pr_data = json.load(f)

            recalls = pr_data.get("recalls", [])
            precisions = pr_data.get("precisions", [])
            if not recalls:
                continue

            label = key.split("_", 1)[-1].replace("_", " ").title()
            ax.plot(recalls, precisions, color=_COLORS[i % len(_COLORS)],
                    linewidth=1.5, label=label, alpha=0.8)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{group_name} — PR Curves")
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = output_dir / f"ablation{group_idx + 1}_pr_overlay.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s PR overlay to %s", group_name, out_path)


# Main


def generate_all_plots() -> None:
    """Find all saved results and generate every report figure."""
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    training_dir = RESULTS / "training"
    if training_dir.exists():
        for hist_path in sorted(training_dir.glob("*_history.json")):
            plot_training_curves(hist_path, _PLOTS_DIR)

    metrics_dir = RESULTS / "metrics"
    pr_path = metrics_dir / "pr_curve.json"
    if pr_path.exists():
        plot_pr_curve(pr_path, _PLOTS_DIR, title="Baseline — Precision-Recall Curve")

    ablation_dir = RESULTS / "ablations"
    all_results_path = ablation_dir / "all_results.json"
    if all_results_path.exists():
        plot_ablation_comparison(all_results_path, _PLOTS_DIR)

    if ablation_dir.exists():
        plot_ablation_pr_overlay(ablation_dir, _PLOTS_DIR)

    logger.info("All plots saved to %s", _PLOTS_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    generate_all_plots()
