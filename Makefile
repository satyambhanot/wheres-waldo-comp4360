# =============================================================================
# Makefile - Where's Waldo Object Detection (COMP 4360)
# University of Manitoba - Machine Learning
#
# Single entry point for the entire project pipeline.
# Run `make help` to see all available commands.
# =============================================================================

PYTHON = python
VENV   = venv/bin/activate

.PHONY: help setup data train evaluate ablation ablation-single \
        check fix test-data test-model test-pipeline \
        status clean clean-data clean-checkpoints clean-results \
        all reproduce

# ---- Help ----

help: ## Show all available commands with descriptions
	@echo ""
	@echo "Where's Waldo - COMP 4360"
	@echo "========================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---- Setup ----

setup: ## Create venv with Python 3.11 and install all dependencies
	$(PYTHON) -m venv venv
	. $(VENV) && pip install --upgrade pip
	. $(VENV) && pip install -r requirements.txt
	@echo "Setup complete. Activate with: source venv/bin/activate"

# ---- Data ----

data: ## Run tiling pipeline (19 images -> tiles at 64/128/256)
	. $(VENV) && $(PYTHON) -m src.preprocess

# ---- Training ----

train: ## Train baseline model (256px, color, Sobel ON, 200 epochs)
	. $(VENV) && $(PYTHON) -m src.train

# ---- Evaluation ----

evaluate: ## Evaluate best model on test set + save visual examples
	. $(VENV) && $(PYTHON) -m src.evaluate

# ---- Ablation Studies ----

ablation: ## Run all 4 ablation experiments (all 11 conditions)
	. $(VENV) && $(PYTHON) -m src.ablation

ablation-single: ## Run one experiment (usage: make ablation-single EXP=abl2_res128)
	@if [ -z "$(EXP)" ]; then \
		echo "Usage: make ablation-single EXP=<experiment_key>"; \
		echo "Available keys:"; \
		echo "  abl1_sobel_on, abl1_sobel_off"; \
		echo "  abl2_res256, abl2_res128, abl2_res64"; \
		echo "  abl3_color, abl3_grayscale, abl3_bw"; \
		echo "  abl4_6heads, abl4_4heads, abl4_pruned"; \
		exit 1; \
	fi
	. $(VENV) && $(PYTHON) -m src.ablation $(EXP)

# ---- Code Quality ----

check: ## Run 8 code quality checks (read-only)
	. $(VENV) && ./scripts/check.sh

fix: ## Auto-fix formatting then run checks
	. $(VENV) && ./scripts/check.sh --fix

# ---- Sanity Checks ----

test-data: ## Sanity check: verify dataset loads correctly
	. $(VENV) && $(PYTHON) -m src.dataset

test-model: ## Sanity check: verify model forward pass
	. $(VENV) && $(PYTHON) -m src.model

test-pipeline: ## Run all sanity checks in sequence (data, model)
	@echo "Running dataset sanity check..."
	. $(VENV) && $(PYTHON) -m src.dataset
	@echo ""
	@echo "Running model sanity check..."
	. $(VENV) && $(PYTHON) -m src.model
	@echo ""
	@echo "All sanity checks passed."

# ---- Status ----

status: ## Show project status (file counts, checkpoints, disk, git log)
	@echo ""
	@echo "=== Project Status ==="
	@echo ""
	@echo "Source files:"
	@find src -name "*.py" | wc -l | xargs -I{} echo "  {} Python files in src/"
	@echo ""
	@echo "Processed tiles:"
	@for res in 64 128 256; do \
		if [ -d "data/processed/$$res" ]; then \
			count=$$(ls data/processed/$$res/*.jpg 2>/dev/null | wc -l | tr -d ' '); \
			echo "  $$res px: $$count tiles"; \
		else \
			echo "  $$res px: not generated"; \
		fi; \
	done
	@echo ""
	@echo "Checkpoints:"
	@if [ -d "results/checkpoints" ]; then \
		ls -lh results/checkpoints/*.pth 2>/dev/null || echo "  (none)"; \
	else \
		echo "  (none)"; \
	fi
	@echo ""
	@echo "Ablation results:"
	@if [ -d "results/ablations" ]; then \
		ls results/ablations/*.json 2>/dev/null | wc -l | xargs -I{} echo "  {} JSON files"; \
	else \
		echo "  (none)"; \
	fi
	@echo ""
	@echo "Disk usage:"
	@du -sh data/processed 2>/dev/null || echo "  data/processed: 0"
	@du -sh results 2>/dev/null || echo "  results: 0"
	@echo ""
	@echo "Recent commits:"
	@git log --oneline -5
	@echo ""

# ---- Cleanup ----

clean: ## Delete ALL generated files (tiles, checkpoints, results)
	rm -rf data/processed/64 data/processed/128 data/processed/256
	rm -f data/processed/splits.json
	rm -rf results/checkpoints/*.pth
	rm -rf results/ablations
	rm -rf results/plots/*.jpg results/plots/*.json
	@echo "Cleaned all generated files."

clean-data: ## Delete processed tiles only (keeps raw images)
	rm -rf data/processed/64 data/processed/128 data/processed/256
	rm -f data/processed/splits.json
	@echo "Cleaned processed tiles."

clean-checkpoints: ## Delete model checkpoints only
	rm -rf results/checkpoints/*.pth
	@echo "Cleaned checkpoints."

clean-results: ## Delete ablation results and plots only
	rm -rf results/ablations
	rm -rf results/plots/*.jpg results/plots/*.json
	@echo "Cleaned results and plots."

# ---- Composite Targets ----

all: data train evaluate ## Full pipeline: data -> train -> evaluate

reproduce: clean data train evaluate ablation ## Full reproducible run: clean -> data -> train -> evaluate -> ablation
