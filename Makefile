.PHONY: setup run clean sync

# Setup the vast.ai instance
setup:
	chmod +x setup.sh
	./setup.sh

# Run the training script
run:
	uv run python main.py

# Clean up generated files (logs, wandb, outputs, pycache)
clean:
	rm -rf outputs/ wandb/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Force re-sync dependencies with uv
sync:
	uv sync
