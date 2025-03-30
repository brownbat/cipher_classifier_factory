```markdown
# Cipher Classifier

A system that builds and analyzes models to classify various types of classical ciphers.

## Overview

This project implements Transformer-based models (primarily using PyTorch) to classify text samples into different cipher categories (e.g., English plaintext, Caesar cipher, Vigenère cipher, Playfair, etc.). The system provides tools for defining, queueing, training, analyzing, and managing experiments with different hyperparameter configurations. A central `experiment_id` (format: `YYYYMMDD-N`) links all artifacts for a given run.

## Project Structure

-   **`data/`**: Default location for all generated data and artifacts.
    -   `pending_experiments.json`: Queue of experiments to be run.
    -   `completed_experiments.json`: Log of completed (or failed) experiments with parameters and metrics.
    *   `models/`: Stores trained model state dictionaries (`<id>.pt`) and metadata (`<id>_metadata.json`).
    *   `cm_history/`: Stores confusion matrix history (`<id>_cm_history.npy`) for animation.
    *   `loss_graphs/`: Saved plots of training/validation loss curves.
    *   `cm/`: Saved confusion matrix GIFs and final plots.
    *   `trend_matrix/`: Saved trend matrix plots (final confusion matrix).
-   **Core Workflow Tools**:
    *   `suggest_experiments.py`: Analyzes past results and suggests new hyperparameter combinations to explore.
    *   `manage_queue.py`: Adds, lists, clears, or replaces experiments in the `pending_experiments.json` queue. Validates parameters and checks for duplicates.
    *   `researcher.py`: Executes experiments sequentially from the pending queue, orchestrating training and artifact saving.
-   **Model Implementation**:
    *   `models/transformer/`: Transformer model architecture (`model.py`), training loop (`train.py`), and inference logic (`inference.py`).
    *   `models/common/`: Shared utilities and data processing components.
-   **Analysis & Visualization Tools**:
    *   `visualization.py`: Interactive Dash application to explore completed experiment results, comparing metrics based on parameter changes.
    *   `generate_gifs.py`: Generates animated GIFs of the confusion matrix evolution during training from `.npy` history files.
    *   `generate_loss_graphs.py`: Generates static plots of training and validation loss/accuracy curves.
    *   `generate_trend_matrix.py`: Generates a static plot of the final confusion matrix for an experiment.
-   **Utilities**:
    *   `query_model.py`: Classifies text samples using a specified trained model (by `experiment_id`).
    *   `purge_experiments.py`: Deletes experiments and all associated artifacts based on `experiment_id` or other criteria.
    *   `ciphers.py`: Implementation of various classical ciphers for data generation.
    *   `prep_samples.py`: Script for generating and caching datasets (if needed separately).
-   **`tests/`**: Unit tests (e.g., `test_ciphers.py`).

## Usage

The typical workflow involves suggesting or defining experiments, adding them to the queue, running the researcher to process the queue, and then analyzing the results.

### 1. Suggesting Experiments (Optional)

```bash
# Suggest N experiments based on current results (trend-following/exploration)
python suggest_experiments.py --suggest 10

# Generate a set of initial experiments if none exist (cold start)
python suggest_experiments.py --cold-start 30
```
*(Suggestions are printed to the console, often piped to `manage_queue.py`)*

### 2. Managing the Experiment Queue

```bash
# Add experiments by specifying parameter ranges/values (creates combinations)
# (Example output from suggest_experiments.py might look like this)
python manage_queue.py --d_model 128,256 --nhead 4,8 --learning_rate 1e-4,3e-4 --patience 10

# Add specific, single experiments (use '=' for single values)
python manage_queue.py --d_model=512 --nhead=8 --num_encoder_layers=6

# List pending experiments
python manage_queue.py --list

# Show N most recently completed experiments
python manage_queue.py --show-completed 10

# Clear the pending queue
python manage_queue.py --clear

# Replace the entire queue with newly specified experiments
python manage_queue.py --replace --d_model 64 --nhead 2
```

### 3. Running Experiments

```bash
# Process all experiments currently in the pending queue
python researcher.py
```
*(This is the primary way to run training. It picks up where it left off if stopped.)*

```bash
# Run a single, specific experiment immediately (bypassing queue)
# Note: Requires specifying *all* necessary hyperparameters
python researcher.py --immediate --num_samples 10000 --sample_length 500 --d_model 128 --nhead 4 --num_encoder_layers 2 --dim_feedforward 512 --batch_size 32 --dropout_rate 0.1 --learning_rate 1e-4 --patience 10
```

### 4. Analyzing Results

```bash
# Launch the interactive Dash visualization app
python visualization.py

# Generate animated confusion matrix GIFs for specific experiments by ID
python generate_gifs.py --id 20250330-1 20250330-5

# Generate GIFs for the 5 most recent experiments
python generate_gifs.py --recent 5

# Generate loss graphs using a filter (e.g., specific model size)
python generate_loss_graphs.py --filter "d_model=256"

# Generate final confusion matrix plot for a specific experiment
python generate_trend_matrix.py --id 20250330-3
```

### 5. Querying Models

```bash
# Classify stdin using a specific model
echo "This is some sample text to classify" | python query_model.py --id 20250330-1

# Classify text directly
python query_model.py --id 20250330-1 --text "uryyb jbeyq"
```

### 6. Purging Experiments

```bash
# See which files *would* be deleted for an experiment (dry run)
python purge_experiments.py --id 20250330-2 --dry-run

# Delete a specific experiment and all its artifacts
python purge_experiments.py --id 20250330-2

# Delete all experiments matching a filter (use with caution!)
python purge_experiments.py --filter "dropout_rate=0.5"
```

### 7. Running Tests

```bash
# Run all tests in the tests directory
python -m unittest discover tests/

# Run tests within a specific file
python -m unittest tests/test_ciphers.py

# Run a specific test class or method
python -m unittest tests.test_ciphers.TestCiphers.test_vigenere
```

## Research

The project facilitates research into transformer model capabilities for cryptanalysis, hyperparameter tuning strategies, and the characteristics of different classical ciphers. Key results and observations are often documented in `findings.md`.
