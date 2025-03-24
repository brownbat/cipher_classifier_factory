# Cipher Classifier

A system that builds models to classify various types of classical ciphers.

## Overview

This project implements Transformer-based models to classify text samples into different cipher categories (e.g., English plaintext, Caesar cipher, Vigenère cipher, etc.). It is designed to make it easy to build and compare models with different hyperparameters.

## Project Structure

- **Entry point**: `researcher.py` - Main tool to run experiments and train models
- **Management tools**:
  - `manage_queue.py` - Tool to manage experiment queue
  - `generate_gifs.py` - Tool to generate animated confusion matrix visualizations
- **Model implementations**:
  - `models/transformer/` - Transformer model architecture
  - `models/common/` - Shared utilities and data processing
- **Utilities**:
  - `visualization.py` - Flask interface for results visualization
  - `query_model.py` - Tool to classify text samples using trained models
  - `ciphers.py` - Implementation of various classical ciphers

## Usage

### Running Experiments

```bash
# Run with default settings
python researcher.py

# Run with specific parameters
python researcher.py --num_samples 100000 --epochs 30 --d_model 256 --nhead 8
```

### Managing Experiment Queue

```bash
# Add experiments to queue
python manage_queue.py --d_model 128,256 --nhead 4,8

# Replace entire queue
python manage_queue.py --replace --d_model 128,256 --nhead 4,8

# List pending experiments
python manage_queue.py --list

# Clear experiment queue
python manage_queue.py --clear
```

### Generating Visualizations

```bash
# Generate GIFs for all completed experiments
python generate_gifs.py

# Process only recent experiments
python generate_gifs.py --recent 5

# Process a specific experiment
python generate_gifs.py --experiment exp_123
```

### Running Tests

```bash
# Run all tests
python -m unittest tests/test_ciphers.py

# Run a specific test
python -m unittest tests.test_ciphers.TestCiphers.test_vigenere
```

## Research

The project includes extensive research on how transformer models identify different cipher types and their effectiveness across various hyperparameter configurations.

## Demo

![demo](https://github.com/brownbat/cipher_classifier_factory/assets/26754/0f89f7a5-14b5-496e-ac74-6d21d8b2180d)

You can play with a subset of the collected data [here](https://brownbat.pythonanywhere.com/).