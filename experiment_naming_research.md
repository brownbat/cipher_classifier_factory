# Experiment Naming Conventions Research

## Overview

This document collects research on how the cipher classifier system names experiments and uses or modifies those names throughout the codebase. It serves as a guide for a refactor/bugcheck to ensure all experiment names are handled with consistent logic.

## Summary of Current Status

There appears to be a transition in progress from an older naming convention (`exp_X`) to a newer date-based naming format (`YYYYMMDD-N`). The `manage_queue.py` file has been updated to use the new format, but other parts of the system still use or expect the older format. This has led to inconsistencies in how experiment identifiers are generated, stored, and referenced throughout the system.

Additionally, a recent change added architecture hashing in `models/transformer/train.py` to prevent checkpoint architecture mismatches, which adds another dimension to the naming system. This hash-based system is important for ensuring that only compatible checkpoints are loaded, but it needs to be consistently integrated with the overall naming convention.

## Current Naming Patterns

Multiple naming conventions appear to be in use:

1. **Sequential ID pattern**: `exp_X` (e.g., `exp_1`, `exp_72`)
2. **Date-based pattern**: `YYYYMMDD-N` (e.g., `20250326-1`)
3. **Full timestamp pattern**: `exp_X_YYYYMMDD_HHMMSS` (e.g., `exp_72_20250325_193117`)
4. **UID pattern**: Used as a unique identifier, sometimes identical to `experiment_id`, sometimes a combination of ID and timestamp

## File Naming Conventions

The following file naming patterns are in use:

1. **Model files**: `data/models/{experiment_id}.pt` or `data/models/exp_X_YYYYMMDD_HHMMSS.pt`
2. **Metadata files**: `data/models/{experiment_id}_metadata.pkl` or `data/models/exp_X_YYYYMMDD_HHMMSS_metadata.pkl`
3. **Checkpoint files**: `data/checkpoints/{experiment_id}_latest.pt` or `data/checkpoints/exp_X_YYYYMMDD_HHMMSS_latest.pt`
4. **Confusion matrix GIFs**: `data/cm/{uid}_conf_matrix.gif`
5. **Loss graphs**: `data/loss_graphs/{uid}_loss.png` or `data/loss_graphs/{uid}_loss_animated.gif`
6. **Trend matrix**: `data/trend_matrix/{uid}_trend_matrix.png`

## Key Files and Their Handling of Experiment Names

### 1. manage_queue.py

This file contains the newest naming logic, creating experiment IDs in the `YYYYMMDD-N` format:

```python
# Gets the next counter for today's date
def get_next_experiment_counter(date_prefix):
    """Get the next experiment counter for a given date prefix."""
    highest_counter = 0
    
    # Load all completed experiments
    all_experiments = []
    completed_experiments = safe_json_load('data/completed_experiments.json')
    if completed_experiments:
        all_experiments.extend(completed_experiments)
    
    # Also check pending experiments
    pending_experiments = safe_json_load('data/pending_experiments.json')
    if pending_experiments:
        all_experiments.extend(pending_experiments)
    
    # Find the highest counter for this date
    for exp in all_experiments:
        exp_id = exp.get('experiment_id', '')
        
        # Check if this experiment ID matches our date prefix
        if exp_id.startswith(date_prefix):
            # Extract the counter part
            try:
                # Split by hyphen and convert the last part to int
                counter_str = exp_id.split('-')[-1]
                counter = int(counter_str)
                highest_counter = max(highest_counter, counter)
            except (ValueError, IndexError):
                # Skip if we can't parse the counter
                continue
    
    # Return the next available counter
    return highest_counter + 1

# Creates date-based experiment IDs
def generate_experiments(params):
    # Generate a date-based session ID to avoid experiment name collisions
    session_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # Get the starting counter for today's experiments
    start_counter = get_next_experiment_counter(session_date)
    
    # Create experiment configurations
    experiments = []
    for i, combination in enumerate(combinations):
        experiment = {
            'experiment_id': f'{session_date}-{start_counter + i}',
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_params': data_params,
            'hyperparams': hyperparams
        }
        experiments.append(experiment)
```

### 2. researcher.py

This file appears to use multiple naming conventions:

1. Sets UID from experiment_id:
```python
exp['uid'] = exp["experiment_id"]  # Use the experiment ID as the UID
```

2. Creates model filenames:
```python
model_filename = f'data/models/{exp["experiment_id"]}.pt'
metadata_filename = f'data/models/{exp["experiment_id"]}_metadata.pkl'
```

3. Creates confusion matrix GIFs:
```python
unique_id = f'{exp["experiment_id"]}_{exp["training_time"]}'
gif_filename = f'data/cm/{unique_id}_conf_matrix.gif'
```

4. Tracks checkpoints:
```python
# Processes checkpoints in clean_old_checkpoints function
for filename in os.listdir(checkpoint_dir):
    if filename.endswith('_latest.pt'):
        experiment_id = filename.split('_latest.pt')[0]
        clean_old_checkpoints(experiment_id, keep_n=1)
```

### 3. models/transformer/train.py

This file handles checkpointing and introduces architecture hashing for checkpoint compatibility:

```python
def generate_config_hash(hyperparams):
    """
    Generate a hash from the model configuration parameters.
    This helps ensure checkpoints are only loaded for matching architectures.
    """
    # Extract key parameters that define the model architecture
    key_params = {
        'd_model': hyperparams.get('d_model', 128),
        'nhead': hyperparams.get('nhead', 8),
        'num_encoder_layers': hyperparams.get('num_encoder_layers', 2),
        'dim_feedforward': hyperparams.get('dim_feedforward', 512),
        'vocab_size': hyperparams.get('vocab_size', 27)
    }
    # Create a stable string representation and hash it
    param_str = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]  # First 8 chars of hash
```

Checkpoint paths now include the architecture hash:

```python
def get_checkpoint_path(experiment_id, hyperparams=None, epoch=None):
    """
    Get checkpoint path for a given experiment ID and hyperparameters.
    Includes config hash if hyperparams are provided to ensure architecture matching.
    """
    if hyperparams:
        # Generate a hash of the configuration to prevent architecture mismatches
        config_hash = generate_config_hash(hyperparams)
        return os.path.join(CHECKPOINT_DIR, f"{experiment_id}_{config_hash}_latest.pt")
    else:
        # Fallback for backward compatibility
        return os.path.join(CHECKPOINT_DIR, f"{experiment_id}_latest.pt")
```

The checkpoint cleaning function tries to handle both old and new ID formats:

```python
def clean_old_checkpoints(experiment_id, keep_n=1, completed=False):
    # Get base experiment ID (without timestamp)
    base_id = experiment_id.split('_20')[0]
    
    # Get all checkpoints for this experiment (including those with hash codes)
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) 
                  if (f.startswith(experiment_id) or f.startswith(base_id)) and 
                  f.endswith('.pt')]
```

This splitting approach assumes the experiment ID format contains a timestamp that starts with `_20`, which may not be true for the new date-based format (`YYYYMMDD-N`).

### 4. Visualization Tools (generate_gifs.py, generate_loss_graphs.py, generate_trend_matrix.py)

These files use the `uid` field for creating output files:

```python
# In generate_gifs.py
gif_filename = f"{output_dir}/{experiment['uid']}_conf_matrix.gif"

# In generate_loss_graphs.py
output_filename = f"{experiment['uid']}_loss.png"

# In generate_trend_matrix.py 
output_path = os.path.join(OUTPUT_DIR, f"{uid}_trend_matrix.png")
```

## Actual Field Usage in Completed Experiments

Examining actual data in the completed_experiments.json file:

```python
# Sample of experiment_id and uid fields:
First 3 sample experiment IDs: ['exp_1', 'exp_2', 'exp_3']
First 3 sample UIDs: ['exp_1_20250324_185607', 'exp_2_20250324_190300', 'exp_3_20250324_190953']
```

This shows:
1. `experiment_id` uses the simple sequential format (`exp_X`)
2. `uid` combines the experiment_id with a timestamp (`exp_X_YYYYMMDD_HHMMSS`)

These experiments likely predate the new naming convention implemented in `manage_queue.py`. The completed_experiments.json file contains a mix of experiments from before and after the naming convention change. Newer experiments should follow the date-based format (YYYYMMDD-N) for their experiment_id.

## Inconsistencies and Issues

1. **Old vs. New Naming Conventions**: The manage_queue.py file has been updated to use a date-based experiment ID format (YYYYMMDD-N), but the rest of the system still references or expects the older formats.

2. **UID vs. experiment_id**: In researcher.py, UID is set to be identical to the experiment_id:
   ```python
   exp['uid'] = exp["experiment_id"]  # Use the experiment ID as the UID
   ```
   
   This is a recent change to use the experiment_id directly as the UID. With the new date-based naming format, this makes sense since the experiment_id already contains a date. However, for older experiments in the completed_experiments.json file, UIDs follow the format `exp_X_YYYYMMDD_HHMMSS`.
   
   Additionally, in plot_confusion_matrices(), a separate identifier is created:
   ```python
   unique_id = f'{exp["experiment_id"]}_{exp["training_time"]}'
   gif_filename = f'data/cm/{unique_id}_conf_matrix.gif'
   ```
   
   This construction method may need to be updated to handle the new naming convention.

3. **File Path Generation**: File paths are constructed differently in different parts of the code, sometimes using experiment_id directly, sometimes using uid, and sometimes constructing a custom identifier.

4. **Model Saving**: The researcher.py file saves models using `{exp["experiment_id"]}.pt` format, which might not account for the new date-based format.

5. **Checkpointing**: The checkpoint system needs to be verified for compatibility with the new naming scheme.

## Key Areas Needing Attention

1. **Checkpoint System in train.py**: The architecture hash is an important addition, but the current implementation assumes experiment IDs have a timestamp portion that can be separated with `split('_20')`, which won't work with the new date-based format.

2. **Visualization Tools**: All visualization tools (generate_gifs.py, generate_loss_graphs.py, generate_trend_matrix.py) depend on the 'uid' field, which is now being set to be identical to experiment_id. The file naming logic should be updated to handle this change.

3. **Immediate Mode in researcher.py**: The immediate mode still creates experiments with the 'immediate_exp' ID, which doesn't follow either naming convention.

4. **Model File Naming**: The code that saves model files uses the experiment_id directly, which should work with the new format but may cause confusion when mixing old and new formats.

## Next Steps for Refactoring

1. Standardize the experiment ID format across the codebase, preferably using the new date-based format (YYYYMMDD-N)

2. Determine a consistent approach to UID generation:
   - Either always set UID = experiment_id
   - Or always construct UID as experiment_id + timestamp

3. Standardize file naming conventions:
   - Model files: `data/models/{experiment_id}.pt`
   - Metadata files: `data/models/{experiment_id}_metadata.pkl`
   - Checkpoint files: `data/checkpoints/{experiment_id}_latest.pt`
   - Visualization files: Use a consistent convention for all visualization outputs

4. Update all file operations to use the standardized naming patterns:
   - Model saving/loading
   - Checkpoint saving/loading
   - Visualization file generation
   - File cleanup operations

5. Add validation to ensure experiment IDs and UIDs are valid and match the expected format

6. Update the checkpoint system in train.py to fully support the new naming format:
   - Modify the `clean_old_checkpoints` function to work with date-based experiment IDs
   - Make sure the architecture hash logic works with the new format

7. Update all visualization tools to use the new naming format exclusively:
   - Update generate_gifs.py, generate_loss_graphs.py, and generate_trend_matrix.py
   - Consider adding a utility function for constructing file paths consistently

8. Document the new naming convention in the README or CONTRIBUTING guide for future developers