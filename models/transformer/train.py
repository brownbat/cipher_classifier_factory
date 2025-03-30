import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time
import os
import itertools
import json
import signal
import sys
import hashlib
import math
from typing import Optional, Dict, Any, Tuple, List

# Ensure imports from project structure work when train.py is called
# (Assuming train_model is called from researcher.py at the root)
try:
    from .model import TransformerClassifier # Relative import within the same package
    from models.common.data import load_and_preprocess_data, create_data_loaders
    from models.common.utils import clear_gpu_memory
except ImportError:
    # Fallback for potential direct execution or different call structure
    from model import TransformerClassifier
    # Adjust path if common is not directly accessible
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from models.common.data import load_and_preprocess_data, create_data_loaders
    from models.common.utils import clear_gpu_memory


# Constants
EPSILON = 1e-7  # Small constant to prevent log(0)
CHECKPOINT_DIR = "data/checkpoints" # Relative to project root
CHECKPOINT_FREQ = 1  # Save checkpoint every N epochs
MAX_SAFETY_EPOCHS = 2000 # Safety limit for training loop

# Ensure checkpoint directory exists
# Use absolute path based on this file's location for robustness during directory creation
_TRAIN_PY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT_FROM_TRAIN = os.path.abspath(os.path.join(_TRAIN_PY_DIR, '..', '..'))
_ABS_CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT_FROM_TRAIN, CHECKPOINT_DIR)
os.makedirs(_ABS_CHECKPOINT_DIR, exist_ok=True)
# Use relative path CHECKPOINT_DIR for filenames stored in logs/metadata


def generate_config_hash(hyperparams: Dict[str, Any], num_classes: int, vocab_size: int) -> str:
    """
    Generate an 8-char MD5 hash from key architectural hyperparameters.

    Includes parameters affecting model structure: d_model, nhead, num_layers,
    dim_feedforward, vocab_size, and num_classes.

    Args:
        hyperparams: Dictionary containing model hyperparameters.
        num_classes: Number of output classes (affects final layer).
        vocab_size: Size of the vocabulary (affects embedding layer).

    Returns:
        An 8-character hexadecimal hash string.
    """
    key_params = {
        # Core Transformer architecture params
        'd_model': hyperparams.get('d_model'),
        'nhead': hyperparams.get('nhead'),
        'num_encoder_layers': hyperparams.get('num_encoder_layers'),
        'dim_feedforward': hyperparams.get('dim_feedforward'),
        # Input/Output layer params
        'vocab_size': vocab_size,
        'num_classes': num_classes
    }
    # Create a stable string representation (sorted keys) and hash it
    param_str = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]


def get_checkpoint_path(experiment_id: str, hyperparams: Dict[str, Any], num_classes: int, vocab_size: int, type: str = 'latest') -> str:
    """
    Construct the checkpoint filename including experiment ID, config hash, and type.

    Args:
        experiment_id: The canonical experiment ID (e.g., "YYYYMMDD-N").
        hyperparams: Dictionary of hyperparameters to generate the config hash.
        num_classes: Number of classes (part of config hash).
        vocab_size: Vocabulary size (part of config hash).
        type: The type of checkpoint ('latest' or 'best').

    Returns:
        The full relative path for the checkpoint file (e.g., "data/checkpoints/YYYYMMDD-N_hash_latest.pt").
    """
    config_hash = generate_config_hash(hyperparams, num_classes, vocab_size)
    filename = f"{experiment_id}_{config_hash}_{type}.pt"
    return os.path.join(CHECKPOINT_DIR, filename) # Return relative path


def clean_old_checkpoints(experiment_id: str, completed: bool = False):
    """
    Clean up checkpoint files (.pt) for a given experiment ID.

    If completed=True, removes ALL checkpoints starting with the experiment_id.
    If completed=False (default), finds the most recent '_latest.pt' and '_best.pt'
    files (based on modification time, across any config hash) starting with
    the experiment_id and removes all other checkpoints for that ID.

    Args:
        experiment_id: The canonical experiment ID (e.g., "YYYYMMDD-N").
        completed: If True, remove all checkpoints for this ID.
    """
    checkpoints_found = []
    abs_checkpoint_dir = os.path.join(_PROJECT_ROOT_FROM_TRAIN, CHECKPOINT_DIR) # Use absolute path for listing

    if not os.path.exists(abs_checkpoint_dir):
        # print(f"Checkpoint directory not found: {abs_checkpoint_dir}. No cleanup needed.")
        return

    # Find all .pt files starting with the exact experiment ID followed by an underscore
    prefix = f"{experiment_id}_"
    for filename in os.listdir(abs_checkpoint_dir):
        if filename.startswith(prefix) and filename.endswith('.pt'):
            checkpoints_found.append(filename)

    if not checkpoints_found:
        # print(f"No checkpoints found for experiment {experiment_id}.")
        return

    # If experiment is completed, remove all associated checkpoints
    if completed:
        removed_count = 0
        print(f"Cleaning up ALL checkpoints for completed experiment: {experiment_id}")
        for chk_name in checkpoints_found:
            chk_path = os.path.join(abs_checkpoint_dir, chk_name)
            try:
                os.remove(chk_path)
                removed_count += 1
            except Exception as e:
                print(f"  Error removing checkpoint {chk_name}: {e}")
        if removed_count > 0:
            print(f"  Removed {removed_count} checkpoint files.")
        return

    # If experiment is NOT completed, keep only the single most recent 'latest' and 'best'
    latest_files = [f for f in checkpoints_found if '_latest.pt' in f]
    best_files = [f for f in checkpoints_found if '_best.pt' in f]

    # Sort by modification time (newest first) using absolute paths for getmtime
    latest_files.sort(key=lambda x: os.path.getmtime(os.path.join(abs_checkpoint_dir, x)), reverse=True)
    best_files.sort(key=lambda x: os.path.getmtime(os.path.join(abs_checkpoint_dir, x)), reverse=True)

    keep_files = set()
    if latest_files:
        keep_files.add(latest_files[0]) # Keep the absolute most recent 'latest' file
    if best_files:
        keep_files.add(best_files[0]) # Keep the absolute most recent 'best' file

    # Remove checkpoints not in the keep_files set
    removed_count = 0
    for chk_name in checkpoints_found:
        if chk_name not in keep_files:
            chk_path = os.path.join(abs_checkpoint_dir, chk_name)
            try:
                os.remove(chk_path)
                removed_count += 1
            except Exception as e:
                print(f"  Error removing old checkpoint {chk_name}: {e}")

    # No need to print excessive logs during training unless removing files
    # if removed_count > 0:
    #     print(f"Removed {removed_count} old checkpoints for {experiment_id}, keeping {len(keep_files)}")


def save_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                      scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, global_step: int,
                      training_metrics: Dict, token_dict: Dict, label_encoder: Any,
                      hyperparams: Dict, is_best: bool = False):
    """
    Save training checkpoint (model state, optimizer, scheduler, metrics, metadata).
    Also triggers cleanup of older checkpoints for the same experiment ID if saving a 'latest' checkpoint.

    Args:
        checkpoint_path: Full path where the checkpoint will be saved.
        model: The model (potentially wrapped in DataParallel).
        optimizer: The optimizer instance.
        scheduler: The learning rate scheduler instance.
        epoch: The last completed epoch index (0-based).
        global_step: The total number of optimizer steps taken.
        training_metrics: Dictionary containing running lists of metrics.
        token_dict: Vocabulary mapping tokens to indices.
        label_encoder: Fitted sklearn LabelEncoder.
        hyperparams: Dictionary of hyperparameters for this run (must include 'experiment_id').
        is_best: Boolean indicating if this checkpoint represents the best performance so far.
    """
    experiment_id = hyperparams.get('experiment_id', 'unknown_save')
    abs_checkpoint_path = os.path.join(_PROJECT_ROOT_FROM_TRAIN, checkpoint_path) # Use absolute path for saving

    # Get model state dict (handle DataParallel)
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    checkpoint_data = {
        'experiment_id': experiment_id, # Store ID within checkpoint for reference
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_metrics': training_metrics, # Saves the running metrics dict
        'token_dict': token_dict,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'hyperparams': hyperparams # Includes architecture and training params
    }

    try:
        # Save checkpoint using pickle protocol 4 and ensure directory exists
        os.makedirs(os.path.dirname(abs_checkpoint_path), exist_ok=True)
        torch.save(checkpoint_data, abs_checkpoint_path, _use_new_zipfile_serialization=True, pickle_protocol=4)

        status = "Best" if is_best else "Latest"
        print(f"✅ {status} checkpoint saved: {checkpoint_path} (Epoch {epoch + 1})")

        # Clean up older checkpoints *only* when saving a 'latest' one to ensure
        # we don't delete the latest needed for resuming when only saving 'best'.
        if not is_best:
            clean_old_checkpoints(experiment_id, completed=False)

    except Exception as e:
        print(f"❌ ERROR saving checkpoint to {abs_checkpoint_path}: {e}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                      scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Optional[Dict]:
    """
    Load training checkpoint state into model, optimizer, and scheduler.

    Args:
        checkpoint_path: Full path to the checkpoint file.
        model: The model instance (must match checkpoint architecture).
        optimizer: The optimizer instance to load state into.
        scheduler: The scheduler instance to load state into.

    Returns:
        The loaded checkpoint dictionary if successful, otherwise None.
    """
    abs_checkpoint_path = os.path.join(_PROJECT_ROOT_FROM_TRAIN, checkpoint_path) # Use absolute path for loading

    if not os.path.exists(abs_checkpoint_path):
        # print(f"No checkpoint found at {abs_checkpoint_path}") # This is checked before calling usually
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load checkpoint data onto the target device
        checkpoint = torch.load(abs_checkpoint_path, map_location=device, weights_only=False) # Use weights_only=False

        # --- Load Model State ---
        state_dict = checkpoint['model_state_dict']
        try:
            # Load state dict (handle DataParallel wrapper potentially)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            print("   Model state loaded successfully.")
        except RuntimeError as e:
             # This indicates a mismatch, likely caught by hash check, but good to handle
             print(f"⚠️ WARNING: Checkpoint model architecture mismatch: {e}.")
             print("   Cannot resume training from this checkpoint. Starting fresh.")
             return None # Signal failure to load incompatible checkpoint

        # --- Load Optimizer State ---
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("   Optimizer state loaded.")
            except Exception as e:
                 print(f"   Warning: Could not load optimizer state: {e}. Optimizer will start fresh.")
        elif optimizer is not None:
             print("   Warning: Optimizer state not found in checkpoint.")

        # --- Load Scheduler State ---
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("   Scheduler state loaded.")
            except Exception as e:
                 print(f"   Warning: Could not load scheduler state: {e}. Scheduler will start fresh.")
        elif scheduler is not None:
             print("   Warning: Scheduler state not found in checkpoint.")

        # --- Report Loaded State ---
        completed_epoch = checkpoint.get('epoch', -1) # 0-based index
        global_step = checkpoint.get('global_step', 0)
        print(f"   Checkpoint state indicates epoch {completed_epoch + 1} completed, global step {global_step}.")

        return checkpoint # Return the full checkpoint data

    except Exception as e:
        print(f"❌ Error loading checkpoint from {abs_checkpoint_path}: {e}")
        # Consider deleting the corrupted checkpoint?
        # try:
        #     os.remove(abs_checkpoint_path)
        #     print(f"   Removed potentially corrupted checkpoint file.")
        # except Exception as rem_e:
        #     print(f"   Failed to remove corrupted checkpoint: {rem_e}")
        return None


def train_model(data, hyperparams: Dict[str, Any]) -> Tuple[Optional[nn.Module], Optional[Dict], Optional[Dict]]:
    """
    Trains the transformer model, handling checkpoints, early stopping, and metrics.

    Args:
        data (pd.DataFrame): DataFrame with 'text' and 'cipher' columns.
        hyperparams (Dict[str, Any]): Dictionary of hyperparameters for training.
                                       Must include 'experiment_id'.

    Returns:
        Tuple containing:
        - model (Optional[nn.Module]): Trained model loaded to best state, or None if training failed.
        - final_metrics (Optional[Dict]): Dictionary of summarized metrics, or None if training failed.
        - model_metadata (Optional[Dict]): Dictionary of metadata for inference, or None if training failed.
    """
    experiment_id = hyperparams.get('experiment_id')
    if not experiment_id:
        print("❌ ERROR: 'experiment_id' missing in hyperparams. Cannot proceed.")
        return None, None, None

    # --- Parameter Extraction ---
    learning_rate = hyperparams.get('learning_rate', 1e-4)
    batch_size = hyperparams.get('batch_size', 32)
    d_model = hyperparams.get('d_model', 128)
    nhead = hyperparams.get('nhead', 4)
    num_encoder_layers = hyperparams.get('num_encoder_layers', 2)
    dim_feedforward = hyperparams.get('dim_feedforward', 512)
    dropout_rate = hyperparams.get('dropout_rate', 0.1)
    early_stopping_patience = hyperparams.get('early_stopping_patience', 10)
    early_stopping_metric = hyperparams.get('early_stopping_metric', 'val_loss')
    early_stopping_min_delta = hyperparams.get('early_stopping_min_delta', 0.0001)
    monitor_mode = 'min' if early_stopping_metric == 'val_loss' else 'max'
    warmup_steps_config = hyperparams.get('warmup_steps') # Can be None

    print(f"--- Training Experiment: {experiment_id} ---")
    print(f"Hyperparams: {hyperparams}") # Log effective hyperparams
    print(f"Early Stopping: Monitor='{early_stopping_metric}', Patience={early_stopping_patience}, Mode='{monitor_mode}', MinDelta={early_stopping_min_delta if monitor_mode=='min' else 'N/A'}")

    # --- Data Preparation ---
    try:
        # vocab_size is determined by the fixed character set used in tokenizer
        vocab_size = 27 # 26 lowercase + 1 for padding token 0 (or unknown)
        # num_classes is determined by the unique labels in the current dataset
        num_classes = len(data['cipher'].unique())
        print(f"Data: {len(data)} samples, Vocab Size: {vocab_size}, Num Classes: {num_classes}")

        X, y, token_dict, label_encoder = load_and_preprocess_data(data)
        train_loader, val_loader = create_data_loaders(X, y, batch_size)

        if len(train_loader) == 0 or len(val_loader) == 0:
            print("❌ ERROR: Training or Validation DataLoader is empty. Cannot train.")
            return None, None, None
    except Exception as e:
        print(f"❌ ERROR during data preparation for {experiment_id}: {e}")
        return None, None, None

    # --- Model Initialization ---
    try:
        # Pass necessary architecture params
        model = TransformerClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
            dropout=dropout_rate
        )
    except Exception as e:
        print(f"❌ ERROR initializing Transformer model: {e}")
        return None, None, None

    # Use GPU if available, handle DataParallel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # LR Scheduler Setup
    estimated_max_steps = MAX_SAFETY_EPOCHS * len(train_loader)
    if warmup_steps_config is None:
        # Default warmup: 10% of safety limit steps, capped at 1000
        warmup_steps = min(estimated_max_steps // 10, 1000)
    else:
        warmup_steps = int(warmup_steps_config)
    total_steps_for_scheduler = estimated_max_steps # Use large estimate for cosine decay

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps_for_scheduler - warmup_steps))
        progress = min(progress, 1.0) # Clamp progress
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) # Use math.cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- State Initialization ---
    start_epoch = 0
    global_step = 0
    cumulative_duration = 0.0
    epochs_no_improve = 0
    best_metric_value = math.inf if monitor_mode == 'min' else -math.inf

    # Use the new function signature to get checkpoint paths
    # Pass num_classes and vocab_size for hash generation
    latest_checkpoint_path = get_checkpoint_path(experiment_id, hyperparams, num_classes, vocab_size, type='latest')
    best_checkpoint_path = get_checkpoint_path(experiment_id, hyperparams, num_classes, vocab_size, type='best')

    # Running metrics dictionary (re-initialized on each run/resume)
    running_metrics = {
        'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'conf_matrix': [],
        'learning_rates': [], 'epochs_completed': 0, 'stopped_early': False,
        'best_epoch': -1, 'best_val_loss': None, 'best_val_accuracy': None
    }

    # --- Load Checkpoint If Exists ---
    if os.path.exists(latest_checkpoint_path):
        print(f"Attempting to load latest checkpoint: {latest_checkpoint_path}")
        # Pass model, optimizer, scheduler to load state into them
        checkpoint = load_checkpoint(latest_checkpoint_path, model, optimizer, scheduler)
        if checkpoint:
            # Check if hyperparams match (redundant if hash works, but good safeguard)
            # stored_hyperparams = checkpoint.get('hyperparams', {})
            # if stored_hyperparams != hyperparams: # Simple dict comparison might fail on minor diffs
            #    print("Warning: Hyperparameters in checkpoint differ from current config. Checkpoint may be incompatible.")
            #    checkpoint = None # Treat as incompatible

            if checkpoint: # Proceed if checkpoint is valid
                start_epoch = checkpoint.get('epoch', -1) + 1 # Start from the next epoch
                global_step = checkpoint.get('global_step', 0)

                # Restore early stopping state from loaded metrics history
                loaded_metric_history = checkpoint.get('training_metrics', {}).get(early_stopping_metric, [])
                if loaded_metric_history:
                    if monitor_mode == 'min':
                        current_best = min(loaded_metric_history)
                        best_idx = loaded_metric_history.index(current_best)
                    else:
                        current_best = max(loaded_metric_history)
                        best_idx = loaded_metric_history.index(current_best)

                    # Check if the recorded best_metric_value is better than history implies
                    saved_best_value = checkpoint.get('training_metrics',{}).get(f'best_{early_stopping_metric}')
                    if saved_best_value is not None:
                         if (monitor_mode == 'min' and saved_best_value < current_best) or \
                            (monitor_mode == 'max' and saved_best_value > current_best):
                              current_best = saved_best_value # Trust the saved best value if better

                    best_metric_value = current_best
                    epochs_no_improve = len(loaded_metric_history) - 1 - best_idx
                    print(f"   Resuming early stopping state: Best '{early_stopping_metric}' = {best_metric_value:.4f}, epochs_no_improve = {epochs_no_improve}")
                else:
                     print("   No previous metric history found in checkpoint for early stopping state.")

                # Restore cumulative duration if saved
                cumulative_duration = checkpoint.get('training_metrics', {}).get('training_duration', 0.0)
                print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch + 1}. Prior duration: {cumulative_duration:.1f}s")

        else: # load_checkpoint returned None (error or incompatibility)
            print(f"❌ Checkpoint at {latest_checkpoint_path} unusable. Starting fresh training.")
            start_epoch = 0 # Ensure reset
            global_step = 0
            cumulative_duration = 0.0
    else:
        print(f"No latest checkpoint found for experiment {experiment_id}. Starting fresh training.")

    # --- Training Loop ---
    session_start_time = time.time()
    print(f"\nStarting training loop from epoch {start_epoch + 1}...")

    training_failed = False # Flag to track NaN/error states
    for epoch in itertools.count(start_epoch):
        if epoch >= MAX_SAFETY_EPOCHS:
            print(f"WARNING: Reached safety limit of {MAX_SAFETY_EPOCHS} epochs. Stopping.")
            running_metrics['stopped_early'] = True
            break # Exit loop

        clear_gpu_memory()
        model.train()
        epoch_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:>3} Trn', leave=False, unit='batch')

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True) # Use non_blocking for potential overlap
            optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential performance gain

            logits = model(inputs)
            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                 print(f"\n❌ NaN/Inf detected in training loss at epoch {epoch+1}, step {global_step}. Stopping training.")
                 running_metrics['stopped_early'] = True
                 training_failed = True
                 break # Exit inner loop

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
            optimizer.step()
            scheduler.step() # Step scheduler after optimizer

            epoch_train_loss += loss.item()
            global_step += 1
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"}, refresh=False) # Format loss value *before* putting in dict

        if training_failed: break # Exit outer loop if inner loop failed

        avg_train_loss = epoch_train_loss / len(train_loader)
        running_metrics['train_loss'].append(avg_train_loss)
        current_lr = scheduler.get_last_lr()[0]
        running_metrics['learning_rates'].append(current_lr)

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0.0
        correct_predictions = 0
        all_preds_list = []
        all_labels_list = []
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1:>3} Val', leave=False, unit='batch')

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                     print(f"\n❌ NaN/Inf detected in validation loss at epoch {epoch+1}. Stopping training.")
                     running_metrics['stopped_early'] = True
                     training_failed = True
                     break # Exit inner validation loop

                epoch_val_loss += loss.item()
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(probabilities, 1)
                correct_predictions += (predicted == labels).sum().item()
                all_preds_list.append(predicted.cpu())
                all_labels_list.append(labels.cpu())

        if training_failed: break # Exit outer loop

        # Concatenate results from all validation batches
        all_predictions_np = torch.cat(all_preds_list).numpy()
        all_true_labels_np = torch.cat(all_labels_list).numpy()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = correct_predictions / len(all_true_labels_np) if len(all_true_labels_np) > 0 else 0.0
        conf_matrix = confusion_matrix(all_true_labels_np, all_predictions_np)

        running_metrics['val_loss'].append(avg_val_loss)
        running_metrics['val_accuracy'].append(val_accuracy)
        running_metrics['conf_matrix'].append(conf_matrix) # Store numpy array directly
        running_metrics['epochs_completed'] = epoch + 1

        # Print epoch summary
        print(f"Epoch {epoch+1:>3} | Trn Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.2e}")

        # --- Early Stopping Check ---
        current_metric_value = avg_val_loss if early_stopping_metric == 'val_loss' else val_accuracy
        improved = False
        if not math.isnan(current_metric_value) and not math.isinf(current_metric_value):
            if monitor_mode == 'min':
                if current_metric_value < best_metric_value - early_stopping_min_delta: improved = True
            else: # mode == 'max'
                if current_metric_value > best_metric_value: improved = True

        if improved:
            print(f"   -> Improvement: {early_stopping_metric} {best_metric_value:.4f} -> {current_metric_value:.4f}. Saving best checkpoint.")
            best_metric_value = current_metric_value
            epochs_no_improve = 0
            running_metrics['best_epoch'] = epoch + 1
            running_metrics['best_val_loss'] = avg_val_loss
            running_metrics['best_val_accuracy'] = val_accuracy
            # Save the 'best' checkpoint state
            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, epoch, global_step,
                            running_metrics, token_dict, label_encoder, hyperparams, is_best=True)
        else:
            epochs_no_improve += 1
            # print(f"   No improvement detected for {epochs_no_improve} epochs.") # Can be verbose

        if epochs_no_improve >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered after epoch {epoch + 1}. Patience={early_stopping_patience}.")
            running_metrics['stopped_early'] = True
            break # Exit training loop

        # --- Save Latest Checkpoint Periodically ---
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            # Save 'latest' checkpoint state for resuming
            save_checkpoint(latest_checkpoint_path, model, optimizer, scheduler, epoch, global_step,
                            running_metrics, token_dict, label_encoder, hyperparams, is_best=False)

    # --- End of Training Loop ---

    # --- Post-Training ---
    session_end_time = time.time()
    session_duration = session_end_time - session_start_time
    total_duration = cumulative_duration + session_duration
    # Note: training_duration is added to final_metrics_summary later

    status_message = "failed (NaN/Inf)" if training_failed else \
                     "stopped early" if running_metrics['stopped_early'] else \
                     "completed safety limit" if epoch >= MAX_SAFETY_EPOCHS else \
                     "finished successfully" # Clarified success message

    print(f"\nTraining {status_message}. Total epochs run: {running_metrics.get('epochs_completed', 0)}. Total duration: {total_duration:.1f}s")

    # If training failed, return None immediately
    if training_failed:
         print(f"❌ Training failed for {experiment_id}. No model or metrics returned.")
         clean_old_checkpoints(experiment_id, completed=True) # Clean up any checkpoints
         # <<< CHANGE: Return None for all expected values >>>
         return None, None, None, None

    # Load the best model state if it exists and training didn't fail
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best model state from epoch {running_metrics.get('best_epoch', 'N/A')}...")
        # Use a temporary variable for loaded data to avoid shadowing model
        loaded_best_checkpoint_data = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        try:
             state_dict_to_load = loaded_best_checkpoint_data['model_state_dict']
             if isinstance(model, nn.DataParallel):
                 model.module.load_state_dict(state_dict_to_load)
             else:
                 model.load_state_dict(state_dict_to_load)
             print("   Best model weights loaded successfully.")
        except Exception as e:
             print(f"⚠️ WARNING: Failed to load best model weights from {best_checkpoint_path}: {e}. Returning final model state.")
    elif running_metrics.get('best_epoch', -1) != -1: # Check if a best epoch was recorded
         print(f"⚠️ WARNING: Best epoch was {running_metrics['best_epoch']} but best checkpoint not found at {best_checkpoint_path}. Returning final model state.")
    else:
         print("   No improvement recorded during training or best checkpoint missing. Returning final model state.")

    cm_history_list = running_metrics.get('conf_matrix', [])
    cm_history_path = None # Initialize path variable

    if cm_history_list:
        print(f"   Preparing to save confusion matrix history ({len(cm_history_list)} epochs)...")
        try:
            # Determine shape and validate consistency
            num_epochs = len(cm_history_list)
            first_valid_cm = next((np.array(cm) for cm in cm_history_list if cm is not None), None)

            if first_valid_cm is None or first_valid_cm.ndim != 2 or first_valid_cm.shape[0] != first_valid_cm.shape[1]:
                 print(f"   ERROR: No valid 2D confusion matrix found in history for {experiment_id}. Cannot save.")
                 cm_history_path = None
            else:
                num_classes = first_valid_cm.shape[0]
                final_shape = (num_epochs, num_classes, num_classes)
                # <<< CHANGE: Pre-allocate the final numeric array >>>
                cm_history_final_array = np.zeros(final_shape, dtype=np.int64) # Use int64
                valid_save = True

                for i, cm in enumerate(cm_history_list):
                    try:
                         if cm is None: # Handle potential None entries if needed
                             # Option: fill with zeros or a marker like -1? Let's use zeros.
                             cm_array = np.zeros((num_classes, num_classes), dtype=np.int64)
                         else:
                             cm_array = np.array(cm, dtype=np.int64)

                         if cm_array.shape != (num_classes, num_classes):
                              print(f"   ERROR: CM at epoch {i+1} has inconsistent shape {cm_array.shape} (expected {(num_classes, num_classes)}). Aborting save.")
                              valid_save = False
                              break
                         # <<< CHANGE: Fill the pre-allocated array >>>
                         cm_history_final_array[i] = cm_array
                    except (TypeError, ValueError) as conv_err:
                         print(f"   ERROR: Could not convert CM at epoch {i+1} to numeric array: {conv_err}. Aborting save.")
                         valid_save = False
                         break

                if valid_save:
                    # Define save path relative to project root
                    cm_history_dir = os.path.join(_PROJECT_ROOT_FROM_TRAIN, "data", "cm_history")
                    os.makedirs(cm_history_dir, exist_ok=True)
                    cm_history_filename = f"{experiment_id}_cm_history.npy"
                    abs_cm_history_path = os.path.join(cm_history_dir, cm_history_filename)

                    # Save the numeric numpy array (this should NOT require pickle)
                    np.save(abs_cm_history_path, cm_history_final_array)
                    cm_history_path = os.path.join("data", "cm_history", cm_history_filename)
                    print(f"   ✅ Saved CM history to: {cm_history_path} (Shape: {cm_history_final_array.shape}, Dtype: {cm_history_final_array.dtype})")
                else:
                     cm_history_path = None # Save aborted

        except Exception as e:
            print(f"❌ ERROR during preparation or saving of confusion matrix history for {experiment_id}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            cm_history_path = None
    else:
         print("   No confusion matrix history found in running_metrics to save.")


    final_metrics_summary = {
        # Curves (kept in JSON for now)
        'train_loss_curve': running_metrics.get('train_loss', []),
        'val_loss_curve': running_metrics.get('val_loss', []),
        'val_accuracy_curve': running_metrics.get('val_accuracy', []),
        'learning_rates': running_metrics.get('learning_rates', []),

        # Summary Scalars
        'epochs_completed': running_metrics.get('epochs_completed', 0),
        'training_duration': total_duration, # Use calculated total_duration
        'stopped_early': running_metrics.get('stopped_early', False),
        'best_epoch': running_metrics.get('best_epoch', -1),
        'best_val_loss': running_metrics.get('best_val_loss'),
        'best_val_accuracy': running_metrics.get('best_val_accuracy'),
    }

    # Prepare model metadata dictionary to return (keeping summary metrics for now)
    config_hash = generate_config_hash(hyperparams, num_classes, vocab_size)
    model_metadata = {
        'experiment_id': experiment_id,
        'config_hash': config_hash,
        'token_dict': token_dict,
        'label_encoder_classes': label_encoder.classes_.tolist(), # Store as list for JSON
        'hyperparams': hyperparams, # Save the exact hyperparams used
        # Specific architectural params (convenience)
        'd_model': d_model, 'nhead': nhead, 'num_encoder_layers': num_encoder_layers,
        'dim_feedforward': dim_feedforward, 'vocab_size': vocab_size, 'num_classes': num_classes,
        # Performance summary (redundant with final_metrics_summary, but kept for context within metadata)
        'best_val_accuracy_achieved': running_metrics.get('best_val_accuracy'),
        'best_val_loss_achieved': running_metrics.get('best_val_loss'),
        'epoch_of_best': running_metrics.get('best_epoch', -1)
    }

    # Final checkpoint cleanup after successful completion or early stop
    print("\nCleaning up checkpoint files for completed/stopped experiment...")
    clean_old_checkpoints(experiment_id, completed=True) # Remove all .pt files
    print("Checkpoint cleanup complete.")

    # <<< CHANGE: Return the 4 values including cm_history_path >>>
    print("--- Training function returning results ---")
    return model, final_metrics_summary, model_metadata, cm_history_path
