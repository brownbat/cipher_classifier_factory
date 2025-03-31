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
from torch_lr_finder import LRFinder
import matplotlib
matplotlib.use('Agg') # <<< ADDED: Set backend *before* importing pyplot
import matplotlib.pyplot as plt # <<< Import pyplot AFTER setting backend
import pandas as pd
from torch.utils.data import DataLoader

# --- IMPORT TESTS ---
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


# --- CONSTANTS ---
EPSILON = 1e-7  # Small constant to prevent log(0)
CHECKPOINT_DIR = "data/checkpoints" # Relative to project root
CHECKPOINT_FREQ = 1  # Save checkpoint every N epochs
MAX_SAFETY_EPOCHS = 2000 # Safety limit for training loop
ES_COEFFICIENT = 3 # How many times longer should the early stopping patience be than the LR optimization patience
# Default LR if finder fails or is skipped
DEFAULT_INITIAL_LR = 1e-4
# Max LR safety cap
MAX_INITIAL_LR_CAP = 1e-2


# Ensure checkpoint directory exists
# Use absolute path based on this file's location for robustness during directory creation
_TRAIN_PY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT_FROM_TRAIN = os.path.abspath(os.path.join(_TRAIN_PY_DIR, '..', '..'))
_ABS_CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT_FROM_TRAIN, CHECKPOINT_DIR)
os.makedirs(_ABS_CHECKPOINT_DIR, exist_ok=True)
# Use relative path CHECKPOINT_DIR for filenames stored in logs/metadata


# --- CHECKPOINT HELPER FUNCTIONS ---
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
        # <<< MODIFIED >>>: Removed 'learning_rate' from hash generation
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


def save_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                      scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], epoch: int, global_step: int,
                      training_metrics: Dict, token_dict: Dict, label_encoder: Any,
                      hyperparams: Dict, is_best: bool = False):
    """
    Save training checkpoint (model state, optimizer, scheduler, metrics, metadata).
    Also triggers cleanup of older checkpoints for the same experiment ID if saving a 'latest' checkpoint.

    Args:
        checkpoint_path: Full path where the checkpoint will be saved.
        model: The model (potentially wrapped in DataParallel).
        optimizer: The optimizer instance (optional).
        scheduler: The learning rate scheduler instance (optional).
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
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
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
        if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("   Optimizer state loaded.")
            except Exception as e:
                 print(f"   Warning: Could not load optimizer state: {e}. Optimizer will start fresh.")
        elif optimizer is not None:
             print("   Warning: Optimizer state not found or null in checkpoint.")

        # --- Load Scheduler State ---
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("   Scheduler state loaded.")
            except Exception as e:
                 print(f"   Warning: Could not load scheduler state: {e}. Scheduler will start fresh.")
        elif scheduler is not None:
             print("   Warning: Scheduler state not found or null in checkpoint.")

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
# --- END CHECKPOINT HELPER FUNCTIONS ---

def setup_experiment_parameters(hyperparams: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts, validates, and derives training parameters from hyperparameters.

    Args:
        hyperparams: The raw hyperparameter dictionary provided to the training run.
                     Must include 'experiment_id' and model architecture details.

    Returns:
        A dictionary containing validated and derived training configuration parameters,
        or None if critical parameters like 'experiment_id' are missing.
    """
    experiment_id = hyperparams.get('experiment_id')
    if not experiment_id:
        print("❌ ERROR: 'experiment_id' missing in hyperparams. Cannot setup parameters.")
        return None

    print(f"--- Setting up Experiment: {experiment_id} ---")

    # --- Parameter Extraction & Defaults ---
    batch_size = hyperparams.get('batch_size', 32)
    d_model = hyperparams.get('d_model', 128)
    nhead = hyperparams.get('nhead', 4)
    num_encoder_layers = hyperparams.get('num_encoder_layers', 2)
    dim_feedforward = hyperparams.get('dim_feedforward', 512)
    dropout_rate = hyperparams.get('dropout_rate', 0.1)
    lr_scheduler_factor = hyperparams.get('lr_scheduler_factor', 0.2)
    lr_scheduler_min_lr = hyperparams.get('lr_scheduler_min_lr', 1e-7)
    early_stopping_metric = hyperparams.get('early_stopping_metric', 'val_loss') # Default monitor metric
    early_stopping_min_delta = hyperparams.get('early_stopping_min_delta', 0.0001)

    # --- Base Patience Validation and Derived Values ---
    base_patience = hyperparams.get('base_patience', 5) # Default base patience
    if not isinstance(base_patience, int) or base_patience <= 0:
         print(f"Warning: Invalid base_patience ({base_patience}). Using default 5.")
         base_patience = 5
    # Derive scheduler and early stopping patience
    lr_scheduler_patience = base_patience
    early_stopping_patience = base_patience * ES_COEFFICIENT # Ensure ES waits longer

    # --- Determine Monitor Mode ---
    if early_stopping_metric not in ['val_loss', 'val_accuracy']:
        print(f"Warning: Invalid early_stopping_metric '{early_stopping_metric}'. Defaulting to 'val_loss'.")
        early_stopping_metric = 'val_loss'
    monitor_mode = 'min' if early_stopping_metric == 'val_loss' else 'max'

    # --- Log Configuration ---
    print(f"Raw Hyperparameters: {hyperparams}")
    print(f"Using Base Patience: {base_patience}")
    print(f"Derived LR Scheduler Patience: {lr_scheduler_patience}")
    print(f"Derived Early Stopping Patience: {early_stopping_patience}")
    print(f"Early Stopping: Monitor='{early_stopping_metric}', Mode='{monitor_mode}', MinDelta={early_stopping_min_delta if monitor_mode=='min' else 'N/A'}")
    print(f"LR Scheduler: Factor={lr_scheduler_factor}, MinLR={lr_scheduler_min_lr}")

    # --- Compile Training Configuration Dictionary ---
    training_config = {
        'experiment_id': experiment_id,
        'batch_size': batch_size,
        'd_model': d_model,
        'nhead': nhead,
        'num_encoder_layers': num_encoder_layers,
        'dim_feedforward': dim_feedforward,
        'dropout_rate': dropout_rate,
        'base_patience': base_patience, # Keep original base for reference if needed
        'lr_scheduler_patience': lr_scheduler_patience,
        'early_stopping_patience': early_stopping_patience,
        'early_stopping_metric': early_stopping_metric,
        'early_stopping_min_delta': early_stopping_min_delta,
        'monitor_mode': monitor_mode,
        'lr_scheduler_factor': lr_scheduler_factor,
        'lr_scheduler_min_lr': lr_scheduler_min_lr,
        # Add other hyperparams that might be needed later but aren't derived
        # Example: weight_decay, optimizer_type if you add options later
    }

    print("--- Experiment Setup Complete ---")
    return training_config


def prepare_data_for_training(data: pd.DataFrame, batch_size: int) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[int], Optional[int], Optional[Dict], Optional[Any]]:
    """
    Prepares data for training: determines vocab/class size, preprocesses, and creates DataLoaders.

    Args:
        data (pd.DataFrame): DataFrame with 'text' and 'cipher' columns.
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        Tuple containing:
        - train_loader (Optional[DataLoader]): Training DataLoader.
        - val_loader (Optional[DataLoader]): Validation DataLoader.
        - vocab_size (Optional[int]): Determined vocabulary size.
        - num_classes (Optional[int]): Determined number of classes.
        - token_dict (Optional[Dict]): Token-to-index mapping.
        - label_encoder (Optional[Any]): Fitted sklearn LabelEncoder instance.
        Returns (None, None, None, None, None, None) if any critical step fails.
    """
    print("--- Preparing Data ---")
    try:
        # --- Determine Vocab Size and Number of Classes ---
        # Vocab size is fixed based on the tokenizer's character set
        vocab_size = 27 # 26 lowercase + 1 for padding/unknown (Assuming fixed char set)
        # Num classes is derived from the unique labels in the provided data slice
        if 'cipher' not in data.columns or data['cipher'].isna().any():
            print("❌ ERROR: 'cipher' column missing or contains NaN values in input data.")
            return None, None, None, None, None, None
        num_classes = len(data['cipher'].unique())
        if num_classes <= 1:
             print(f"❌ ERROR: Found {num_classes} unique classes in 'cipher' column. Need at least 2.")
             return None, None, None, None, None, None

        print(f"Input Data: {len(data)} samples")
        print(f"Determined Vocab Size: {vocab_size} (fixed)")
        print(f"Determined Num Classes: {num_classes} (from data)")

        # --- Load, Preprocess, and Tokenize ---
        # This function should handle tokenization and label encoding
        X, y, token_dict, label_encoder = load_and_preprocess_data(data)
        print(f"Data preprocessed: X shape {X.shape}, y shape {y.shape}") # Log shapes

        # --- Create DataLoaders ---
        train_loader, val_loader = create_data_loaders(X, y, batch_size)

        # --- Validation ---
        if len(train_loader) == 0 or len(val_loader) == 0:
            print("❌ ERROR: Training or Validation DataLoader is empty after creation.")
            # Potential cause: Very small dataset and large batch size, or split issues.
            return None, None, None, None, None, None

        print(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
        print("--- Data Preparation Complete ---")
        return train_loader, val_loader, vocab_size, num_classes, token_dict, label_encoder

    except KeyError as e:
        print(f"❌ ERROR during data preparation: Missing expected column - {e}")
        return None, None, None, None, None, None
    except ValueError as e:
        print(f"❌ ERROR during data preparation: {e}") # e.g., from LabelEncoder if labels change unexpectedly
        return None, None, None, None, None, None
    except Exception as e:
        print(f"❌ An unexpected ERROR occurred during data preparation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        return None, None, None, None, None, None

def initialize_model_and_device(vocab_size: int,
                                num_classes: int,
                                training_config: Dict[str, Any]
                               ) -> Tuple[Optional[nn.Module], Optional[torch.device]]:
    """
    Initializes the TransformerClassifier model, detects the device, and handles DataParallel.

    Args:
        vocab_size (int): The size of the vocabulary.
        num_classes (int): The number of output classes.
        training_config (Dict[str, Any]): Dictionary containing model architecture parameters
                                           (d_model, nhead, num_encoder_layers, dim_feedforward,
                                           dropout_rate).

    Returns:
        Tuple containing:
        - model (Optional[nn.Module]): The initialized model (potentially wrapped in DataParallel),
                                       moved to the correct device. Returns None on init failure.
        - device (Optional[torch.device]): The torch.device being used (e.g., 'cuda:0' or 'cpu').
                                          Returns None on init failure.
    """
    print("--- Initializing Model and Device ---")
    try:
        # --- Extract necessary architecture parameters ---
        d_model = training_config['d_model']
        nhead = training_config['nhead']
        num_encoder_layers = training_config['num_encoder_layers']
        dim_feedforward = training_config['dim_feedforward']
        dropout_rate = training_config['dropout_rate']

        # --- Instantiate the Model ---
        model = TransformerClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers, # Note: param name is num_layers in class
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
            dropout=dropout_rate
        )
        print(f"Model instantiated: {type(model).__name__}")
        # Optional: Print model summary or number of parameters here if desired
        # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"   - Trainable Parameters: {num_params:,}")

        # --- Determine Device and Handle DataParallel ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            print(f"   - Detected {torch.cuda.device_count()} GPUs. Using DataParallel.")
            model = nn.DataParallel(model)

        # --- Move Model to Device ---
        model.to(device)
        print(f"Model moved to {device}.")

        print("--- Model and Device Initialization Complete ---")
        return model, device

    except KeyError as e:
        print(f"❌ ERROR initializing model: Missing key in training_config - {e}")
        return None, None
    except Exception as e:
        print(f"❌ An unexpected ERROR occurred during model initialization: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def find_initial_learning_rate(model: nn.Module,
                               criterion: nn.Module,
                               train_loader: DataLoader,
                               device: torch.device,
                               experiment_id: str,
                               project_root: str,
                               config_hash: str # Added: Needed for consistent plot naming
                               ) -> float:
    """
    Uses torch-lr-finder to find a suggested initial learning rate.

    Includes range test, manual calculation of steepest gradient, optional plotting,
    error handling, and fallbacks.

    Args:
        model: The model instance (unwrapped if DataParallel was used).
        criterion: The loss function.
        train_loader: Training DataLoader.
        device: The torch device.
        experiment_id: The experiment ID for logging and plot naming.
        project_root: Absolute path to the project root directory.
        config_hash: Unique hash for this model config, for plot naming.

    Returns:
        The suggested initial learning rate (float). Falls back to DEFAULT_INITIAL_LR.
    """
    print("--- Running LR Finder ---")
    initial_lr = DEFAULT_INITIAL_LR # Start with the default fallback
    lr_finder = None
    temp_optimizer = None
    fig = None # Initialize fig for finally clause

    # Use the underlying model if wrapped in DataParallel for LRFinder
    model_to_find = model.module if isinstance(model, nn.DataParallel) else model

    try:
        # Setup LR Finder dependencies
        # Use a low starting LR for the finder's optimizer
        temp_optimizer = torch.optim.AdamW(model_to_find.parameters(), lr=1e-7, weight_decay=0.01)
        lr_finder = LRFinder(model_to_find, temp_optimizer, criterion, device=device)

        # Determine number of iterations for the test
        # Avoid running on the very last batch if only one batch exists.
        # Run for at most 100 iterations or dataset size - 1.
        num_total_batches = len(train_loader)
        if num_total_batches <= 1:
            print("   WARNING: Too few batches in train_loader (<=1). Skipping LR Finder.")
            return initial_lr # Return default

        num_iter = min(100, num_total_batches - 1)
        print(f"   Will run LR range test for {num_iter} iterations.")

        # --- Run the Range Test ---
        print("   Running range test...")
        # The range_test populates lr_finder.history. end_lr=1 is often sufficient.
        lr_finder.range_test(train_loader, end_lr=1, num_iter=num_iter, step_mode="exp")
        print("   Range test finished.")

        # --- Manual LR Suggestion Calculation (using steepest gradient) ---
        manual_suggested_lr = None
        calculated_loss_at_lr = None
        calculated_idx_at_lr = None

        try:
            print("   Calculating LR suggestion from history (steepest gradient)...")
            # Check if history has the necessary keys and enough data points
            if (lr_finder.history and "lr" in lr_finder.history and "loss" in lr_finder.history and
                    len(lr_finder.history["lr"]) > 1 and len(lr_finder.history["loss"]) > 1):

                lrs_hist = np.array(lr_finder.history["lr"])
                losses_hist = np.array(lr_finder.history["loss"])

                # Filter out NaNs/Infs from losses which can corrupt gradient calculation
                valid_indices = np.isfinite(losses_hist)
                num_valid = np.sum(valid_indices)

                if num_valid > 1: # Need at least 2 valid points for gradient
                    lrs_valid = lrs_hist[valid_indices]
                    losses_valid = losses_hist[valid_indices]

                    min_grad_idx = None
                    try:
                        # Calculate the gradient of the log of the losses (often smoother)
                        # Add small epsilon to prevent log(0)
                        log_losses = np.log(losses_valid + 1e-10)
                        grads = np.gradient(log_losses)
                        # Find the index *within the valid arrays* where the gradient is most negative
                        min_grad_idx = grads.argmin()
                    except (ValueError, IndexError, FloatingPointError) as e_grad:
                        print(f"   WARNING: Failed to compute gradients: {e_grad}. Trying without log.")
                        try: # Fallback to gradient of raw losses
                             grads = np.gradient(losses_valid)
                             min_grad_idx = grads.argmin()
                        except (ValueError, IndexError, FloatingPointError) as e_grad_raw:
                             print(f"   WARNING: Failed to compute raw gradients: {e_grad_raw}")


                    # Ensure index is valid for the *valid* arrays
                    if min_grad_idx is not None and min_grad_idx < len(lrs_valid):
                        manual_suggested_lr = lrs_valid[min_grad_idx]
                        calculated_loss_at_lr = losses_valid[min_grad_idx]
                        calculated_idx_at_lr = min_grad_idx # Store index relative to valid points
                        print(f"   Calculated best LR (steepest gradient): {manual_suggested_lr:.2e} at index {min_grad_idx} (of {num_valid} valid points)")
                    else:
                        print("   WARNING: Could not determine steepest gradient index or index out of bounds.")
                else:
                    print(f"   WARNING: Not enough valid (finite) loss points ({num_valid}) in history to calculate gradient.")
            else:
                print("   WARNING: 'lr'/'loss' missing or insufficient data in lr_finder.history.")

        except Exception as e_manual_sugg:
            print(f"   WARNING: Error during manual LR suggestion calculation: {e_manual_sugg}")
            import traceback
            traceback.print_exc()
        # --- End Manual LR Suggestion Calculation ---

        # --- Plotting (Optional but Recommended) ---
        '''try:
            print("   Plotting LR range test results...")
            # Check if history exists before trying to plot
            if lr_finder.history and "lr" in lr_finder.history and "loss" in lr_finder.history:
                 fig, ax = plt.subplots()
                 # Plot the losses vs LRs using the full history
                 ax.plot(lr_finder.history["lr"], lr_finder.history["loss"])
                 ax.set_xlabel("Learning Rate")
                 ax.set_ylabel("Loss")
                 ax.set_xscale("log")
                 ax.set_title(f"LR Range Test ({experiment_id} - {config_hash})")
                 ax.grid(True, which='both', linestyle='--', linewidth=0.5)

                 # Add a marker for the calculated LR if found
                 if manual_suggested_lr is not None and calculated_loss_at_lr is not None:
                    ax.plot(manual_suggested_lr, calculated_loss_at_lr, 'ro', markersize=6,
                            label=f'Steepest Grad LR ({manual_suggested_lr:.2e})')
                    ax.legend()

                 # Create directory if it doesn't exist using absolute path
                 lr_plot_dir = os.path.join(project_root, "data", "loss_graphs")
                 os.makedirs(lr_plot_dir, exist_ok=True)
                 # Save the figure using a unique name including config hash
                 lr_plot_filename = f"{experiment_id}_{config_hash}_lr_finder_plot.png"
                 lr_plot_path = os.path.join(lr_plot_dir, lr_plot_filename)
                 fig.savefig(lr_plot_path)
                 print(f"   LR Finder plot saved to relative path: data/loss_graphs/{lr_plot_filename}")
            else:
                 print("   Skipping plot: History data missing or incomplete.")

        except Exception as e_plot:
             print(f"   WARNING: LR Finder plot generation failed: {e_plot}")
             # import traceback # Uncomment for deep debugging of plot issues
             # traceback.print_exc()'''
        # --- End Plotting ---

        # --- Determine Final Initial LR ---
        if manual_suggested_lr is not None:
            # Use suggested LR divided by a factor (e.g., 2 or 10 are common)
            # Dividing by 2 was used originally.
            derived_lr = manual_suggested_lr / 2
            print(f"   Using derived initial_lr: {derived_lr:.2e} (Calculated Steepest / 2)")

            # Apply safety cap
            if derived_lr > MAX_INITIAL_LR_CAP:
                print(f"   WARNING: Derived initial_lr ({derived_lr:.2e}) exceeds cap ({MAX_INITIAL_LR_CAP:.2e}).")
                initial_lr = MAX_INITIAL_LR_CAP
                print(f"   Using capped initial_lr: {initial_lr:.2e}")
            elif derived_lr <= 0: # Check for invalid LR
                 print(f"   WARNING: Derived initial_lr ({derived_lr:.2e}) is non-positive. Falling back to default.")
                 initial_lr = DEFAULT_INITIAL_LR
            else:
                initial_lr = derived_lr # Use the derived LR
        else:
            # Fallback logic (if manual suggestion failed)
            print(f"   LR suggestion calculation failed. Falling back to default initial LR {DEFAULT_INITIAL_LR:.2e}.")
            initial_lr = DEFAULT_INITIAL_LR

    # --- End of main `try` block for LR Finder ---
    except ImportError:
         print("   WARNING: 'torch-lr-finder' or 'numpy' not installed. Skipping LR Finder.")
         initial_lr = DEFAULT_INITIAL_LR
    except AttributeError as e:
         print(f"   WARNING: Attribute error during LR Finder (possibly internal issue): {e}. Skipping.")
         initial_lr = DEFAULT_INITIAL_LR
    except Exception as e_finder:
        print(f"❌ ERROR during LR Finder setup or execution: {e_finder}. Falling back to default {DEFAULT_INITIAL_LR:.2e}.")
        import traceback
        traceback.print_exc()
        initial_lr = DEFAULT_INITIAL_LR # Ensure fallback on major error

    finally:
        # --- Cleanup ---
        print("   Cleaning up LR Finder resources...")
        if lr_finder is not None:
             try:
                 # Reset the model's state (important!)
                 lr_finder.reset()
                 print("      LR Finder model state reset.")
             except Exception as e_reset:
                 print(f"      Warning: Failed to reset LR Finder state: {e_reset}")
        if temp_optimizer is not None:
             del temp_optimizer # Allow garbage collection
             print("      Temporary optimizer deleted.")
        if fig is not None: # Check if fig was assigned during plotting
            try:
                 plt.close(fig) # Close the plot figure to free memory
                 print("      Closed LR Finder plot figure.")
            except Exception as e_close:
                 print(f"      Warning: Failed to close plot figure: {e_close}")

    print(f"--- LR Finder finished. Final initial_lr: {initial_lr:.2e} ---")
    return initial_lr


def setup_optimizer_and_scheduler(model: nn.Module,
                                  initial_lr: float,
                                  training_config: Dict[str, Any]
                                 ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Creates the AdamW optimizer and ReduceLROnPlateau learning rate scheduler.

    Args:
        model (nn.Module): The model whose parameters will be optimized.
        initial_lr (float): The initial learning rate to use for the optimizer.
                            This is typically determined by the LR finder or a default.
        training_config (Dict[str, Any]): Dictionary containing scheduler parameters:
                                           - lr_scheduler_factor (float)
                                           - lr_scheduler_min_lr (float)
                                           - lr_scheduler_patience (int)
                                           - early_stopping_metric (str) - Used to align monitor
                                           - monitor_mode (str) - ('min' or 'max')
                                           - Optionally: 'weight_decay' (float, default 0.01)

    Returns:
        Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
            The configured optimizer and learning rate scheduler.
    """
    print("--- Setting up Optimizer and LR Scheduler ---")

    # --- Extract parameters from training_config ---
    # Optimizer params
    # AdamW's default weight decay is often 0.01, provide it or allow override
    weight_decay = training_config.get('weight_decay', 0.01)

    # Scheduler params
    lr_scheduler_factor = training_config['lr_scheduler_factor']
    lr_scheduler_min_lr = training_config['lr_scheduler_min_lr']
    lr_scheduler_patience = training_config['lr_scheduler_patience'] # Derived patience
    scheduler_monitor_metric = training_config['early_stopping_metric'] # Align monitor metric
    scheduler_mode = training_config['monitor_mode'] # Align mode ('min' or 'max')

    # --- Create Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    print(f"Optimizer: AdamW (Initial LR={initial_lr:.2e}, Weight Decay={weight_decay})")

    # --- Create LR Scheduler ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=lr_scheduler_min_lr,
        verbose=True # Set verbose=True to get scheduler step notifications
    )
    print(f"LR Scheduler: ReduceLROnPlateau (Monitor='{scheduler_monitor_metric}', Mode='{scheduler_mode}', "
          f"Factor={lr_scheduler_factor}, Patience={lr_scheduler_patience}, MinLR={lr_scheduler_min_lr})")

    print("--- Optimizer and Scheduler Setup Complete ---")
    return optimizer, scheduler


def load_training_state(latest_checkpoint_path: str,
                          model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          scheduler: torch.optim.lr_scheduler._LRScheduler,
                          training_config: Dict[str, Any],
                          project_root: str
                         ) -> Dict[str, Any]:
    """
    Attempts to load training state from the latest checkpoint.

    If successful, it restores model, optimizer, and scheduler states via
    the load_checkpoint helper and sets up training progress variables.
    Otherwise, it initializes state for a fresh run.

    Args:
        latest_checkpoint_path (str): Relative path to the 'latest' checkpoint file.
        model (nn.Module): The model instance to load state into.
        optimizer (torch.optim.Optimizer): The optimizer instance to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler instance to load state into.
        training_config (Dict[str, Any]): Contains 'early_stopping_metric', 'monitor_mode'.
        project_root (str): Absolute path to the project root for finding the checkpoint file.

    Returns:
        Dict[str, Any]: A dictionary containing the training state:
            - 'start_epoch': Epoch to resume from (int, 0-based index).
            - 'global_step': Global step count to resume from (int).
            - 'cumulative_duration': Training duration loaded from checkpoint (float).
            - 'best_metric_value': Best validation metric value seen so far (float).
            - 'epochs_no_improve': Counter for early stopping patience (int).
            - 'running_metrics': Metrics dictionary loaded or initialized (Dict).
            - 'checkpoint_loaded': Boolean indicating if loading was successful.
    """
    print("--- Attempting to Load Training State ---")

    # Initialize default state for a fresh run
    monitor_mode = training_config.get('monitor_mode', 'min')
    initial_best_metric = math.inf if monitor_mode == 'min' else -math.inf
    training_state = {
        'start_epoch': 0,
        'global_step': 0,
        'cumulative_duration': 0.0,
        'best_metric_value': initial_best_metric,
        'epochs_no_improve': 0,
        'running_metrics': { # Initialize structure expected by later functions
             'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'conf_matrix': [],
             'learning_rates': [], 'epochs_completed': 0, 'stopped_early': False,
             'best_epoch': -1, # Use -1 to indicate no best epoch recorded yet
             'best_val_loss': None, 'best_val_accuracy': None, 'training_duration': 0.0
        },
        'checkpoint_loaded': False
    }

    # --- Check for and Load Checkpoint ---
    # NOTE: Assumes load_checkpoint helper handles the file existence check and absolute path logic
    print(f"Checking for checkpoint at relative path: {latest_checkpoint_path}")
    # We pass project_root to load_checkpoint so it can construct the absolute path
    checkpoint_data = load_checkpoint(latest_checkpoint_path, model, optimizer, scheduler) # Removed project_root arg here, assuming load_checkpoint handles it internally now based on comments. Add it back if load_checkpoint requires it explicitly.

    if checkpoint_data:
        print(f"✅ Checkpoint loaded successfully from {latest_checkpoint_path}.")
        training_state['checkpoint_loaded'] = True

        # --- Restore Training Progress ---
        # Epoch is saved as the last *completed* epoch (0-based)
        completed_epoch = checkpoint_data.get('epoch', -1)
        training_state['start_epoch'] = completed_epoch + 1
        training_state['global_step'] = checkpoint_data.get('global_step', 0)

        # Restore metrics history and duration
        loaded_metrics = checkpoint_data.get('training_metrics', {})
        # Merge loaded metrics with default structure to ensure all keys exist
        training_state['running_metrics'].update(loaded_metrics)
        # Specifically restore duration from the correct field if it exists
        training_state['cumulative_duration'] = training_state['running_metrics'].get('training_duration', 0.0)

        print(f"   Resuming from Epoch: {training_state['start_epoch'] + 1} (Completed {completed_epoch + 1})")
        print(f"   Global Step: {training_state['global_step']}")
        print(f"   Cumulative Duration: {training_state['cumulative_duration']:.1f}s")
        if optimizer:
             print(f"   Optimizer LR loaded: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Restore Early Stopping State ---
        print("   Restoring early stopping state...")
        early_stopping_metric = training_config.get('early_stopping_metric', 'val_loss')
        metric_curve_for_restore = training_state['running_metrics'].get(early_stopping_metric, [])

        # Try restoring from explicitly saved best value first
        saved_best_value = training_state['running_metrics'].get(f'best_{early_stopping_metric}')
        best_epoch_recorded = training_state['running_metrics'].get('best_epoch', -1) # Already updated from loaded metrics

        if saved_best_value is not None and not math.isnan(saved_best_value) and not math.isinf(saved_best_value):
            training_state['best_metric_value'] = saved_best_value
            if best_epoch_recorded > 0: # best_epoch is 1-based in saved metrics
                # Calculate epochs passed since the best epoch was recorded
                # start_epoch is 0-based index of the *next* epoch to run
                # best_epoch_recorded is 1-based index of the epoch where best occurred
                epochs_since_best = (training_state['start_epoch']) - (best_epoch_recorded -1)
                training_state['epochs_no_improve'] = max(0, epochs_since_best) # Ensure non-negative
            else:
                # Should not happen if saved_best_value exists, but as fallback
                training_state['epochs_no_improve'] = 0
            print(f"   Restored from saved state: Best '{early_stopping_metric}' = {training_state['best_metric_value']:.4f} "
                  f"at epoch {best_epoch_recorded}. Epochs since improve: {training_state['epochs_no_improve']}")

        # Fallback: Recalculate from the curve if saved best value is missing/invalid or no best epoch recorded
        elif metric_curve_for_restore:
            print("   Warning: Could not restore best metric value from saved field or best_epoch missing. Recalculating from history.")
            current_best = math.inf
            best_idx = -1
            try:
                 if monitor_mode == 'min':
                     filtered_curve = [v for v in metric_curve_for_restore if not (math.isnan(v) or math.isinf(v))]
                     if filtered_curve:
                         current_best = min(filtered_curve)
                         # Find first occurrence index in original list
                         best_idx = metric_curve_for_restore.index(current_best)
                 else: # mode == 'max'
                     filtered_curve = [v for v in metric_curve_for_restore if not (math.isnan(v) or math.isinf(v))]
                     if filtered_curve:
                         current_best = max(filtered_curve)
                         # Find first occurrence index in original list
                         best_idx = metric_curve_for_restore.index(current_best)

                 if best_idx != -1:
                     training_state['best_metric_value'] = current_best
                     # epochs_no_improve is the count of epochs *after* the best one
                     training_state['epochs_no_improve'] = len(metric_curve_for_restore) - 1 - best_idx
                     # Update the best_epoch field in running_metrics if recalculating
                     training_state['running_metrics']['best_epoch'] = best_idx + 1
                     print(f"   Resuming early stopping state (recalculated): Best '{early_stopping_metric}' = {training_state['best_metric_value']:.4f} "
                           f"at epoch {best_idx + 1}. Epochs since improve: {training_state['epochs_no_improve']}")
                 else:
                      print("   Warning: Could not find valid values in metric history to recalculate best.")
                      # Keep default initial values for best_metric_value and epochs_no_improve
            except Exception as e_recalc:
                 print(f"   Error recalculating best metric: {e_recalc}. Using initial state.")
                 training_state['best_metric_value'] = initial_best_metric
                 training_state['epochs_no_improve'] = 0


        else:
             print("   No previous metric history found in checkpoint to restore early stopping state.")
             # Keep default initial values for best_metric_value and epochs_no_improve

    else:
        # This case handled by load_checkpoint returning None
        print(f"No valid checkpoint found at {latest_checkpoint_path} or loading failed. Starting fresh.")
        # training_state remains as initialized defaults

    print("--- Load Training State Attempt Complete ---")
    return training_state


def train_single_epoch(model: nn.Module,
                       loader: DataLoader,
                       criterion: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       device: torch.device,
                       epoch_num: int,
                       global_step: int,
                       max_grad_norm: float = 1.0 # Add option for grad norm clipping value
                      ) -> Tuple[float, int, bool]:
    """
    Performs a single training epoch.

    Iterates over the training data, performs forward and backward passes,
    updates model weights, and calculates the average training loss.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to train on (CPU or CUDA).
        epoch_num (int): The current epoch number (1-based, for logging).
        global_step (int): The current global step count.
        max_grad_norm (float): The maximum norm for gradient clipping.

    Returns:
        Tuple[float, int, bool]:
            - avg_epoch_loss (float): Average training loss for the epoch.
            - updated_global_step (int): The global step count after the epoch.
            - training_failed (bool): True if NaN/Inf loss was detected, False otherwise.
    """
    model.train()  # Set model to training mode
    epoch_train_loss = 0.0
    training_failed = False
    num_batches = len(loader)

    # Optional: Clear GPU memory at the start of the epoch if needed
    # clear_gpu_memory() # Uncomment if you face memory issues between epochs

    # Setup progress bar
    train_pbar = tqdm(loader, desc=f'Epoch {epoch_num:>3} Trn', leave=False, unit='batch')

    for inputs, labels in train_pbar:
        # Move data to the target device
        # Using non_blocking=True can slightly improve performance on GPU
        # by overlapping data transfer with computation, but requires pinned memory
        # in the DataLoader (which might be default or configurable).
        try:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        except Exception as e:
             print(f"\n❌ ERROR moving batch to {device} in epoch {epoch_num}: {e}")
             training_failed = True
             break # Stop processing this epoch

        # --- Forward Pass ---
        # Zero gradients before the forward pass
        # Using set_to_none=True can be slightly faster
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        # --- Check for Invalid Loss ---
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ NaN/Inf detected in training loss at epoch {epoch_num}, step {global_step}. Stopping epoch.")
            training_failed = True
            break # Exit inner loop for this epoch

        # --- Backward Pass & Optimization ---
        loss.backward()

        # Gradient Clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        # --- Accumulate Loss & Update Step ---
        # .item() gets the Python number from the tensor, detaching it from the graph
        epoch_train_loss += loss.item()
        global_step += 1

        # Update progress bar postfix (optional, but helpful)
        # Use refresh=False to update less frequently if needed
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"}, refresh=True)

    # --- End of Epoch Loop ---
    train_pbar.close() # Ensure the progress bar is closed

    if training_failed:
        # Return NaN or some indicator of failure if preferred,
        # but the flag is usually sufficient. Return 0 loss to avoid downstream NaN issues.
        avg_epoch_loss = 0.0
    elif num_batches > 0:
         avg_epoch_loss = epoch_train_loss / num_batches
    else:
         avg_epoch_loss = 0.0 # Avoid division by zero if loader is empty

    # Optionally log end of epoch training
    # print(f"Epoch {epoch_num} Training Phase Complete. Avg Loss: {avg_epoch_loss:.4f}")

    return avg_epoch_loss, global_step, training_failed


def validate_single_epoch(model: nn.Module,
                            loader: DataLoader,
                            criterion: nn.Module,
                            device: torch.device,
                            epoch_num: int # For logging
                           ) -> Tuple[float, float, np.ndarray, bool]:
    """
    Performs a single validation epoch.

    Iterates over the validation data, calculates loss, accuracy, and confusion matrix.
    Operates in no_grad mode.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run evaluation on.
        epoch_num (int): The current epoch number (1-based, for logging).

    Returns:
        Tuple[float, float, np.ndarray, bool]:
            - val_loss (float): Average validation loss.
            - val_accuracy (float): Overall validation accuracy.
            - conf_matrix (np.ndarray): Confusion matrix for the epoch.
            - validation_failed (bool): True if NaN/Inf loss was detected, False otherwise.
    """
    model.eval()  # Set model to evaluation mode
    epoch_val_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds_list = []
    all_labels_list = []
    validation_failed = False
    num_batches = len(loader)

    # Setup progress bar
    val_pbar = tqdm(loader, desc=f'Epoch {epoch_num:>3} Val', leave=False, unit='batch')

    with torch.no_grad(): # Disable gradient calculations
        for inputs, labels in val_pbar:
            # Move data to the target device
            try:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            except Exception as e:
                 print(f"\n❌ ERROR moving batch to {device} during validation in epoch {epoch_num}: {e}")
                 validation_failed = True
                 break # Stop processing this epoch

            # --- Forward Pass ---
            logits = model(inputs)
            loss = criterion(logits, labels)

            # --- Check for Invalid Loss ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ NaN/Inf detected in validation loss at epoch {epoch_num}. Stopping epoch.")
                validation_failed = True
                break # Exit inner loop for this epoch

            # --- Accumulate Loss ---
            epoch_val_loss += loss.item()

            # --- Calculate Predictions & Accuracy ---
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probabilities, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0) # Keep track of total samples evaluated

            # Store predictions and labels for confusion matrix (move to CPU)
            all_preds_list.append(predicted.cpu())
            all_labels_list.append(labels.cpu())

            # Update progress bar postfix (optional)
            # val_pbar.set_postfix({'loss': f"{loss.item():.4f}"}, refresh=True) # Can be verbose

    # --- End of Epoch Loop ---
    val_pbar.close()

    # Handle failure case
    if validation_failed:
        # Return dummy values to avoid downstream errors
        val_loss = 0.0
        val_accuracy = 0.0
        # Create an empty or zero matrix matching expected dimensions if possible,
        # otherwise None or handle appropriately upstream. Returning None might be safer.
        conf_matrix = np.array([[]]) # Or potentially None
        return val_loss, val_accuracy, conf_matrix, validation_failed

    # --- Calculate Final Metrics ---
    if num_batches > 0:
        val_loss = epoch_val_loss / num_batches
    else:
        val_loss = 0.0 # Avoid division by zero

    if total_samples > 0:
        val_accuracy = correct_predictions / total_samples
    else:
        val_accuracy = 0.0 # Avoid division by zero

    # Concatenate results from all batches for confusion matrix
    if all_preds_list and all_labels_list:
        try:
            all_predictions_np = torch.cat(all_preds_list).numpy()
            all_true_labels_np = torch.cat(all_labels_list).numpy()
            # Calculate confusion matrix
            # Ensure labels cover the range expected by confusion_matrix if num_classes is known
            # num_classes = model.module.num_classes if isinstance(model, nn.DataParallel) else model.num_classes
            # conf_matrix = confusion_matrix(all_true_labels_np, all_predictions_np, labels=np.arange(num_classes))
            # Simpler version if labels are guaranteed to cover all classes present in the batch:
            conf_matrix = confusion_matrix(all_true_labels_np, all_predictions_np)
        except Exception as e_cm:
            print(f"❌ ERROR calculating confusion matrix in epoch {epoch_num}: {e_cm}")
            conf_matrix = np.array([[]]) # Return empty on error
            # Potentially set validation_failed = True here as well?
    else:
        print(f"Warning: No predictions/labels collected in epoch {epoch_num} for confusion matrix.")
        conf_matrix = np.array([[]]) # Return empty if no data collected

    # Optionally log end of epoch validation
    # print(f"Epoch {epoch_num} Validation Phase Complete. Avg Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    return val_loss, val_accuracy, conf_matrix, validation_failed


def update_training_state_after_epoch(training_state: Dict[str, Any],
                                      training_config: Dict[str, Any],
                                      epoch_metrics: Dict[str, Any],
                                      scheduler: torch.optim.lr_scheduler._LRScheduler,
                                      epoch_num: int # 1-based epoch number
                                     ) -> Tuple[Dict[str, Any], bool, bool]:
    """
    Updates the training state after an epoch completes.

    Appends metrics, steps scheduler, checks early stopping criteria,
    updates best metric, and determines if training should stop.

    Args:
        training_state (Dict[str, Any]): The current state including:
            - 'running_metrics' (Dict): Holds lists of metrics (train_loss, val_loss, etc.)
            - 'best_metric_value' (float): The best validation metric achieved so far.
            - 'epochs_no_improve' (int): Counter for early stopping.
        training_config (Dict[str, Any]): Configuration containing:
            - 'early_stopping_metric' (str): Metric to monitor (e.g., 'val_loss').
            - 'monitor_mode' (str): 'min' or 'max'.
            - 'early_stopping_patience' (int): Patience for early stopping.
            - 'early_stopping_min_delta' (float): Minimum change to qualify as improvement for 'min' mode.
        epoch_metrics (Dict[str, Any]): Metrics collected during the current epoch:
            - 'train_loss' (float)
            - 'val_loss' (float)
            - 'val_accuracy' (float)
            - 'conf_matrix' (np.ndarray or None)
            - 'current_lr' (float): LR used in this epoch.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The LR scheduler instance.
        epoch_num (int): The current epoch number (1-based).

    Returns:
        Tuple[Dict[str, Any], bool, bool]:
            - updated_training_state (Dict): The state dictionary with updated metrics and ES counters.
            - should_stop (bool): True if early stopping is triggered.
            - is_best (bool): True if this epoch achieved the best performance so far.
    """
    # --- Retrieve necessary values ---
    running_metrics = training_state['running_metrics']
    best_metric_value = training_state['best_metric_value']
    epochs_no_improve = training_state['epochs_no_improve']

    es_metric_name = training_config['early_stopping_metric']
    monitor_mode = training_config['monitor_mode']
    es_patience = training_config['early_stopping_patience']
    es_min_delta = training_config['early_stopping_min_delta']

    # --- Append current epoch's metrics ---
    running_metrics['train_loss'].append(epoch_metrics['train_loss'])
    running_metrics['val_loss'].append(epoch_metrics['val_loss'])
    running_metrics['val_accuracy'].append(epoch_metrics['val_accuracy'])
    running_metrics['learning_rates'].append(epoch_metrics['current_lr'])
    # Append confusion matrix (handle potential None or empty array)
    cm = epoch_metrics.get('conf_matrix')
    if cm is not None and isinstance(cm, np.ndarray) and cm.size > 0:
        running_metrics['conf_matrix'].append(cm)
    else:
        # Append a placeholder if CM is invalid/missing for this epoch
        # This ensures subsequent processing (like saving) doesn't fail on length mismatch
        # Determine shape from previous valid entry if possible, else use None
        # For simplicity now, append None, but saving logic needs to handle this.
        running_metrics['conf_matrix'].append(None)

    running_metrics['epochs_completed'] = epoch_num

    # --- Step the LR Scheduler ---
    # Use the metric determined earlier for both early stopping and scheduler
    current_metric_value = epoch_metrics.get(es_metric_name)

    scheduler_stepped = False
    if current_metric_value is not None and not math.isnan(current_metric_value) and not math.isinf(current_metric_value):
        scheduler.step(metrics=current_metric_value)
        scheduler_stepped = True
        # Note: ReduceLROnPlateau logs its own steps if verbose=True
    else:
        print(f"   WARNING: Invalid metric value ({current_metric_value}) for '{es_metric_name}' "
              f"at epoch {epoch_num}. Skipping scheduler step and improvement check.")

    # --- Check for Improvement (Early Stopping Logic) ---
    is_best = False
    improved = False
    # Only check improvement if the metric value is valid
    if scheduler_stepped: # Re-use the check that metric was valid
        if monitor_mode == 'min':
            # Improvement if current is less than best by at least min_delta
            if current_metric_value < best_metric_value - es_min_delta:
                improved = True
        else: # monitor_mode == 'max'
            # Improvement if current is strictly greater than best
            # (Typically min_delta isn't used for 'max' mode in basic ES)
            if current_metric_value > best_metric_value:
                improved = True

    if improved:
        delta = abs(best_metric_value - current_metric_value)
        print(f"   ✅ Improvement detected! {es_metric_name}: {best_metric_value:.4f} -> {current_metric_value:.4f} (Δ {delta:.4f})")
        best_metric_value = current_metric_value
        epochs_no_improve = 0 # Reset counter
        is_best = True
        # Update best metrics in the running dict
        running_metrics['best_epoch'] = epoch_num # Record 1-based epoch number
        running_metrics['best_val_loss'] = epoch_metrics['val_loss']
        running_metrics['best_val_accuracy'] = epoch_metrics['val_accuracy']
        # Store the specific best value for the monitored metric (consistent naming helps)
        running_metrics[f'best_{es_metric_name}'] = best_metric_value
    elif scheduler_stepped: # Only increment if metric was valid but no improvement
        epochs_no_improve += 1
        print(f"   No improvement in '{es_metric_name}' for {epochs_no_improve} epochs (Patience: {es_patience}).")

    # --- Check if Early Stopping Patience Exceeded ---
    should_stop = False
    if epochs_no_improve >= es_patience:
        print(f"\n🛑 Early stopping triggered after epoch {epoch_num}. No improvement for {epochs_no_improve} epochs.")
        should_stop = True
        running_metrics['stopped_early'] = True # Mark in metrics

    # --- Update state dictionary ---
    training_state['best_metric_value'] = best_metric_value
    training_state['epochs_no_improve'] = epochs_no_improve
    # running_metrics were updated in place

    return training_state, should_stop, is_best


def manage_checkpoints(is_best: bool,
                       save_latest: bool, # Determined by caller logic
                       checkpoint_paths: Dict[str, str], # Contains 'latest' and 'best' relative paths
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,
                       epoch: int, # 0-based completed epoch index
                       global_step: int,
                       running_metrics: Dict, # Passed to save_checkpoint, contains duration
                       token_dict: Dict,
                       label_encoder: Any, # Stores classes needed by save_checkpoint
                       hyperparams: Dict, # Contains experiment_id, needed by save_checkpoint
                       project_root: str # Needed if helpers don't handle abs path internally
                       ):
    """
    Saves 'latest' and/or 'best' checkpoints based on flags and triggers cleanup.
    Assumes running_metrics['training_duration'] is updated before calling.

    Args:
        is_best (bool): Whether the current epoch achieved the best performance.
        save_latest (bool): Whether to save the 'latest' checkpoint this epoch (due to frequency or interruption).
        checkpoint_paths (Dict[str, str]): Relative paths for 'latest' and 'best' checkpoints.
        model, optimizer, scheduler: Objects to save state from.
        epoch (int): The completed epoch number (0-based).
        global_step (int): Current global step count.
        running_metrics (Dict): Dictionary containing metric history (including duration).
        token_dict (Dict): Vocabulary mapping.
        label_encoder (Any): Fitted LabelEncoder instance.
        hyperparams (Dict): Hyperparameters dictionary (must include 'experiment_id').
        project_root (str): Absolute path to project root.
                             (May not be needed if save_checkpoint handles path logic internally).
    """
    # Save the 'best' checkpoint if this epoch was the best
    if is_best:
        print(f"   Saving best checkpoint for epoch {epoch + 1}...")
        save_checkpoint(
            checkpoint_path=checkpoint_paths['best'], # Pass relative path
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            training_metrics=running_metrics,
            token_dict=token_dict,
            label_encoder=label_encoder,
            hyperparams=hyperparams,
            is_best=True
        )

    # Save the 'latest' checkpoint if requested (either by frequency or interruption)
    if save_latest:
        # Add clarity if saving 'latest' because it was also the 'best' vs just frequency/interrupt
        reason = "(Best)" if is_best else "(Frequency/Interrupt)"
        print(f"   Saving latest checkpoint for epoch {epoch + 1} {reason}...")
        save_checkpoint(
            checkpoint_path=checkpoint_paths['latest'], # Pass relative path
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            training_metrics=running_metrics,
            token_dict=token_dict,
            label_encoder=label_encoder,
            hyperparams=hyperparams,
            is_best=False # Mark as 'latest' type checkpoint file
        )


def load_best_model_weights(model: nn.Module,
                              best_checkpoint_path: str, # Relative path
                              device: torch.device,
                              project_root: str
                             ) -> bool:
    """
    Loads the model state dictionary from the 'best' checkpoint file.

    Args:
        model (nn.Module): The model instance to load weights into.
        best_checkpoint_path (str): Relative path to the 'best' checkpoint.
        device (torch.device): The device to map the loaded weights to.
        project_root (str): Absolute path to project root for constructing the full path.

    Returns:
        bool: True if loading was successful, False otherwise.
    """
    print("--- Loading Best Model Weights ---")
    abs_best_checkpoint_path = os.path.join(project_root, best_checkpoint_path)

    if not os.path.exists(abs_best_checkpoint_path):
        print(f"⚠️ WARNING: Best checkpoint not found at {abs_best_checkpoint_path}. Cannot load best weights.")
        print("   (Returning model in its final training state).")
        return False

    try:
        # Load the entire checkpoint dictionary first
        # Use weights_only=False as we need the full state_dict structure
        print(f"Loading best model state from: {abs_best_checkpoint_path}")
        checkpoint_data = torch.load(abs_best_checkpoint_path, map_location=device, weights_only=False)

        # Extract the model state dictionary
        if 'model_state_dict' not in checkpoint_data:
             print(f"❌ ERROR: 'model_state_dict' key not found in checkpoint: {abs_best_checkpoint_path}")
             return False

        state_dict_to_load = checkpoint_data['model_state_dict']

        # Load state dict (handle DataParallel wrapper)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict_to_load)
        else:
            model.load_state_dict(state_dict_to_load)

        print("✅ Best model weights loaded successfully.")
        return True

    except FileNotFoundError:
        # This case is technically handled by the os.path.exists check, but good practice
        print(f"❌ ERROR: File not found error during load: {abs_best_checkpoint_path}")
        return False
    except Exception as e:
        print(f"❌ ERROR loading best model weights from {abs_best_checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_confusion_matrix_history(cm_history_list: List[Optional[np.ndarray]],
                                  experiment_id: str,
                                  project_root: str
                                 ) -> Optional[str]:
    """
    Saves the history of confusion matrices collected during training to a .npy file.

    Handles potential None entries or inconsistent shapes.

    Args:
        cm_history_list (List[Optional[np.ndarray]]): List of confusion matrices per epoch.
                                                       Can contain None if validation/CM failed for an epoch.
        experiment_id (str): The experiment ID for naming the output file.
        project_root (str): Absolute path to the project root directory for saving.

    Returns:
        Optional[str]: The relative path to the saved .npy file (e.g., "data/cm_history/...")
                       if successful, otherwise None.
    """
    print("--- Saving Confusion Matrix History ---")

    if not cm_history_list:
        print("   No confusion matrix history found to save.")
        return None

    # --- Validate and Prepare Data ---
    num_epochs = len(cm_history_list)
    first_valid_cm = None
    first_valid_cm_idx = -1

    # Find the first valid CM to determine expected shape
    for i, cm in enumerate(cm_history_list):
        if cm is not None and isinstance(cm, np.ndarray) and cm.ndim == 2 and cm.shape[0] == cm.shape[1] and cm.shape[0] > 0:
            first_valid_cm = cm
            first_valid_cm_idx = i
            break

    if first_valid_cm is None:
        print(f"   ERROR: No valid 2D confusion matrix found in history for {experiment_id} across {num_epochs} epochs. Cannot save.")
        return None

    num_classes_cm = first_valid_cm.shape[0]
    expected_shape = (num_classes_cm, num_classes_cm)
    print(f"   Detected expected CM shape: {expected_shape} from epoch {first_valid_cm_idx + 1}.")

    # Create the final array, initialized with zeros (or potentially NaNs if preferred)
    final_shape = (num_epochs, num_classes_cm, num_classes_cm)
    cm_history_final_array = np.zeros(final_shape, dtype=np.int64) # Use int64 for counts
    valid_save = True

    # Fill the final array, validating each entry
    for i, cm in enumerate(cm_history_list):
        try:
            # If cm is None or invalid, keep the zeros (or fill with NaN if dtype=float)
            if cm is None or not isinstance(cm, np.ndarray) or cm.shape != expected_shape:
                 if cm is not None: # Log specific error if it's not None but still invalid
                      print(f"   WARNING: CM at epoch {i+1} has inconsistent shape {getattr(cm, 'shape', 'N/A')} "
                            f"(expected {expected_shape}). Filling with zeros.")
                 # Else: it was None, zeros are already there.
                 continue # Keep the default zero matrix for this epoch

            # Convert valid CM to the target dtype and assign
            cm_array = np.array(cm, dtype=np.int64)
            cm_history_final_array[i] = cm_array

        except (TypeError, ValueError) as conv_err:
            print(f"   ERROR: Could not convert CM at epoch {i+1} to numeric array: {conv_err}. Aborting save.")
            valid_save = False
            break

    if not valid_save:
        return None # Abort saving if any error occurred during processing

    # --- Save the NumPy Array ---
    try:
        # Define relative and absolute paths
        relative_cm_history_dir = os.path.join("data", "cm_history")
        abs_cm_history_dir = os.path.join(project_root, relative_cm_history_dir)
        os.makedirs(abs_cm_history_dir, exist_ok=True)

        cm_history_filename = f"{experiment_id}_cm_history.npy"
        abs_cm_history_path = os.path.join(abs_cm_history_dir, cm_history_filename)

        # Save the array
        np.save(abs_cm_history_path, cm_history_final_array)

        # Return the relative path for logging/metadata storage
        relative_cm_history_path = os.path.join(relative_cm_history_dir, cm_history_filename)
        print(f"   ✅ Saved CM history to: {relative_cm_history_path} "
              f"(Shape: {cm_history_final_array.shape}, Dtype: {cm_history_final_array.dtype})")
        return relative_cm_history_path

    except Exception as e:
        print(f"❌ ERROR during saving of confusion matrix history numpy array: {e}")
        import traceback
        traceback.print_exc()
        return None


def compile_final_results(running_metrics: Dict[str, Any],
                            hyperparams: Dict[str, Any],
                            training_config: Dict[str, Any],
                            data_details: Dict[str, Any],
                            initial_lr: float, # LR used initially (after finder/default)
                            total_duration: float
                           ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compiles the final metrics summary and model metadata dictionaries.

    Args:
        running_metrics (Dict[str, Any]): Final state of collected metrics history.
        hyperparams (Dict[str, Any]): Original hyperparameters provided to the run.
        training_config (Dict[str, Any]): Validated and derived training parameters.
        data_details (Dict[str, Any]): Details about the data (vocab_size, num_classes, etc.).
        initial_lr (float): The initial learning rate used (post-finder or default).
        total_duration (float): Total training duration in seconds.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - final_metrics_summary: Dictionary summarizing performance curves and outcomes.
            - model_metadata: Dictionary with configuration and details for inference/archiving.
    """
    print("--- Compiling Final Results ---")

    # --- Final Metrics Summary ---
    final_metrics_summary = {
        # Curves
        'train_loss_curve': running_metrics.get('train_loss', []),
        'val_loss_curve': running_metrics.get('val_loss', []),
        'val_accuracy_curve': running_metrics.get('val_accuracy', []),
        'learning_rates': running_metrics.get('learning_rates', []),
        # Outcomes
        'epochs_completed': running_metrics.get('epochs_completed', 0),
        'training_duration': total_duration,
        'stopped_early': running_metrics.get('stopped_early', False),
        'best_epoch': running_metrics.get('best_epoch', -1), # 1-based epoch num
        'best_val_loss': running_metrics.get('best_val_loss'),
        'best_val_accuracy': running_metrics.get('best_val_accuracy'),
    }
    print("   Compiled final metrics summary.")

    # --- Model Metadata ---
    num_classes = data_details['num_classes']
    vocab_size = data_details['vocab_size']
    experiment_id = training_config['experiment_id']

    # Regenerate config hash based on actual params used
    config_hash = generate_config_hash(hyperparams, num_classes, vocab_size)

    model_metadata = {
        # Identifiers
        'experiment_id': experiment_id,
        'config_hash': config_hash,
        # Data/Preprocessing Details
        'token_dict': data_details['token_dict'],
        'label_encoder_classes': data_details['label_encoder_classes'],
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        # Original Hyperparameters (includes base_patience, etc.)
        'hyperparams': hyperparams,
        # Key Architecture Params (from training_config for convenience)
        'd_model': training_config['d_model'],
        'nhead': training_config['nhead'],
        'num_encoder_layers': training_config['num_encoder_layers'],
        'dim_feedforward': training_config['dim_feedforward'],
        'dropout_rate': training_config['dropout_rate'],
         # Training Setup Details (from training_config)
        'initial_learning_rate': initial_lr,
        'lr_scheduler_type': 'ReduceLROnPlateau', # Hardcoded for now
        'lr_scheduler_factor': training_config['lr_scheduler_factor'],
        'lr_scheduler_patience': training_config['lr_scheduler_patience'], # Derived value
        'lr_scheduler_min_lr': training_config['lr_scheduler_min_lr'],
        'lr_scheduler_monitor': training_config['early_stopping_metric'],
        'early_stopping_patience': training_config['early_stopping_patience'], # Derived value
        'early_stopping_metric': training_config['early_stopping_metric'],
        'early_stopping_min_delta': training_config['early_stopping_min_delta'],
        # Performance Summary (from running_metrics)
        'best_val_accuracy_achieved': running_metrics.get('best_val_accuracy'),
        'best_val_loss_achieved': running_metrics.get('best_val_loss'),
        'epoch_of_best': running_metrics.get('best_epoch', -1) # 1-based epoch num
    }
    print("   Compiled model metadata.")
    print("--- Final Results Compilation Complete ---")

    return final_metrics_summary, model_metadata


def train_model(data: pd.DataFrame,
                hyperparams: Dict[str, Any],
                check_continue_func: callable # Function to check for interruption signal
               ) -> Tuple[Optional[nn.Module], Optional[Dict], Optional[Dict], Optional[str]]:
    """
    Trains the transformer model using a modular structure.

    Handles parameter setup, data prep, model init, LR finding, checkpointing,
    training loop, early stopping, metric collection, results compilation,
    and cooperative interruption handling.

    Args:
        data (pd.DataFrame): DataFrame with 'text' and 'cipher' columns.
        hyperparams (Dict[str, Any]): Dictionary of hyperparameters for training.
                                       Must include 'experiment_id'.
        check_continue_func (callable): A function that returns False if training should stop
                                        due to an external signal (e.g., Ctrl+C).

    Returns:
        Tuple containing:
        - model (Optional[nn.Module]): Trained model loaded to best state, or None if failed/interrupted.
        - final_metrics (Optional[Dict]): Summarized metrics dictionary, or None if failed/interrupted.
        - model_metadata (Optional[Dict]): Metadata dictionary for inference, or None if failed/interrupted.
        - cm_history_path (Optional[str]): Relative path to saved confusion matrix history, or None.
    """
    # --- 1. Setup Experiment ---
    training_config = setup_experiment_parameters(hyperparams)
    if training_config is None:
        return None, None, None, None # Critical setup failed
    experiment_id = training_config['experiment_id']
    project_root = _PROJECT_ROOT_FROM_TRAIN # Use global constant

    # --- 2. Prepare Data ---
    train_loader, val_loader, vocab_size, num_classes, token_dict, label_encoder = \
        prepare_data_for_training(data, training_config['batch_size'])
    if train_loader is None:
        print(f"❌ Data preparation failed for experiment {experiment_id}. Aborting.")
        return None, None, None, None
    data_details = {
        'vocab_size': vocab_size, 'num_classes': num_classes,
        'token_dict': token_dict, 'label_encoder': label_encoder, # Keep encoder itself
        'label_encoder_classes': label_encoder.classes_.tolist()
    }

    # --- 3. Initialize Model & Device ---
    model, device = initialize_model_and_device(vocab_size, num_classes, training_config)
    if model is None or device is None:
        print(f"❌ Model initialization failed for experiment {experiment_id}. Aborting.")
        return None, None, None, None

    # --- 4. Checkpoints & LR Finder ---
    config_hash = generate_config_hash(hyperparams, num_classes, vocab_size)
    checkpoint_paths = {
        'latest': get_checkpoint_path(experiment_id, hyperparams, num_classes, vocab_size, 'latest'),
        'best': get_checkpoint_path(experiment_id, hyperparams, num_classes, vocab_size, 'best')
    }
    abs_latest_checkpoint_path = os.path.join(project_root, checkpoint_paths['latest'])

    effective_initial_lr = DEFAULT_INITIAL_LR # Start with default
    run_lr_finder = not os.path.exists(abs_latest_checkpoint_path)

    if run_lr_finder:
        print("No checkpoint found, running LR Finder...")
        # Create a temporary criterion JUST for the LR finder
        criterion_for_lr = nn.CrossEntropyLoss()
        effective_initial_lr = find_initial_learning_rate(
            model=model, # Pass the potentially wrapped model
            criterion=criterion_for_lr,
            train_loader=train_loader,
            device=device,
            experiment_id=experiment_id,
            project_root=project_root,
            config_hash=config_hash
        )
        del criterion_for_lr # Clean up temporary criterion
        print(f"LR Finder determined initial LR: {effective_initial_lr:.2e}")
    else:
        print(f"Checkpoint found at {checkpoint_paths['latest']}, skipping LR Finder.")
        # LR will be loaded from checkpoint by load_training_state if resuming.

    # --- 5. Setup Optimizer & Scheduler ---
    # This defines the structure; state might be overwritten by checkpoint load
    optimizer, scheduler = setup_optimizer_and_scheduler(model, effective_initial_lr, training_config)

    # --- 6. Load Training State (if checkpoint exists) ---
    # This modifies model, optimizer, scheduler in-place if load successful
    training_state = load_training_state(
        latest_checkpoint_path=checkpoint_paths['latest'],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        project_root=project_root
    )
    # Unpack state needed for the loop and post-processing
    start_epoch = training_state['start_epoch'] # 0-based index of next epoch
    global_step = training_state['global_step']
    # Note: cumulative_duration, best_metric_value, epochs_no_improve, running_metrics
    # are accessed directly via training_state dictionary below for simplicity

    # --- 7. Training Loop ---
    print(f"\nStarting training loop from epoch {start_epoch + 1}...")
    criterion = nn.CrossEntropyLoss() # Main loss function
    session_start_time = time.time()
    training_failed = False # Flag for catastrophic failure (NaN/Inf)
    training_interrupted = False # <<< New flag for signal interruption

    for epoch in itertools.count(start_epoch):
        epoch_1_based = epoch + 1 # Use 1-based for logging & checks

        # --- V V V Check for interruption signal FIRST V V V ---
        if not check_continue_func():
            print(f"\n🛑 Interruption signal detected before epoch {epoch_1_based}. Saving latest state and stopping.")
            training_interrupted = True
            # --- Save Checkpoint NOW (state from end of previous epoch) ---
            current_session_duration = time.time() - session_start_time
            training_state['running_metrics']['training_duration'] = training_state['cumulative_duration'] + current_session_duration
            # Save state reflecting the last *completed* epoch (index 'epoch-1')
            save_epoch_idx = max(0, epoch - 1) # Index of last completed epoch
            print(f"   Saving latest state (Epoch {save_epoch_idx + 1}) due to interruption.")
            manage_checkpoints(
                is_best=False, # Not necessarily best
                save_latest=True, # Force save latest
                checkpoint_paths=checkpoint_paths,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=save_epoch_idx, # Save state as of this completed epoch
                global_step=global_step, # Use current global_step
                running_metrics=training_state['running_metrics'],
                token_dict=data_details['token_dict'],
                label_encoder=data_details['label_encoder'],
                hyperparams=hyperparams,
                project_root=project_root
            )
            # --- Break AFTER saving ---
            break # Exit outer loop
        # --- ^ ^ ^ End interruption check ^ ^ ^ ---

        # --- Safety Break ---
        if epoch >= MAX_SAFETY_EPOCHS:
            print(f"WARNING: Reached safety limit of {MAX_SAFETY_EPOCHS} epochs. Stopping.")
            training_state['running_metrics']['stopped_early'] = True # Mark as stopped (though not 'early' in the typical sense)
            break

        # --- Optional: Clear GPU Memory ---
        # clear_gpu_memory() # Uncomment if needed

        # --- Train ---
        train_loss, global_step, train_fail = train_single_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch_1_based,
            global_step=global_step
            # max_grad_norm=training_config.get('max_grad_norm', 1.0) # Can pass from config
        )
        if train_fail:
            print(f"❌ Training failed in epoch {epoch_1_based}. Stopping run.")
            training_failed = True
            break # Exit the main training loop

        # --- Validate ---
        val_loss, val_accuracy, conf_matrix, val_fail = validate_single_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch_num=epoch_1_based
        )
        if val_fail:
            print(f"❌ Validation failed in epoch {epoch_1_based}. Stopping run.")
            training_failed = True
            break # Exit main loop

        # --- Update State & Check Stopping ---
        current_lr = optimizer.param_groups[0]['lr'] # Get LR before potential scheduler step
        epoch_metrics_dict = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'conf_matrix': conf_matrix,
            'current_lr': current_lr
        }

        # Pass the whole training_state dict; it holds running_metrics, best_metric_value etc.
        # It will be updated internally.
        training_state, should_stop, is_best = update_training_state_after_epoch(
            training_state=training_state, # Pass the dict
            training_config=training_config,
            epoch_metrics=epoch_metrics_dict,
            scheduler=scheduler,
            epoch_num=epoch_1_based
        )

        # Log epoch summary
        print(f"Epoch {epoch_1_based:>3} Summary | Trn Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.2e}")

        # --- Manage Checkpoints (Regular end-of-epoch) ---
        save_latest_flag = (epoch_1_based % CHECKPOINT_FREQ == 0)
        # Update total duration within running_metrics before potentially saving
        current_session_duration = time.time() - session_start_time
        training_state['running_metrics']['training_duration'] = training_state['cumulative_duration'] + current_session_duration

        # Only save if frequency matches OR if it's the best epoch
        if save_latest_flag or is_best:
            manage_checkpoints(
                is_best=is_best,
                save_latest=save_latest_flag, # Pass the frequency flag
                checkpoint_paths=checkpoint_paths,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch, # Pass current completed epoch (0-based)
                global_step=global_step, # Pass updated global_step
                running_metrics=training_state['running_metrics'], # Pass updated metrics
                token_dict=data_details['token_dict'],
                label_encoder=data_details['label_encoder'], # Pass encoder object
                hyperparams=hyperparams,
                project_root=project_root
            )

        # --- Check for Early Stop ---
        if should_stop:
             # message already printed by update_training_state_after_epoch
             break # Exit training loop

    # --- 8. Post-Training ---
    final_total_duration = training_state['running_metrics'].get('training_duration', 0.0)
    final_epochs_completed = training_state['running_metrics'].get('epochs_completed', 0)

    # Determine final status message
    # --- V V V UPDATED STATUS MESSAGE V V V ---
    status_message = "failed (NaN/Inf)" if training_failed else \
                     "interrupted by signal" if training_interrupted else \
                     "stopped early" if training_state['running_metrics'].get('stopped_early', False) else \
                     "reached safety limit" if epoch >= MAX_SAFETY_EPOCHS else \
                     "finished successfully"
    print(f"\n--- Training {status_message} for experiment {experiment_id} ---")
    print(f"Total epochs completed in this run: {final_epochs_completed}. Total duration: {final_total_duration:.1f}s")

    # --- V V V Handle Failure or Interruption V V V ---
    if training_failed or training_interrupted:
         print(f"❌ Training ended prematurely for {experiment_id}. No results returned.")
         # Don't clean up all checkpoints, the 'latest' one saved might be useful for resuming
         print("   Skipping final checkpoint cleanup to preserve latest state.")
         return None, None, None, None # Return None tuple

    # --- V V V Proceed only if training completed normally (finished or stopped early) V V V ---

    # Load Best Model Weights
    _ = load_best_model_weights(
        model=model,
        best_checkpoint_path=checkpoint_paths['best'],
        device=device,
        project_root=project_root
    ) # We modify model in-place

    # Save CM History
    cm_history_path = save_confusion_matrix_history(
        cm_history_list=training_state['running_metrics'].get('conf_matrix', []),
        experiment_id=experiment_id,
        project_root=project_root
    )

    # Compile Final Results
    final_metrics, model_metadata = compile_final_results(
        running_metrics=training_state['running_metrics'],
        hyperparams=hyperparams,
        training_config=training_config,
        data_details=data_details,
        initial_lr=effective_initial_lr, # Pass the LR used for *starting* this run
        total_duration=final_total_duration
    )

    # --- Final Cleanup ---
    # Clean checkpoints only after successful completion or successful early stop
    print("\nCleaning up final checkpoint files (keeping only final model)...")
    clean_old_checkpoints(experiment_id, completed=True) # Remove all intermediate checkpoints
    print("Checkpoint cleanup complete.")

    print(f"--- Training function returning results for {experiment_id} ---")
    # Return the model (with best weights loaded), metrics, metadata, and cm path
    return model, final_metrics, model_metadata, cm_history_path
