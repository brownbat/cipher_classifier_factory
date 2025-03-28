"""
Training functionality for transformer model.
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time
import os
import json
import signal
import sys
import hashlib

from models.transformer.model import TransformerClassifier
from models.common.data import load_and_preprocess_data, create_data_loaders
from models.common.utils import clear_gpu_memory

# Constants
EPSILON = 1e-7  # Small constant to prevent log(0)
CHECKPOINT_DIR = "data/checkpoints"
CHECKPOINT_FREQ = 1  # Save checkpoint every N epochs

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def clean_old_checkpoints(experiment_id, keep_n=1, completed=False):
    """
    Clean checkpoints for an experiment.
    For incomplete experiments: keep only the latest checkpoint.
    For completed experiments: remove all checkpoints.
    """
    # Get base experiment ID (without timestamp)
    base_id = experiment_id.split('_20')[0]
    
    # Get all checkpoints for this experiment (including those with hash codes)
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) 
                  if (f.startswith(experiment_id) or f.startswith(base_id)) and 
                  f.endswith('.pt')]
    
    # If experiment is completed, remove all checkpoints
    if completed:
        removed_count = 0
        for chk in checkpoints:
            try:
                chk_path = os.path.join(CHECKPOINT_DIR, chk)
                os.remove(chk_path)
                removed_count += 1
            except Exception as e:
                print(f"Error removing checkpoint {chk}: {e}")
        
        if removed_count > 0:
            print(f"Removed all {removed_count} checkpoints for completed experiment {experiment_id}")
        return
    
    # For incomplete experiments, keep only the latest checkpoint
    if len(checkpoints) > keep_n:
        # Sort by modification time (newest first)
        sorted_checkpoints = sorted(checkpoints, 
                                   key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)),
                                   reverse=True)
        
        # Keep only the newest checkpoint file
        keep_checkpoints = sorted_checkpoints[:keep_n]
        
        # Remove older checkpoints
        removed_count = 0
        for chk in sorted_checkpoints[keep_n:]:
            try:
                chk_path = os.path.join(CHECKPOINT_DIR, chk)
                os.remove(chk_path)
                removed_count += 1
            except Exception as e:
                print(f"Error removing checkpoint {chk}: {e}")
        
        if removed_count > 0:
            print(f"Removed {removed_count} old checkpoints for experiment {experiment_id}")


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, global_step, training_metrics, token_dict, label_encoder, hyperparams):
    """Save training checkpoint to resume later"""
    # Get model state dict (handle DataParallel)
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_metrics': training_metrics,
        'token_dict': token_dict,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'hyperparams': hyperparams
    }
    
    # Save checkpoint using pickle protocol 4 for better compatibility
    torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True, pickle_protocol=4)
    
    # Remove any other checkpoint files for this experiment
    clean_old_checkpoints(hyperparams.get('experiment_id', 'unknown_experiment'), keep_n=1, completed=False)
    
    # Don't print every checkpoint save to reduce log noise
    print(f"Checkpoint updated for experiment {hyperparams.get('experiment_id', 'unknown_experiment')}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint to resume training"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    try:
        # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
        # This is safe in our controlled environment
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        
        # Load model weights (handle DataParallel)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler if provided
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Print checkpoint info - store 0-based internally, display 1-based
        completed_epoch = checkpoint['epoch'] + 1  # Convert to 1-based for display
        total_epochs = checkpoint.get('hyperparams', {}).get('epochs', '?')
        training_progress = f"Epoch {completed_epoch}/{total_epochs}"
        accuracy = checkpoint.get('training_metrics', {}).get('val_accuracy', [])
        accuracy_info = f", Accuracy: {accuracy[-1]:.4f}" if accuracy else ""
        
        print(f"Checkpoint details: {training_progress}{accuracy_info} (completed)")
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("   Try using a newer checkpoint or start fresh")
        return None


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


def train_model(data, hyperparams):
    """
    Trains the transformer model using the provided data and hyperparameters.

    Args:
        data (DataFrame): The data to train the model on.
        hyperparams (dict): Hyperparameters for model training, including model
            architecture details.

    Returns:
        model (nn.Module): The trained model.
        training_metrics (dict): Metrics and stats from the training process.
        model_metadata (dict): Metadata needed for inference.
    """
    # Note: Checkpointing automatically happens based on CHECKPOINT_FREQ
    
    # Get experiment ID for checkpointing
    experiment_id = hyperparams.get('experiment_id', 'unknown_experiment')
    
    # Extract hyperparameters
    epochs = hyperparams['epochs']
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    
    # Transformer-specific hyperparameters
    d_model = hyperparams.get('d_model', 128)  # Embedding dimension
    nhead = hyperparams.get('nhead', 8)  # Number of attention heads
    num_encoder_layers = hyperparams.get('num_encoder_layers', 2)  # Number of transformer layers
    dim_feedforward = hyperparams.get('dim_feedforward', 512)  # Hidden dimension in feed forward network
    dropout_rate = hyperparams.get('dropout_rate', 0.1)  # Dropout rate
    
    # Other parameters
    vocab_size = hyperparams.get('vocab_size', 27)  # Default for character-level model
    num_classes = len(np.unique(data['cipher']))  # Number of cipher classes

    # Preprocess data
    X, y, token_dict, label_encoder = load_and_preprocess_data(data)
    train_loader, val_loader = create_data_loaders(X, y, batch_size)
    
    # Check if the train_loader has batches
    if len(train_loader) == 0:
        print("Training DataLoader is empty.")
    else:
        print(f"Training DataLoader contains {len(train_loader)} batches.")        

    # Initialize model
    try:
        model = TransformerClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, 
            num_classes=num_classes,
            dropout=dropout_rate
        )
        print("Transformer model initialized successfully.")
    except Exception as e:
        print(f"Transformer model initialization failed: {e}")
        raise

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler - use cosine annealing with warmup for transformers
    warmup_steps = min(epochs * len(train_loader) // 10, 1000)
    total_steps = epochs * len(train_loader)
    
    def lr_lambda(current_step):
        # Linear warmup for warmup_steps, then cosine decay
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Check for existing checkpoint
    start_epoch = 0
    global_step = 0
    
    # Metrics tracking
    training_metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'conf_matrix': [],
        'learning_rates': []
    }
    
    # Check for any checkpoint for this experiment with matching architecture
    print(f"Looking for checkpoints for experiment {experiment_id}...")
    
    # Generate config hash for this experiment
    config_hash = generate_config_hash(hyperparams)
    print(f"Architecture hash: {config_hash}")
    
    # Look for checkpoints with matching experiment ID base and architecture hash
    base_id = experiment_id.split('_20')[0]  # Remove timestamp portion
    
    # Only look for exact architecture matches using the hash
    exp_checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) 
                      if ((f.startswith(experiment_id) or f.startswith(base_id)) and 
                         f"_{config_hash}_" in f and 
                         f.endswith('.pt'))]
    
    # No fallback to incompatible checkpoints - this would only cause load errors
    
    if exp_checkpoints:
        # Sort by modification time (newest first)
        sorted_checkpoints = sorted(exp_checkpoints, 
                                    key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)),
                                    reverse=True)
        
        # Get current target epochs
        current_target_epochs = hyperparams.get('epochs', 0)
        valid_checkpoint_path = None
        
        # Find the newest checkpoint that doesn't exceed our target epoch count
        for checkpoint_file in sorted_checkpoints:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
            try:
                # Quickly peek at checkpoint to check epoch
                checkpoint_data = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
                checkpoint_epoch = checkpoint_data.get('epoch', 0)
                
                # Only use if epoch count is less than our target
                if checkpoint_epoch < current_target_epochs:
                    valid_checkpoint_path = checkpoint_path
                    break
            except Exception:
                # Skip problematic checkpoints
                continue
        
        # If we found a valid checkpoint, use it
        if valid_checkpoint_path:
            print(f"✅ Found checkpoint at {valid_checkpoint_path}")
            checkpoint = load_checkpoint(valid_checkpoint_path, model, optimizer, scheduler)
            if checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                global_step = checkpoint['global_step']
                training_metrics = checkpoint['training_metrics']
                
                # Extract the previous cumulative duration if available
                if 'training_duration' in training_metrics:
                    cumulative_duration = training_metrics['training_duration']
                    print(f"✅ Resuming with epoch {start_epoch+1}/{epochs} (previous duration: {cumulative_duration:.1f}s)")
                else:
                    print(f"✅ Resuming with epoch {start_epoch+1}/{epochs}")
            else:
                print("❌ Failed to load checkpoint. Starting from scratch.")
        else:
            print("No suitable checkpoints found (all checkpoints at/beyond target epoch count)")
            print("Starting fresh training")
    else:
        print(f"No checkpoints found for experiment {experiment_id}. Starting fresh training.")
        
    # Initialize cumulative duration (will be updated from checkpoint if available)
    cumulative_duration = 0.0
    
    # Track starting time for accurate duration calculation on interruption
    session_start_time = time.time()
    
    # Get experiment ID for cleanup later
    current_experiment_id = hyperparams.get('experiment_id', 'unknown_experiment')
    
    for epoch in range(start_epoch, epochs):
            
        clear_gpu_memory()  # Clear GPU memory before each epoch

        # Training phase
        model.train()
        train_loss = 0
        
        for inputs, labels in tqdm(
                train_loader,
                desc=f'Epoch {epoch+1}/{epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(inputs)
            logits = torch.clamp(logits, min=EPSILON, max=1-EPSILON)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()

            # Gradient clipping (important for transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            train_loss += loss.item()
            training_metrics['learning_rates'].append(scheduler.get_last_lr()[0])
            
            global_step += 1
            

        # Validation phase
        model.eval()
        val_loss = 0
        correct_predictions = 0
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(
                    val_loader,
                    desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                logits = model(inputs)
                logits = torch.clamp(logits, min=EPSILON, max=1-EPSILON)
                
                # Calculate loss
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Calculate accuracy
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(probabilities, dim=1)
                correct_predictions += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().tolist())
                all_true_labels.extend(labels.cpu().tolist())
                

        # Calculate metrics for this epoch
        val_accuracy = correct_predictions / len(val_loader.dataset)
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)

        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        training_metrics['train_loss'].append(avg_train_loss)
        training_metrics['val_loss'].append(avg_val_loss)
        training_metrics['val_accuracy'].append(val_accuracy)
        training_metrics['conf_matrix'].append(conf_matrix)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: "
              + f"{avg_train_loss:.4f}, Validation Loss: "
              + f"{avg_val_loss:.4f}, Validation Accuracy: "
              + f"{val_accuracy:.4f}")

        # Save only the latest checkpoint periodically
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            # Include architecture hash in the checkpoint path
            latest_path = get_checkpoint_path(experiment_id, hyperparams)
            
            # Save only the latest checkpoint (no epoch-specific checkpoints)
            save_checkpoint(
                latest_path, model, optimizer, scheduler, 
                epoch, global_step, training_metrics, 
                token_dict, label_encoder, hyperparams
            )

        # Stop training if NaN values are detected
        if torch.isnan(torch.tensor(avg_train_loss)) or torch.isnan(torch.tensor(avg_val_loss)):
            print(f"NaN loss detected at epoch {epoch+1}. Stopping training.")
            break

    # Record training duration
    end_time = time.time()
    session_duration = end_time - session_start_time
    
    # Add this session's duration to any previously accumulated duration
    total_duration = cumulative_duration + session_duration
    training_metrics['training_duration'] = total_duration
    
    print(f"Training completed. Session duration: {session_duration:.1f}s, Total duration: {total_duration:.1f}s")
    
    # Save model metadata along with the model
    model_metadata = {
        'token_dict': token_dict,
        'label_encoder_classes': label_encoder.classes_.tolist(), 
        'hyperparams': hyperparams,
        'd_model': d_model,
        'nhead': nhead,
        'num_encoder_layers': num_encoder_layers,
        'dim_feedforward': dim_feedforward,
        'checkpoint_path': get_checkpoint_path(experiment_id, hyperparams),  # Add checkpoint path to metadata
        'config_hash': generate_config_hash(hyperparams)  # Include the config hash for reference
    }
    
    # Clean up all checkpoints for completed experiment
    print("\nCleaning up checkpoints for completed experiment...")
    clean_old_checkpoints(current_experiment_id, completed=True)
    print("Checkpoint cleanup complete.")
    
    return model, training_metrics, model_metadata