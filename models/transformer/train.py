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

from models.transformer.model import TransformerClassifier
from models.common.data import load_and_preprocess_data, create_data_loaders
from models.common.utils import clear_gpu_memory

# Constants
EPSILON = 1e-7  # Small constant to prevent log(0)
CHECKPOINT_DIR = "data/checkpoints"
CHECKPOINT_FREQ = 1  # Save checkpoint every N epochs

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint to resume training"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
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
    
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return checkpoint


def get_checkpoint_path(experiment_id, epoch=None):
    """Get checkpoint path for a given experiment ID and optional epoch"""
    if epoch is not None:
        return os.path.join(CHECKPOINT_DIR, f"{experiment_id}_epoch_{epoch}.pt")
    else:
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
    latest_checkpoint_path = get_checkpoint_path(experiment_id)
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
    
    # Try to load checkpoint if exists
    if os.path.exists(latest_checkpoint_path):
        print(f"Found checkpoint at {latest_checkpoint_path}. Attempting to resume training...")
        checkpoint = load_checkpoint(latest_checkpoint_path, model, optimizer, scheduler)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            global_step = checkpoint['global_step']
            training_metrics = checkpoint['training_metrics']
            print(f"Resuming training from epoch {start_epoch}")

    # Training loop
    start_time = time.time()
    
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

        # Save checkpoint periodically
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            checkpoint_path = get_checkpoint_path(experiment_id, epoch+1)
            latest_path = get_checkpoint_path(experiment_id)
            
            # Save epoch-specific and latest checkpoint
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, 
                epoch, global_step, training_metrics, 
                token_dict, label_encoder, hyperparams
            )
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
    training_duration = end_time - start_time
    training_metrics['training_duration'] = training_duration
    
    # Save model metadata along with the model
    model_metadata = {
        'token_dict': token_dict,
        'label_encoder_classes': label_encoder.classes_.tolist(), 
        'hyperparams': hyperparams,
        'd_model': d_model,
        'nhead': nhead,
        'num_encoder_layers': num_encoder_layers,
        'dim_feedforward': dim_feedforward,
        'checkpoint_path': get_checkpoint_path(experiment_id)  # Add checkpoint path to metadata
    }
    
    return model, training_metrics, model_metadata