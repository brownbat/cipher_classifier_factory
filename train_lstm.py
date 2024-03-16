import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import os
from datetime import datetime
import string
import re
import json
import hashlib
from sklearn.metrics import confusion_matrix
import prep_samples
import numpy as np
import traceback
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage
from IPython.display import Image as IPImage
import imageio
from ciphers import _get_cipher_functions, _get_cipher_names
import time
import subprocess
import gc

# TODO: feature engineering, index of coincidence, digraphs, trigraphs,
#   skipgraphs, ioc for subsets

# Constants
EPOCHS = 5
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
VOCAB_SIZE = 27


# Define LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

def file_hash(filename):
    """Generate a hash for a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def custom_text_tokenizer(text, token_dict):
    if type(text) is not str:
        text = str(text)
    return [token_dict.get(char, 0) for char in text]  # 0 for unknown chars


def load_and_preprocess_data(data):
    """
    Preprocesses the given DataFrame for training.

    Args:
    - data (DataFrame): The DataFrame containing the sample text and labels.

    Returns:
    - X_padded (Tensor): The tokenized and padded text sequences.
    - y (Tensor): The encoded labels.
    - token_dict (dict): The token dictionary used for text tokenization.
    - label_encoder (LabelEncoder): The label encoder used for encoding labels.
    """
    # Create a character-level tokenizer dictionary
    unique_chars = string.ascii_lowercase # set(''.join(data['text']))
    # +1 to reserve 0 for padding
    token_dict = {char: i+1 for i, char in enumerate(unique_chars)}

    # Tokenize text samples
    test_string = "example"
    tokenized = custom_text_tokenizer(test_string, token_dict)
    
    X = [
        torch.tensor(custom_text_tokenizer(
            sample,
            token_dict), dtype=torch.long) for sample in data['text']]

    X_padded = pad_sequence(X, batch_first=True, padding_value=0)

    # Encode labels and convert to Long type
    label_encoder = LabelEncoder()
    y = torch.tensor(
        label_encoder.fit_transform(data['cipher'].values), dtype=torch.long)

    return X_padded, y, token_dict, label_encoder


def create_data_loaders(X, y, batch_size=BATCH_SIZE, validation_split=0.2):
    """
    Splits data into training and validation sets and
        creates DataLoader objects for both.

    Args:
    - X (Tensor): The input features (tokenized and padded text sequences).
    - y (Tensor): The target labels.
    - batch_size (int, optional): The size of each batch for training.
        Default is set to BATCH_SIZE.
    - validation_split (float, optional):
        The proportion of the dataset to include in the validation split.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    """
    # Split data into training and validation sets
    num_train = int((1 - validation_split) * len(X))
    indices = torch.randperm(len(X)).tolist()
    train_indices, val_indices = indices[:num_train], indices[num_train:]

    # Create Tensor datasets
    train_data = TensorDataset(X[train_indices], y[train_indices])
    val_data = TensorDataset(X[val_indices], y[val_indices])

    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    return train_loader, val_loader

# WARNING: activation functions are implied by this model and by crossentropy
# and should not be specified
def is_valid_activation_function(func):
    # List of known PyTorch activation function classes
    valid_activation_funcs = [nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.SiLU]
    return any(isinstance(func, cls) for cls in valid_activation_funcs)


# WARNING: activation functions are implied by this model and by crossentropy
# and should not be specified
def get_activation_function(activation_input):
    """
    Returns the PyTorch activation function based on the input.
    If a string is provided, it maps to the corresponding activation function.
    If a callable is provided, it returns the callable directly.

    Args:
    - activation_input (str or callable): The name of the activation function or
    the activation function itself.

    Returns:
    - activation_func (callable): The PyTorch activation function.
    """
    if callable(activation_input):
        if is_valid_activation_function(activation_input):
            return activation_input
        else:
            raise ValueError(
                "Invalid activation function or unrecognized "
                + "function name.")
    activation_functions = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'swish': nn.SiLU(),  # Swish is known as SiLU in PyTorch
        # Add more activations if needed
    }
    return activation_functions.get(activation_input.lower(), None)


def get_gpu_temp():
    process = subprocess.Popen(["sensors"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output, _ = process.communicate()

    # Convert bytes to string
    sensor_data = output.decode()

    # Regex to find temperature lines
    # edge_temp = re.search(r'edge:\s+\+(\d+\.\d+)°C', sensor_data)
    junction_temp = re.search(r'junction:\s+\+(\d+\.\d+)°C', sensor_data)

    # Extract temperatures
    # edge_temp_val = edge_temp.group(1) if edge_temp else -1
    junction_temp_val = float(junction_temp.group(1)) if junction_temp else -1

    return junction_temp_val


def train_model(data, hyperparams):
    """
    Trains the LSTM model using the provided data and hyperparameters.

    Args:
    - data (DataFrame): The data to train the model on.
    - hyperparams (dict): Hyperparameters for model training, including model
        architecture details.

    Returns:
    - model (nn.Module): The trained model.
    - training_metrics (dict): Metrics and stats from the training process,
        like loss and accuracy.
    """
    # Extract hyperparameters - NO DEFAULTS, must be fully specified
    epochs = hyperparams['epochs']
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    embedding_dim = hyperparams['embedding_dim']
    hidden_dim = hyperparams['hidden_dim']

    # ok vocab size can be a default it's basically fixed
    vocab_size = hyperparams.get('vocab_size', 27)
    num_classes = len(np.unique(data['cipher']))

    # WARNING: activation functions are implied by this model and by crossentropy
    # and should not be specified

    # activation_name = hyperparams.get('activation_func', None)
    # activation_func = get_activation_function(activation_name)

    # Preprocess data
    X, y, token_dict, label_encoder = load_and_preprocess_data(data)
    train_loader, val_loader = create_data_loaders(X, y, batch_size)
    # Check the first batch in the train_loader
    for inputs, labels in train_loader:
        break  # Just to check the first batch
    # Check if the train_loader has batches
    if len(train_loader) == 0:
        print("Training DataLoader is empty.")
    else:
        print(f"Training DataLoader contains {len(train_loader)} batches.")        

    # Initialize model
    try:
        model = LSTMClassifier(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_classes)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        raise  # Optionally re-raise the exception to stop the script

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'conf_matrix': []}

    # Training loop
    start_time = time.time()
    confusion_matrices = []
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()  # add explicit garbage collection to prevent memory leaks
        # ~ while get_gpu_temp() > 102:  # to address runaway heating issues on rocm5.6 that seem to have been resolved in rocm5.7
            # ~ print("WARNING: High GPU Temp!")
            # ~ time.sleep(20)
        model.train()
        train_loss = 0
        correct_predictions = 0
        all_predictions = []
        all_true_labels = []
        for inputs, labels in tqdm(
                train_loader,
                desc=f'Epoch {epoch+1}/{epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)  # Get logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            probabilities = model(inputs)  # Get probabilities for accuracy calculation
            _, predicted = torch.max(probabilities.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            all_predictions.extend(predicted.tolist())
            all_true_labels.extend(labels.tolist())

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
                logits = model(inputs)  # Get logits
                loss = criterion(logits, labels)
                val_loss += loss.item()


                # Calculate accuracy
                probabilities = model(inputs)
                _, predicted = torch.max(probabilities.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                all_predictions.extend(predicted.tolist())
                all_true_labels.extend(labels.tolist())

        val_accuracy = correct_predictions / len(val_loader.dataset)
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)

        # Record training and validation loss
        training_metrics['train_loss'].append(train_loss / len(train_loader))
        training_metrics['val_loss'].append(val_loss / len(val_loader))
        training_metrics['val_accuracy'].append(val_accuracy)
        training_metrics['conf_matrix'].append(conf_matrix)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: "
              + f"{train_loss/len(train_loader):.4f}, Validation Loss: "
              + f"{val_loss/len(val_loader):.4f}, Validation Accuracy: "
              + f"{val_accuracy:.4f}")

    end_time = time.time()
    training_duration = end_time - start_time
    training_metrics['training_duration'] = training_duration
    return model, training_metrics


def conf_matrix_to_gif(conf_matrices, filename='confusion_matrices.gif', duration=1000, loop=0):
    with imageio.get_writer(filename, mode='I', duration=duration, loop=loop) as writer:
        for epoch, conf_matrix in enumerate(conf_matrices, 1):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Epoch {epoch}')
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')

            # Convert to image array and append to the GIF
            with io.BytesIO() as buf:
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_array = np.array(PIL.Image.open(buf))
                writer.append_data(img_array)

            plt.close(fig)


def save_model(
        model, token_dict, label_encoder, params, hyperparams,
        training_metrics, metadata_file="model/model_metadata.json"):
    # Generate unique identifier for the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"model_{timestamp}"
    model_filename = f"data/models/{model_id}.pth"

    # Save the model
    torch.save(model.state_dict(), model_filename)

    # Save token dictionary and label encoder as JSON
    token_dict_filename = f"data/models/{model_id}_token_dict.json"
    label_encoder_filename = f"data/models/{model_id}_label_encoder.json"

    with open(token_dict_filename, 'w') as file:
        json.dump(token_dict, file)

    label_mapping = dict(zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_)))
    with open(label_encoder_filename, 'w') as file:
        json.dump(label_mapping, file)

    # Calculate file hash for model
    model_hash = file_hash(model_filename)

    # Load or initialize metadata
    if not os.path.exists(metadata_file) or os.stat(metadata_file).st_size == 0:
        metadata = {}
    else:
        with open(metadata_file, "r") as file:
            metadata = json.load(file)

    # Update metadata with new model details and file hashes
    metadata[model_id] = {
        "filename": model_filename,
        "token_dict": token_dict_filename,
        "label_encoder": label_encoder_filename,
        "model_hash": model_hash,
        "params": params,
        "hyperparams": hyperparams,
        "training_metrics": training_metrics,
        "timestamp": timestamp
    }

    with open(metadata_file, "w") as file:
        json.dump(metadata, file, indent=4)

    cleanup_model_metadata(metadata_file)
    return model_filename


def load_model(model_id, metadata_file="model/model_metadata.json"):
    cleanup_model_metadata(metadata_file)

    # Load metadata
    with open(metadata_file, "r") as file:
        metadata = json.load(file)

    if model_id in metadata:
        model_filename = metadata[model_id]["filename"]
        token_dict_filename = metadata[model_id]["token_dict"]
        label_encoder_filename = metadata[model_id]["label_encoder"]

        # Load token dictionary and label encoder
        with open(token_dict_filename, 'r') as file:
            token_dict = json.load(file)

        with open(label_encoder_filename, 'r') as file:
            label_mapping = json.load(file)
            # Logic to reconstruct label encoder from label_mapping
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(list(label_mapping.keys()))

        # Reconstruct and load the model
        # This step assumes you have the architecture details in metadata
        # Example:
        # embedding_dim = metadata[model_id]["hyperparams"]["embedding_dim"]
        # hidden_dim = metadata[model_id]["hyperparams"]["hidden_dim"]
        # num_classes = len(label_encoder.classes_)
        # model = LSTMClassifier(VOCAB_SIZE, embedding_dim, hidden_dim,
        #    num_classes)
        # model.load_state_dict(torch.load(model_filename))

        return model, token_dict, label_encoder
    else:
        raise FileNotFoundError(f"No model found with ID: {model_id}")


def cleanup_model_metadata(metadata_file="data/models/model_metadata.json"):
    """Synchronizes the model metadata file with the actual model files in the
    directory."""
    if not os.path.exists(metadata_file):
        print("Model metadata file not found.")
        return

    with open(metadata_file, "r") as file:
        metadata = json.load(file)

    valid_metadata = {}
    for model_id, info in metadata.items():
        if os.path.exists(info["filename"]):
            valid_metadata[model_id] = info

    with open(metadata_file, "w") as file:
        json.dump(valid_metadata, file, indent=4)

    print("Model metadata cleanup complete.")


def classify_text(model, text, token_dict, label_encoder):
    """
    Classifies the given text using the trained LSTM model.

    Args:
    - model (nn.Module): The trained LSTM model.
    - text (str): The text to be classified.
    - token_dict (dict): The token dictionary used for text tokenization.
    - label_encoder (LabelEncoder): The label encoder for decoding the
        predicted label.

    Returns:
    - predicted_label (str): The predicted label of the text.
    """
    if len(text) == 0:
        return "Invalid input: Text is empty"
    print("Sanitizing input to lowercase letters only")
    text = re.sub(r'[^a-zA-Z]', '', text).lower()
    # Tokenize and pad the new text
    tokenized_text = custom_text_tokenizer(text)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(tokenized_text)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()

    # Decode the predicted index to get the label
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label


def get_data(params):
    """
    Retrieves or generates the necessary data for training based on specified
    parameters.

    Args:
    - params (dict): Parameters for data generation.

    Returns:
    - data (DataFrame): The data ready for training.
    """
    cipher_names = params.get('ciphers', _get_cipher_names())
    num_samples = params['num_samples']
    sample_length = params['sample_length']
    metadata_file = params.get(
        'metadata_file',
        "data/sample_feathers_metadata.json")

    # Call manage_sample_data from prep_samples.py
    filename, data_generated = prep_samples.manage_sample_data(
        cipher_names, num_samples, sample_length, metadata_file)

    # Load the data from the feather file
    data = pd.read_feather(filename)

    return data


def run_training_experiment(params, hyperparams):
    """
    Orchestrates the training of an LSTM model, generating and saving necessary
    data, saves the model, and manages metadata.

    Args:
    - params (dict): Dictionary containing parameters for data generation and
    model configuration.
    - hyperparams (dict): Dictionary containing hyperparameters for model
    training.

    Returns:
    - model_path (str): File path to the saved model.
    - training_metrics (dict): Metrics and stats from the training process.
    """

    # Ensure data is ready
    data = get_data(params)

    # Train the model
    model, training_metrics = train_model(data, hyperparams)

    # Save the model and manage metadata
    model_path = save_model(model, params, hyperparams, training_metrics)

    return model_path, training_metrics


def run_basic_tests():
    """
    Runs basic tests to ensure that primary functionalities are working.
    """
    print("Running basic tests...")

    # Test: Model Initialization
    try:
        model = LSTMClassifier(
            vocab_size=27,
            embedding_dim=128,
            hidden_dim=128,
            output_dim=10)
        print("Model initialization test passed.")
    except Exception as e:
        print(f"Model initialization test failed: {e}")

    try:
        # Generate a small dataset
        cipher_names = ['caesar', 'vigenere', 'english', 'columnar_transposition']
        num_samples = 10000
        sample_length = 500
        filename, _ = prep_samples.manage_sample_data(
            cipher_names, num_samples, sample_length)

        # Load the generated data
        data = pd.read_feather(filename)
        print(data.head())

        # Define hyperparameters for the test
        hyperparams = {
            'epochs': 3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'embedding_dim': 128,
            'hidden_dim': 128,
        }


        # Run a basic training cycle
        trained_model, training_metrics = train_model(data, hyperparams)
        print("Data generation and model training test passed.")
    except Exception as e:
        print(f"Data generation and model training test failed: {e}")
        traceback.print_exc()

    print("Basic tests completed.")


if __name__ == "__main__":
    run_basic_tests()
