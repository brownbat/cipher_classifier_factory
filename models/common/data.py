"""
Common data handling functions for the cipher classifier.
"""
import torch
import pandas as pd
import numpy as np
import os
import sys
import string
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
# Import cipher names for data generation
from ciphers import _get_cipher_names


# Setup project root for imports
# Assumes models/common/data.py is two levels down from project root
_COMMON_DATA_FILE_PATH = os.path.abspath(__file__)
_MODELS_COMMON_DIR = os.path.dirname(_COMMON_DATA_FILE_PATH)
_MODELS_DIR = os.path.dirname(_MODELS_COMMON_DIR)
_PROJECT_ROOT_FROM_DATA = os.path.dirname(_MODELS_DIR)

# Add project root to path if not already there
if _PROJECT_ROOT_FROM_DATA not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_FROM_DATA)
    # print(f"DEBUG [data.py]: Added {_PROJECT_ROOT_FROM_DATA} to sys.path")


def custom_text_tokenizer(text, token_dict):
    """
    Tokenize text using a character-level token dictionary.

    Args:
        text (str): The text to tokenize
        token_dict (dict): Mapping from characters to token IDs

    Returns:
        list: List of token IDs
    """
    if type(text) is not str:
        text = str(text)
    return [token_dict.get(char, 0) for char in text]  # 0 for unknown chars


def load_and_preprocess_data(data):
    """
    Preprocesses the given DataFrame for training.

    Args:
        data (DataFrame): The DataFrame containing the sample text and labels.

    Returns:
        X_padded (Tensor): The tokenized and padded text sequences.
        y (Tensor): The encoded labels.
        token_dict (dict): The token dictionary used for text tokenization.
        label_encoder (LabelEncoder): The label encoder used for encoding labels.
    """
    # Create a character-level tokenizer dictionary
    unique_chars = string.ascii_lowercase  # set(''.join(data['text']))
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


def create_data_loaders(X, y, batch_size=32, validation_split=0.2):
    """
    Splits data into training and validation sets and creates DataLoader objects for both.

    Args:
        X (Tensor): The input features (tokenized and padded text sequences).
        y (Tensor): The target labels.
        batch_size (int, optional): The size of each batch for training.
        validation_split (float, optional): The proportion of the dataset to include in the validation split.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
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


def get_data(data_params: dict) -> pd.DataFrame | None:
    """
    Ensures sample data exists or generates it, then loads and returns it.

    Uses prep_samples.manage_sample_data to handle data generation logic,
    including caching, parallel processing, and interruption signals via
    the global shutdown_event mechanism.

    Args:
        data_params (dict): A dictionary containing the parameters needed
                            for data generation, typically including:
                            - 'cipher_names' (list): List of cipher identifiers.
                            - 'num_samples' (int): Total number of samples.
                            - 'sample_length' (int): Length of each text sample.
                            - 'force_regenerate' (bool, optional): Force regeneration.
                            - 'metadata_file' (str, optional): Path to metadata JSON.

    Returns:
        pd.DataFrame | None: A DataFrame containing the 'text' and 'cipher'
                              columns, or None if data generation or loading
                              failed or was interrupted.
    """
    # Extract necessary parameters
    cipher_names = data_params.get('cipher_names')
    num_samples = data_params.get('num_samples')
    sample_length = data_params.get('sample_length')
    force_regenerate = data_params.get('force_regenerate', False) # Default to False

    # Define path for metadata file (could be made configurable via data_params too)
    metadata_file_path = os.path.join(_PROJECT_ROOT_FROM_DATA, 'data', 'feathers', 'sample_feathers_metadata.json')
    metadata_file_param = data_params.get('metadata_file', metadata_file_path)

    # Basic validation of required parameters
    if not all([cipher_names, num_samples, sample_length]):
        print("ERROR [get_data]: Missing required data parameters: 'cipher_names', 'num_samples', or 'sample_length'.")
        return None
    if not isinstance(cipher_names, list) or not cipher_names:
        print("ERROR [get_data]: 'cipher_names' must be a non-empty list.")
        return None
    if not isinstance(num_samples, int) or num_samples <= 0:
        print("ERROR [get_data]: 'num_samples' must be a positive integer.")
        return None
    if not isinstance(sample_length, int) or sample_length <= 0:
        print("ERROR [get_data]: 'sample_length' must be a positive integer.")
        return None


    # Import prep_samples here to avoid circular imports
    import prep_samples
    
    # Call manage_sample_data from prep_samples module
    # It uses the global shutdown_event implicitly and handles parallelism.
    # It returns: (relative_filename | None, bool, status_message | None)
    relative_filename, data_was_generated, status_message = prep_samples.manage_sample_data(
        cipher_names=cipher_names,
        num_samples=num_samples,
        sample_length=sample_length,
        metadata_file=metadata_file_param,
        force_regenerate=force_regenerate
    )

    # --- Handle the outcome ---
    if status_message == 'interrupted':
        print("ERROR [get_data]: Data generation was interrupted by shutdown signal.")
        return None # Signal interruption upwards
    elif status_message == 'worker_errors':
        # Data was generated but with issues. We might still proceed but log a clear warning.
        print(f"WARNING [get_data]: Data generated with worker errors. Proceeding with file: {relative_filename}")
        # Continue to load data below, but the warning is logged.
    elif status_message is not None and status_message.startswith('Error:'):
        # Specific error occurred during generation or management
        print(f"ERROR [get_data]: Failed to prepare data. Reason: {status_message}")
        return None # Signal error upwards
    elif relative_filename is None:
        # Catch-all for unexpected cases where filename is None without a specific error message
        print("ERROR [get_data]: Failed to get data filename from manage_sample_data for unknown reasons.")
        return None # Signal error upwards

    # --- Load the data ---
    # If we reached here, status was None (success) or 'worker_errors' (proceed with warning)
    try:
        # Construct absolute path from project root and relative filename
        absolute_filename = os.path.join(_PROJECT_ROOT_FROM_DATA, relative_filename)
        print(f"INFO [get_data]: Loading data from: {absolute_filename}")
        if not os.path.exists(absolute_filename):
             print(f"ERROR [get_data]: Feather file not found at expected path: {absolute_filename}")
             return None

        data = pd.read_feather(absolute_filename)
        print(f"INFO [get_data]: Successfully loaded {len(data)} samples.")

        # Basic validation of loaded data
        if data.empty:
             print("WARNING [get_data]: Loaded data file is empty.")
             # Depending on requirements, you might return None here or allow empty DataFrame.
             # Let's return None to indicate failure to get usable data.
             return None
        if not {'text', 'cipher'}.issubset(data.columns):
             print(f"ERROR [get_data]: Loaded data missing required columns ('text', 'cipher'). Found: {list(data.columns)}")
             return None

        return data

    except FileNotFoundError:
        print(f"ERROR [get_data]: Feather file reported by manage_sample_data not found: {absolute_filename}")
        return None
    except Exception as e:
        # Catch errors during feather loading or validation
        print(f"ERROR [get_data]: Failed to load or validate data from {relative_filename}: {e}")
        import traceback
        traceback.print_exc()
        return None
