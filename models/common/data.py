"""
Common data handling functions for the cipher classifier.
"""
import torch
import pandas as pd
import numpy as np
import string
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import prep_samples
from ciphers import _get_cipher_names


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


def get_data(params):
    """
    Retrieves or generates the necessary data for training based on specified parameters.

    Args:
        params (dict): Parameters for data generation.

    Returns:
        data (DataFrame): The data ready for training.
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