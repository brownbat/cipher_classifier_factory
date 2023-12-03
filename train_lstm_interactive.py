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

# Constants
MAX_SEQUENCE_LENGTH = 500  # Length of samples
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
VOCAB_SIZE = 256  # Assuming 256 different characters


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


def custom_text_tokenizer(text, token_dict):
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
    unique_chars = set(''.join(data['sample']))
    # +1 to reserve 0 for padding
    token_dict = {char: i+1 for i, char in enumerate(unique_chars)}

    # Tokenize and truncate text samples
    X = [
        torch.tensor(custom_text_tokenizer(
            sample[:MAX_SEQUENCE_LENGTH],
            token_dict)) for sample in data['sample']]
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)

    # Encode labels and convert to Long type
    label_encoder = LabelEncoder()
    y = torch.tensor(
        label_encoder.fit_transform(data['label'].values), dtype=torch.long)

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


def train_model(model, train_loader, val_loader, epochs=5, learning_rate=0.001):
    """
    Trains the LSTM model and evaluates it on the validation set
        with progress bars.

    Args:
    - model (nn.Module): The LSTM model to be trained.
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    - epochs (int, optional): Number of training epochs. Default is 5.
    - learning_rate (float, optional): Learning rate for the optimizer.
        Default is 0.001.

    Returns:
    - model: The trained LSTM model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Training phase with tqdm progress bar
        model.train()
        train_loss = 0
        for inputs, labels in tqdm(
                train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase with tqdm progress bar
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in tqdm(
                    val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Print epoch stats
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: " +
            f"{train_loss/len(train_loader)}, Validation Loss: " +
            f"{val_loss/len(val_loader)}")

    return model


def save_model(model, token_dict, label_encoder, file_path):
    """
    Saves the model, token dictionary,
        and label encoder to specified file paths.

    Args:
    - model (nn.Module): The trained model to be saved.
    - token_dict (dict): The token dictionary used for text tokenization.
    - label_encoder (LabelEncoder): The label encoder used for encoding labels.
    - file_path (str): The base file path for saving the artifacts.
    """
    # Save the model
    torch.save(model.state_dict(), f'{file_path}_model.pth')

    # Save token dictionary and label encoder
    with open(f'{file_path}_token_dict.pkl', 'wb') as f:
        pickle.dump(token_dict, f)
    with open(f'{file_path}_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save metadata with current date and time
    metadata = {
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f'{file_path}_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def load_model(file_path):
    """
    Loads the LSTM model and associated artifacts
        (token dictionary and label encoder).

    Args:
    - file_path (str): The base file path for loading the artifacts.

    Returns:
    - model (nn.Module): The loaded LSTM model.
    - token_dict (dict): The token dictionary used for text tokenization.
    - label_encoder (LabelEncoder): The label encoder used for encoding labels.
    """

    # Load token dictionary and label encoder
    with open(f'{file_path}_token_dict.pkl', 'rb') as f:
        token_dict = pickle.load(f)
    with open(f'{file_path}_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Determine the number of classes from label_encoder
    num_classes = len(label_encoder.classes_)

    # Load the model with the dynamically determined number of classes
    model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, num_classes)
    model.load_state_dict(torch.load(f'{file_path}_model.pth'))

    # Load metadata
    with open(f'{file_path}_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    return model, token_dict, label_encoder, metadata


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
    tokenized_text = [
        token_dict.get(char, 0) for char in text[:MAX_SEQUENCE_LENGTH]]
    padded_sequence = pad_sequence(
        [torch.tensor(tokenized_text)], batch_first=True, padding_value=0)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(padded_sequence)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()

    # Decode the predicted index to get the label
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label


def feathers_to_df(directory):
    """
    Merges all .feather files in the specified directory into one DataFrame.

    Args:
    - directory (str): The directory to search for .feather files.

    Returns:
    - DataFrame: The merged data from all .feather files.
    """
    feather_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory) if f.endswith('.feather')]
    df_list = [pd.read_feather(file) for file in feather_files]
    merged_df = pd.concat(df_list, ignore_index=True).drop_duplicates()
    return merged_df


def generate_data(ciphers, num_samples, sample_length, data_path):
    """
    Generates a dataset with specified parameters and saves it as a feather file.

    Args:
    - ciphers (list of str): List of ciphers to be used in data generation.
    - num_samples (int): Total number of samples to generate.
    - sample_length (int): Length of each sample.
    - data_path (str): Path to save the generated feather file.

    Returns:
    - DataFrame: The generated data in DataFrame format.

    TODO: allow for varying length samples, and different lengths for training,
        validation
    TODO: check if this exists... 
    """

    # Call to `prep_samples.py` functionality
    # This may involve modifying `prep_samples.py` to accept these parameters
    # and return data or save it directly as a feather file.
    
    # Example pseudocode:
    # generated_data = prep_samples.generate(ciphers, num_samples, sample_length)
    # generated_data.to_feather(os.path.join(data_path, 'generated_data.feather'))

    # Save the data as /data/(timestamp).feather
    # Record parameters and filename in .json
    # TODO flesh this out
    return generated_data


def main():
    """
    Main function to load a trained model and classify text,
    or train a new model if not available.
    """
    model_path = 'model/'  # Modify with your model's file path
    data_path = 'data/'

    model_loaded = False

    # Check if model exists
    if os.path.exists(f'{model_path}_model.pth'):
        print("Loading saved model...")
        model, token_dict, label_encoder, metadata = load_model(model_path)
        model_loaded = True
        NUM_CLASSES = len(label_encoder.classes_)
        # Display model details (modify this based on available info)
        print("Model details: ...")  # Replace with actual model details
        # Display model details
        print("Model details:")
        print(f"Model trained on: {metadata['training_date']}")
        print(f" - Embedding dimensions: {EMBEDDING_DIM}")
        print(f" - Hidden dimensions: {HIDDEN_DIM}")
        print(f" - Number of classes: {NUM_CLASSES}")
        print(
            f" - Total parameters: " +
            f"{sum(p.numel() for p in model.parameters())}")

        # Option to retrain
        retrain_option = input(
            "Do you want to retrain the model? (y/n): ").strip().lower()
        if retrain_option == 'y':
            model_loaded = False

    if not model_loaded:
        print("Training a new model...")
        data = feathers_to_df(data_path)
        X, y, token_dict, label_encoder = load_and_preprocess_data(data)
        NUM_CLASSES = len(label_encoder.classes_)
        train_loader, val_loader = create_data_loaders(X, y)
        model = LSTMClassifier(
            VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
        model = train_model(model, train_loader, val_loader)
        save_model(model, token_dict, label_encoder, model_path)

    # Prompt for text classification
    classify_option = input(
        "Do you want to classify text? (y/n): ").strip().lower()
    if classify_option == 'y':
        text_to_classify = ''
        while text_to_classify.strip().lower() != 'x':
            text_to_classify = input(
                "Enter the text to classify (or 'x' to exit): ")
            if text_to_classify.strip().lower() == 'x':
                break
            predicted_label = classify_text(
                model, text_to_classify, token_dict, label_encoder)
            print(f"Predicted Label: {predicted_label}")

    print("Exiting program.")


if __name__ == "__main__":
    main()
