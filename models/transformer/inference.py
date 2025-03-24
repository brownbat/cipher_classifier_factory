"""
Inference functionality for transformer model.
"""
import torch
import re
import numpy as np
from models.common.data import custom_text_tokenizer


def classify_text(model, text, token_dict, label_encoder, device=None):
    """
    Classifies the given text using the trained transformer model.

    Args:
        model (nn.Module): The trained transformer model.
        text (str): The text to be classified.
        token_dict (dict): The token dictionary used for text tokenization.
        label_encoder (LabelEncoder): The label encoder for decoding the predicted label.
        device (torch.device, optional): The device to run inference on. If None, uses cuda if available.

    Returns:
        dict: Contains predicted label, probabilities, and input info
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not isinstance(text, str) or len(text) == 0:
        return {"error": "Invalid input: Text is empty or not a string"}
    
    # Sanitize input to lowercase letters only
    sanitized_text = re.sub(r'[^a-zA-Z]', '', text).lower()
    if len(sanitized_text) == 0:
        return {"error": "Invalid input: Text contains no valid characters"}
    
    # Tokenize the text
    tokenized_text = torch.tensor([custom_text_tokenizer(sanitized_text, token_dict)], 
                                 dtype=torch.long).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(tokenized_text)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    # Get all class probabilities
    probs_dict = {}
    for i, prob in enumerate(probabilities[0].cpu().numpy()):
        class_name = label_encoder.inverse_transform([i])[0]
        probs_dict[class_name] = float(prob)
        
    # Decode the predicted index to get the label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    return {
        "predicted_class": predicted_label,
        "confidence": float(probabilities[0][predicted_class].item()),
        "all_probabilities": probs_dict,
        "input_length": len(text),
        "sanitized_length": len(sanitized_text)
    }


def load_model(model_path, metadata_path, device=None):
    """
    Load a saved transformer model and its metadata.
    
    Args:
        model_path (str): Path to the saved model file
        metadata_path (str): Path to the saved metadata file
        device (torch.device, optional): Device to load the model to
        
    Returns:
        dict: Contains model, token_dict, label_encoder and other metadata
    """
    import pickle
    from sklearn.preprocessing import LabelEncoder
    from models.transformer.model import TransformerClassifier
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Extract metadata components
    token_dict = metadata['token_dict']
    
    # Reconstruct label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
    
    # Get hyperparameters
    hyperparams = metadata['hyperparams']
    
    # Load the model
    model = torch.load(model_path, map_location=device)
    
    return {
        "model": model,
        "token_dict": token_dict,
        "label_encoder": label_encoder,
        "hyperparams": hyperparams
    }