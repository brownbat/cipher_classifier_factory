"""
Inference functionality for transformer model.
"""
import torch
import re
import numpy as np
import json # Added
import os   # Added
import sys  # Added
from pathlib import Path # Added
import random # Added
from sklearn.preprocessing import LabelEncoder # Added
# Assuming these package imports work when run with -m from root
from models.common.data import custom_text_tokenizer
from models.transformer.model import TransformerClassifier
import models.common.utils as utils
import prep_samples # Added
import ciphers      # Added


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


def load_model(model_path_rel, metadata_path_rel, device=None):
    """
    Loads a saved transformer model state dict and its JSON metadata.

    Instantiates the model architecture based on metadata before loading the state dict.

    Args:
        model_path_rel (str): Relative path (from project root) to the saved model state dict (.pt file).
        metadata_path_rel (str): Relative path (from project root) to the saved metadata (.json file).
        device (torch.device, optional): Device to load the model onto ('cuda', 'cpu', etc.).
                                         Defaults to cuda if available, else cpu.

    Returns:
        Optional[Dict]: A dictionary containing the loaded 'model' (nn.Module),
                        'token_dict', 'label_encoder', 'hyperparams', and 'experiment_id'
                        if successful, otherwise None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Attempting to load model onto device: {device}")

    # Construct absolute paths using project root from utils
    try:
        abs_metadata_path = os.path.join(utils._PROJECT_ROOT, metadata_path_rel)
        abs_model_path = os.path.join(utils._PROJECT_ROOT, model_path_rel)
    except AttributeError:
         print("ERROR: Could not find project root via utils._PROJECT_ROOT. Ensure utils initializes correctly.")
         # Fallback to assuming relative paths are correct from current working dir (less robust)
         abs_metadata_path = metadata_path_rel
         abs_model_path = model_path_rel
         print(f"Warning: Assuming paths are relative to current directory: {os.getcwd()}")


    # --- 1. Load Metadata from JSON ---
    try:
        print(f"Loading metadata from: {abs_metadata_path}")
        with open(abs_metadata_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {abs_metadata_path}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Failed to decode JSON from metadata file: {abs_metadata_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load metadata: {e}")
        return None

    # --- 2. Extract Necessary Components from Metadata ---
    try:
        token_dict = metadata['token_dict']
        label_encoder_classes = metadata['label_encoder_classes']
        hyperparams = metadata.get('hyperparams', {}) # Use .get for safety
        experiment_id = metadata.get('experiment_id', 'Unknown') # Get ID if present

        # Extract architectural hyperparameters needed for instantiation
        # Use metadata top-level keys first (saved by train.py), fallback to hyperparams dict
        vocab_size = metadata.get('vocab_size', 27) # Default if missing
        num_classes = metadata.get('num_classes')
        d_model = metadata.get('d_model', hyperparams.get('d_model'))
        nhead = metadata.get('nhead', hyperparams.get('nhead'))
        num_layers = metadata.get('num_encoder_layers', hyperparams.get('num_encoder_layers')) # Key name mismatch
        dim_feedforward = metadata.get('dim_feedforward', hyperparams.get('dim_feedforward'))
        dropout = metadata.get('dropout_rate', hyperparams.get('dropout_rate', 0.1)) # Key name mismatch, add default

        # Validate required architectural params
        required_arch_params = {
             'num_classes': num_classes, 'd_model': d_model, 'nhead': nhead,
             'num_layers': num_layers, 'dim_feedforward': dim_feedforward}
        if any(v is None for v in required_arch_params.values()):
            missing_keys = [k for k, v in required_arch_params.items() if v is None]
            print(f"ERROR: Missing required architectural parameters in metadata: {missing_keys}")
            return None

    except KeyError as e:
        print(f"ERROR: Missing expected key in metadata file: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Failed processing metadata content: {e}")
        return None

    # --- 3. Reconstruct Label Encoder ---
    try:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(label_encoder_classes)
    except Exception as e:
        print(f"ERROR: Failed to reconstruct LabelEncoder: {e}")
        return None

    # --- 4. Instantiate Model Architecture ---
    try:
        print("Instantiating model architecture...")
        model = TransformerClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
            dropout=dropout # Pass dropout rate
        )
        model.to(device) # Move model to target device BEFORE loading state dict
    except Exception as e:
        print(f"ERROR: Failed to instantiate TransformerClassifier model: {e}")
        return None

    # --- 5. Load State Dict ---
    try:
        print(f"Loading model state_dict from: {abs_model_path}")
        if not os.path.exists(abs_model_path):
             print(f"ERROR: Model state dict file not found at {abs_model_path}")
             return None

        # Load the state dictionary
        state_dict = torch.load(abs_model_path, map_location=device)

        # Load state dict into the instantiated model
        model.load_state_dict(state_dict)
        model.eval() # Set model to evaluation mode
        print("Model state_dict loaded successfully.")

    except Exception as e:
        print(f"ERROR: Failed to load model state_dict: {e}")
        return None

    # --- 6. Return Loaded Components ---
    return {
        "model": model,
        "token_dict": token_dict,
        "label_encoder": label_encoder,
        "hyperparams": hyperparams, # Return original hyperparams dict for reference
        "experiment_id": experiment_id # Return ID for context
    }


if __name__ == "__main__":

    print("--- Running Inference Test ---")

    # --- Dynamic Setup ---
    # 1. Find Project Root (utils should handle this, but verify)
    if utils._PROJECT_ROOT is None:
        print("ERROR: Project root not found by utils. Cannot proceed.")
        sys.exit(1)
    print(f"Using Project Root: {utils._PROJECT_ROOT}")

    # 2. Identify Target Experiment ID
    #    Option A: Hardcode a recent, valid ID (simplest for now)
    #    <<< USER: CHOOSE A VALID, RECENTLY COMPLETED EXPERIMENT ID >>>
    TEST_EXPERIMENT_ID = "20250329-13" # Replace with an ID like 20250329-10, -11, -12, -13 etc.
    #    Option B: Implement logic to find best/latest ID (more complex)
    #    e.g., load completed_experiments.json, sort by accuracy/timestamp, get ID
    #    best_exp = find_best_experiment_id(...) # Requires implementing this helper
    #    if not best_exp: sys.exit("No suitable experiment found.")
    #    TEST_EXPERIMENT_ID = best_exp['experiment_id']
    # -------------------
    print(f"Testing with Experiment ID: {TEST_EXPERIMENT_ID}")

    # Construct relative paths
    model_path_rel = f"data/models/{TEST_EXPERIMENT_ID}.pt"
    metadata_path_rel = f"data/models/{TEST_EXPERIMENT_ID}_metadata.json"

    # --- Test load_model ---
    print("\nStep 1: Loading model and metadata...")
    loaded_data = load_model(model_path_rel, metadata_path_rel) # Uses relative paths

    if loaded_data is None:
        print("\n--- TEST FAILED: load_model returned None ---")
        sys.exit(1)
    else:
        print("   load_model successful.")
        print(f"   Loaded experiment: {loaded_data.get('experiment_id', 'N/A')}")

    # Extract components needed for classification
    model = loaded_data['model']
    token_dict = loaded_data['token_dict']
    label_encoder = loaded_data['label_encoder']
    known_ciphers = label_encoder.classes_ # Get list of ciphers the model knows

    # --- Test classify_text ---
    print("\nStep 2: Generating and classifying sample texts...")

    # 3. Generate Sample Texts using prep_samples
    sample_texts_to_generate = 5
    generated_samples = []
    try:
        # Get available cipher functions
        available_cipher_funcs = {f.__name__: f for f in ciphers._get_cipher_functions()}
        # Use only ciphers the loaded model was trained on
        target_cipher_funcs = [available_cipher_funcs[name] for name in known_ciphers if name in available_cipher_funcs]

        if not target_cipher_funcs:
             print("ERROR: No matching cipher functions found for model's known classes.")
             sys.exit(1)

        print(f"Generating {sample_texts_to_generate} samples using known ciphers...")
        # Generate one sample for N random known ciphers
        for _ in range(sample_texts_to_generate):
            cipher_func = random.choice(target_cipher_funcs)
            # Use generate_batches to get one sample (num_batches=1, single func)
            # Note: generate_batches expects a list of functions
            df_sample = prep_samples.generate_batches(cipher_funcs=[cipher_func], sample_length=200, num_batches=1) # Shorter sample length for testing
            if not df_sample.empty:
                 generated_samples.append({
                     "text": df_sample.iloc[0]['text'],
                     "true_cipher": df_sample.iloc[0]['cipher']
                 })

        # Add a plain English example manually
        generated_samples.append({
            "text": "this is a sample of plain english text to see how the classifier handles it",
            "true_cipher": "english" # Assuming 'english' is one of the classes
        })

    except Exception as e:
        print(f"ERROR generating samples: {e}")
        sys.exit(1)

    if not generated_samples:
         print("ERROR: Failed to generate any samples.")
         sys.exit(1)

    # 4. Classify Generated Samples
    all_passed = True
    correct_predictions = 0
    for i, sample in enumerate(generated_samples):
        text = sample["text"]
        true_label = sample["true_cipher"]
        print(f"\nInput {i+1} (True: {true_label}): \"{text[:60]}...\"")
        result = classify_text(model, text, token_dict, label_encoder)

        if "error" in result:
            print(f"   ERROR during classification: {result['error']}")
            all_passed = False
        else:
            predicted = result.get('predicted_class', 'N/A')
            confidence = result.get('confidence', 0.0)
            print(f"   Predicted: {predicted} (Confidence: {confidence:.4f})")
            if predicted == true_label:
                 correct_predictions += 1
            # print(f"   All Probabilities: {result.get('all_probabilities')}")

    # --- Summary ---
    print("-" * 30)
    if all_passed:
        print("Classification completed for all samples without errors.")
        accuracy = correct_predictions / len(generated_samples)
        print(f"Basic Accuracy on Generated Samples: {correct_predictions}/{len(generated_samples)} ({accuracy:.2%})")
        print("--- Inference Test PASSED (Functionality check) ---")
    else:
        print("--- Inference Test FAILED (Errors occurred during classification) ---")
