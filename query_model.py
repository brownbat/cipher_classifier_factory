"""
Inference tool for the transformer-based cipher classifier.
This tool loads the best performing models and evaluates text samples.
"""
import json
import pickle
import torch
import string
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from models.transformer.model import TransformerClassifier
from models.common.data import custom_text_tokenizer

DEFAULT_COMPLETED_FILE = 'data/completed_experiments.json'

# Sample test texts
E1 = """When the people of America reflect that they are now called upon to decide a question, which, in its consequences, must prove one of the most important that ever engaged their attention, the propriety of their taking a very comprehensive, as well as a very serious, view of it, will be evident.
"""

K1 = """EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJ
YQTQUXQBQVYUVLLTREVJYQTMKYRDMFD"""
K2 = """VFPJUDEEHZWETZYVGWHKKQETGFQJNCE
GGWHKK?DQMCPFQZDQMMIAGPFXHQRLG
TIMVMZJANQLVKQEDAGDVFRPJUNGEUNA
QZGZLECGYUXUEENJTBJLBQCRTBJDFHRR
YIZETKZEMVDUFKSJHKFWHKUWQLSZFTI
HHDDDUVH?DWKBFUFPWNTDFIYCUQZERE
EVLDKFEZMOQQJLTTUGSYQPFEUNLAVIDX
FLGGTEZ?FKZBSFDQVGOGIPUFXHHDRKF
FHQNTGPUAECNUVPDJMQCLQUMUNEDFQ
ELZZVRRGKFFVOEEXBDMVPNFQXEZLGRE
DNQFMPNZGLFLPMRJQYALMGNUVPDXVKP
DQUMEBEDMHDAFMJGZNUPLGEWJLLAETG"""
K3 = """ENDYAHROHNLSRHEOCPTEOIBIDYSHNAIA
CHTNREYULDSLLSLLNOHSNOSMRWXMNE
TPRNGATIHNRARPESLNNELEBLPIIACAE
WMTWNDITEENRAHCTENEUDRETNHAEOE
TFOLSEDTIWENHAEIOYTEYQHEENCTAYCR
EIFTBRSPAMHHEWENATAMATEGYEERLB
TEEFOASFIOTUETUAEOTOARMAEERTNRTI
BSEDDNIAAHTTMSTEWPIEROAGRIEWFEB
AECTDDHILCEIHSITEGOEAOSDDRYDLORIT
RKLMLEHAGTDHARDPNEOHMGFMFEUHE
ECDMRIPFEIMEHNLSSTTRTVDOHW?"""
K4 = """OBKR
UOXOGHULBSOLIFBBWFLRVQQPRNGKSSO
TWTQSJQSSEKZZWATJKLUDIAWINFBNYP
VTTMZFPKWGDKZXTJCDIGKUHUAUEKCAR"""

DEFAULT_TEXTS = {"english": E1, "k1": K1, "k2": K2, "k3": K3, "k4": K4}


def get_top_experiments(completed_experiments_file=DEFAULT_COMPLETED_FILE, top_n=5, sort_key='val_accuracy'):
    """
    Get the top performing experiments sorted by a specified metric.
    
    Args:
        completed_experiments_file (str): Path to the completed experiments JSON file
        top_n (int): Number of top experiments to return
        sort_key (str): Metric to sort by ('val_accuracy' or 'val_loss')
        
    Returns:
        list: UIDs of the top performing experiments
    """
    with open(completed_experiments_file, 'r') as f:
        experiments = json.load(f)

    # Warn if using a non-standard metric for sorting
    standard_metrics = ['val_accuracy', 'val_loss']
    if sort_key not in standard_metrics:
        print(f"WARNING: Sorting models by '{sort_key}' may produce unexpected results.")
        print("For quality ranking, 'val_accuracy' or 'val_loss' are recommended.")
    
    # Sorting experiments based on the last value of the specified metric
    sorted_experiments = sorted(
        experiments,
        key=lambda x: x['metrics'][sort_key][-1],
        reverse=(sort_key != 'val_loss' and sort_key != 'train_loss')  # Reverse for accuracy, not for losses
    )

    sorted_uids = [experiment['uid'] for experiment in sorted_experiments]

    return sorted_uids[:top_n]


def get_experiment(model_uid, completed_experiments_file=DEFAULT_COMPLETED_FILE):
    """
    Get experiment details by UID.
    
    Args:
        model_uid (str): Experiment UID to look up
        completed_experiments_file (str): Path to the completed experiments JSON file
        
    Returns:
        dict: Experiment details
    """
    with open(completed_experiments_file, 'r') as f:
        experiments = json.load(f)
    target_experiment = next((exp for exp in experiments if exp['uid'] == model_uid), None)

    if not target_experiment:
        raise FileNotFoundError(f"Experiment with UID {model_uid} not found.")

    return target_experiment


def load_model(model_uid, completed_experiments_file=DEFAULT_COMPLETED_FILE):
    """
    Load a transformer model and its metadata by experiment UID.
    
    Args:
        model_uid (str): Experiment UID
        completed_experiments_file (str): Path to completed experiments file
        
    Returns:
        dict: Contains model, token_dict, label_encoder, and hyperparams
    """
    # Get experiment details
    target_experiment = get_experiment(model_uid)
    
    # Get file paths
    model_filename = target_experiment.get('model_filename')
    metadata_filename = target_experiment.get('metadata_filename')
    
    if not model_filename or not metadata_filename:
        raise ValueError(f"Model or metadata filename not found for experiment {model_uid}")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU")
    
    # Load model to the appropriate device
    model = torch.load(model_filename, map_location=device)
    model.eval()
    
    # Load metadata
    with open(metadata_filename, 'rb') as f:
        metadata = pickle.load(f)
    
    # Extract necessary components
    token_dict = metadata['token_dict']
    
    # Reconstruct label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = metadata['label_encoder_classes']
    
    return {
        "model": model,
        "token_dict": token_dict,
        "label_encoder": label_encoder,
        "hyperparams": metadata['hyperparams'],
        "device": device
    }


def preprocess_text(input_text, token_dict, max_length=500):
    """
    Preprocess text for transformer model input.
    
    Args:
        input_text (str): Text to preprocess
        token_dict (dict): Token dictionary for tokenization
        max_length (int): Maximum sequence length
        
    Returns:
        torch.Tensor: Preprocessed text tensor
    """
    # Convert input_text to lowercase
    input_text = input_text.lower()
    
    # Tokenize using custom_text_tokenizer
    tokenized_text = custom_text_tokenizer(input_text, token_dict)[:max_length]
    
    # Convert to tensor and add batch dimension
    padded_text = torch.tensor(tokenized_text, dtype=torch.long).unsqueeze(0)
    
    return padded_text


def predict_with_model(model_uid, text_to_test):
    """
    Make a prediction using a transformer model.
    
    Args:
        model_uid (str): Experiment UID
        text_to_test (str): Text to classify
        
    Returns:
        str: Predicted cipher class
    """
    # Load model and metadata
    model_data = load_model(model_uid)
    model = model_data["model"]
    token_dict = model_data["token_dict"]
    label_encoder = model_data["label_encoder"]
    device = model_data["device"]
    
    # Preprocess text
    preprocessed_text = preprocess_text(text_to_test, token_dict)
    # Move input to the same device as the model
    preprocessed_text = preprocessed_text.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(preprocessed_text)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_index = output.argmax(dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    
    return predicted_label


def main():
    """Run predictions on default texts with top models."""
    print("Top performing models and their predictions:")
    top_experiments = get_top_experiments(top_n=5)
    
    for experiment_uid in top_experiments:
        print(f"\nExperiment: {experiment_uid}")
        print('-' * 40)
        
        # Get experiment details
        experiment = get_experiment(experiment_uid)
        accuracy = experiment['metrics']['val_accuracy'][-1]
        print(f"Validation accuracy: {accuracy:.4f}")
        
        # Transformer hyperparameters
        hyperparams = experiment['hyperparams']
        print(f"d_model: {hyperparams.get('d_model')}, nhead: {hyperparams.get('nhead')}")
        print(f"num_encoder_layers: {hyperparams.get('num_encoder_layers')}")
        print(f"dim_feedforward: {hyperparams.get('dim_feedforward')}")
        
        # Make predictions on default texts
        print("\nPredictions:")
        for text_key, text_value in DEFAULT_TEXTS.items():
            try:
                prediction = predict_with_model(experiment_uid, text_value)
                print(f"  {text_key}: {prediction}")
            except Exception as e:
                print(f"  {text_key}: Error - {str(e)}")
        
        print("")


if __name__ == "__main__":
    main()