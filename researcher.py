import json
import yaml
from train_lstm import train_model, get_data
import numpy as np


'''
a test model specified with:
        hyperparams = {
            'epochs': 3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'embedding_dim': 128,
            'hidden_dim': 128,
            'activation_func': 'relu'
        }
in train_model we have
    # Extract hyperparameters
    epochs = hyperparams.get('epochs', 5)
    learning_rate = hyperparams.get('learning_rate', 0.001)
    batch_size = hyperparams.get('batch_size', BATCH_SIZE)
    embedding_dim = hyperparams.get('embedding_dim', EMBEDDING_DIM)
    hidden_dim = hyperparams.get('hidden_dim', HIDDEN_DIM)
    vocab_size = hyperparams.get('vocab_size', VOCAB_SIZE)
    num_classes = len(np.unique(data['cipher']))
    activation_name = hyperparams.get('activation_func', None)
    activation_func = get_activation_function(activation_name)

no loss function, assuming crossentropy would be all we'd need.
we can add it later. we'll use yaml and group things as data parameters, hyperparametercs, and performance metrics.
some performance metrics should be grouped by epoch.

we currently collect
     training_metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'conf_matrix': []}
but could do more.

for data params, we have 
    # Example test call
    test_cipher_names = ["caesar", "vigenere"]  # Replace with available ciphers
    test_num_samples = 2500
    test_sample_length = 500

    try:
        test_filename, generated = manage_sample_data(test_cipher_names, test_num_samples, test_sample_length)
'''


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]  # Recursively apply to items
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}  # Apply to dictionary values
    else:
        return obj


def print_experiment_details(exp):
    data_params = exp.get('data_params', {})
    hyperparams = exp.get('hyperparams', {})
    metrics = exp.get('metrics', {})

    print(f"Experiment ID: {exp.get('experiment_id', 'N/A')}")
    ciphers_used = ', '.join(data_params.get('ciphers', []))
    print(f"Ciphers Used: {ciphers_used}")
    print(f"Sample Length: {data_params.get('sample_length', 'N/A')}, Number of Samples: {data_params.get('num_samples', 'N/A')}")
    print(f"Epochs: {hyperparams.get('epochs', 'N/A')}")
    if metrics:
        print(f"Final Loss: {metrics.get('train_loss', ['N/A'])[-1]:.4f}")
    else:
        print("Metrics not available.")


def read_experiment(file_path, exp_id=None):
    with open(file_path, 'r') as file:
        experiments = yaml.safe_load(file) or []
    
    for exp in experiments:
        if exp_id is None or exp['experiment_id'] == exp_id:
            data_params = exp.get('data_params', {})  # Define data_params here
            hyperparams = exp.get('hyperparams', {})
            metrics = exp.get('metrics', {})

            if not metrics:
                print(f"Running experiment: {exp['experiment_id']}...")
                data = get_data(data_params)  # Correctly use data_params
                _, metrics = train_model(data, hyperparams)
                exp['metrics'] = convert_ndarray_to_list(metrics)
                update_experiment_file(file_path, experiments)

                print(f"Experiment {exp['experiment_id']} completed.")
                print_experiment_details(exp)
            else:
                print(f"Experiment {exp['experiment_id']} already run.")
                print_experiment_details(exp)

            if exp_id is not None:
                return exp

    return None if exp_id else experiments


def update_experiment_file(file_path, experiments):
    # Convert all ndarray objects in experiments to lists
    experiments_converted = convert_ndarray_to_list(experiments)

    with open(file_path, 'w') as file:
        yaml.dump(experiments_converted, file)

def main():
    experiment_details = read_experiment('data/experiments.yaml')
    if not experiment_details:
        print("Experiment not found.")

if __name__ == "__main__":
    main()
