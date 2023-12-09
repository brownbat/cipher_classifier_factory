import json
import yaml
from train_lstm import train_model, get_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_confusion_matrices(file_path, experiment_ids):
    with open(file_path, 'r') as file:
        experiments = yaml.safe_load(file) or []

    for exp in experiments:
        if exp['experiment_id'] in experiment_ids:
            metrics = exp.get('metrics', {}) or {}
            conf_matrices = metrics.get('conf_matrix', [])
            
            for epoch, cm in enumerate(conf_matrices, start=1):
                # print(cm)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - Epoch {epoch}')
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
                plt.show()


def query_experiments_metrics(file_path='data/experiments.yaml'):
    with open(file_path, 'r') as file:
        experiments = yaml.safe_load(file) or []

    for exp in experiments:
        exp_id = exp.get('experiment_id', 'N/A')
        metrics = exp.get('metrics', {})
        train_loss = metrics.get('train_loss', ['N/A'] * 5)  # Assuming 5 epochs as default
        val_accuracy = metrics.get('val_accuracy', ['N/A'] * 5)  # Assuming 5 epochs as default

        loss_str = ', '.join([f"{loss:.4f}" for loss in train_loss])
        acc_str = ', '.join([f"{accuracy:.4f}" for accuracy in val_accuracy])
        print(f"{exp_id}: Loss: {loss_str} | Accuracy: {acc_str}")


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

    print("Experiment details")
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
    experiments = read_experiment('data/experiments.yaml')
    if not experiments:
        print("No experiments found.")
    experiments = [exp['experiment_id'] for exp in experiments]
    plot_confusion_matrices('data/experiments.yaml', experiments)

    print("Experiment metrics:")
    query_experiments_metrics()


if __name__ == "__main__":
    main()
