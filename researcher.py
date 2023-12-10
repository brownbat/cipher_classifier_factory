import json
import yaml
from train_lstm import train_model, get_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import notifications
import imageio.v2 as imageio
from ciphers import _get_cipher_names
import datetime
import torch


def plot_confusion_matrices(file_path):
    print('Plotting...')
    with open(file_path, 'r') as file:
        experiments = yaml.safe_load(file) or []

    for exp in experiments:
        # Confirm file empty
        gif_filename = f'data/cm/{exp["experiment_id"]}_conf_matrix.gif'
        if os.path.exists(gif_filename):
            print(f"GIF already exists at {gif_filename}")
            continue  # skip to next exp if this one has gif

        # Gather info
        metrics = exp.get('metrics', {}) or {}
        conf_matrices = metrics.get('conf_matrix', [])
        cipher_names = exp['data_params'].get('ciphers', _get_cipher_names())
        hyperparams = exp.get('hyperparams', {})

        # Init list of file paths for frames
        frames = []

        for epoch, cm in enumerate(conf_matrices, start=1):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=cipher_names, yticklabels=cipher_names)

            # Adding accuracy and loss at the bottom
            accuracy = metrics['val_accuracy'][epoch-1]
            loss = metrics['val_loss'][epoch-1]
            plt.xlabel(f'Predicted Labels\nLoss: {loss:.4f}, Accuracy: {accuracy:.4f}')
            plt.ylabel('True Labels')

            # Add experiment ID and hyperparameters as title
            title = f"Experiment: {exp['experiment_id']} - Epoch {epoch}/{len(conf_matrices)}\n"
            title += ', '.join([f"{key}: {value}" for key, value in hyperparams.items()])
            plt.title(title)
            
            frame_filename = f"data/cm/tmp_frame_{epoch}.png"
            plt.savefig(frame_filename, bbox_inches='tight')
            frames.append(frame_filename)
            plt.close()

        # Create a GIF from the frames
        with imageio.get_writer(gif_filename, mode='I', duration=500, loop=0) as writer:
            for frame_filename in frames:
                image = imageio.imread(frame_filename)
                writer.append_data(image)

            # Add the last frame again with a longer duration
            last_frame = imageio.imread(frames[-1])
            for i in range(3):
                writer.append_data(last_frame)

        # Remove temporary image files
        for frame_filename in frames:
            os.remove(frame_filename)

        print(f"GIF created for experiment {exp['experiment_id']}")


def query_experiments_metrics(file_path='data/experiments.yaml'):
    with open(file_path, 'r') as file:
        experiments = yaml.safe_load(file) or []

    outstr = ""
    for exp in experiments:
        exp_id = exp.get('experiment_id', 'N/A')
        metrics = exp.get('metrics', {})
        train_loss = metrics.get('train_loss', ['N/A'] * 5)  # Assuming 5 epochs as default
        val_accuracy = metrics.get('val_accuracy', ['N/A'] * 5)  # Assuming 5 epochs as default

        loss_str = ', '.join([f"{loss:.4f}" for loss in train_loss])
        acc_str = ', '.join([f"{accuracy:.4f}" for accuracy in val_accuracy])
        training_time = metrics.get('training_time', 0)
        outstr += (f"{exp_id}: Loss: {loss_str} | Accuracy: {acc_str} | Training time: {training_time}")
        outstr += "\n"
    return outstr
        


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
    if 'model_filename' in exp:
        print(f"Model saved as: {exp['model_filename']}")
    ciphers_used = ', '.join(data_params.get('ciphers', []))
    print(f"Ciphers Used: {ciphers_used}")
    print(f"Sample Length: {data_params.get('sample_length', 'N/A')}, Number of Samples: {data_params.get('num_samples', 'N/A')}")
    print(f"Epochs: {hyperparams.get('epochs', 'N/A')}")
    if metrics:
        print(f"Final Loss: {metrics.get('train_loss', ['N/A'])[-1]:.4f}")
    else:
        print("Metrics not available.")


def read_experiment(file_path):
    trained_experiments = []  # List to store IDs of experiments that were trained
    with open(file_path, 'r') as file:
        experiments = yaml.safe_load(file) or []
    
    for exp in experiments:
        data_params = exp.get('data_params', {})
        hyperparams = exp.get('hyperparams', {})
        metrics = exp.get('metrics', {})

        if not metrics:
            print(f"Running experiment: {exp['experiment_id']}...")
            data = get_data(data_params)
            model, metrics = train_model(data, hyperparams)

            # Generate model filename with experiment ID and current date
            current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f'data/models/model_{exp["experiment_id"]}_{current_date}.pt'
            torch.save(model.state_dict(), model_filename)

            # Store model filename in experiment's dictionary
            exp['metrics'] = convert_ndarray_to_list(metrics)
            exp['model_filename'] = model_filename  # Add model filename

            update_experiment_file(file_path, experiments)  # Update the YAML file
            trained_experiments.append(exp['experiment_id'])  # Add trained experiment ID

            print(f"Experiment {exp['experiment_id']} completed.")
            print_experiment_details(exp)
        else:
            print(f"Experiment {exp['experiment_id']} already run.")
            print_experiment_details(exp)

    return trained_experiments


def update_experiment_file(file_path, experiments):
    # Convert all ndarray objects in experiments to lists
    experiments_converted = convert_ndarray_to_list(experiments)

    with open(file_path, 'w') as file:
        yaml.dump(experiments_converted, file)


def main():
    trained_experiments = read_experiment('data/experiments.yaml')
    if not trained_experiments:
        print("No new training occurred.")

    metrics = "Experiment metrics:\n"
    metrics += query_experiments_metrics()
    print(metrics)

    if trained_experiments:
        notifications.send_discord_notification(metrics)
    plot_confusion_matrices('data/experiments.yaml')


if __name__ == "__main__":
    main()
