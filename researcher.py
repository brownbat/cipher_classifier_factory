import datetime
import json
import notifications
import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from train_lstm import train_model, get_data
from ciphers import _get_cipher_names


def plot_confusion_matrices(file_path='data/completed_experiments.yaml'):
    print('Plotting...')
    with open(file_path, 'r') as file:
        experiments = yaml.safe_load(file) or []

    for exp in experiments:
        # Check if metrics and training_time are available
        if ('metrics' in exp
                and 'training_time' in exp['metrics']
                and 'conf_matrix' in exp['metrics']):

            unique_id = f'{exp["experiment_id"]}_{exp["training_time"]}'
            gif_filename = f'data/cm/{unique_id}_conf_matrix.gif'

            if os.path.exists(gif_filename):
                print(
                    f"GIF already exists for experiment {exp['experiment_id']}")
                continue  # skip if gif exists

            # Gather info
            metrics = exp.get('metrics', {}) or {}
            conf_matrices = metrics.get('conf_matrix', [])
            cipher_names = exp['data_params'].get(
                'ciphers', _get_cipher_names())
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
                plt.xlabel(
                    f"Predicted Labels\nLoss: {loss:.4f}, "
                    + f"Accuracy: {accuracy:.4f}")
                plt.ylabel('True Labels')

                # Add experiment ID and hyperparameters as title
                title = f"Experiment: {exp['experiment_id']}"
                title += " - Epoch {epoch}/{len(conf_matrices)}\n"
                title += ', '.join([f"{key}: {value}" for key, value in hyperparams.items()])
                plt.title(title)
                
                frame_filename = f"data/cm/tmp_frame_{epoch}.png"
                plt.savefig(frame_filename, bbox_inches='tight')
                frames.append(frame_filename)
                plt.close()

            # Create a GIF from the frames
            with imageio.get_writer(
                    gif_filename,
                    mode='I',
                    duration=500,
                    loop=0) as writer:
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
            exp['cm_gif_filename'] = gif_filename

            print(f"GIF created for experiment {exp['experiment_id']}")
        else:
            print(f"No confusion matrix found for experiment {exp['experiment_id']}, skipping GIF creation")
    # Write the updated experiments data back to the file
    with open(file_path, 'w') as file:
        experiments = convert_ndarray_to_list(experiments)
        yaml.dump(experiments, file)

    print("Updated experiment data saved to file.")


def query_experiments_metrics(file_path='data/experiments.yaml'):
    with open(file_path, 'r') as file:
        experiments = yaml.safe_load(file) or []

    outstr = ""
    for exp in experiments:
        exp_id = exp.get('experiment_id', 'N/A')
        metrics = exp.get('metrics', {})
        # Assuming 5 epochs as default
        train_loss = metrics.get('train_loss', ['N/A'] * 5)
        val_accuracy = metrics.get('val_accuracy', ['N/A'] * 5)

        loss_str = ', '.join([f"{loss:.4f}" for loss in train_loss])
        acc_str = ', '.join([f"{accuracy:.4f}" for accuracy in val_accuracy])
        training_time = metrics.get('training_time', 0)
        outstr += (
            f"{exp_id}: Loss: {loss_str} | Accuracy: {acc_str}"
            + f" | Training time: {training_time}")
        outstr += "\n"
    return outstr


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, list):
        # Recursively apply to items
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        # Apply to dictionary values
        return {
            key: convert_ndarray_to_list(value) for key, value in obj.items()}
    else:
        return obj


def get_experiment_details(exp):
    '''
    Returns experiment details as a formatted string.
    '''
    details = []
    details.append("Experiment details")

    details.append(f"Experiment ID: {exp.get('experiment_id', 'N/A')}")
    if 'model_filename' in exp:
        details.append(f"Model saved as: {exp['model_filename']}")
    ciphers_used = ', '.join(exp.get('data_params', {}).get('ciphers', []))
    details.append(f"Ciphers used: {ciphers_used}")
    sample_length = exp.get('data_params', {}).get('sample_length', 'N/A')
    num_samples = exp.get('data_params', {}).get('num_samples', 'N/A')
    details.append(f"Sample length: {sample_length}, Number of Samples: {num_samples}")
    epochs = exp.get('hyperparams', {}).get('epochs', 'N/A')
    details.append(f"Epochs: {epochs}")

    metrics = exp.get('metrics', {})
    if metrics:
        final_loss = metrics.get('val_loss', ['N/A'])[-1]
        details.append(f"Final Loss: {final_loss:.4f}")
        final_accuracy = metrics.get('val_accuracy', ['N/A'])[-1]
        details.append(f"Final Accuracy: {final_loss:.4f}")
    else:
        details.append("Metrics not available.")

    return '\n'.join(details)


def run_experiments(pending_file, completed_file):
    '''
    Runs experiments from pending_experiments.yaml and writes results to
    completed_experiments.yaml
    '''
    trained_experiments = []  # List to store IDs of experiments that were trained

    with open(pending_file, 'r') as file:
        experiments = yaml.safe_load(file) or []

    for exp in experiments.copy():  # Iterate over a copy of the list
        data_params = exp.get('data_params', {})
        hyperparams = exp.get('hyperparams', {})

        # Remove the experiment from the pending list before modifying it
        experiments.remove(exp)

        # Append timestamp to the experiment ID for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Running experiment: {exp['experiment_id']}...")
        data = get_data(data_params)
        model, metrics = train_model(data, hyperparams)
        exp['metrics'] = convert_ndarray_to_list(metrics)

        # Update experiment details with results
        exp['training_time'] = timestamp
        unique_id = f'{exp["experiment_id"]}_{exp["training_time"]}'
        exp['uid'] = unique_id
        model_filename = f'data/models/{unique_id}.pt'
        exp['model_filename'] = model_filename
        torch.save(model.state_dict(), model_filename)
    
        # Append this experiment to the completed experiments file
        append_to_experiment_file(completed_file, exp)

        # Add trained experiment ID to the list
        trained_experiments.append(exp['experiment_id'])

        print(f"Experiment {exp['experiment_id']} completed.")
        print(get_experiment_details(exp))

        # Update the pending experiments file with the remaining experiments
        rewrite_experiment_file(pending_file, experiments)

    return trained_experiments



def read_experiments(file_path, exp_id=None):
    '''
    Fetches experiment results from completed_experiments.yaml and returns their details.
    '''
    experiment_details = []

    with open(file_path, 'r') as file:
        experiments = yaml.safe_load(file) or []

    for exp in experiments:
        if exp_id is not None and exp['experiment_id'] != exp_id:
            continue  # Skip if a specific experiment ID is provided and does not match

        metrics = exp.get('metrics', {})
        if not metrics:
            warning_message = f"WARNING: No metrics available for experiment {exp['experiment_id']}!"
            experiment_details.append(warning_message)
        else:
            details = get_experiment_details(exp)
            experiment_details.append(details)

    return '\n\n'.join(experiment_details)  # Join all experiment details with a double newline


def rewrite_experiment_file(file_path, experiments):
    # Convert all ndarray objects in experiments to lists
    experiments_converted = convert_ndarray_to_list(experiments)

    with open(file_path, 'w') as file:
        yaml.dump(experiments_converted, file)


def append_to_experiment_file(file_path, experiment):
    # Convert the experiment data (if it's an ndarray) to a list
    experiment_converted = convert_ndarray_to_list(experiment)

    # Read existing data from the file
    with open(file_path, 'r') as file:
        existing_data = yaml.safe_load(file) or []

    # Append the new experiment to the existing data
    existing_data.append(experiment_converted)

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        yaml.dump(existing_data, file)


def clean_up_files(
        model_directory='data/models',
        gif_directory='data/cm',
        experiments_file='data/completed_experiments.yaml',
        test_mode=True):
    with open(experiments_file, 'r') as file:
        experiments = yaml.safe_load(file) or []

    model_filenames = {
        os.path.basename(exp.get('model_filename', '')) for exp in experiments}
    gif_filenames = {
        os.path.basename(exp.get('cm_gif_filename', '')) for exp in experiments}

    files_to_delete = []

    # Check model files
    for filename in os.listdir(model_directory):
        if filename.endswith('.pt') and filename not in model_filenames:
            files_to_delete.append(os.path.join(model_directory, filename))

    # Check GIF files
    for filename in os.listdir(gif_directory):
        if filename.endswith('.gif') and filename not in gif_filenames:
            files_to_delete.append(os.path.join(gif_directory, filename))

    print("Files proposed for deletion:")
    for f in files_to_delete:
        print(f)
    if test_mode:
        print("Testing only, no files deleted")
    else:
        confirmed = input("Confirm you want to delete these files by typing 'DELETE' in all caps: ")
        if confirmed:
            for f in files_to_delete:
                os.remove(f)
        else:
            print("No files deleted")

    return files_to_delete


def main():
    # Run experiments from the pending file
    trained_experiments = run_experiments(
        'data/pending_experiments.yaml', 'data/completed_experiments.yaml')

    if not trained_experiments:
        print("No new training occurred.")
    else:
        # Fetch and print the details of newly trained experiments
        new_experiment_results = read_experiments(
            'data/completed_experiments.yaml', exp_id=trained_experiments)
        print("New Experiment Results:\n" + new_experiment_results)

        # Send a notification with the results of the new experiments
        print("new experiment results to send to discord")
        print("*****")
        print(new_experiment_results) # DEBUG
        print("*****")
        notifications.send_discord_notification(new_experiment_results)

        # Plot confusion matrices for the new experiments
        plot_confusion_matrices()
    clean_up_files()


if __name__ == "__main__":
    main()
