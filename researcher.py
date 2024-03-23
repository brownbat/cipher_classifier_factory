import datetime
import itertools
import json
import notifications
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from train_lstm import train_model, get_data
from ciphers import _get_cipher_names
import torch

import signal
import time
import argparse

# look at speed throughout
# todo -- add additional ciphers, such as ADFGVX, trifid, VIC, enigma, railfence
# move from LSTM to transformers

# TODO -- INVESTIGATE accuracy, overheating, crashes

#   LOW ACCURACY
# why is {'epochs': 30, 'num_layers': 32, 'batch_size': 256, 'embedding_dim': 64, 'hidden_dim': 192, 'dropout_rate': 0.2, 'learning_rate': 0.001}
# so awful compared to neighbors? random bad luck on dropout? re-run
# it only had 55.7 accuracy, but we get 
# 96 accuracy from {'epochs': 30, 'num_layers': 32, 'batch_size': 256, 'embedding_dim': 64, 'hidden_dim': 192, 'dropout_rate': 0.1, 'learning_rate': 0.003}

# add option to run same experiment multiple times, generating fresh data or using consistent seed

#   OVERHEATING
# Overheating with {'epochs': 60, 'num_layers': 128, 'batch_size': 128, 'embedding_dim': 64, 'hidden_dim': 512, 'dropout_rate': 0.3, 'learning_rate': 0.004}
# No overheating with {'epochs': 45, 'num_layers': 64, 'batch_size': 64, 'embedding_dim': 32, 'hidden_dim': 256, 'dropout_rate': 0.3, 'learning_rate': 0.003}
# overheating with num_samples 1,000,000, not with 10,000
# overheating seems tied to complexity, especially hidden_dim, suggesting vram issue - file bug?

#   CRASHES
# crashes at 100000 samples, batch 256, embedding 128, hidden 256


# set file location as working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Global flag and queue for communication
should_continue = True

# set these parameters with alternatives to run combination of all alternatives
default_params = {
    'ciphers': [_get_cipher_names()],
    'num_samples': [10000],
    'sample_length': [500],
    'epochs': [30],
    'num_layers': [64, 128],
    'batch_size': [128],
    'embedding_dim': [64],
    'hidden_dim': [64, 128],
    'dropout_rate': [0.2],
    'learning_rate': [0.004]
}

def safe_json_load(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        # Handle empty or invalid JSON file
        print(f"WARNING: {file_path} is empty or invalid. An empty list will be used.")
        return []
    except FileNotFoundError:
        # Handle file not found and create a new empty file
        print(f"WARNING: {file_path} not found. Creating a new file.")
        with open(file_path, 'w') as file:
            json.dump([], file)
        return []


def signal_handler(sig, frame):
    global should_continue
    print('Ctrl+C pressed, preparing to exit...')
    should_continue = False


def plot_confusion_matrices(file_path='data/completed_experiments.json'):
    print('Plotting...')
    experiments = safe_json_load(file_path)

    for exp in experiments:
        # TODO: This is too slow for large datasets. consider instead reworking
        # to call with a specific UID in mind. cm_to_gif(uid, filename) or sim
        # Check if metrics and training_time are available
        if ('metrics' in exp and 'training_time' in exp
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
                title += f" - Epoch {epoch}/{len(conf_matrices)}\n"
                title += ', '.join([f"{key}: {value}" for key, value in hyperparams.items()])
                plt.title(title)
                
                frame_filename = f"data/cm/tmp_frame_{epoch}.png"
                frame_dir = os.path.dirname(frame_filename)
                os.makedirs(frame_dir, exist_ok=True)  # make directory if it does not exist
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
        json.dump(experiments, file, indent=4)

    print("Updated experiment data saved to file.")


def query_experiments_metrics(file_path='data/experiments.json'):
    # TODO ensure compatible with current data model
    experiments = safe_json_load(file_path)

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
    # TODO ensure compatible with current data model
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
        details.append(f"Final Accuracy: {final_accuracy:.4f}")
    else:
        details.append("Metrics not available.")

    return '\n'.join(details)


def run_experiment(exp):
    '''
    Runs one experiment and writes results to completed_experiments.json
    '''
    data_params = exp.get('data_params', {})
    hyperparams = exp.get('hyperparams', {})

    if 'num_samples' not in data_params:
        raise ValueError(f"'num_samples' is missing in data_params for experiment {exp.get('experiment_id')}")

    # Append timestamp to the experiment ID for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data = get_data(data_params)
    
    
    print(f"Running experiment: {exp['experiment_id']}...")
    model, metrics = train_model(data, hyperparams)
    exp['metrics'] = convert_ndarray_to_list(metrics)

    # Update experiment details with results
    exp['training_time'] = timestamp
    unique_id = f'{exp["experiment_id"]}_{exp["training_time"]}'
    exp['uid'] = unique_id
    print(f"Experiment {exp['experiment_id']} completed")
    print(f"  now called: {exp['uid']}")
    
    model_filename = f'data/models/{unique_id}.pt'
    model_dir = os.path.dirname(model_filename)
    os.makedirs(model_dir, exist_ok=True) # create directory if it doesn't exist
    exp['model_filename'] = model_filename
    torch.save(model.state_dict(), model_filename)

    return exp


def read_experiments(file_path, exp_uid=[]):
    '''
    Fetches experiment results from completed_experiments.json and returns their details.
    '''
    experiment_details = []

    experiments = safe_json_load(file_path)

    for exp in experiments:
        # report on matching experiments, or ALL if exp_uid is []
        if exp_uid == [] or exp['uid'] in exp_uid:  
            metrics = exp.get('metrics', {})
            if not metrics:
                warning_message = f"WARNING: No metrics available for experiment {exp['experiment_id']}!"
                experiment_details.append(warning_message)
            else:
                details = get_experiment_details(exp)
                experiment_details.append(details)

    return '\n\n'.join(experiment_details)  # Join all experiment details with a double newline


def rewrite_experiment_file(file_path, experiments):
    #  Convert all ndarray objects in experiments to lists
    experiments_converted = convert_ndarray_to_list(experiments)

    with open(file_path, 'w') as file:
        json.dump(experiments_converted, file)


def append_to_experiment_file(file_path, experiment):
    # Convert the experiment data (if it's an ndarray) to a list
    experiment_converted = convert_ndarray_to_list(experiment)

    # Read existing data from the file
    existing_data = safe_json_load(file_path)

    # Append the new experiment to the existing data
    existing_data.append(experiment_converted)

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file)


def clean_up_files(
        model_directory='data/models',
        gif_directory='data/cm',
        experiments_file='data/completed_experiments.json',
        test_mode=True):
    experiments = safe_json_load(experiments_file)

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
        # confirmed = input("Confirm you want to delete these files by typing 'DELETE' in all caps: ")
        confirmed = True
        if confirmed and files_to_delete:
            for f in files_to_delete:
                os.remove(f)
            print("All files deleted")
        else:
            print("No files deleted")

    return files_to_delete


def get_pending_experiments(pending_file='data/pending_experiments.json'):
    with open(pending_file, 'r') as file:
        return safe_json_load(pending_file)


def reformat_completed_experiments(file_path):
    experiments = safe_json_load(file_path)

    reformatted_experiments = []

    for exp in experiments:
        uid = exp.get('uid', 'unknown_uid')
        reformatted_experiment = {uid: exp}
        reformatted_experiments.append(reformatted_experiment)

    with open(file_path, 'w') as file:
        json.dump(reformatted_experiments, file)


def experiment_key(experiment):
    data_params = experiment.get('data_params', {})
    hyperparams = experiment.get('hyperparams', {})

    key_parts = [
        json.dumps(data_params, sort_keys=True),
        json.dumps(hyperparams, sort_keys=True)
    ]

    return '|'.join(key_parts)


def experiment_exists(key, pending_file, completed_file):
    # Check in pending experiments
    with open(pending_file, 'r') as file:
        pending_experiments = safe_json_load(pending_file)
        for exp in pending_experiments:
            if experiment_key(exp) == key:
                return "PENDING"
    # Check in completed experiments
    with open(completed_file, 'r') as file:
        completed_experiments = safe_json_load(completed_file)
        for exp in completed_experiments:
            if experiment_key(exp) == key:
                return "COMPLETED"
    return False


def get_completed_experiment_keys(file_path="data/completed_experiments.json"):
    completed_experiment_keys = set()

    experiments = safe_json_load(file_path)

    for exp in experiments:
        key = experiment_key(exp)
        completed_experiment_keys.add(key)

    return completed_experiment_keys


def reset_pending_experiments(file_path="data/pending_experiments.json"):
    default_experiment = {
        'data_params': {
            'ciphers': ['english', 'vigenere', 'columnar_transposition'],
            'num_samples': 1000,
            'sample_length': 200
        },
        'experiment_id': 'exp_1',
        'hyperparams': {
            'batch_size': 32,
            'dropout_rate': 0.015,
            'embedding_dim': 32,
            'epochs': 3,
            'hidden_dim': 64,
            'learning_rate': 0.003,
            'num_layers': 10
        }
    }

    with open(file_path, 'w') as file:
        json.dump([default_experiment], file)


def load_experiment_keys(file_path):
    experiments = safe_json_load(file_path)
    return {experiment_key(exp) for exp in experiments}


def generate_experiments(settings={}, pending_file='data/pending_experiments.json', completed_file='data/completed_experiments.json'):
    # TODO - consider hashing the experiment_key and using it in a hash table to
    # speed lookups for experiment collisions
    # TODO - responsibly overwrite an old experiment if rerun=True 
    print("Generating new experiments to run based off of settings:")
    print(settings)
    print()

    # TODO: change defaults to lists to avoid nonlist check
    default_params = {
        'num_samples': 1000,
        'sample_length': 200,
        'epochs': 3,
        'num_layers': 10,
        'batch_size': 32,
        'embedding_dim': 32,
        'hidden_dim': 64,
        'dropout_rate': 0.015,
        'learning_rate': 0.002,
        'ciphers': ['english', 'vigenere', 'caesar', 'columnar_transposition', 'random_noise']
    }
    default_params.update(settings)
    settings.update(default_params)
    ciphers = settings.pop('ciphers')
    if not isinstance(ciphers[0], list):
        ciphers = [ciphers]

    # Separate list and non-list parameters
    list_params = {k: v for k, v in settings.items() if isinstance(v, (list, tuple))}
    non_list_params = {k: v for k, v in settings.items() if not isinstance(v, (list, tuple))}

    if list_params:
        # Compute all combinations of list parameters
        keys, values = zip(*list_params.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        print("No variable parameters found. Adding a single experiment.")
        combinations = [{}]

    experiment_id_counter = 1
    new_experiments = []

    for cipher_set in ciphers:
        for combination in combinations:
            experiment_params = {**non_list_params, **combination}
            experiment_params['ciphers'] = list(cipher_set)
            
            experiment = {
                'data_params': {
                    'ciphers': experiment_params.pop('ciphers'),
                    'num_samples': experiment_params.pop('num_samples'),
                    'sample_length': experiment_params.pop('sample_length')
                },
                'hyperparams': experiment_params,
                'experiment_id': f'exp_{experiment_id_counter}'
            }
            new_experiments.append(experiment)
            experiment_id_counter += 1


    # Check for duplicates
    pending_keys = load_experiment_keys(pending_file)
    completed_keys = load_experiment_keys(completed_file)

    experiments_to_add = [exp for exp in new_experiments if experiment_key(exp) not in pending_keys and experiment_key(exp) not in completed_keys]
    
    print(f"Considering {len(new_experiments)} experiments for duplicates.")
    print(f"{len(experiments_to_add)} new experiments to add.")

    # Add new experiments to pending file
    if experiments_to_add:
        existing_pending_experiments = safe_json_load(pending_file)
        existing_pending_experiments.extend(experiments_to_add)
        with open(pending_file, 'w') as file:
            json.dump(existing_pending_experiments, file)
    else:
        print("No new experiments to add.")
    pending_keys = load_experiment_keys(pending_file)
    print(f"{len(pending_keys)} total experiments to run.")


def argument_parser(default_params):
    parser = argparse.ArgumentParser(description="Generate LSTM models with various configurations.")
    parser.add_argument('--ciphers', nargs='*', default='all', help="List of ciphers to use, such as 'vigenere' or 'caesar', or 'all' for all ciphers.")
    parser.add_argument('--samples', nargs='*', type=int, default=None, help="Number of samples to generate.")
    parser.add_argument('--sample_length', nargs='*', type=int, default=None, help="Length of samples to generate.")
    parser.add_argument('--epochs', nargs='*', type=int, default=None, help="Number of epochs to train.")
    parser.add_argument('--layers', nargs='*', type=int, default=None, help="Number of layers.")
    parser.add_argument('--batch_size', nargs='*', type=int, default=None, help="Batch size.")
    parser.add_argument('--embedding_dimensions', nargs='*', type=int, default=None, help="Embedding dimensions.")
    parser.add_argument('--hidden_dimensions', nargs='*', type=int, default=None, help="Hidden dimensions.")
    parser.add_argument('--dropout_rate', nargs='*', type=float, default=None, help="Dropout rate.")  # Changed to float for rates
    parser.add_argument('--learning_rate', nargs='*', type=float, default=None, help="Learning rate.")  # Changed to float for rates

    args = parser.parse_args()

    # Copy default_params to avoid modifying the original
    params = default_params.copy()

    # Update params with command line arguments if provided
    if args.ciphers is not None:
        if args.ciphers == 'all':
            params['ciphers'] = [_get_cipher_names()]
        else:
            params['ciphers'] = args.ciphers
    if args.samples is not None:
        params['num_samples'] = args.samples
    if args.sample_length is not None:
        params['sample_length'] = args.sample_length
    if args.epochs is not None:
        params['epochs'] = args.epochs
    if args.layers is not None:
        params['num_layers'] = args.layers
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.embedding_dimensions is not None:
        params['embedding_dim'] = args.embedding_dimensions
    if args.hidden_dimensions is not None:
        params['hidden_dim'] = args.hidden_dimensions
    if args.dropout_rate is not None:
        params['dropout_rate'] = args.dropout_rate
    if args.learning_rate is not None:
        params['learning_rate'] = args.learning_rate

    return params


def main():
    global should_continue
    global default_params
    build_cm_gifs = True
    params = argument_parser(default_params)

    print(f"Using torch version {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is available")
    else:
        print(f"CUDA is not available")
    print("Current CUDA device:")
    print(torch.cuda.get_device_name(0))

    signal.signal(signal.SIGINT, signal_handler)
    generate_experiments(params)

    # Run experiments from the pending file
    pending_experiments = get_pending_experiments()
    completed_keys = get_completed_experiment_keys()

    trained_experiments = []
    for exp in pending_experiments.copy():
        # Count pending and completed experiments
        num_pending = len(get_pending_experiments())
        num_completed = len(get_completed_experiment_keys())

        print("Press `Ctrl+C` at any time to exit cleanly after this experiment completes.")
        if not should_continue:
            print("Exit command received. Stopping further processing.")
            break

        exp_key = experiment_key(exp)
        if exp_key in completed_keys:
            print(f"Experiment with key {exp_key} already completed. Removing from pending list.")
            pending_experiments.remove(exp)
            continue
            
        updated_exp = run_experiment(exp)
        experiment_details = get_experiment_details(updated_exp)
        experiment_details += f"\n{num_completed} experiments completed, {num_pending} remaining\n"
        print("***")
        print(experiment_details)
        print("EXP:")
        print(exp['hyperparams'])
        print("***")

        notification_msg = "Training experiment completed"
        notification_msg += experiment_details
        notifications.send_discord_notification(notification_msg)

        trained_experiments.append(updated_exp['uid'])
        append_to_experiment_file(
            'data/completed_experiments.json', updated_exp)
        pending_experiments.remove(exp)
        # Update the pending experiments file with the remaining experiments
        rewrite_experiment_file(
            'data/pending_experiments.json', pending_experiments)

    if not trained_experiments:
        print("No new training occurred.")
    else:
        # Notification training batch is complete
        notifications.send_discord_notification(
            "Pending experiments batch complete")

    if not should_continue:
        print("Skipping building confusion matrix .gif files...")
    elif build_cm_gifs:
        # Plot confusion matrices for the new experiments
        # TODO: just plot for the best performer of the batch?
        plot_confusion_matrices()


    print("Cleaning up files")
    clean_up_files(test_mode=False)
    # reformat_completed_experiments('data/completed_experiments.json')


if __name__ == "__main__":
    main()
