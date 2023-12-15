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

import signal
import time

# TODO duplication checks are very slow.
# look at speed throughout

# Global flag and queue for communication
should_continue = True


def safe_json_load(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        # Handle empty or invalid JSON file
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


def run_experiments(pending_file, completed_file):
    '''
    Runs experiments from pending_experiments.json and writes results to
    completed_experiments.json
    '''
    # TODO align with data structure
    trained_experiments = []  # List to store IDs of experiments that were trained

    with open(pending_file, 'r') as file:
        experiments = safe_json_load(pending_file)

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
        trained_experiments.append(exp['uid'])

        print(f"Experiment {exp['experiment_id']} completed.")
        print(get_experiment_details(exp))

        # Update the pending experiments file with the remaining experiments
        rewrite_experiment_file(pending_file, experiments)

    return trained_experiments


def run_experiment(exp):
    '''
    Runs one experiment and writes results to completed_experiments.json
    '''
    data_params = exp.get('data_params', {})
    hyperparams = exp.get('hyperparams', {})

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

    default_params = {
        'num_samples': 1000,
        'sample_length': 200,
        'epochs': 3,
        'num_layers': 10,
        'batch_size': 32,
        'embedding_dim': 32,
        'hidden_dim': 64,
        'dropout_rate': 0.015,
        'learning_rate': 0.002
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


def main():
    # for experiments with previously run keys (settings) --
    # options: SKIP (current), re-run and add, re-run and overwrite, prompt user
    # if you skip it, should you delete it from researcher.py?
    # TODO: add a key or hash of key to the exp in the file

    # add default experiments to the pending_experiments file

    global should_continue
    build_cm_gifs = False
    
    signal.signal(signal.SIGINT, signal_handler)

    params = {
            'ciphers': ['english', 'vigenere', 'caesar'],
            'num_samples': [1000,2000,3000],
            'sample_length': [200,400,600],
            'epochs': [10, 20, 30],
            'num_layers': 10,
            'batch_size': 64,
            'embedding_dim': 32,
            'hidden_dim': 64,
            'dropout_rate': 0.015,
            'learning_rate': 0.002
        }


    generate_experiments(params)
    
    # Run experiments from the pending file
    pending_experiments = get_pending_experiments()
    completed_keys = get_completed_experiment_keys()



    trained_experiments = []
    for exp in pending_experiments.copy():
        print("Press `Ctrl+C` at any time to exit cleanly after this experiment completes.")
        if not should_continue:
            print("Exit command received. Stopping further processing.")
            break
        
        exp_key = experiment_key(exp)
        if exp_key in completed_keys:
            print(f"Skipping experiment already run with key:\n{exp_key}\n")
            continue
        updated_exp = run_experiment(exp)
        experiment_details = get_experiment_details(updated_exp)
        print(experiment_details)

        notification_msg = "Training experiment completed"
        notification_msg += experiment_details
        notifications.send_discord_notification(notification_msg)

        trained_experiments.append(updated_exp['uid'])
        append_to_experiment_file(
            'data/completed_experiments.json', updated_exp)
        pending_experiments.remove(exp)
        # Update the pending experiments file with the remaining experiments
        # need to get experiments from somewhere
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
        # plot_confusion_matrices()
        pass  # do not plot, it's too slow

    print("Cleaning up files")
    clean_up_files(test_mode=False)
    # reformat_completed_experiments('data/completed_experiments.json')


if __name__ == "__main__":
    main()
