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
import torch
import signal
import sys
import time
import argparse

# Import from our new modular structure
from models import train_model, get_data
from models.common.utils import safe_json_load, convert_ndarray_to_list
from ciphers import _get_cipher_names



# set file location as working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Global flag and queue for communication
should_continue = True

# set these parameters with alternatives to run combination of all alternatives
default_params = {
    'ciphers': [_get_cipher_names()],
    'num_samples': [100000],
    'sample_length': [500],
    'epochs': [30],
    # Transformer hyperparameters
    'd_model': [128, 256],  # Embedding dimension
    'nhead': [4, 8],  # Number of attention heads
    'num_encoder_layers': [2, 4],  # Number of transformer layers
    'dim_feedforward': [512, 1024],  # Hidden dimension in feed forward network
    'batch_size': [32, 64],
    'dropout_rate': [0.1, 0.2],
    'learning_rate': [1e-4, 3e-4]  # Lower learning rates for transformers
}


# Using safe_json_load from models.common.utils


def signal_handler(sig, frame):
    """
    Handle Ctrl+C to immediately save checkpoint and exit
    """
    global should_continue
    print('\nCtrl+C pressed. Saving checkpoint and exiting...')
    print('Use `python researcher.py` later to resume from checkpoint.')
    should_continue = False
    
    # Keep only the latest checkpoint for incomplete experiment
    from models.transformer.train import clean_old_checkpoints
    checkpoint_dir = os.path.join("data", "checkpoints")
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('_latest.pt'):
                experiment_id = filename.split('_latest.pt')[0]
                clean_old_checkpoints(experiment_id, keep_n=1)
                break
    
    # Exit with a clean status code
    # Training code will save checkpoint before exiting
    sys.exit(0)


def plot_confusion_matrices(file_path='data/completed_experiments.json'):
    print('Plotting...')
    experiments = safe_json_load(file_path)

    for exp in experiments:
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


# Using convert_ndarray_to_list from models.common.utils


def get_experiment_details(exp):
    '''
    Returns experiment details as a formatted string.
    '''
    details = []
    details.append("Experiment details")

    details.append(f"Experiment ID: {exp.get('experiment_id', 'N/A')}")
    if 'model_filename' in exp:
        details.append(f"Model saved as: {exp['model_filename']}")
    data_params = exp.get('data_params', {})
    ciphers_used = ', '.join(data_params.get('ciphers', []))
    sample_length = data_params.get('sample_length', 'N/A')
    num_samples = data_params.get('num_samples', 'N/A')

    details.append(f"Ciphers used: {ciphers_used}")
    details.append(f"Sample length: {sample_length}, Number of Samples: {num_samples}")

    hyperparams = exp.get('hyperparams', {})
    epochs = hyperparams.get('epochs', 'N/A')
    batch_size = hyperparams.get('batch_size', 'N/A')
    dropout_rate = hyperparams.get('dropout_rate', 'N/A')
    learning_rate = hyperparams.get('learning_rate', 'N/A')

    details.append(f"Epochs: {epochs}")
    details.append(f"Batch size: {batch_size}")
    details.append(f"Dropout rate: {dropout_rate}")
    details.append(f"Learning rate: {learning_rate}")
    
    # Transformer hyperparameters
    d_model = hyperparams.get('d_model', 'N/A')
    nhead = hyperparams.get('nhead', 'N/A')
    num_encoder_layers = hyperparams.get('num_encoder_layers', 'N/A')
    dim_feedforward = hyperparams.get('dim_feedforward', 'N/A')
    
    details.append(f"Model dimension: {d_model}")
    details.append(f"Attention heads: {nhead}")
    details.append(f"Encoder layers: {num_encoder_layers}")
    details.append(f"Feedforward dim: {dim_feedforward}")

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

    # Get timestamp for recording when the experiment actually ran
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data = get_data(data_params)
    
    # Store the timestamp but use the original experiment ID
    # (which is already unique with our new date-based naming convention)
    exp['uid'] = exp["experiment_id"]  # Use the experiment ID as the UID
    exp['run_timestamp'] = timestamp   # Just record when it was run
    
    # Add experiment ID to hyperparams for checkpointing
    hyperparams['experiment_id'] = exp["experiment_id"]

    print(f"Running experiment: {exp['experiment_id']}...")
    model, metrics, model_metadata = train_model(data, hyperparams)
    exp['metrics'] = convert_ndarray_to_list(metrics)

    # Update experiment details with results
    exp['training_time'] = timestamp
    print(f"Experiment {exp['experiment_id']} completed")

    # Save the model - using experiment ID which is already unique
    model_filename = f'data/models/{exp["experiment_id"]}.pt'
    model_dir = os.path.dirname(model_filename)
    os.makedirs(model_dir, exist_ok=True)  # create directory if it doesn't exist
    exp['model_filename'] = model_filename
    torch.save(model, model_filename)
    
    # Save model metadata
    metadata_filename = f'data/models/{exp["experiment_id"]}_metadata.pkl'
    exp['metadata_filename'] = metadata_filename
    with open(metadata_filename, 'wb') as f:
        import pickle
        pickle.dump(model_metadata, f)

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
        checkpoint_directory='data/checkpoints',
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

    # Limit checkpoint files by experiment
    if os.path.exists(checkpoint_directory):
        experiment_checkpoints = {}
        # Group checkpoints by experiment ID
        for filename in os.listdir(checkpoint_directory):
            if filename.endswith('.pt') and '_epoch_' in filename and not filename.endswith('_latest.pt'):
                exp_id = filename.split('_epoch_')[0]
                if exp_id not in experiment_checkpoints:
                    experiment_checkpoints[exp_id] = []
                experiment_checkpoints[exp_id].append(filename)
        
        # For each experiment, keep only the 3 newest checkpoints
        for exp_id, checkpoints in experiment_checkpoints.items():
            if len(checkpoints) > 3:
                # Sort by modification time (newest first)
                sorted_checkpoints = sorted(checkpoints, 
                                          key=lambda x: os.path.getmtime(os.path.join(checkpoint_directory, x)),
                                          reverse=True)
                # Mark older checkpoints for deletion
                for old_checkpoint in sorted_checkpoints[3:]:
                    files_to_delete.append(os.path.join(checkpoint_directory, old_checkpoint))

    print(f"Files proposed for deletion: {len(files_to_delete)}")
    if files_to_delete:
        print(f"Including {sum(1 for f in files_to_delete if 'checkpoints' in f)} checkpoint files")
    
    if test_mode:
        print("Testing only, no files deleted")
    else:
        # confirmed = input("Confirm you want to delete these files by typing 'DELETE' in all caps: ")
        confirmed = True
        if confirmed and files_to_delete:
            for f in files_to_delete:
                os.remove(f)
            print(f"Deleted {len(files_to_delete)} files")
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
            'dropout_rate': 0.1,
            'epochs': 3,
            'd_model': 128,
            'nhead': 4,
            'num_encoder_layers': 2,
            'dim_feedforward': 512,
            'learning_rate': 1e-4
        }
    }

    with open(file_path, 'w') as file:
        json.dump([default_experiment], file)


def load_experiment_keys(file_path):
    experiments = safe_json_load(file_path)
    return {experiment_key(exp) for exp in experiments}


def generate_experiments(settings={}, pending_file='data/pending_experiments.json', completed_file='data/completed_experiments.json'):
    """
    This function has been removed. Please use manage_queue.py instead.
    """
    print("⚠️  Queue management has been moved to manage_queue.py")
    print("To add experiments to the queue, use:")
    print("  python manage_queue.py --d_model 128,256 --nhead 4,8")
    print("For more options run: python manage_queue.py --help")
    print("")
    
    # Show current queue status
    pending_experiments = safe_json_load(pending_file)
    print(f"Current queue has {len(pending_experiments)} experiments.")
    
    return  # Early return - don't actually generate anything

    default_params = {
        'num_samples': 10000,
        'sample_length': 200,
        'epochs': 3,
        'batch_size': 32,
        'dropout_rate': 0.1,
        'learning_rate': 1e-4,
        'ciphers': ['english', 'vigenere', 'caesar', 'columnar_transposition', 'random_noise'],
        # Transformer-specific defaults
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 2,
        'dim_feedforward': 512,
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


def argument_parser():
    parser = argparse.ArgumentParser(description="Generate transformer models with various configurations.")
    
    # Execution mode
    parser.add_argument('--immediate', action='store_true', help="Run a single experiment immediately, bypassing the queue.")
    
    # Experiment parameters
    parser.add_argument('--ciphers', nargs='*', default='all', help="List of ciphers to use, such as 'vigenere' or 'caesar', or 'all' for all ciphers.")
    parser.add_argument('--num_samples', nargs='*', type=int, default=None, help="Number of samples to generate.")
    parser.add_argument('--sample_length', nargs='*', type=int, default=None, help="Length of samples to generate.")
    parser.add_argument('--epochs', nargs='*', type=int, default=None, help="Number of epochs to train.")
    
    # Transformer parameters
    parser.add_argument('--d_model', nargs='*', type=int, default=None, help="Model dimension.")
    parser.add_argument('--nhead', nargs='*', type=int, default=None, help="Number of attention heads.")
    parser.add_argument('--num_encoder_layers', nargs='*', type=int, default=None, help="Number of encoder layers.")
    parser.add_argument('--dim_feedforward', nargs='*', type=int, default=None, help="Dimension of feedforward network.")
    
    # Other parameters
    parser.add_argument('--batch_size', nargs='*', type=int, default=None, help="Batch size.")
    parser.add_argument('--dropout_rate', nargs='*', type=float, default=None, help="Dropout rate.")
    parser.add_argument('--learning_rate', nargs='*', type=float, default=None, help="Learning rate.")

    return parser.parse_args()


def process_args(args, default_params):
    """Process command line arguments into parameters dictionary"""
    # Copy default_params to avoid modifying the original
    params = default_params.copy()

    # Update params with command line arguments if provided
    if args.ciphers is not None:
        if args.ciphers == 'all':
            params['ciphers'] = [_get_cipher_names()]
        else:
            params['ciphers'] = args.ciphers
    if args.num_samples is not None:
        params['num_samples'] = args.num_samples
    if args.sample_length is not None:
        params['sample_length'] = args.sample_length
    if args.epochs is not None:
        params['epochs'] = args.epochs
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.dropout_rate is not None:
        params['dropout_rate'] = args.dropout_rate
    if args.learning_rate is not None:
        params['learning_rate'] = args.learning_rate
    
    # Transformer parameters
    if args.d_model is not None:
        params['d_model'] = args.d_model
    if args.nhead is not None:
        params['nhead'] = args.nhead
    if args.num_encoder_layers is not None:
        params['num_encoder_layers'] = args.num_encoder_layers
    if args.dim_feedforward is not None:
        params['dim_feedforward'] = args.dim_feedforward

    return params


def create_single_experiment(params):
    """Create a single experiment from parameters without permutation."""
    # Convert lists to first value if needed
    single_params = {}
    for key, value in params.items():
        if isinstance(value, list) and len(value) > 0:
            single_params[key] = value[0] 
        else:
            single_params[key] = value
    
    # Create experiment structure
    data_params = {
        'ciphers': [single_params['ciphers']] if isinstance(single_params['ciphers'], list) else single_params['ciphers'],
        'num_samples': single_params['num_samples'],
        'sample_length': single_params['sample_length']
    }
    
    hyperparams = {
        'epochs': single_params['epochs'],
        'd_model': single_params['d_model'],
        'nhead': single_params['nhead'],
        'num_encoder_layers': single_params['num_encoder_layers'],
        'dim_feedforward': single_params['dim_feedforward'],
        'batch_size': single_params['batch_size'],
        'dropout_rate': single_params['dropout_rate'],
        'learning_rate': single_params['learning_rate']
    }
    
    experiment = {
        'experiment_id': 'immediate_exp',
        'data_params': data_params,
        'hyperparams': hyperparams,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return experiment


def main():
    global should_continue
    global default_params
    build_cm_gifs = True
    
    # Parse arguments
    args = argument_parser()
    params = process_args(args, default_params)

    print(f"Using torch version {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is available")
    else:
        print(f"CUDA is not available")
    print("Current CUDA device:")
    print(torch.cuda.get_device_name(0))

    # Register signal handlers for both Ctrl+C and kill commands
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Handle immediate experiment mode
    if args.immediate:
        print("Running a single experiment immediately (bypassing queue)...")
        experiment = create_single_experiment(params)
        print(f"Experiment parameters:")
        print(f"- d_model: {experiment['hyperparams']['d_model']}")
        print(f"- nhead: {experiment['hyperparams']['nhead']}")
        print(f"- num_encoder_layers: {experiment['hyperparams']['num_encoder_layers']}")
        print(f"- dim_feedforward: {experiment['hyperparams']['dim_feedforward']}")
        print(f"- batch_size: {experiment['hyperparams']['batch_size']}")
        print(f"- dropout_rate: {experiment['hyperparams']['dropout_rate']}")
        print(f"- learning_rate: {experiment['hyperparams']['learning_rate']}")
        
        updated_exp = run_experiment(experiment)
        experiment_details = get_experiment_details(updated_exp)
        print("***")
        print(experiment_details)
        print("***")
        
        # Save the experiment as completed
        append_to_experiment_file('data/completed_experiments.json', updated_exp)
        return
    
    # Check for existing checkpoint files (before starting)
    checkpoint_dir = os.path.join("data", "checkpoints")
    if os.path.exists(checkpoint_dir):
        existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if existing_checkpoints:
            # Count unique experiment IDs
            exp_ids = set()
            for chk in existing_checkpoints:
                if '_epoch_' in chk:
                    exp_id = chk.split('_epoch_')[0]
                    exp_ids.add(exp_id)
            
            print(f"Found {len(existing_checkpoints)} checkpoints for {len(exp_ids)} experiments")
            print("Checkpoints will be automatically loaded when running experiments")
    
    # Normal queue processing mode - remove the generate_experiments call
    # Just check the pending experiments
    pending_experiments = get_pending_experiments()
    completed_keys = get_completed_experiment_keys()
    
    # Check if queue is empty and provide guidance
    if not pending_experiments:
        print("\n📋 The experiment queue is empty.")
        print("To add experiments to the queue, use the manage_queue.py tool:")
        print("  python manage_queue.py --d_model 128,256 --nhead 4,8")
        print("  python manage_queue.py --replace --epochs 30 --batch_size 32,64")
        print("  python manage_queue.py --list")
        print("  python manage_queue.py --clear")
        print("\nOr run a single experiment immediately:")
        print("  python researcher.py --immediate --d_model 128 --nhead 8")
        return

    trained_experiments = []
    for exp in pending_experiments.copy():
        # Count pending and completed experiments
        num_pending = len(get_pending_experiments())
        num_completed = len(get_completed_experiment_keys())

        print("Press `Ctrl+C` at any time to save checkpoint and exit.")
        
        exp_key = experiment_key(exp)
        if exp_key in completed_keys:
            print(f"Experiment with key {exp_key} already completed. Removing from pending list.")
            pending_experiments.remove(exp)
            continue

        updated_exp = run_experiment(exp)
        experiment_details = get_experiment_details(updated_exp)
        num_completed += 1
        num_pending -= 1
        
        # Calculate estimated time to completion
        est_time_str = ""
        if num_pending > 0:
            # Get durations from completed experiments
            completed_experiments = safe_json_load('data/completed_experiments.json')
            durations = [exp.get('metrics', {}).get('training_duration', 0) 
                        for exp in completed_experiments 
                        if 'metrics' in exp and 'training_duration' in exp.get('metrics', {})]
            
            # Use the most recent 10 experiments (or all if fewer than 10)
            recent_durations = durations[-10:] if len(durations) >= 10 else durations
            
            if recent_durations:
                avg_duration = sum(recent_durations) / len(recent_durations)
                total_est_seconds = avg_duration * num_pending
                
                # Convert to hours and minutes
                est_hours = int(total_est_seconds // 3600)
                est_minutes = int((total_est_seconds % 3600) // 60)
                
                # Format the time string based on duration
                if est_hours > 0:
                    est_time_str = f"\nEstimated time to complete remaining experiments: {est_hours}h {est_minutes}m"
                else:
                    est_time_str = f"\nEstimated time to complete remaining experiments: {est_minutes}m"
        
        experiment_details += f"\n{num_completed} experiments completed, {num_pending} remaining{est_time_str}"
        print("***")
        print(experiment_details)
        print("***\n")

        notification_msg = "Training experiment completed\n"
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
        
        # Suggest using generate_gifs.py instead of generating them here
        if trained_experiments:
            latest_exp = trained_experiments[-1]
            print("\n📊 Experiment visualization:")
            print("To generate confusion matrix GIFs, use:")
            print(f"  python generate_gifs.py --experiment {latest_exp}")
            print("  python generate_gifs.py --recent 3")
            print("  python generate_gifs.py --all")


    print("Cleaning up files")
    clean_up_files(test_mode=False)
    # reformat_completed_experiments('data/completed_experiments.json')


if __name__ == "__main__":
    main()
