import pandas as pd
import ciphers
from tqdm import tqdm
import os
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
import json
import hashlib


def file_hash(filename):
    """Generate a hash for a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def feather_file_statistics(file_path):
    """
    Load a specified feather file, print some sample data,
    return number of samples, memory usage, and file size
    """
    df = pd.read_feather(file_path)
    num_samples = len(df)
    file_size = os.path.getsize(file_path)

    print(f"Samples: {num_samples}")
    memory_usage = df.memory_usage(deep=True).sum()
    print(f"Memory usage: {memory_usage / 1024 ** 2:.2f} MB")
    print(f"File size: {file_size}")
    print(df.sample(3))

    return {"samples": num_samples,
            "memory": f"{memory_usage / 1024 ** 2:.2f} MB",
            "filesize": file_size}


def generate_batches(cipher_funcs=None, sample_length=500, num_batches=1, progress_queue=None):
    """
    Generates a batch of samples, one for each provided cipher.

    Args:
    - cipher_funcs (list): List of cipher functions to use for generating samples.
    - sample_length (int): Length of the samples to generate.

    Returns:
    - List of tuples, where each tuple contains the sample and its label.
    """
    # TODO - fix generation error, shifted two cols to the right
    samples = []
    if cipher_funcs is None:
        cipher_funcs = ciphers._get_cipher_functions()

    for _ in range(num_batches):
        for cipher_func in cipher_funcs:
            enciphered_text = cipher_func(sample_length)
            samples.append((enciphered_text, cipher_func.__name__))
            if progress_queue:
                progress_queue.put(1)

    # Specify column names when creating the DataFrame
    df_samples = pd.DataFrame(samples, columns=['text', 'cipher'])

    return df_samples


def generate_batches_parallel(cipher_funcs=None, total_samples=10000, sample_length=500, progress_queue=None):
    num_workers = cpu_count()
    if cipher_funcs is None:
        cipher_funcs = ciphers._get_cipher_functions()

    # Calculate the number of batches each worker will generate
    batches_per_worker = (total_samples + len(cipher_funcs) - 1) // len(cipher_funcs) // num_workers
    remaining_batches = (total_samples + len(cipher_funcs) - 1) // len(cipher_funcs) % num_workers

    samples = pd.DataFrame(columns=['text', 'cipher'])
    with Manager() as manager:
        if progress_queue is None:
            progress_queue = manager.Queue()

        with Pool(processes=num_workers) as pool:
            results = []
            # Distribute the workload among worker processes
            for i in range(num_workers):
                num_batches = batches_per_worker + (1 if i < remaining_batches else 0)
                result = pool.apply_async(generate_batches, args=(cipher_funcs, sample_length, num_batches, progress_queue))
                results.append(result)

            pool.close()

            # Initialize progress bar and total progress counter
            total_progress = 0
            with tqdm(total=total_samples) as pbar:
                while total_progress < total_samples:
                    progress_update = progress_queue.get()
                    total_progress += progress_update
                    pbar.update(progress_update)

            pool.join()

            # Collect results from all batches
            for result in results:
                batch_samples = result.get()
                samples = pd.concat([samples, batch_samples])
    return samples


def cleanup_metadata(metadata_file="data/sample_feathers_metadata.json"):
    """Synchronizes the metadata file with the actual feather files in the directory."""
    if not os.path.exists(metadata_file):
        print("Metadata file not found.")
        return

    with open(metadata_file, "r") as file:
        metadata = json.load(file)

    valid_metadata = {}
    for dataset_id, info in metadata.items():
        if os.path.exists(info["filename"]):
            valid_metadata[dataset_id] = info

    with open(metadata_file, "w") as file:
        json.dump(valid_metadata, file, indent=4)

    print("Metadata cleanup complete.")


def manage_sample_data(cipher_names=None, num_samples=1500, sample_length=500, metadata_file="data/sample_feathers_metadata.json"):
    """
    Manages the generation and retrieval of sample data, avoiding duplication.

    Args:
    - cipher_funcs (list): List of cipher functions to use for generating samples.
    - num_samples (int): Total number of samples to generate.
    - sample_length (int): Length of the samples to generate.
    - metadata_file (str): JSON file path for storing metadata about generated datasets.

    Returns (filename, generated):
    - filename: The filename of the .feather file with these samples
    - generated: whether the file was newly generated or retrieved (preexisting)
    """
    # Initialize metadata if file doesn't exist or is empty
    if not os.path.exists(metadata_file) or os.stat(metadata_file).st_size == 0:
        metadata = {}
    else:
        try:
            with open(metadata_file, "r") as file:
                metadata = json.load(file)
        except json.JSONDecodeError:
            metadata = {}

    if cipher_names is None:
        cipher_names = ciphers._get_cipher_names()

    # Create a unique identifier for the dataset
    cipher_names_string = "_".join(cipher_names)
    dataset_id = f"{cipher_names_string}_{num_samples}_{sample_length}"

    data_generated = False

    if dataset_id in metadata:
        # Load existing dataset
        df = pd.read_feather(metadata[dataset_id]["filename"])
        filename = metadata[dataset_id]["filename"]
    else:
        data_generated = True
        # Generate new dataset
        all_ciphers = ciphers._get_cipher_functions()
        cipher_funcs = [cipher for cipher in all_ciphers if cipher.__name__ in cipher_names]

        df = generate_batches_parallel(cipher_funcs, num_samples, sample_length)
        df.columns = df.columns.map(str)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/feathers/{timestamp}.feather"
        df.to_feather(filename)

        # Update metadata
        metadata[dataset_id] = {
            "filename": filename,
            "ciphers": cipher_names,
            "samples": num_samples,
            "sample_length": sample_length,
            "hash": file_hash(filename),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        with open(metadata_file, "w") as file:
            json.dump(metadata, file, indent=4)
    cleanup_metadata()
    return filename, data_generated


if __name__ == "__main__":
    # Example test call
    test_cipher_names = ["caesar", "vigenere"]  # Replace with available ciphers
    test_num_samples = 2500
    test_sample_length = 500

    try:
        print(
            f"Collecting data for: \n"
            + f"Ciphers: {test_cipher_names} \n"
            + f"Number of samples: {test_num_samples} \n"
            + f"Sample lengths: {test_sample_length}\n")
        test_filename, generated = manage_sample_data(
            test_cipher_names,
            test_num_samples,
            test_sample_length)
        action = "generation" if generated else "retrieval"
        print(f"Data {action} successful. File saved as {test_filename}")
        feather_file_statistics(test_filename)
    except ValueError as e:
        print(f"Error: {e}")
