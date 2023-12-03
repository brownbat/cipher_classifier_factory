import pandas as pd
import random
import book_processing  # Make sure this is correctly imported
import ciphers
from tqdm import tqdm
import os
from datetime import datetime
import subprocess
import math
import string
import inspect
from multiprocessing import Pool, Manager
import time
from itertools import cycle

def get_cipher_functions():
    """
    Retrieves cipher functions from the ciphers module, excluding helper
    functions, which start with _
    """
    return [func for name, func in inspect.getmembers(
        ciphers, inspect.isfunction) if not name.startswith('_')]


def clear_screen():
    # Clear the screen based on the operating system
    if os.name == 'nt':  # Windows
        subprocess.call('cls', shell=True)
    else:  # Unix/Linux/Mac
        subprocess.call('clear', shell=True)


def count_feather_files(directory='data'):
    return len([f for f in os.listdir(directory) if f.endswith('.feather')])


status_message = ""


def main_menu():
    global status_message
    invalid_choice = False
    while True:
        clear_screen()
        feather_file_count = count_feather_files()
        if status_message:
            print(status_message)
            status_message = ""  # Reset status message after displaying

        print(f"\nFeather files in 'data' directory: {feather_file_count}")
        print("\nMain menu:")
        print("1) Generate samples and store as feather files")
        print("2) Merge feather files")
        print("3) Split large feather file")
        print("4) Display feather file statistics")
        print("5) Delete all feather files")
        print("6) Generate one batch of samples (for display only)")
        print("7) Exit")

        if invalid_choice:
            print("\nInvalid choice. Please try again.")
            invalid_choice = False

        choice = input("\nEnter your choice: ")

        if choice == '1':
            generate_samples_menu()
        elif choice == '2':
            merge_feathers_menu()
        elif choice == '3':
            split_feather_menu()
        elif choice == '4':
            display_feather_statistics_menu()
        elif choice == '5':
            delete_all_feathers_menu()
        elif choice == '6':
            display_samples_menu()
        elif choice.lower() in ('7', 'exit', 'x', 'q', 'quit'):
            print("Exiting...")
            break
        else:
            invalid_choice = True


def calc_samples_per_file(total_samples, num_files):
    """Calculates the distribution of samples across a specified number of
    files, ensuring each file's sample count is divisible by the number of
    ciphers.
    """
    if total_samples < num_files:
        raise ValueError(
            "Number of samples cannot be less than the number of files.")

    num_ciphers = len(get_cipher_functions())
    base_samples = (total_samples // num_files) // num_ciphers * num_ciphers
    remainder = total_samples - base_samples * num_files

    # Distribute the remainder, ensuring divisibility by number of ciphers
    samples_per_file = [base_samples for _ in range(num_files)]
    for i in range(num_files):
        if remainder >= num_ciphers:
            samples_per_file[i] += num_ciphers
            remainder -= num_ciphers
        elif remainder > 0:
            extra = remainder // num_ciphers * num_ciphers
            samples_per_file[i] += extra
            remainder -= extra

    return samples_per_file


def parallel_generate_samples(args, progress_queue):
    batch_size, chunk_size, _ = args
    samples = []
    for _ in range(0, batch_size, chunk_size):
        chunk = generate_sample_chunk(min(chunk_size, batch_size - len(samples)))
        samples.extend(chunk)
        progress_queue.put(len(chunk))  # Update progress for each chunk
    return samples


def generate_sample_chunk(chunk_size, sample_length=500):
    """
    Generates a chunk of samples using available cipher functions in a cycle.

    Args:
    - chunk_size (int): Number of samples to generate in this chunk.
    - sample_length (int): Length of each sample.

    Returns:
    - List of tuples, where each tuple contains the sample and its label.
    """
    samples = []
    cipher_functions = get_cipher_functions()
    # Create a cyclic iterator over the cipher functions
    cyclic_ciphers = cycle(cipher_functions)

    for _ in range(chunk_size):
        cipher_func = next(cyclic_ciphers)  # Get the next cipher function
        sample = cipher_func(length=sample_length) # Generate a sample using the cipher function
        samples.append((sample, cipher_func.__name__))

    return samples


def generate_samples_menu(batch_size=1000, total_samples=30000, num_workers=None):
    global status_message

    num_batches = math.ceil(total_samples / batch_size)
    chunk_size = 10  # Number of samples to generate before updating progress

    while True:
        samples_per_file = calc_samples_per_file(total_samples, num_batches)
        unique_batch_sizes = set(samples_per_file)
        batch_size_message = str(min(unique_batch_sizes))
        if len(unique_batch_sizes) > 1:
            batch_size_message += " or " + str(max(unique_batch_sizes))

        print(
            f"{total_samples} samples will be generated across {num_batches} "
            + f"files, each containing {batch_size_message} samples")
        print("1) Confirm and generate samples")
        print("2) Advanced menu (adjust number of samples and files to generate)")
        print("3) Cancel")
        choice = input("Enter your choice: ")

        if choice == '1':
            break
        elif choice == '2':
            total_samples, num_batches = advanced_sample_settings()
        elif choice == '3':
            status_message = "Sample generation canceled."
            return None
        else:
            clear_screen()
            print(f"Invalid choice. Please try again.")

    with Manager() as manager:
        progress_queue = manager.Queue()  # Queue for progress updates
        args = [(size, chunk_size, idx) for idx, size in enumerate(samples_per_file)]
        
        with Pool(processes=num_workers) as pool:  # Specify num_workers or leave it to default
            results = []

            for arg in args:
                result = pool.apply_async(parallel_generate_samples, (arg, progress_queue))
                results.append(result)

            with tqdm(total=total_samples, desc="Generating Samples") as pbar:
                while any(not r.ready() for r in results):
                    while not progress_queue.empty():
                        progress = progress_queue.get()
                        pbar.update(progress)
                    time.sleep(0.1)  # Adjust sleep time as needed

            samples_list = [r.get() for r in results]

        for idx, samples in enumerate(samples_list):
            save_batch_samples(samples, idx)

    status_message = f"{total_samples} samples generated across {num_batches} files"


def advanced_sample_settings():
    new_total_samples = int(input("Enter new total samples count: "))
    new_num_batches = int(input("Spread across how many files: "))

    clear_screen()
    return new_total_samples, new_num_batches


def generate_samples(sample_length, num_samples=None, pbar=None):
    """
    Generates samples using the provided cipher functions.

    Args:
    - cipher_functions (list): List of cipher functions to use for generating
        samples.
    - sample_length (int): Length of the samples to generate.
    - num_samples (int, optional): Number of samples to generate.
        If None, generates one sample per cipher function.

    Returns:
    - List of samples.
    """
    samples = []
    cipher_functions = get_cipher_functions()
    for cipher in cipher_functions:
        if num_samples is None:
            samples.append((cipher(sample_length), cipher.__name__))
        else:
            for _ in range(num_samples):
                samples.append((cipher(sample_length), cipher.__name__))
                if pbar is not None:
                    pbar.update(1)

    return samples


def generate_batch_samples(batch_size, pbar=None):
    cipher_functions = get_cipher_functions()
    num_samples_per_cipher = batch_size // len(cipher_functions)
    samples = generate_samples(
        sample_length=500, num_samples=num_samples_per_cipher, pbar=pbar)
    return samples


def display_samples_menu():
    choice = 'y'
    while choice.lower() == 'y':
        sample_length = random.randint(80, 600)
        samples = generate_samples(sample_length)

        for sample in samples:
            print(sample)
            print()

        print(f"Continue (generate another) (y/n)?")
        choice = input()
    return samples


def save_batch_samples(samples, batch_number):
    padded_batch_number = str(batch_number + 1).zfill(3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file_path = f"data/{timestamp}_batch_{padded_batch_number}_samples.feather"
    save_samples_to_feather(samples, batch_file_path)


def merge_feathers_menu():
    global status_message
    print("Merging all feathers in the 'data' directory.")
    print("WARNING: This will delete small merged files. Continue (y/n)?")
    choice = input()
    if choice.lower() != 'y':
        status_message = "Merging files canceled"
        return None

    feather_files = [f for f in os.listdir('data') if f.endswith('.feather')]
    if not feather_files:
        status_message = "Attempted to merge but no files found"
        return None
    if len(feather_files) == 1:
        status_message = "Attempted to merge but only one file found"
        return None
    all_data_frames = []

    for file in feather_files:
        file_path = os.path.join('data', file)
        df = pd.read_feather(file_path)
        all_data_frames.append(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = f'data/merged_{timestamp}.feather'
    combined_df = pd.concat(
        all_data_frames, ignore_index=True).drop_duplicates()
    combined_df.to_feather(merged_path)

    for file in feather_files:
        file_path = os.path.join('data', file)
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
    status_message = f"Feathers merged into {merged_path}"
    status_message += "\nSmaller merged files deleted"


def split_feather_menu(directory='data'):
    global status_message
    largest_file = find_largest_feather_file(directory)
    if not largest_file:
        status_message = "Attempted to split but no feather files found"
        return

    largest_file_path = os.path.join(directory, largest_file)
    largest_size = os.path.getsize(largest_file_path)
    print(
        f"The largest file is {largest_file} with a size of {largest_size} "
        + f"bytes.")

    confirmation = input("Do you want to split this file? (y/n): ")
    if confirmation.lower() != 'y':
        print("Splitting cancelled.")
        return

    num_splits = int(
        input("How many pieces do you want to split the file into? "))
    delete_original = input("Delete the original file after splitting? (y/n): ")

    split_feather_file(largest_file_path, num_splits)

    if delete_original.lower() == 'y':
        os.remove(largest_file_path)
        print(f"Deleted the original file: {largest_file}")
    status_message = (
        f"Feather file {largest_file} split into {num_splits} pieces")
    if delete_original.lower() == 'y':
        status_message += f"\nOriginal file ( {largest_file} ) deleted"


def find_largest_feather_file(directory):
    feather_files = [f for f in os.listdir(directory) if f.endswith('.feather')]
    if not feather_files:
        return None  # Or handle this case as needed
    largest_file = max(
        feather_files,
        key=lambda f: os.path.getsize(os.path.join(directory, f)))
    return largest_file


def split_feather_file(file_path, num_splits):
    df = pd.read_feather(file_path)
    split_size = len(df) // num_splits + (1 if len(df) % num_splits else 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(num_splits):
        split_df = df.iloc[i * split_size:(i + 1) * split_size]
        split_file_path = f'data/split_{timestamp}_p{str(i+1).zfill(3)}.feather'
        split_df.to_feather(split_file_path)
        print(f"Saved split file: {split_file_path}")


def quick_duplicate_check(directory_path):
    largest_file = find_largest_feather_file(directory_path)
    if not largest_file:
        return []

    largest_df = pd.read_feather(os.path.join(directory_path, largest_file))
    # Check if the largest DataFrame is empty
    if largest_df.empty:
        print(f"DataFrame from {largest_file} is empty.")
        return []

    duplicates_found = []
    for this_file in os.listdir(directory_path):
        if this_file.endswith('.feather') and this_file != largest_file:
            df = pd.read_feather(os.path.join(directory_path, this_file))
            # Check if the current DataFrame is empty
            if not df.empty:
                first_sample = df.iloc[0]['sample']
                if first_sample in largest_df['sample'].values:
                    duplicates_found.append(this_file)
            else:
                print(f"DataFrame from {this_file} is empty.")

    if duplicates_found:
        duplicates_found.insert(0, largest_file)

    return duplicates_found


def display_random_samples_from_feather(feather_path, num_samples=3):
    """
    Displays a few random samples from a Feather file to help inspect the data.

    Args:
    feather_path (str): Path to the Feather file.
    num_samples (int): Number of random samples to display.

    Returns:
    None
    """
    df = pd.read_feather(feather_path)
    if not df.empty:
        random_samples = df.sample(n=num_samples)
        print(f"\nRandom samples from {feather_path}:")
        for index, row in random_samples.iterrows():
            print(f"Sample {index}\nLabel: {row['label']}\n{row['sample']}")
    else:
        print(f"ERROR: No samples found in {feather_path}!")


def display_feather_statistics_menu(directory_path='data'):
    display_limit = 3
    feather_files = [
        f for f in os.listdir(directory_path) if f.endswith('.feather')]
    total_samples = 0

    print("Feather Files Statistics:")

    # Calculate total samples independently
    for feather_file in feather_files:
        file_path = os.path.join(directory_path, feather_file)
        df = pd.read_feather(file_path)
        total_samples += len(df)

    # Randomly sample a limited number of files to display
    files_to_display = random.sample(
        feather_files, min(display_limit, len(feather_files)))

    for feather_file in files_to_display:
        file_path = os.path.join(directory_path, feather_file)
        df = pd.read_feather(file_path)

        num_samples = len(df)
        memory_usage = df.memory_usage(deep=True).sum()
        file_size = os.path.getsize(file_path)

        print(f"\n{feather_file}:")
        print(f"  Number of samples: {num_samples}")
        print(f"  Memory usage: {memory_usage / 1024 ** 2:.2f} MB")
        print(f"  File size on disk: {file_size / 1024 ** 2:.2f} MB")

        # Display random samples
        print(f"Sample data")
        display_random_samples_from_feather(file_path)

    print(f"\nTotal Number of Feather Files: {len(feather_files)}")
    print(f"Total Number of Samples: {total_samples}")

    duplicates = quick_duplicate_check(directory_path)
    if duplicates:
        print("Duplicate samples found in the following files:")
        for file in duplicates:
            print(file)
    input("Press Enter to return to the main menu...")


def delete_all_feathers_menu(directory='data'):
    global status_message
    print("WARNING: This will delete all Feather files in the directory.")
    confirmation = input("Type 'DELETE' to confirm: ")
    if confirmation == 'DELETE':
        for file in os.listdir(directory):
            if file.endswith('.feather'):
                os.remove(os.path.join(directory, file))
                print(f"Deleted {file}")
        print("All Feather files have been deleted.")
        status_message = "All feather files have been deleted."
    else:
        status_message = "Deletion cancelled."


def feather_files_sample_count_and_memory(directory_path):
    feather_files = [
        f for f in os.listdir(directory_path) if f.endswith('.feather')]
    total_samples = 0

    print("Feather Files, Sample Counts, Memory usage:")
    for feather_file in feather_files:
        file_path = os.path.join(directory_path, feather_file)
        df = pd.read_feather(file_path)
        num_samples = len(df)
        total_samples += num_samples
        print(f"{feather_file}: {num_samples} samples")
        memory_usage = df.memory_usage(deep=True).sum()
        print(f"{feather_file}: {memory_usage / 1024 ** 2:.2f} MB")

    print(f"\nTotal Number of Feather Files: {len(feather_files)}")
    print(f"Total Number of Samples: {total_samples}")
    input("Press Enter to return to the main menu...")


def examine_feather(file_path, num_samples=15, random=False):
    """
    Load a Feather file and display a specified number of samples, display
    memory usage.
    """
    df = pd.read_feather(file_path)
    if random:
        print(df.sample(num_samples))
    else:
        print(df.head(num_samples))
    memory_usage = df.memory_usage(deep=True).sum()
    print()
    print(f"{file_path}: {memory_usage / 1024 ** 2:.2f} MB")


def save_samples_to_feather(data, file_path):
    try:
        existing_data = pd.read_feather(file_path)
        new_data = pd.DataFrame(data, columns=['sample', 'label'])
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        combined_data = pd.DataFrame(data, columns=['sample', 'label'])

    combined_data.to_feather(file_path)


def load_samples_from_feather(file_path):
    return pd.read_feather(file_path)


if __name__ == "__main__":
    main_menu()
