# --- prep_samples.py ---
import time
import pandas as pd
import random
import multiprocessing as mp
from multiprocessing import Pool, Manager # Event is no longer needed here, will be imported
import os
import json
import hashlib
from tqdm import tqdm
import queue # For queue.Empty exception
import sys # For potentially adding project root to path
import traceback # For detailed error logging

# Import ciphers.py and book_processing.py
try:
    from ciphers import _get_cipher_functions, _get_cipher_names
    import book_processing  # Import as module - we'll use our own load_books and sample_text functions
except ImportError as e:
    print(f"ERROR: Could not import cipher or book_processing modules: {e}")
    print("Ensure these files exist and are importable.")
    sys.exit(1)


def load_books(dir_path):
    """
    Load processed text files from the specified directory.
    
    Args:
        dir_path (str): Path to directory containing processed book files.
        
    Returns:
        list: List of text strings from the processed books.
    """
    book_texts = []
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
        print("No books loaded - fetch books first.")
        return book_texts
    
    for filename in os.listdir(dir_path):
        if filename.endswith("_processed.txt"):
            try:
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text:
                        book_texts.append(text)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(book_texts)} books from {dir_path}")
    return book_texts


def sample_text(book_texts, length):
    """
    Sample text of specified length from a random position in one of the provided books.
    
    Args:
        book_texts (list): List of text strings from books.
        length (int): Length of the sample to extract.
        
    Returns:
        str: Sampled text of specified length.
    """
    if not book_texts:
        # Fallback to book_processing's random text fetching if no books loaded
        return book_processing.get_random_text_passage(length)
    
    # Select a random book with sufficient length
    valid_books = [book for book in book_texts if len(book) >= length]
    if not valid_books:
        print("No books with sufficient length found, falling back to random passage.")
        return book_processing.get_random_text_passage(length)
    
    selected_book = random.choice(valid_books)
    start_pos = random.randint(0, len(selected_book) - length)
    return selected_book[start_pos:start_pos + length]


def list_cipher_names():
    """Alias for _get_cipher_names() for backward compatibility."""
    return _get_cipher_names()


# --- Setup Project Root ---
# Determine the project root directory dynamically.
# Assumes prep_samples.py is located directly within the project root directory.
_PREP_SAMPLES_FILE_PATH = os.path.abspath(__file__)
_PREP_SAMPLES_DIR = os.path.dirname(_PREP_SAMPLES_FILE_PATH)
_PROJECT_ROOT_FROM_PREP = _PREP_SAMPLES_DIR

# Add project root to sys.path if needed
if _PROJECT_ROOT_FROM_PREP not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_FROM_PREP)

# Create a local shutdown event that can be set by researcher.py later
# This avoids the circular import problem
from multiprocessing import Event
shutdown_event = Event() # Local event that can be replaced at runtime


# --- Constants ---
BOOK_DIR = os.path.join(_PROJECT_ROOT_FROM_PREP, 'local_library')
METADATA_FILE = os.path.join(_PROJECT_ROOT_FROM_PREP, 'data', 'sample_feathers_metadata.json')
DEFAULT_NUM_PROCESSES = max(1, mp.cpu_count() - 1)  # Use CPU count-1, but at least 1

# Note: load_books, sample_text, and list_cipher_names functions are already defined above


def load_metadata(metadata_file):
    """Load the metadata from the file or create empty dict if file doesn't exist."""
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"WARNING: Metadata file {metadata_file} is corrupted. Creating new.")
            return {}
    return {}


def save_metadata(metadata, metadata_file):
    """Save the metadata to the file, creating directory if needed."""
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    with open(metadata_file, 'w') as file:
        json.dump(metadata, file, indent=4)


def get_params_hash(cipher_names, num_samples, sample_length):
    """Generate a hash based on parameters to use as a cache key."""
    params_str = f"{sorted(cipher_names)}_{num_samples}_{sample_length}"
    return hashlib.md5(params_str.encode()).hexdigest()


def generate_filename(cipher_names, num_samples, sample_length):
    """Generate a filename for the feather file based on parameters."""
    params_hash = get_params_hash(cipher_names, num_samples, sample_length)
    feathers_dir = os.path.join(_PROJECT_ROOT_FROM_PREP, 'data', 'feathers')
    os.makedirs(feathers_dir, exist_ok=True)
    return os.path.join(feathers_dir, f"{params_hash}.feather")



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


def generate_batches(args):
    """
    Worker function to generate a batch of samples for a specific cipher.
    Handles potential errors during cipher generation or text sampling.

    Args:
        args (tuple): Contains (cipher_name_str, num_samples_for_cipher,
                       sample_length, book_texts, process_id, progress_queue)
                       Note: cipher_name_str is the string name of the cipher.

    Returns:
        dict: A dictionary containing either the generated data or an error status.
              {'status': 'success', 'data': list_of_dicts}
              {'status': 'error', 'message': str, 'cipher': cipher_name_str, 'process_id': process_id}
    """
    cipher_name_str, num_samples_for_cipher, sample_length, book_texts, process_id, progress_queue = args
    batch_data = []
    try:
        # Get the specific cipher function object from its name string
        all_funcs_list = _get_cipher_functions()
        name_to_func_map = {func.__name__: func for func in all_funcs_list}
        cipher_func = name_to_func_map.get(cipher_name_str)

        if cipher_func is None:
            err_msg = f"Cipher function name '{cipher_name_str}' not found"
            return {'status': 'error', 'message': err_msg, 'cipher': cipher_name_str, 'process_id': process_id}

        samples_generated = 0
        attempts = 0
        max_attempts = num_samples_for_cipher * 5

        while samples_generated < num_samples_for_cipher and attempts < max_attempts:
            attempts += 1
            plaintext = sample_text(book_texts, sample_length)
            if not plaintext:
                continue

            try:
                ciphertext = cipher_func(text=plaintext)
                if ciphertext:
                    batch_data.append({'text': ciphertext, 'cipher': cipher_name_str})
                    samples_generated += 1
                    if progress_queue and samples_generated % 10 == 0:
                        progress_queue.put(10)
            except ValueError:
                continue
            except Exception:
                continue

        # Report any remaining progress
        final_progress_chunk = samples_generated % 10
        if progress_queue and final_progress_chunk > 0:
            progress_queue.put(final_progress_chunk)

        return {'status': 'success', 'data': batch_data}

    except Exception as e:
        return {'status': 'error', 'message': str(e), 'cipher': cipher_name_str, 'process_id': process_id}


def generate_batches_parallel(cipher_names, num_samples_per_cipher, sample_length, book_texts, num_processes):
    """
    Generates samples in parallel using a process pool.
    Checks the global shutdown_event (imported from researcher) for graceful interruption.
    Handles worker errors and reports progress.

    Args:
        cipher_names (list): List of cipher names.
        num_samples_per_cipher (dict): Dict mapping cipher name to number of samples.
        sample_length (int): Length of each text sample.
        book_texts (list): List of strings (book contents).
        num_processes (int): Number of worker processes.

    Returns:
        dict: A dictionary indicating the outcome.
              {'status': 'success', 'data': list_of_dicts}
              {'status': 'error', 'message': str}
              {'status': 'interrupted'}
              {'status': 'worker_errors', 'message': str, 'errors': list_of_errors}
    """
    all_samples = []
    total_samples_to_generate = sum(num_samples_per_cipher.values())
    worker_errors = []

    # Use Manager queue for progress reporting across processes
    with Manager() as manager:
        progress_queue = manager.Queue()
        results = [] # List to store AsyncResult objects
        pool = Pool(processes=num_processes)
        pbar = None # Initialize progress bar variable

        try:
            # Prepare arguments for each task chunk
            tasks = []
            for cipher_name in cipher_names:
                n_cipher = num_samples_per_cipher[cipher_name]
                # Calculate chunk size to balance overhead and load distribution
                chunk_size = min(n_cipher, max(10, n_cipher // (num_processes * 2)))
                num_chunks = (n_cipher + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    samples_in_chunk = min(chunk_size, n_cipher - i * chunk_size)
                    if samples_in_chunk > 0:
                         process_id = f"{cipher_name}_{i+1}"
                         tasks.append((cipher_name, samples_in_chunk, sample_length, book_texts, process_id, progress_queue))

            # Shuffle tasks to distribute cipher types
            random.shuffle(tasks)

            # Submit tasks asynchronously
            for task_args in tasks:
                # Check for shutdown signal before submitting each task
                if shutdown_event.is_set():
                    pool.terminate()
                    pool.join()
                    return {'status': 'interrupted', 'message': 'Shutdown during task submission', 'data': None, 'errors': None}

                res = pool.apply_async(generate_batches, args=(task_args,))
                results.append(res)

            pool.close() # No more tasks will be submitted

            # --- Monitoring Loop ---
            pbar = tqdm(total=total_samples_to_generate, desc="Generating Samples", unit="samples", smoothing=0.1)
            active_results = list(results) # Copy results list to modify while iterating

            while active_results:
                # --- Check for Shutdown Signal ---
                if shutdown_event.is_set():
                    if pbar: pbar.close()
                    pool.terminate()
                    pool.join()
                    return {'status': 'interrupted', 'message': 'Shutdown during processing', 'data': all_samples, 'errors': worker_errors}

                # --- Update Progress Bar from Queue ---
                try:
                    while True: # Drain the queue without blocking
                        progress_increment = progress_queue.get_nowait()
                        if progress_increment > 0:
                            pbar.update(progress_increment)
                except queue.Empty:
                    pass # No progress update available right now

                # --- Check for Completed Tasks (Non-blocking) ---
                remaining_results = []
                for res in active_results:
                    if res.ready():
                        try:
                            # Task finished, get the result dictionary
                            result_dict = res.get()
                            if result_dict['status'] == 'success':
                                generated_data = result_dict.get('data', [])
                                all_samples.extend(generated_data)
                            elif result_dict['status'] == 'error':
                                err_msg = (f"\nWORKER ERROR: Process {result_dict.get('process_id', 'N/A')}, "
                                           f"Cipher {result_dict.get('cipher', 'N/A')}")
                                worker_errors.append(result_dict)
                            else:
                                # Handle unexpected status from worker
                                worker_errors.append({'status': 'unknown', 'raw': result_dict})

                        except Exception as e:
                            # Catch potential errors from res.get() itself
                            worker_errors.append({'status': 'retrieval_error', 'exception': str(e)})
                    else:
                        # Task not ready, keep it in the list for the next check
                        remaining_results.append(res)

                active_results = remaining_results

                # Prevent busy-waiting
                if active_results:
                    time.sleep(0.1) # Short sleep (100ms) to yield CPU

            # Ensure the pool is properly joined after loop finishes naturally
            pool.join()

            # Final progress update to ensure bar reaches the actual count
            if pbar:
                pbar.update(len(all_samples) - pbar.n) # Update with the difference
                pbar.close()

            # --- Check for Worker Errors and Final Status ---
            if worker_errors:
                error_summary = f"Completed with {len(worker_errors)} worker errors. Generated {len(all_samples)} samples."
                return {'status': 'worker_errors', 'message': error_summary, 'errors': worker_errors, 'data': all_samples}
            elif not all_samples and total_samples_to_generate > 0:
                # Handle case where no errors occurred but no samples were generated
                no_sample_msg = "Completed successfully but generated 0 samples."
                return {'status': 'success_empty', 'message': no_sample_msg, 'errors': None, 'data': []}
            else:
                # Normal successful completion
                return {'status': 'success', 'message': 'Completed successfully', 'errors': None, 'data': all_samples}

        except (KeyboardInterrupt, SystemExit) as e:
             # Catch signals that might bypass the main handler's event setting initially
             if pbar: pbar.close()
             pool.terminate()
             pool.join()
             shutdown_event.set()
             return {'status': 'interrupted', 'message': f'Caught {type(e).__name__}', 'data': all_samples, 'errors': worker_errors}

        except Exception as e:
             # Catch unexpected errors within the coordinator logic
             traceback.print_exc()
             if pbar: pbar.close()
             # Attempt to clean up the pool
             try:
                 pool.terminate()
                 pool.join()
             except Exception:
                 pass
             return {'status': 'error', 'message': f"Critical error in coordinator logic: {e}", 'data': None, 'errors': None}


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


def manage_sample_data(cipher_names, num_samples, sample_length, metadata_file=METADATA_FILE, force_regenerate=False):
    """
    Manages the creation and loading of sample data. Checks metadata,
    generates data if needed (using parallel processing), and returns the filename.
    Uses the global shutdown_event imported from researcher.

    Args:
        cipher_names (list): List of cipher names.
        num_samples (int): Total number of samples desired.
        sample_length (int): Length of each sample.
        metadata_file (str): Path to the metadata JSON file.
        force_regenerate (bool): If True, always regenerate data.

    Returns:
        tuple: (filename, data_was_generated, status_message)
               - filename (str or None): Relative path to the feather file from project root,
                                         or None on failure/interruption.
               - data_was_generated (bool): True if data generation was attempted in this call.
               - status_message (str or None): None on success, 'interrupted' if stopped by event,
                                               'worker_errors' if completed with errors,
                                               or an error message string starting with 'Error: '.
    """
    # Ensure the directory for metadata exists
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    metadata = load_metadata(metadata_file)
    params_hash = get_params_hash(cipher_names, num_samples, sample_length)

    # --- Check Cache ---
    if not force_regenerate and params_hash in metadata:
        entry = metadata[params_hash]
        filename_relative = entry.get('filename')
        filename_abs = os.path.join(_PROJECT_ROOT_FROM_PREP, filename_relative) if filename_relative else None

        # Validate cache entry
        if filename_abs and os.path.exists(filename_abs) and \
           entry.get('num_samples') == num_samples and \
           entry.get('sample_length') == sample_length and \
           set(entry.get('cipher_names', [])) == set(cipher_names):
            # Return cached file
            return filename_relative, False, None
        else:
            # Cache invalid - regenerate
            data_was_generated = True
    else:
        data_was_generated = True

    # --- Data Generation ---
    try:
        # Load book texts
        book_texts = load_books(BOOK_DIR)
        if not book_texts:
            return None, True, "Error: No book texts found or loaded."

        # Distribute samples among ciphers
        num_ciphers = len(cipher_names)
        if num_ciphers == 0:
            return None, True, "Error: No cipher names specified."

        # Calculate samples per cipher
        samples_per_cipher = num_samples // num_ciphers
        remainder = num_samples % num_ciphers
        num_samples_per_cipher_dict = {}
        for i, name in enumerate(cipher_names):
            count = samples_per_cipher + (1 if i < remainder else 0)
            num_samples_per_cipher_dict[name] = count

        # Call parallel generation
        num_processes = DEFAULT_NUM_PROCESSES
        generation_result = generate_batches_parallel(
            cipher_names,
            num_samples_per_cipher_dict,
            sample_length,
            book_texts,
            num_processes
        )

        # Process generation result
        gen_status = generation_result.get('status', 'error')
        gen_message = generation_result.get('message', 'Unknown generation outcome')
        all_samples = generation_result.get('data', [])
        worker_errors = generation_result.get('errors', None)

        if gen_status in ['success', 'success_empty', 'worker_errors']:
            if not all_samples and gen_status != 'success_empty':
                status_msg = f"Error: Generation process '{gen_status}' but produced 0 samples."
                if worker_errors:
                    status_msg += f" Worker errors may be relevant: {gen_message}"
                return None, True, status_msg

            # Create DataFrame and save
            df = pd.DataFrame(all_samples) if all_samples else pd.DataFrame(columns=['text', 'cipher'])
            if not df.empty:
                df = df.sample(frac=1).reset_index(drop=True)

            # Save to file
            filename_abs = generate_filename(cipher_names, num_samples, sample_length)
            os.makedirs(os.path.dirname(filename_abs), exist_ok=True)
            df.to_feather(filename_abs)
            filename_relative = os.path.relpath(filename_abs, _PROJECT_ROOT_FROM_PREP)

            # Update metadata
            metadata_entry = {
                'filename': filename_relative,
                'cipher_names': cipher_names,
                'num_samples': num_samples,
                'actual_samples': len(df),
                'sample_length': sample_length,
                'generation_timestamp': time.time(),
                'generation_status': gen_status,
            }
            if worker_errors:
                metadata_entry['worker_errors_summary'] = [
                    {k: v for k, v in err.items() if k not in ['raw', 'data']}
                    for err in worker_errors
                ]
            metadata[params_hash] = metadata_entry
            save_metadata(metadata, metadata_file)

            # Determine return status
            return_status_message = None
            if gen_status == 'worker_errors':
                return_status_message = 'worker_errors'
            elif gen_status == 'success_empty':
                return_status_message = 'Error: Generated empty dataset'

            return filename_relative, True, return_status_message

        elif gen_status == 'interrupted':
            return None, True, 'interrupted'
        else:
            error_msg_combined = f"Error: Data generation failed with status '{gen_status}': {gen_message}"
            return None, True, error_msg_combined

    except Exception as e:
        # Catch unexpected errors
        traceback.print_exc()
        return None, True, f"Error: Unexpected error in manage_sample_data: {e}"


# --- Main Execution / Testing ---
if __name__ == "__main__":
    # This block uses the `shutdown_event` imported or created at the top of the file.
    print("Running prep_samples.py directly for testing...")
    print(f"Using shutdown_event: {shutdown_event}") # Show which event is being used

    # Example usage parameters:
    # Use only a few available ciphers for faster testing
    available_ciphers = list_cipher_names()
    test_cipher_names = available_ciphers[:min(len(available_ciphers), 3)]
    if not test_cipher_names:
        print("ERROR: No ciphers found via list_cipher_names(). Cannot run tests.")
        sys.exit(1)

    test_num_samples = 500  # Keep relatively small for testing
    test_sample_length = 40 # Keep relatively short for testing

    # --- Test graceful shutdown utility function ---
    def delayed_shutdown(event_to_set, delay_seconds):
        """Helper function to trigger the shutdown event after a delay."""
        print(f"[Test Runner] Will trigger shutdown event in {delay_seconds} seconds...")
        time.sleep(delay_seconds)
        print("[Test Runner] Triggering shutdown event NOW!")
        event_to_set.set() # Set the event (global or dummy)

    # --- Test Case 1: Normal Generation ---
    print("\n--- Test Case 1: Normal Generation (forced) ---")
    # Ensure event is clear before starting
    shutdown_event.clear()
    filename_case1, generated_case1, status_case1 = manage_sample_data(
        test_cipher_names,
        test_num_samples,
        test_sample_length,
        force_regenerate=True # Force generation for this test
        # No event argument is passed; uses imported global event
    )
    print(f"Result -> Filename: {filename_case1}, Generated: {generated_case1}, Status: {status_case1}")
    # Basic assertions for success
    assert generated_case1 is True
    assert filename_case1 is not None
    assert status_case1 is None or status_case1 == 'worker_errors' # Allow success or success with warnings

    # --- Test Case 2: Using Cached Data ---
    if filename_case1 and status_case1 is None: # Only run if previous step likely succeeded cleanly
        print("\n--- Test Case 2: Using Cached Data ---")
        shutdown_event.clear() # Ensure event is clear
        filename_case2, generated_case2, status_case2 = manage_sample_data(
            test_cipher_names,
            test_num_samples,
            test_sample_length,
            force_regenerate=False # Should use cache this time
            # No event argument passed
        )
        print(f"Result -> Filename: {filename_case2}, Generated: {generated_case2}, Status: {status_case2}")
        # Assertions for cache hit
        assert generated_case2 is False
        assert status_case2 is None
        assert filename_case1 == filename_case2 # Should be the same file path
    else:
        print("\n--- Skipping Test Case 2 (Cached Data) due to previous run status ---")


    # --- Test Case 3: Interruption (Requires uncommenting thread start) ---
    print("\n--- Test Case 3: Interruption Test (Requires uncommenting thread start below) ---")
    # To test this:
    # 1. Increase test_num_samples significantly (e.g., * 10) to ensure generation takes time.
    # 2. Set a short delay for delayed_shutdown (e.g., 3 seconds).
    # 3. Uncomment the shutdown_thread.start() line.
    shutdown_event.clear() # Ensure event is clear before starting test
    interruption_delay = 3 # Seconds
    # Use the imported shutdown_event for the test trigger
    import threading
    shutdown_thread = threading.Thread(target=delayed_shutdown, args=(shutdown_event, interruption_delay))
    # shutdown_thread.start() # <--- UNCOMMENT THIS LINE TO ACTUALLY TEST INTERRUPTION

    filename_case3, generated_case3, status_case3 = manage_sample_data(
        test_cipher_names,
        test_num_samples * 5, # Use a larger number to ensure it's running when interrupted
        test_sample_length,
        force_regenerate=True
        # No event argument passed
    )
    print(f"Result -> Filename: {filename_case3}, Generated: {generated_case3}, Status: {status_case3}")
    # If shutdown_thread was started, assert interruption occurred
    # if shutdown_thread.is_alive(): # Or check if it was started
    #    assert status_case3 == 'interrupted'
    #    assert filename_case3 is None
    #    shutdown_thread.join() # Clean up the thread if testing
    # else:
    #    print("(Skipped interruption assertion as thread was not started)")
    # Need a way to know if thread was started to assert correctly, otherwise just print status


    # --- Test Case 4: Worker error (Requires modifying a cipher to fail sometimes) ---
    # To test this, manually modify one of the cipher functions in ciphers.py
    # to raise an exception under certain conditions (e.g., if plaintext length > N).
    # Then run this script. Expect status_case4 == 'worker_errors'.
    print("\n--- Test Case 4: Worker Error (Manual setup required in ciphers.py) ---")
    # print("Modify a cipher function to sometimes fail, then run.")
    # filename_case4, generated_case4, status_case4 = manage_sample_data(
    #      test_cipher_names, test_num_samples, test_sample_length, force_regenerate=True
    # )
    # print(f"Result -> Filename: {filename_case4}, Generated: {generated_case4}, Status: {status_case4}")
    # assert status_case4 == 'worker_errors' # Or check for specific error messages


    # --- Test Case 5: Bad Input ---
    print("\n--- Test Case 5: Bad Input (No Ciphers) ---")
    shutdown_event.clear()
    filename_case5, generated_case5, status_case5 = manage_sample_data(
        [], # Empty cipher list
        test_num_samples,
        test_sample_length,
        force_regenerate=True
    )
    print(f"Result -> Filename: {filename_case5}, Generated: {generated_case5}, Status: {status_case5}")
    assert filename_case5 is None
    assert status_case5 is not None and status_case5.startswith("Error:")


    print("\nprep_samples.py testing finished.")
