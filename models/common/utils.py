"""
Common utility functions for the cipher classifier project, including experiment ID
management and configuration lookup. Dynamically finds the project root using markers.
"""
import hashlib
import re
import subprocess
import gc
import torch
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path # Import pathlib

# --- Dynamic Project Root Finding ---

def find_project_root(start_path: Path, markers: List[str]) -> Optional[Path]:
    """
    Find the project root directory by searching upwards from the start_path
    for a specific marker file or directory from the provided list.

    Args:
        start_path (Path): The starting path (usually Path(__file__).resolve()).
        markers (List[str]): A list of filenames or directory names to search for,
                             in order of preference.

    Returns:
        Optional[Path]: The absolute path to the project root directory if found,
                        otherwise None.
    """
    current_path = start_path
    # Limit search depth to avoid checking the entire filesystem root in edge cases
    max_depth = 30 # Adjust as needed, usually sufficient
    depth = 0

    for parent in current_path.parents:
        if depth > max_depth:
            break
        for marker in markers:
            # Check if the marker exists as a file or directory in the parent
            if (parent / marker).exists():
                # Found a marker, return this parent directory as the root
                return parent
        depth += 1

    # If no marker was found after checking parents
    return None

# Define potential root markers in order of preference
# Standard markers are generally more robust than project-specific names
# Add 'cipher_classifier' as a directory name marker if desired, but place it lower priority
# Note: Searching for a directory name like this checks parent.name == 'cipher_classifier' - less direct with Path
# A better project-specific marker might be the presence of key files/dirs like 'researcher.py' and 'data'
ROOT_MARKERS = [
    '.git',                 # Standard repository marker
    'pyproject.toml',       # Standard Python package marker
    'setup.py',             # Older Python package marker
    'requirements.txt',     # Common dependency file at root
    'README.md',            # Common documentation file at root
    # Add more specific markers if needed, e.g.:
    # ('researcher.py', 'data') # Tuple indicating multiple required items
]

# Attempt to find the project root
_PROJECT_ROOT = None
try:
    _PROJECT_ROOT = find_project_root(Path(__file__).resolve(), ROOT_MARKERS)
except Exception as e:
    # Catch potential errors during path resolution or iteration
    print(f"Error during project root search: {e}")

if _PROJECT_ROOT is None:
    # Raise a configuration error if the root wasn't found
    raise FileNotFoundError(
        "Could not automatically detect the project root directory. "
        f"Ensure one of the marker files/directories ({ROOT_MARKERS}) "
        "exists in the project's root directory, or manually configure paths."
    )

# --- Constants for Experiment Files (relative to found root) ---
_DEFAULT_DATA_DIR = _PROJECT_ROOT / 'data'

# Define absolute paths using pathlib (convert to string for functions expecting strings)
PENDING_EXPERIMENTS_FILE = str(_DEFAULT_DATA_DIR / 'pending_experiments.json')
COMPLETED_EXPERIMENTS_FILE = str(_DEFAULT_DATA_DIR / 'completed_experiments.json')



# --- File/System Utilities ---

def file_hash(filename: str) -> str:
    """Generate a SHA256 hash for a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filename, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Warning: File not found for hashing: {filename}")
        return "file_not_found"
    except Exception as e:
        print(f"Error hashing file {filename}: {e}")
        return "hashing_error"


def get_gpu_temp() -> float:
    """
    Get current GPU temperature, attempting to use 'sensors'.
    Returns junction temperature if found, otherwise -1.0.
    Note: This is Linux-specific and depends on 'lm-sensors' being installed
          and configured. May not work for all GPUs or systems.
    """
    try:
        process = subprocess.Popen(["sensors"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        output, _ = process.communicate()

        # Regex to find temperature lines (specifically junction temp if available)
        junction_match = re.search(r'junction:\s*\+?(-?\d+\.\d+)', output, re.IGNORECASE)
        if junction_match:
            return float(junction_match.group(1))

        # Fallback: Look for edge temperature if junction not found
        edge_match = re.search(r'edge:\s*\+?(-?\d+\.\d+)', output, re.IGNORECASE)
        if edge_match:
            return float(edge_match.group(1))

        # Generic fallback (less reliable)
        generic_match = re.search(r'temp\d+_input:\s*(\d+\.\d+)', output)
        if generic_match:
            return float(generic_match.group(1))

    except FileNotFoundError:
        print("Warning: 'sensors' command not found. Cannot get GPU temperature.")
    except Exception as e:
        print(f"Warning: Error getting GPU temperature via sensors: {e}")

    return -1.0


def clear_gpu_memory():
    """Clear PyTorch GPU memory cache and run Python garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# --- JSON/Data Handling Utilities ---

def convert_ndarray_to_list(obj: Any) -> Any:
    """
    Recursively convert numpy ndarrays and numpy numeric types within an object
    (dict, list, or scalar) to native Python types for JSON serialization.
    Compatible with NumPy 2.0+.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # --- Check specific numpy integer types ---
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    # --- Check specific numpy float types ---
    # <<< CHANGE: Replace np.float_ with np.float64, np.float32, etc. >>>
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    # --- Check specific numpy complex types ---
    elif isinstance(obj, (np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}
    # --- Check specific numpy boolean type ---
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    # --- Check for void type (e.g., from structured arrays) ---
    elif isinstance(obj, (np.void)):
        return None # Or suitable representation
    # --- Recursively handle containers ---
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    # --- Return object unchanged if not explicitly handled ---
    else:
        return obj


def safe_json_load(file_path: str) -> List[Dict]:
    """
    Safely load a list of dictionaries from a JSON file.
    Handles FileNotFoundError and JSONDecodeError, returning an empty list
    and printing a warning in those cases. Creates the file if it doesn't exist.
    Uses absolute paths based on the provided file_path.
    """
    abs_file_path = Path(file_path).resolve()
    try:
        # Ensure directory exists
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Use the absolute path for file operations
        with open(abs_file_path, 'r') as file:
            content = file.read()
            # Handle empty file case
            if not content.strip():
                return []
            data = json.loads(content)
            # Expecting a list of experiments (dictionaries)
            if isinstance(data, list):
                return data
            else:
                print(f"WARNING: Expected a list in {abs_file_path}, but found {type(data)}. Returning empty list.")
                return []
    except json.JSONDecodeError:
        print(f"WARNING: {abs_file_path} contains invalid JSON. Returning empty list.")
        return []
    except FileNotFoundError:
        print(f"WARNING: {abs_file_path} not found. Creating a new empty file.")
        try:
            with open(abs_file_path, 'w') as file:
                json.dump([], file) # Create file with an empty list
        except Exception as e:
             print(f"ERROR: Could not create empty file at {abs_file_path}: {e}")
        return []
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading {abs_file_path}: {e}")
        return []

# --- Experiment ID and Configuration Utilities ---

def _find_max_daily_counter(date_prefix: str,
                           pending_file: str = PENDING_EXPERIMENTS_FILE,
                           completed_file: str = COMPLETED_EXPERIMENTS_FILE) -> int:
    """
    Internal helper to find the highest experiment counter ('N' in YYYYMMDD-N)
    for a given date prefix across both pending and completed experiments.
    Uses the globally defined absolute paths for JSON files.
    """
    max_counter = 0

    # Load experiments from both files using their absolute paths
    pending_experiments = safe_json_load(pending_file)
    completed_experiments = safe_json_load(completed_file)
    all_experiments = pending_experiments + completed_experiments

    for exp in all_experiments:
        exp_id = exp.get('experiment_id', '')
        if exp_id.startswith(date_prefix + '-'):
            try:
                # Extract the counter part after the last hyphen
                counter_str = exp_id.split('-')[-1]
                counter = int(counter_str)
                max_counter = max(max_counter, counter)
            except (ValueError, IndexError):
                # Ignore IDs that don't match the expected format
                continue

    return max_counter


def generate_experiment_id() -> str:
    """
    Generates the next unique experiment ID in the format YYYYMMDD-N.
    Checks both pending and completed experiments to ensure the 'N' suffix
    is unique for the current date.
    """
    today_prefix = datetime.now().strftime("%Y%m%d")
    # Pass the absolute file paths to the helper function
    next_counter = _find_max_daily_counter(today_prefix,
                                          pending_file=PENDING_EXPERIMENTS_FILE,
                                          completed_file=COMPLETED_EXPERIMENTS_FILE) + 1
    return f"{today_prefix}-{next_counter}"


def get_experiment_config_by_id(exp_id: str,
                                pending_file: str = PENDING_EXPERIMENTS_FILE,
                                completed_file: str = COMPLETED_EXPERIMENTS_FILE) -> Optional[Dict]:
    """
    Retrieves the full configuration dictionary for a given experiment ID
    by searching both pending and completed experiment files using their absolute paths.

    Args:
        exp_id: The canonical experiment ID (e.g., "20231027-5").

    Returns:
        The experiment configuration dictionary if found, otherwise None.
    """
    # Search pending experiments first using absolute path
    pending_experiments = safe_json_load(pending_file)
    for exp in pending_experiments:
        if exp.get('experiment_id') == exp_id:
            return exp

    # If not found in pending, search completed experiments using absolute path
    completed_experiments = safe_json_load(completed_file)
    for exp in completed_experiments:
        if exp.get('experiment_id') == exp_id:
            return exp

    # Return None if the ID is not found in either file
    return None

# Example Usage (Optional - useful for testing utils.py directly)
if __name__ == "__main__":
    print("\n--- Running utils.py Self-Tests ---")
    print(f"Using Project Root: {_PROJECT_ROOT}")
    print(f"Using Pending File: {PENDING_EXPERIMENTS_FILE}")
    print(f"Using Completed File: {COMPLETED_EXPERIMENTS_FILE}")

    print("\nTesting Experiment ID Generation:")
    # Ensure files exist for testing
    safe_json_load(PENDING_EXPERIMENTS_FILE)
    safe_json_load(COMPLETED_EXPERIMENTS_FILE)

    new_id = generate_experiment_id()
    print(f"Generated new experiment ID: {new_id}")

    print("\nTesting Experiment Config Lookup:")
    # Add a dummy entry to pending for lookup test
    dummy_id = generate_experiment_id()
    dummy_config = {"experiment_id": dummy_id, "hyperparams": {"lr": 0.01}, "data_params": {}}
    temp_pending = safe_json_load(PENDING_EXPERIMENTS_FILE)
    temp_pending.append(dummy_config)
    try:
        with open(PENDING_EXPERIMENTS_FILE, 'w') as f:
             json.dump(temp_pending, f)
        print(f"Added dummy experiment {dummy_id} to pending file for test.")
    except Exception as e:
        print(f"Error adding dummy data: {e}")


    config = get_experiment_config_by_id(dummy_id)
    if config:
        print(f"Found config for {dummy_id}:")
        print(json.dumps(config, indent=2))
    else:
        print(f"Config for {dummy_id} not found (unexpected).")

    non_existent_id = "19990101-1"
    config_non = get_experiment_config_by_id(non_existent_id)
    if not config_non:
        print(f"Correctly did not find config for non-existent ID {non_existent_id}.")
    else:
         print(f"Incorrectly found config for non-existent ID {non_existent_id}.")

    # Clean up dummy entry (optional)
    final_pending = [exp for exp in safe_json_load(PENDING_EXPERIMENTS_FILE) if exp.get('experiment_id') != dummy_id]
    with open(PENDING_EXPERIMENTS_FILE, 'w') as f: json.dump(final_pending, f)
    print(f"Removed dummy experiment {dummy_id} from pending file.")


    print("\nTesting GPU Temp:")
    temp = get_gpu_temp()
    if temp != -1.0:
        print(f"GPU Temperature: {temp}°C")
    else:
        print("Could not retrieve GPU temperature.")

    print("\n--- utils.py Self-Tests Complete ---")

    print("\nTesting convert_ndarray_to_list:")
    # Create sample numpy data
    sample_data = {
        'a': np.array([1, 2, 3]),
        'b': np.int64(10),
        'c': np.float32(3.14),
        'd': True, # Normal python type
        'e': np.bool_(True),
        'f': [np.array([4.5, 5.5]), np.int16(100)],
        'g': {'nested_array': np.arange(4).reshape(2,2), 'nested_float': np.float64(9.9)}
    }
    print("  Original Sample Data (with NumPy types):")
    print(f"    {sample_data}")

    try:
        converted_data = convert_ndarray_to_list(sample_data)
        print("\n  Converted Data (should be JSON serializable):")
        # Try to dump/load via json to be sure
        try:
            json_str = json.dumps(converted_data, indent=4)
            print(json_str)
            # Double check types after conversion
            print("\n  Verifying types in converted data:")
            print(f"    type(converted_data['a']): {type(converted_data['a'])}")
            print(f"    type(converted_data['b']): {type(converted_data['b'])}")
            print(f"    type(converted_data['c']): {type(converted_data['c'])}")
            print(f"    type(converted_data['e']): {type(converted_data['e'])}")
            print(f"    type(converted_data['f'][0]): {type(converted_data['f'][0])}")
            print(f"    type(converted_data['f'][1]): {type(converted_data['f'][1])}")
            print(f"    type(converted_data['g']['nested_array']): {type(converted_data['g']['nested_array'])}")
            print(f"    type(converted_data['g']['nested_float']): {type(converted_data['g']['nested_float'])}")
            print("  ✅ convert_ndarray_to_list test PASSED (JSON serializable).")
        except TypeError as json_err:
            print(f"  ❌ convert_ndarray_to_list test FAILED. Data not JSON serializable: {json_err}")
            print("     Converted data was:", converted_data)

    except Exception as convert_err:
        # Catch errors during the conversion itself (like the old np.float_ error)
        print(f"  ❌ convert_ndarray_to_list test FAILED during conversion: {convert_err}")

    print("\n--- utils.py Self-Tests Complete ---")
