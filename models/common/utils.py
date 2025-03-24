"""
Common utility functions used by both LSTM and transformer implementations.
"""
import hashlib
import re
import subprocess
import gc
import torch
import numpy as np
import json
from datetime import datetime


def file_hash(filename):
    """Generate a hash for a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_gpu_temp():
    """Get current GPU temperature from sensors."""
    process = subprocess.Popen(["sensors"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output, _ = process.communicate()

    # Convert bytes to string
    sensor_data = output.decode()

    # Regex to find temperature lines
    junction_temp = re.search(r'junction:\s+\+(\d+\.\d+)°C', sensor_data)

    # Extract temperatures
    junction_temp_val = float(junction_temp.group(1)) if junction_temp else -1

    return junction_temp_val


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    torch.cuda.empty_cache()
    gc.collect()


def convert_ndarray_to_list(obj):
    """Convert numpy ndarrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    else:
        return obj


def safe_json_load(file_path):
    """Safely load JSON from a file, handling errors and creating the file if needed."""
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