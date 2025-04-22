#!/usr/bin/env python3
"""
Quick script to remove today's experiments from completed_experiments.json
"""
import json
import os
import datetime

# Get today's date in YYYYMMDD format
today = datetime.datetime.now().strftime("%Y%m%d")
print(f"Today's date prefix: {today}")

# Path to completed experiments
file_path = "data/completed_experiments.json"

try:
    # Load existing data
    with open(file_path, 'r') as f:
        all_experiments = json.load(f)
    
    # Count before filtering
    initial_count = len(all_experiments)
    print(f"Initial experiment count: {initial_count}")
    
    # Filter out experiments with today's IDs
    filtered_experiments = [exp for exp in all_experiments 
                           if not exp.get('experiment_id', '').startswith(f"{today}-")]
    
    # Count after filtering
    final_count = len(filtered_experiments)
    removed_count = initial_count - final_count
    print(f"Removed {removed_count} experiments from today ({today})")
    
    # Save the filtered data
    with open(file_path, 'w') as f:
        json.dump(filtered_experiments, f, indent=2)
    
    print(f"Successfully updated {file_path}")
    print(f"New experiment count: {final_count}")
    
except Exception as e:
    print(f"Error: {e}")