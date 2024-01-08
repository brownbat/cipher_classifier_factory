import json
import os
import glob

def concatenate_and_minify_json(directory, pattern, output_file):
    combined_data = []

    # Construct the full path with pattern
    full_pattern = os.path.join(directory, pattern)

    # Find all files matching the pattern
    file_list = glob.glob(full_pattern)

    # Read and combine JSON data from each file
    for file in file_list:
        with open(file, 'r') as f:
            data = json.load(f)
            combined_data.append(data)

    # Minify JSON by dumping it without indentation
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, separators=(',', ':'))

    return file_list

# Set the directory, pattern, and output file
curr_directory = os.getcwd()
directory = curr_directory + '/data/'
pattern = 'completed_experiments*.json'
output_file = 'completed_all.json'

# Run the function and capture the list of processed files
processed_files = concatenate_and_minify_json(directory, pattern, output_file)
print(processed_files)
