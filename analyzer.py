import yaml

with open(file_path, 'r') as file:
    experiments = yaml.safe_load(file) or []

for exp in experiments:
    data_params = exp.get('data_params', {})
    hyperparams = exp.get('hyperparams', {})
    metrics = exp.get('metrics', {})
