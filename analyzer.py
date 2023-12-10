import yaml
import logging
from visualization import run_app

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error reading YAML file: {e}")
        return None

def transform_data(raw_data):
    transformed_data = {}
    for experiment in raw_data:
        hyperparams = (
            experiment['hyperparams']['batch_size'],
            experiment['hyperparams']['dropout_rate'],
            experiment['hyperparams']['embedding_dim'],
            experiment['hyperparams']['epochs'],
            experiment['hyperparams']['hidden_dim'],
            experiment['hyperparams']['learning_rate'],
            experiment['hyperparams']['num_layers']
        )
        # Taking the last values of validation accuracy and loss as final metrics
        val_accuracy = experiment['metrics']['val_accuracy'][-1]
        val_loss = experiment['metrics']['val_loss'][-1]
        training_duration = experiment['metrics']['training_duration']

        transformed_data[hyperparams] = (training_duration, val_loss)
    return transformed_data


def main():
    logging.basicConfig(level=logging.INFO)
    file_path = "data/completed_experiments.yaml"
    
    raw_data = read_yaml(file_path)
    if raw_data is None:
        return
    
    structured_data = transform_data(raw_data)
    
    run_app(structured_data)

if __name__ == "__main__":
    main()
