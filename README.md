# Cipher_Classifier_Factory
 Builds LSTMs to classify ciphers, allows one to shift hyperparameters to measure effectiveness of competing models.

Set up experiments to run in experiments.yaml in the data directory. 

Each entry of the form:
```
- data_params:
    num_samples: 1000
    sample_length: 500
  experiment_id: exp_all_ciphers_1000_samples
  hyperparams:
    activation_func: relu
    batch_size: 32
    embedding_dim: 128
    epochs: 5
    hidden_dim: 128
    learning_rate: 0.001
  metrics: {}
```

  The experiments will be run in turn, filling in metrics for later analysis.

  (WARNING: This yaml structure will be revised in a future version.)
