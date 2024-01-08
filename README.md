# Cipher_Classifier_Factory
 Builds LSTMs to classify ciphers, allows one to shift hyperparameters to measure effectiveness of competing models.

Set up experiments to run in pending_experiments.json in the data directory.

Entries in this format:
```
[
  {
    "data_params": {
      "ciphers": [
        "english",
        "vigenere",
        "caesar",
        "columnar_transposition",
        "random_noise"
      ],
      "num_samples": 10000,
      "sample_length": 500
    },
    "hyperparams": {
      "epochs": 30,
      "num_layers": 32,
      "batch_size": 64,
      "embedding_dim": 32,
      "hidden_dim": 192,
      "dropout_rate": 0.3,
      "learning_rate": 0.003
    },
    "experiment_id": "exp_9"
  },
  {
    "data_params": {
      "ciphers": [
        "english",
        ...
```

  The experiments will be run in turn, filling in metrics for later analysis.

Alternatively, specify lists of parameters (currently unceremoniously hard coded at the top of researcher.py)
```
params = {
        'ciphers': [['english', 'vigenere', 'caesar', 'columnar_transposition', 'random_noise']],
        'num_samples': [10000],
        'sample_length': [500],
        'epochs': [30],
        'num_layers': [32, 64, 128],
        'batch_size': [64, 128, 256],
        'embedding_dim': [32, 64, 128],
        'hidden_dim': [192, 256, 512],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.002, 0.003]
    }
```
Researcher will calculate all combination of provided options and populate pending_experiments.json with those options, then run through them in sequence.

NOTE: yaml has been deprecated, it was far too slow.

Currently many models are getting into the 97-99% accuracy range after just a minute of training on one 7900 XTX. Further experiments will help identify how much training with what settings will be needed to provide robust identification of many known classical ciphers. Effectiveness may hit ceilings as we approach more complex ciphers that implement some combination of substitution and transposition or multiple rounds of encipherment.

ROADMAP: Add more classical ciphers. Tack on an attention layer in place. Eventually move fully to a transformer model.
  
![demo](https://github.com/brownbat/cipher_classifier_factory/assets/26754/0f89f7a5-14b5-496e-ac74-6d21d8b2180d)

You can play with a subset of the collected data [here](https://brownbat.pythonanywhere.com/).

So far I've been surprised with how nonlinearly performance responds to different hyperparameters, there's not one you can just crank and always get better results, they are all very interdependent. This may indicate that there's high sensitivity to initial conditions, possibly heavily dependent on randomness like how the samples are divided into training and validation sets or something similar.
