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

**Findings**
Performance is extremely nonlinear to different hyperparameters. There's not one you can just crank or set and forget and always get better results, they are all very interdependent. This may indicate that there's high sensitivity to initial conditions, possibly heavily dependent on randomness like how the samples are divided into training and validation sets or something similar.

To test: how much variance is there between runs with the same settings?

It's easy to get to 99% accuracy distinguishing english, random, substitution, columnar, and vigenere. (So chosen so that it recognizes ciphers as neither pure english nor noise, and also distinguishes between substitution and transposition. Adding a mix of substitution and transposition or more diffusion may stump the guesser.)

Adding several other ciphers, even if they aren't very complex, seems to make it hard to learn anything at all. One run it figured out how to distinguish playfair from the others, but was still comlpetely guessing even with english or random noise samples. I'm surprised how rare it is to develop some model for English, start getting that right out of the gate, then slowly get the harder ones, it seems to prefer to have a fully worked out theory before it leaves the safety of random guessing. (Humans seem more content with using very bad theories as stepping stones to better ones.)

I haven't used feature engineering yet, big shortfall. What happens if I tag these with frequencies? IoCs?

System can quickly and easily train a model to distinguish between vigenere encoded with different keys, and I don't think it's decrypting then checking if the output is english. 
This includes keypairs with small internal misspellings, such as 'palimpsest' and 'palinpsest'
That's... weird right? Plaintext is still random samples of English, so the mathematical patterns shouldn't be completely trivial. (Most vigenere attacks rely on some kind of periodicity of frequency anaylsis... is it doing that much? That's a lot of operations to string together before there's value to extract, so that's either notable that it would be reconstructing such a system (through some kind of weird leaps of faith?) or it would also be notable if it saw some much more direct and accessible pattern amateur solvers have simply missed over the years.) It would be interesting to try some interpetability experiments here.

Larger dimensions have led to overheating on a radeon 7900 xtx with rocm 5.6 and even a bit on 5.7. I am hoping that AMD updates will continue to address this.

