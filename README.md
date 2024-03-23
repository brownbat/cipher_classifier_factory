# Cipher_Classifier_Factory
researcher.py is the main entry point.

It builds LSTMs that can classify simple ciphers. It is designed to make it easy to build several different models to compare the effectiveness of various hyperparameters.

It uses a set of default parameters like so:
```
{
    'ciphers': [_get_cipher_names()],
    'num_samples': [10000],
    'sample_length': [500],
    'epochs': [30],
    'num_layers': [64, 128],
    'batch_size': [128],
    'embedding_dim': [64],
    'hidden_dim': [64, 128],
    'dropout_rate': [0.2],
    'learning_rate': [0.004]
}
```

For items with multiple values, it will generate a series of experiments with all combinations of those values. Here, all models will be trained on 10,000 samples and sample lengths of 500 and so on. However, we will plan to generate four models, the combination of 64 and 128 layers with 64 and 128 hidden dimensions.

Researcher will first generate these combinations and populate "pending_experiments.json" and remove any duplicates. You can also add experiments to that json directly if you like, though you may want to clear it if you have generated a large combination of possible features and want to clear it and prioritize a new experiment.

Then the experiments will be run in turn, filling in metrics for later analysis.

Other features

- For analysis of results, the system will generate confidence matrices and also can build a visualization of completed experiments using flask using visualization.py, which will host the visualization locally.
- query_model.py lets you categorize text samples using the five top performing models you've generated, to compare their results in the wild.
- The system can send notifications to you when it completes individual experiments using discord or email if you provide details to notifications.py.

You can set parameters using the default_parameters dictionary at the top of researcher.py or through command line arguments.

ROADMAP: Add more classical ciphers. Implement transformers.
  
![demo](https://github.com/brownbat/cipher_classifier_factory/assets/26754/0f89f7a5-14b5-496e-ac74-6d21d8b2180d)

You can play with a subset of the collected data [here](https://brownbat.pythonanywhere.com/).

**Findings**
Performance is extremely nonlinear to different hyperparameters. There's not one you can just crank or set and forget and always get better results, they are all very interdependent. It seems there's high sensitivity to initial conditions, with heavy dependence on randomness like how the samples are divided into training and validation sets or something similar.

It's easy to quickly get to 99% accuracy distinguishing english, random, substitution, columnar, and vigenere.

Adding several other ciphers, even if they aren't very complex, seems to make it hard to learn anything at all. In sets of all ciphers, Playfair is often identified well before and with lower dimensions than even pure english or random noise.

I haven't used feature engineering yet. What happens if I tag these with frequencies? IoCs? Is that necessary?

The system can also quickly and easily train a model to distinguish between vigenere encoded with different prespecified keys, and I don't think it's decrypting then checking if the output is english...
This includes quickly differentiating between keypairs with small internal misspellings, such as 'palimpsest' and 'palinpsest'
That's... very weird, right? The plaintext is still random samples of English, so the mathematical patterns here shouldn't be so trivial. Most vigenere attacks rely on some kind of periodicity of frequency anaylsis... is it doing that much with small dimensions? That's a lot of operations to string together before there's any value to extract at all, so probably not doing all that, but it's also notable that it might see more direct and accessible patterns that amateur solvers have simply missed over the decades.) It would be interesting to try some interpetability experiments here.

Larger dimensions have led to overheating and crashes on a 7900 XTX, from rocm5.6 up to rocm6.0.2, though I do not have these issues when renting 4090s. I am hoping that AMD updates might help address this.

