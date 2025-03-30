# Cipher Classifier Findings & Thoughts

## Key Findings

Performance is extremely nonlinear to different hyperparameters. There's not one you can just crank or set and forget and always get better results, they are all very interdependent. It seems there's high sensitivity to initial conditions, with heavy dependence on randomness like how the samples are divided into training and validation sets or something similar.

It's easy to quickly get to 99% accuracy distinguishing english, random, substitution, columnar, and vigenere.

Adding several other ciphers, even if they aren't very complex, seems to make it hard to learn anything at all. In sets of all ciphers, Playfair is often identified well before and with lower dimensions than even pure english or random noise.

I haven't used feature engineering yet. What happens if I tag these with frequencies? IoCs? Is that necessary?

The system can also quickly and easily train a model to distinguish between vigenere encoded with different prespecified keys, and I don't think it's decrypting then checking if the output is english...

This includes quickly differentiating between keypairs with small internal misspellings, such as 'palimpsest' and 'palinpsest'

That's... very weird, right? The plaintext is still random samples of English, so the mathematical patterns here shouldn't be so trivial. Most vigenere attacks rely on some kind of periodicity of frequency anaylsis... is it doing that much with small dimensions? That's a lot of operations to string together before there's any value to extract at all, so probably not doing all that, but it's also notable that it might see more direct and accessible patterns that amateur solvers have simply missed over the decades.) It would be interesting to try some interpetability experiments here.

Larger dimensions have led to overheating and crashes on a 7900 XTX, from rocm5.6 up to rocm6.0.2, though I do not have these issues when renting 4090s. I am hoping that AMD updates might help address this.

## Current Model Performance

Our transformer models have reached 78.74% accuracy on a 10-cipher classification task. We're still exploring optimal hyperparameter configurations through the 72 experiments to better understand how different architectural choices affect performance.

We need further experimentation to draw stronger conclusions about the transformer architecture's effectiveness for cipher classification.

## Unusual Observations

The most fascinating finding is how well the models can differentiate between Vigenère ciphers with nearly identical keys. This suggests the models are identifying subtle statistical artifacts that human cryptanalysts might normally miss. It's unclear if the model is finding frequency patterns, building some internal representation of the key, or discovering entirely new statistical features.

When visualizing confusion matrices over training time, certain cipher types (like Playfair) are consistently recognized first, suggesting some cipher types have more distinctive statistical signatures than others.

## Hardware Notes

The transformer models with d_model > 256 tend to cause overheating issues on AMD 7900 XTX GPUs, but run fine on NVIDIA 4090s. This appears to be related to how PyTorch utilizes ROCm drivers versus CUDA drivers, and may improve with future ROCm updates.

Gradient clipping is crucial for stable transformer training, with max_norm=1.0 working well across all experiments.

## Future Research Directions

1. Investigate model interpretability to understand what features the transformer is actually using to distinguish cipher types

2. Test whether explicit feature engineering (IoC calculations, frequency analysis) improves model performance or if transformers already implicitly learn these features

3. Attempt transfer learning: train a model on a subset of ciphers, then fine-tune for new cipher types

4. Explore using the model for partial decryption hints - can the attention patterns reveal key length or other useful decryption parameters?