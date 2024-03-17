import json
from train_lstm import LSTMClassifier
from sklearn.preprocessing import LabelEncoder
import torch
import string
import torch.nn.functional as F


DEFAULT_COMPLETED_FILE = 'data/completed_experiments.json'

# maybe too hard? why can't they get the english though... 

E1 = """When the people of America reflect that they are now called upon to decide a question, which, in its consequences, must prove one of the most important that ever engaged their attention, the propriety of their taking a very comprehensive, as well as a very serious, view of it, will be evident.
"""

K1 = """EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJ
YQTQUXQBQVYUVLLTREVJYQTMKYRDMFD"""
K2 = """VFPJUDEEHZWETZYVGWHKKQETGFQJNCE
GGWHKK?DQMCPFQZDQMMIAGPFXHQRLG
TIMVMZJANQLVKQEDAGDVFRPJUNGEUNA
QZGZLECGYUXUEENJTBJLBQCRTBJDFHRR
YIZETKZEMVDUFKSJHKFWHKUWQLSZFTI
HHDDDUVH?DWKBFUFPWNTDFIYCUQZERE
EVLDKFEZMOQQJLTTUGSYQPFEUNLAVIDX
FLGGTEZ?FKZBSFDQVGOGIPUFXHHDRKF
FHQNTGPUAECNUVPDJMQCLQUMUNEDFQ
ELZZVRRGKFFVOEEXBDMVPNFQXEZLGRE
DNQFMPNZGLFLPMRJQYALMGNUVPDXVKP
DQUMEBEDMHDAFMJGZNUPLGEWJLLAETG"""
K3 = """ENDYAHROHNLSRHEOCPTEOIBIDYSHNAIA
CHTNREYULDSLLSLLNOHSNOSMRWXMNE
TPRNGATIHNRARPESLNNELEBLPIIACAE
WMTWNDITEENRAHCTENEUDRETNHAEOE
TFOLSEDTIWENHAEIOYTEYQHEENCTAYCR
EIFTBRSPAMHHEWENATAMATEGYEERLB
TEEFOASFIOTUETUAEOTOARMAEERTNRTI
BSEDDNIAAHTTMSTEWPIEROAGRIEWFEB
AECTDDHILCEIHSITEGOEAOSDDRYDLORIT
RKLMLEHAGTDHARDPNEOHMGFMFEUHE
ECDMRIPFEIMEHNLSSTTRTVDOHW?"""
K4 = """OBKR
UOXOGHULBSOLIFBBWFLRVQQPRNGKSSO
TWTQSJQSSEKZZWATJKLUDIAWINFBNYP
VTTMZFPKWGDKZXTJCDIGKUHUAUEKCAR"""

DEFAULT_TEXTS = {"english":E1, "k1":K1, "k2":K2, "k3":K3, "k4":K4}


def get_top_experiments(completed_experiments_file=DEFAULT_COMPLETED_FILE, top_n=5, sort_key='val_accuracy'):
    with open(completed_experiments_file, 'r') as f:
        experiments = json.load(f)

    # Sorting experiments based on the last value of the specified metric (assuming improvement over time)
    sorted_experiments = sorted(
        experiments,
        key=lambda x: x['metrics'][sort_key][-1],
        reverse=True  # Assuming higher is better; reverse for loss
    )

    sorted_uids = [experiment['uid'] for experiment in sorted_experiments]

    return sorted_uids[:top_n]


def load_model(model_uid, completed_experiments_file=DEFAULT_COMPLETED_FILE):
    with open(completed_experiments_file, 'r') as f:
        experiments = json.load(f)
    target_experiment = next((exp for exp in experiments if exp['uid'] == model_uid), None)

    if not target_experiment:
        raise FileNotFoundError(f"Experiment with UID {model_uid} not found.")
    
    model_filename = target_experiment['model_filename']
    data_params = target_experiment['data_params']
    hyperparams = target_experiment['hyperparams']
    embedding_dim = hyperparams['embedding_dim']
    hidden_dim = hyperparams['hidden_dim']
    ciphers = data_params['ciphers']
    output_dim = len(ciphers)
    vocab_size = 27

    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_filename))
    model.eval()
        
    return model


def get_experiment(model_uid, completed_experiments_file=DEFAULT_COMPLETED_FILE):
    with open(completed_experiments_file, 'r') as f:
        experiments = json.load(f)
    target_experiment = next((exp for exp in experiments if exp['uid'] == model_uid), None)

    if not target_experiment:
        raise FileNotFoundError(f"Experiment with UID {model_uid} not found.")

    return target_experiment
    

def preprocess_text(input_text, max_length=500):
    # Convert input_text to lowercase to match training preprocessing
    input_text = input_text.lower()
    
    # Create a character-level tokenizer dictionary
    unique_chars = string.ascii_lowercase # set(''.join(data['text']))
    # +1 to reserve 0 for padding
    token_dict = {char: i+1 for i, char in enumerate(unique_chars)}

    # Tokenize using custom_text_tokenizer or similar function
    tokenized_text = [token_dict.get(char, 0) for char in input_text][:max_length]
    
    # Pad sequence
    padded_text = torch.tensor(tokenized_text, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    if len(tokenized_text) < max_length:
        padded_text = F.pad(padded_text, (0, max_length - len(tokenized_text)), "constant", 0)
    
    return padded_text

def predict_with_model(model_uid, text_to_test):
    target_experiment = get_experiment(model_uid)
    model = load_model(model_uid)
    preprocessed_text = preprocess_text(text_to_test)
    
    data_params = target_experiment['data_params']
    hyperparams = target_experiment['hyperparams']
    embedding_dim = hyperparams['embedding_dim']
    hidden_dim = hyperparams['hidden_dim']
    ciphers = data_params['ciphers']
    output_dim = len(ciphers)
    vocab_size = 27

    with torch.no_grad():
        output = model(preprocessed_text)
        predicted_index = output.argmax(dim=1).item()
        label_encoder = LabelEncoder()
        label_encoder.fit(ciphers)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label



top_exp = None
for e in get_top_experiments(top_n=5):
    print(e)
    print('----')
    for default_text_key in DEFAULT_TEXTS.keys():
        prediction = predict_with_model(e, DEFAULT_TEXTS[default_text_key])
        print(f"{default_text_key}: {prediction}")
    print()

