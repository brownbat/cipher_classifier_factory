#!/usr/bin/env python3
"""
Query Tool for Cipher Classification Models

Loads a specified trained model (by experiment ID) or identifies top-performing
models from completed experiments, then classifies provided text samples or
default samples.
"""
import json
import torch
import string
import argparse
import os
import sys

# --- Use utilities ---
import models.common.utils as utils
# --- Use standardized inference functions ---
from models.transformer.inference import load_model, classify_text

# Default path, using utils
DEFAULT_COMPLETED_FILE = utils.COMPLETED_EXPERIMENTS_FILE


# Sample test texts
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

DEFAULT_TEXTS = {"english": E1, "k1": K1, "k2": K2, "k3": K3, "k4": K4}


def find_experiments_for_query(args):
    """
    Identifies experiment IDs to query based on command-line arguments.

    Args:
        args: Parsed argparse arguments.

    Returns:
        list: A list of experiment IDs to query.
              Returns an empty list if none are found or an error occurs.
    """
    target_ids = []
    if args.experiment_id:
        target_ids = args.experiment_id # Use IDs directly provided
        # Optionally verify these IDs exist using utils.get_experiment_config_by_id
        valid_ids = []
        for exp_id in target_ids:
             config = utils.get_experiment_config_by_id(exp_id)
             if config and config.get('model_filename'): # Check if likely completed
                  valid_ids.append(exp_id)
             else:
                  print(f"Warning: Experiment ID '{exp_id}' not found or appears incomplete. Skipping.")
        target_ids = valid_ids

    elif args.top_n:
        print(f"Finding top {args.top_n} experiments based on '{args.sort_key}'...")
        experiments = utils.safe_json_load(DEFAULT_COMPLETED_FILE)
        if not experiments:
            print(f"Error: No completed experiments found in {DEFAULT_COMPLETED_FILE}")
            return []

        # Filter experiments that have the sort key metric and are likely valid
        valid_experiments = []
        for exp in experiments:
            metric_val = exp.get('metrics', {}).get(args.sort_key)
            # Check if metric exists and is a finite number, and model file exists
            if metric_val is not None and isinstance(metric_val, (int, float)) and \
               math.isfinite(metric_val) and exp.get('model_filename'):
                valid_experiments.append(exp)
            # Handle cases where metric is a list (use best or last?) - Use 'best' metric directly
            elif args.sort_key == 'best_val_accuracy' or args.sort_key == 'best_val_loss':
                 best_metric_val = exp.get('metrics', {}).get(args.sort_key)
                 if best_metric_val is not None and isinstance(best_metric_val, (int, float)) and \
                    math.isfinite(best_metric_val) and exp.get('model_filename'):
                      valid_experiments.append(exp)


        if not valid_experiments:
             print(f"Error: No valid experiments found with metric '{args.sort_key}'.")
             return []

        try:
            # Sort based on the specific metric value
            reverse_sort = (args.sort_key == 'best_val_accuracy') # Higher accuracy is better
            sorted_experiments = sorted(
                valid_experiments,
                key=lambda x: x['metrics'][args.sort_key],
                reverse=reverse_sort
            )
            target_ids = [exp['experiment_id'] for exp in sorted_experiments[:args.top_n] if 'experiment_id' in exp]
        except KeyError:
            print(f"Error: Metric '{args.sort_key}' not found consistently in experiments.")
            return []
        except Exception as e:
             print(f"Error sorting experiments: {e}")
             return []

    else:
        # Should not happen if argparse group is required, but handle defensively
        print("Error: No experiment selection criteria provided (--experiment_id or --top_n).")
        return []

    if not target_ids:
         print("No experiments selected to query.")

    return target_ids


def main():
    """Loads model(s) and runs predictions."""
    parser = argparse.ArgumentParser(description="Query trained cipher classification models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-e', '--experiment_id', nargs='+', help='Specific experiment ID(s) to query.')
    group.add_argument('-t', '--top_n', type=int, help="Query the top N performing models.")

    parser.add_argument('-s', '--sort_key', default='best_val_accuracy',
                        choices=['best_val_accuracy', 'best_val_loss'], # Restrict choices
                        help="Metric to use for sorting top models (default: best_val_accuracy).")
    parser.add_argument('--text', type=str, help="Custom text string to classify.")
    parser.add_argument('--file', type=str, help="Path to a text file to classify.")

    args = parser.parse_args()

    # --- Determine Text Input ---
    texts_to_classify = {}
    if args.text:
        texts_to_classify['custom_text'] = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts_to_classify[os.path.basename(args.file)] = f.read()
        except Exception as e:
            print(f"Error reading file '{args.file}': {e}")
            sys.exit(1)
    else:
        print("No specific text/file provided, using default samples.")
        texts_to_classify = DEFAULT_TEXTS

    # --- Find Experiments to Query ---
    experiment_ids = find_experiments_for_query(args)
    if not experiment_ids:
        sys.exit(1)

    # --- Load and Query Each Model ---
    for exp_id in experiment_ids:
        print(f"\n--- Querying Experiment: {exp_id} ---")

        # Construct relative paths
        model_path_rel = f"data/models/{exp_id}.pt"
        metadata_path_rel = f"data/models/{exp_id}_metadata.json"

        # Load model using the standardized function from inference.py
        loaded_data = load_model(model_path_rel, metadata_path_rel)

        if loaded_data is None:
            print(f"Failed to load model for {exp_id}. Skipping.")
            continue

        # Extract components
        model = loaded_data['model']
        token_dict = loaded_data['token_dict']
        label_encoder = loaded_data['label_encoder']
        hyperparams = loaded_data['hyperparams'] # Keep for display if needed
        device = model.classifier.weight.device # Get device model is actually on

        print(f"Model loaded successfully onto {device}.")
        # Optionally print key hyperparams
        # print(f"  Params: d={hyperparams.get('d_model')}, h={hyperparams.get('nhead')}, lyr={hyperparams.get('num_encoder_layers')}")

        print("\nPredictions:")
        for name, text in texts_to_classify.items():
            print(f"  Input: '{name}' ({len(text)} chars)")

            # Classify text using the standardized function
            result = classify_text(model, text, token_dict, label_encoder, device=device)

            if "error" in result:
                print(f"    Error: {result['error']}")
            else:
                predicted = result.get('predicted_class', 'N/A')
                confidence = result.get('confidence', 0.0)
                print(f"    Predicted: {predicted} (Confidence: {confidence:.4f})")
                # Optionally print all probabilities
                # print(f"    All Probabilities: {result.get('all_probabilities')}")

        print("-" * (len(exp_id) + 25)) # Adjust separator length


if __name__ == "__main__":
    # Add math import needed by find_experiments_for_query
    import math
    # Ensure project root is found if running script directly
    print(f"Project Root: {utils._PROJECT_ROOT}")
    main()
