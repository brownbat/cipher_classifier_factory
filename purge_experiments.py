#!/usr/bin/env python3
"""
Utility to remove experiments from completed/pending logs, delete associated artifacts,
and handle partial runs. Supports filtering or purging all experiments.
"""
import json
import os
import sys
import argparse
import datetime  # For --today mode
import shutil # For potentially removing directories if needed later
import glob   # For pattern matching files

# Import common utilities
try:
    import models.common.utils as utils
except ImportError:
    print("ERROR: Cannot import models.common.utils. Ensure it's in the PYTHONPATH.")
    sys.exit(1)

# --- Configuration: Use paths derived from utils ---
COMPLETED_FILE = utils.COMPLETED_EXPERIMENTS_FILE
PENDING_FILE = utils.PENDING_EXPERIMENTS_FILE
MODEL_DIR = os.path.join(utils._PROJECT_ROOT, 'data', 'models')
METADATA_DIR = os.path.join(utils._PROJECT_ROOT, 'data', 'models') # Same dir
CHECKPOINT_DIR = os.path.join(utils._PROJECT_ROOT, 'data', 'checkpoints')
CM_HISTORY_DIR = os.path.join(utils._PROJECT_ROOT, 'data', 'cm_history') # New
CM_GIF_DIR = os.path.join(utils._PROJECT_ROOT, 'data', 'cm')
LOSS_GRAPH_DIR = os.path.join(utils._PROJECT_ROOT, 'data', 'loss_graphs')
# Add other artifact directories if they exist (e.g., data/trend_matrix)

# --- Filter Parsing --- (Keep existing functions, adjusted slightly for clarity)
def parse_filter_string(filter_str):
    """Parses filter string like 'param=val1,val2;param2=val3'"""
    filter_params = {'hyperparams': {}, 'data_params': {}}
    param_groups = filter_str.split(';')
    for group in param_groups:
        if not group.strip() or '=' not in group: continue
        param, values_str = group.split('=', 1)
        param = param.strip()
        value_strings = [v.strip() for v in values_str.split(',') if v.strip()]
        parsed_values = []
        for val in value_strings:
            try: parsed_values.append(int(val))
            except ValueError:
                try: parsed_values.append(float(val))
                except ValueError: parsed_values.append(val)

        # Heuristic to determine if hyperparam or data_param
        # Consider making this explicit if ambiguity arises
        if param in ['ciphers', 'num_samples', 'sample_length']:
             filter_params['data_params'][param] = parsed_values
        else:
             filter_params['hyperparams'][param] = parsed_values
    return filter_params

def load_filter_config(filter_arg):
    """Loads filter from string or JSON file."""
    if os.path.exists(filter_arg) and filter_arg.endswith('.json'):
        try:
            with open(filter_arg, 'r') as f: return json.load(f)
        except Exception as e:
            print(f"ERROR: Could not load filter file '{filter_arg}': {e}"); return None
    else:
        try: return parse_filter_string(filter_arg)
        except Exception as e: print(f"ERROR: Could not parse filter string: {e}"); return None

# --- Matching Logic ---
def match_experiment(exp, filter_params):
    """Check if an experiment matches filter criteria (for --filter mode)."""
    # Helper to check a value against filter list
    def check_match(actual_value, filter_values):
        # Handle list matching specifically for 'ciphers' if needed
        # For now, assume exact match for simplicity
        return actual_value in filter_values

    # Match data parameters
    if 'data_params' in filter_params:
        exp_data = exp.get('data_params', {})
        for param, values in filter_params['data_params'].items():
            if param in exp_data and check_match(exp_data[param], values):
                 return True # Return true if any data param matches

    # Match hyperparameters
    if 'hyperparams' in filter_params:
        exp_hyper = exp.get('hyperparams', {})
        for param, values in filter_params['hyperparams'].items():
            if param in exp_hyper and check_match(exp_hyper[param], values):
                 return True # Return true if any hyperparam matches

    return False # No match found

# --- File Deletion ---
def delete_experiment_artifacts(exp_id, thorough=False, dry_run=False):
    """
    Delete ALL files associated with a specific experiment ID.
    Uses glob for more robust finding of visualization files.
    Returns a list of files that *would be* or *were* deleted.
    """
    if not exp_id:
        print("  Warning: Cannot delete files for missing experiment ID.")
        return []

    files_actioned = []
    action = "Would delete" if dry_run else "Deleting"

    # Helper to attempt removal using absolute path
    def _remove_file(filepath_abs):
        if filepath_abs and os.path.exists(filepath_abs):
            filepath_rel = os.path.relpath(filepath_abs, utils._PROJECT_ROOT)
            files_actioned.append(filepath_rel)
            print(f"  {action}: {filepath_rel}")
            if not dry_run:
                try: os.remove(filepath_abs)
                except Exception as e: print(f"    ERROR removing {filepath_rel}: {e}")
        elif filepath_abs:
             pass # File path constructed but doesn't exist

    # Construct expected paths
    model_path_abs = os.path.join(MODEL_DIR, f"{exp_id}.pt")
    metadata_path_abs = os.path.join(METADATA_DIR, f"{exp_id}_metadata.json")
    cm_history_path_abs = os.path.join(CM_HISTORY_DIR, f"{exp_id}_cm_history.npy")

    # Delete specific files
    _remove_file(model_path_abs)
    _remove_file(metadata_path_abs)
    _remove_file(cm_history_path_abs)

    # Delete checkpoints using glob pattern
    checkpoint_pattern = os.path.join(CHECKPOINT_DIR, f"{exp_id}_*.pt")
    for chk_path in glob.glob(checkpoint_pattern):
         _remove_file(chk_path)

    # Delete visualizations if thorough cleanup requested
    if thorough:
        # CM GIFs using glob pattern
        cm_gif_pattern = os.path.join(CM_GIF_DIR, f"{exp_id}_*.gif")
        for gif_path in glob.glob(cm_gif_pattern):
            _remove_file(gif_path)
        # CM Final PNGs using glob pattern (if generate_trend_matrix created them)
        cm_png_pattern = os.path.join(CM_GIF_DIR, f"{exp_id}_*.png")
        for png_path in glob.glob(cm_png_pattern):
             _remove_file(png_path)


        # <<< --- ADDED: Loss Graphs --- >>>
        # Loss Graphs using glob pattern (PNG and GIF)
        loss_graph_pattern_png = os.path.join(LOSS_GRAPH_DIR, f"{exp_id}_*.png")
        loss_graph_pattern_gif = os.path.join(LOSS_GRAPH_DIR, f"{exp_id}_*.gif")
        for loss_path in glob.glob(loss_graph_pattern_png) + glob.glob(loss_graph_pattern_gif):
            _remove_file(loss_path)

        # Add other visualization patterns here if needed (e.g., trend matrix)

    return files_actioned

# --- Partial Experiment Handling ---
def find_partial_experiments():
    """Find experiment IDs with artifacts but not in completed list."""
    potential_partial_ids = set()

    # Check Checkpoints
    if os.path.exists(CHECKPOINT_DIR):
        for filename in os.listdir(CHECKPOINT_DIR):
            if filename.endswith('.pt'):
                potential_partial_ids.add(filename.split('_')[0]) # Add base ID

    # Check Models
    if os.path.exists(MODEL_DIR):
         for filename in os.listdir(MODEL_DIR):
             if filename.endswith('.pt'):
                 potential_partial_ids.add(filename.replace('.pt',''))
             elif filename.endswith('_metadata.json'):
                 potential_partial_ids.add(filename.replace('_metadata.json',''))

    # Check CM History
    if os.path.exists(CM_HISTORY_DIR):
        for filename in os.listdir(CM_HISTORY_DIR):
             if filename.endswith('_cm_history.npy'):
                 potential_partial_ids.add(filename.replace('_cm_history.npy',''))

    # Load completed IDs
    completed_ids = set()
    if os.path.exists(COMPLETED_FILE):
         try:
             completed_exps = utils.safe_json_load(COMPLETED_FILE)
             completed_ids = {exp.get('experiment_id') for exp in completed_exps if exp.get('experiment_id')}
         except Exception as e: print(f"Warning: Error reading completed file for partial check: {e}")

    # Partial IDs are those found minus those completed
    partial_ids = potential_partial_ids - completed_ids
    return sorted(list(partial_ids))


def cleanup_partial_experiment(exp_id, dry_run=True):
    """Remove ALL artifacts and pending entry for a partial experiment."""
    files_actioned = []
    action = "Would" if dry_run else "Will"

    print(f"  {action} delete ALL artifacts for partial experiment: {exp_id}")
    # Use the main artifact deletion function (ensure thorough is True for partial cleanup)
    deleted_artifacts = delete_experiment_artifacts(exp_id, thorough=True, dry_run=dry_run)
    files_actioned.extend(deleted_artifacts)

    # Remove from pending queue
    if os.path.exists(PENDING_FILE):
        try:
            pending_exps = utils.safe_json_load(PENDING_FILE)
            original_count = len(pending_exps)
            filtered_pending = [exp for exp in pending_exps if exp.get('experiment_id') != exp_id]
            if len(filtered_pending) < original_count:
                print(f"  {action} remove entry from {os.path.basename(PENDING_FILE)}")
                if not dry_run:
                    with open(PENDING_FILE, 'w') as f:
                        # Use utils converter just in case pending has numpy types
                        json.dump(utils.convert_ndarray_to_list(filtered_pending), f, indent=2)
            else:
                 print(f"  Info: Experiment ID {exp_id} not found in pending queue.")
        except Exception as e: print(f"  Error updating {os.path.basename(PENDING_FILE)}: {e}")

    return files_actioned

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description='Purge experiments and associated files.',
        formatter_class=argparse.RawTextHelpFormatter # Keep help text formatting
    )

    # --- Mode Selection ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--filter',
                      help='Filter criteria (param=val) to select experiments for removal.')
    mode_group.add_argument('--partial', action='store_true',
                      help='Clean up partial/incomplete experiments.')
    mode_group.add_argument('--all', action='store_true',
                      help='Remove ALL completed experiments and their artifacts.')
    mode_group.add_argument('--id', nargs='+',
                      help='Specify one or more exact experiment IDs to remove.')
    mode_group.add_argument('--status', nargs='+',
                      help='Specify one or more status values (e.g., crashed failed_or_interrupted) to remove.')
    mode_group.add_argument('--today', action='store_true',
                      help='Remove experiments from today (based on ID prefix) from completed experiments.')


    # --- Options ---
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be done without making changes.')
    parser.add_argument('--no-confirm', action='store_true',
                      help='Skip confirmation prompt (USE WITH CAUTION!).')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Show detailed information about experiments being affected.')
    parser.add_argument('--thorough', action='store_true',
                      help='Also delete associated visualization files. Default for --all and --partial.')
    parser.add_argument('--keep-artifacts', action='store_true',
                      help='Only remove entries from completed/pending files without deleting model files. Useful with --today.')
    # <<< --- REMOVED: Requeue argument (not needed) --- >>>
    # parser.add_argument('--requeue', action='store_true',
    #                   help='Attempt to add removed experiments back to the pending queue with new IDs.')

    args = parser.parse_args()
    action_prefix = "[DRY RUN] " if args.dry_run else ""

    # Determine operation mode and target experiments
    target_experiments = []
    operation_mode = "unknown"
    load_completed = False # Flag to indicate if we need to load completed.json

    if args.all:
         operation_mode = "--all"
         print(f"{action_prefix}--- Purge ALL Completed Experiments Mode ---")
         load_completed = True
    elif args.partial:
         operation_mode = "--partial"
         print(f"{action_prefix}--- Clean Up Partial Experiments Mode ---")
         # Partial mode finds IDs directly
    elif args.filter:
         operation_mode = "--filter"
         print(f"{action_prefix}--- Purge by Filter Mode ---")
         load_completed = True
    elif args.id:
         operation_mode = "--id"
         print(f"{action_prefix}--- Purge by Specific ID(s) Mode ---")
         load_completed = True
    elif args.status:
         operation_mode = "--status"
         print(f"{action_prefix}--- Purge by Status Mode ---")
         load_completed = True
    elif args.today:
         operation_mode = "--today"
         print(f"{action_prefix}--- Remove Today's Experiments Mode ---")
         load_completed = True

    # --- Load Data if Needed ---
    experiments = []
    if load_completed:
        if not os.path.exists(COMPLETED_FILE):
             print(f"Error: Completed experiments file not found: {COMPLETED_FILE}")
             print("Cannot proceed with --all, --filter, --id, or --status mode.")
             sys.exit(1)
        experiments = utils.safe_json_load(COMPLETED_FILE)
        print(f"Loaded {len(experiments)} experiments from '{os.path.basename(COMPLETED_FILE)}'")

    # --- Identify Experiments to Remove ---
    experiments_to_remove = []
    ids_to_remove = set()

    if operation_mode == "--all":
        experiments_to_remove = experiments # Target all loaded completed experiments
        ids_to_remove = {exp.get('experiment_id') for exp in experiments_to_remove if exp.get('experiment_id')}
        # Also find and handle partial experiments later in this block
    elif operation_mode == "--filter":
        filter_params = load_filter_config(args.filter)
        if not filter_params or (not filter_params.get('hyperparams') and not filter_params.get('data_params')):
            print("No valid filter criteria loaded. Exiting.")
            sys.exit(1)
        print(f"Using filter criteria: {filter_params}")
        for exp in experiments:
            if match_experiment(exp, filter_params):
                experiments_to_remove.append(exp)
                if exp.get('experiment_id'): ids_to_remove.add(exp.get('experiment_id'))
    elif operation_mode == "--id":
        target_ids = set(args.id)
        print(f"Targeting specific IDs: {', '.join(target_ids)}")
        for exp in experiments:
             exp_id = exp.get('experiment_id')
             if exp_id and exp_id in target_ids:
                  experiments_to_remove.append(exp)
                  ids_to_remove.add(exp_id)
        # Also add IDs specified but not found in completed log (might be partial/pending)
        ids_to_remove.update(target_ids)
    elif operation_mode == "--status":
         target_statuses = set(args.status)
         print(f"Targeting statuses: {', '.join(target_statuses)}")
         for exp in experiments:
             exp_status = exp.get('status')
             if exp_status and exp_status in target_statuses:
                 experiments_to_remove.append(exp)
                 if exp.get('experiment_id'): ids_to_remove.add(exp.get('experiment_id'))
    elif operation_mode == "--today":
         # Get today's date in YYYYMMDD format (same as experiment IDs)
         today = datetime.datetime.now().strftime("%Y%m%d")
         today_prefix = f"{today}-"
         print(f"Targeting experiments from today with ID prefix: {today_prefix}")
         initial_count = len(experiments)
         for exp in experiments:
             exp_id = exp.get('experiment_id')
             if exp_id and exp_id.startswith(today_prefix):
                 experiments_to_remove.append(exp)
                 ids_to_remove.add(exp_id)
         removed_count = len(experiments_to_remove)
         print(f"Found {removed_count} experiment(s) from today ({removed_count}/{initial_count})")


    # --- Handle --partial Separately or as part of --all ---
    partial_ids_found = find_partial_experiments()
    if operation_mode == "--partial":
        if not partial_ids_found:
            print("No partial experiments found.")
            return
        print(f"Found {len(partial_ids_found)} partial experiment(s): {', '.join(partial_ids_found)}")
        # ... (rest of partial handling, unchanged) ...
        return # Exit after partial cleanup mode

    # --- Handle --all, --filter, --id, --status ---

    # Add partial experiments cleanup to --all mode implicitly
    if operation_mode == "--all" and partial_ids_found:
         print(f"Also found {len(partial_ids_found)} partial experiment(s) to clean up: {', '.join(partial_ids_found)}")
         ids_to_remove.update(partial_ids_found) # Ensure partial IDs are targeted for file deletion

    # Check if any action is needed
    if not ids_to_remove:
        if operation_mode == "--filter": print("No experiments matched the filter criteria.")
        elif operation_mode == "--id": print("Specified IDs not found or have no artifacts.")
        elif operation_mode == "--status": print("No experiments matched the specified status(es).") # Added
        elif operation_mode == "--all": print("No completed or partial experiments found.")
        return

    # Determine if 'thorough' cleanup is active
    is_thorough = args.thorough or operation_mode == "--all" # Always thorough for --all

    # Display targeted experiments/IDs
    print(f"\nTargeting {len(ids_to_remove)} experiment ID(s) for removal (Mode: {operation_mode}):")
    if args.verbose or len(ids_to_remove) <= 20:
        # Optionally show status for --status mode
        ids_with_status = {}
        if operation_mode == "--status":
             ids_with_status = {exp.get('experiment_id'): exp.get('status') for exp in experiments_to_remove}

        for i, exp_id in enumerate(sorted(list(ids_to_remove)), 1):
             status_info = f" (Status: {ids_with_status.get(exp_id, 'N/A')})" if operation_mode == "--status" else ""
             print(f"  {i}. {exp_id}{status_info}")
        if len(ids_to_remove) > 20 and not args.verbose:
            print("  ... (use -v to see all IDs)")
    else: # Only show count if not verbose and many IDs
        pass # Count already printed above

    # --- Dry Run Output ---
    if args.dry_run:
        if args.keep_artifacts:
            print(f"\n[DRY RUN] Would remove {len(ids_to_remove)} experiments from logs (keeping artifacts):")
            # Simulate completed queue removal
            num_to_remove_from_completed = len(experiments_to_remove)
            if num_to_remove_from_completed > 0:
                print(f"  Would remove {num_to_remove_from_completed} entries from {os.path.basename(COMPLETED_FILE)}")
            # Simulate pending queue removal
            print(f"  Would check and remove entries from {os.path.basename(PENDING_FILE)}")
            print(f"\n[DRY RUN] Summary: Would remove {len(ids_to_remove)} experiment entries from logs WITHOUT deleting model files.")
        else:
            print(f"\n[DRY RUN] Listing actions for {len(ids_to_remove)} targeted IDs ({'THOROUGH' if is_thorough else 'Standard'} file cleanup):")
            total_files_to_delete = 0
            for exp_id in sorted(list(ids_to_remove)):
                print(f"\n[DRY RUN] Processing ID: {exp_id}")
                # Simulate artifact deletion for this ID
                files = delete_experiment_artifacts(exp_id, thorough=is_thorough, dry_run=True)
                total_files_to_delete += len(files)
                # Simulate pending queue removal
                print(f"  Would check and remove entry from {os.path.basename(PENDING_FILE)}")
            # Simulate completed queue removal
            num_to_remove_from_completed = len(experiments_to_remove) # Only relevant for modes that load 'experiments'
            if num_to_remove_from_completed > 0:
                 print(f"\n[DRY RUN] Would remove {num_to_remove_from_completed} entries from {os.path.basename(COMPLETED_FILE)}")

            print(f"\n[DRY RUN] Summary: Would target {len(ids_to_remove)} IDs, ~{total_files_to_delete} files, and log entries.")
        
        print("[DRY RUN] No changes made.")
        return

    # --- Confirmation Prompt ---
    if not args.no_confirm:
        print("\n" + "#" * 60)
        print("###                  W A R N I N G                   ###")
        print("#" * 60)
        if args.keep_artifacts:
            print(f"### This will remove {len(ids_to_remove)} experiment ID(s) from logs  ###")
            print(f"### but keep all model files and artifacts.             ###")
        else:
            print(f"### This will PERMANENTLY DELETE artifacts for {len(ids_to_remove)}    ###")
            print(f"### experiment ID(s) and remove log entries.         ###")
            if is_thorough:
                print("### (Including visualizations due to --thorough or --all) ###")
        print("#" * 60)
        confirm_phrase = "DELETE ALL" if operation_mode == "--all" else str(len(ids_to_remove))
        response = input(f"Type '{confirm_phrase}' to confirm deletion based on mode '{operation_mode}': ")
        if response != confirm_phrase:
            print("Confirmation failed. Aborting.")
            return

    # --- Execute Deletion ---
    print("\nProceeding with deletion...")
    total_deleted_files = 0

    # Delete artifacts unless --keep-artifacts is specified
    if not args.keep_artifacts:
        print(f"Deleting artifacts for {len(ids_to_remove)} IDs...")
        for exp_id in sorted(list(ids_to_remove)):
            print(f"Processing {exp_id}...")
            deleted = delete_experiment_artifacts(exp_id, thorough=is_thorough, dry_run=False)
            total_deleted_files += len(deleted)
    else:
        print(f"Skipping artifact deletion (--keep-artifacts specified)")
        print(f"Only removing {len(ids_to_remove)} IDs from experiment logs...")

    # Update pending experiments file
    print(f"Checking and updating {os.path.basename(PENDING_FILE)}...")
    try:
        if os.path.exists(PENDING_FILE):
            pending_exps = utils.safe_json_load(PENDING_FILE)
            original_pending_count = len(pending_exps)
            filtered_pending = [exp for exp in pending_exps if exp.get('experiment_id') not in ids_to_remove]
            removed_pending_count = original_pending_count - len(filtered_pending)
            if removed_pending_count > 0:
                with open(PENDING_FILE, 'w') as f:
                    json.dump(utils.convert_ndarray_to_list(filtered_pending), f, indent=2)
                print(f"  Removed {removed_pending_count} entry/entries from pending queue.")
            else:
                print("  No targeted IDs found in pending queue.")
    except Exception as e:
        print(f"  ERROR updating {os.path.basename(PENDING_FILE)}: {e}")

    # Update completed experiments file
    if load_completed and experiments_to_remove: # Only modify if we loaded and filtered experiments
        print(f"Updating {os.path.basename(COMPLETED_FILE)}...")
        try:
            experiments_to_keep = [exp for exp in experiments if exp.get('experiment_id') not in ids_to_remove]
            removed_completed_count = len(experiments) - len(experiments_to_keep)
            with open(COMPLETED_FILE, 'w') as f:
                json.dump(utils.convert_ndarray_to_list(experiments_to_keep), f, indent=2)
            print(f"  Removed {removed_completed_count} entry/entries from completed log. Kept {len(experiments_to_keep)}.")
        except Exception as e:
            print(f"  ERROR updating {os.path.basename(COMPLETED_FILE)}: {e}")
    elif load_completed:
         print(f"No entries targeted for removal from {os.path.basename(COMPLETED_FILE)} based on mode {operation_mode}.")


    print(f"\n--- Purge Operation Complete ({operation_mode}) ---")
    print(f"Processed {len(ids_to_remove)} experiment ID(s).")
    if args.keep_artifacts:
        print(f"Kept all artifact files (--keep-artifacts specified).")
    else:
        print(f"Deleted {total_deleted_files} associated files.")


if __name__ == "__main__":
    print(f"Project Root: {utils._PROJECT_ROOT}") # Show root at start
    main()
