import unittest
from unittest import mock
import tempfile
import os
import json
import sys
import datetime
import glob
import shutil

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
import purge_experiments

class TestPurgeExperiments(unittest.TestCase):
    """Test the purge_experiments.py functionality"""
    
    def setUp(self):
        """Setup for each test"""
        # Create temporary directory structure for test artifacts
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        
        # Create subdirectories
        self.models_dir = os.path.join(self.root_dir, 'models')
        self.checkpoints_dir = os.path.join(self.root_dir, 'checkpoints')
        self.cm_history_dir = os.path.join(self.root_dir, 'cm_history')
        self.cm_dir = os.path.join(self.root_dir, 'cm')
        self.loss_graphs_dir = os.path.join(self.root_dir, 'loss_graphs')
        
        os.makedirs(self.models_dir)
        os.makedirs(self.checkpoints_dir)
        os.makedirs(self.cm_history_dir)
        os.makedirs(self.cm_dir)
        os.makedirs(self.loss_graphs_dir)
        
        # Create temporary files for completed/pending experiments
        self.completed_file = os.path.join(self.root_dir, 'completed_experiments.json')
        self.pending_file = os.path.join(self.root_dir, 'pending_experiments.json')
        
        # Mock the paths in purge_experiments
        self.original_completed = purge_experiments.COMPLETED_FILE
        self.original_pending = purge_experiments.PENDING_FILE
        self.original_model_dir = purge_experiments.MODEL_DIR
        self.original_metadata_dir = purge_experiments.METADATA_DIR
        self.original_checkpoint_dir = purge_experiments.CHECKPOINT_DIR
        self.original_cm_history_dir = purge_experiments.CM_HISTORY_DIR
        self.original_cm_gif_dir = purge_experiments.CM_GIF_DIR
        self.original_loss_graph_dir = purge_experiments.LOSS_GRAPH_DIR
        
        purge_experiments.COMPLETED_FILE = self.completed_file
        purge_experiments.PENDING_FILE = self.pending_file
        purge_experiments.MODEL_DIR = self.models_dir
        purge_experiments.METADATA_DIR = self.models_dir
        purge_experiments.CHECKPOINT_DIR = self.checkpoints_dir
        purge_experiments.CM_HISTORY_DIR = self.cm_history_dir
        purge_experiments.CM_GIF_DIR = self.cm_dir
        purge_experiments.LOSS_GRAPH_DIR = self.loss_graphs_dir
        
        # Sample experiment data for testing
        today = datetime.datetime.now().strftime("%Y%m%d")
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")
        
        self.sample_completed = [
            {
                "experiment_id": f"{today}-test1",
                "status": "completed",
                "hyperparams": {"d_model": 128, "nhead": 4},
                "data_params": {"cipher_names": ["caesar", "vigenere"], "num_samples": 10000}
            },
            {
                "experiment_id": f"{yesterday}-test2",
                "status": "completed",
                "hyperparams": {"d_model": 256, "nhead": 8},
                "data_params": {"cipher_names": ["caesar", "vigenere"], "num_samples": 10000}
            },
            {
                "experiment_id": f"{yesterday}-test3",
                "status": "failed",
                "hyperparams": {"d_model": 512, "nhead": 8},
                "data_params": {"cipher_names": ["caesar", "vigenere"], "num_samples": 10000}
            }
        ]
        
        # Write the sample data to files
        with open(self.completed_file, 'w') as f:
            json.dump(self.sample_completed, f)
        
        with open(self.pending_file, 'w') as f:
            json.dump([], f)
        
        # Create sample artifacts for test1 and test2
        for exp_id in [f"{today}-test1", f"{yesterday}-test2"]:
            # Create model file
            with open(os.path.join(self.models_dir, f"{exp_id}.pt"), 'w') as f:
                f.write("dummy model")
            
            # Create metadata file
            with open(os.path.join(self.models_dir, f"{exp_id}_metadata.json"), 'w') as f:
                f.write("{}")
            
            # Create CM history file
            with open(os.path.join(self.cm_history_dir, f"{exp_id}_cm_history.npy"), 'w') as f:
                f.write("dummy history")
            
            # Create checkpoints
            with open(os.path.join(self.checkpoints_dir, f"{exp_id}_epoch1.pt"), 'w') as f:
                f.write("dummy checkpoint")
            
            # Create visualizations
            with open(os.path.join(self.cm_dir, f"{exp_id}_cm.gif"), 'w') as f:
                f.write("dummy gif")
            
            with open(os.path.join(self.cm_dir, f"{exp_id}_cm.png"), 'w') as f:
                f.write("dummy png")
            
            with open(os.path.join(self.loss_graphs_dir, f"{exp_id}_loss.png"), 'w') as f:
                f.write("dummy loss graph")
    
    def tearDown(self):
        """Clean up after each test"""
        # Restore original paths
        purge_experiments.COMPLETED_FILE = self.original_completed
        purge_experiments.PENDING_FILE = self.original_pending
        purge_experiments.MODEL_DIR = self.original_model_dir
        purge_experiments.METADATA_DIR = self.original_metadata_dir
        purge_experiments.CHECKPOINT_DIR = self.original_checkpoint_dir
        purge_experiments.CM_HISTORY_DIR = self.original_cm_history_dir
        purge_experiments.CM_GIF_DIR = self.original_cm_gif_dir
        purge_experiments.LOSS_GRAPH_DIR = self.original_loss_graph_dir
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_find_artifacts_on_disk(self):
        """Test that artifacts are correctly identified on disk"""
        artifacts = purge_experiments.find_artifacts_on_disk()
        
        # Should find artifacts for both test1 and test2
        self.assertEqual(len(artifacts), 2)
        
        # Check that all artifact types were found for each experiment
        today = datetime.datetime.now().strftime("%Y%m%d")
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")
        
        test1_id = f"{today}-test1"
        test2_id = f"{yesterday}-test2"
        
        self.assertIn(test1_id, artifacts)
        self.assertIn(test2_id, artifacts)
        
        # Each experiment should have 7 artifacts (model, metadata, cm_history, checkpoint, gif, png, loss graph)
        self.assertEqual(len(artifacts[test1_id]), 7)
        self.assertEqual(len(artifacts[test2_id]), 7)
    
    def test_delete_experiment_artifacts(self):
        """Test that artifacts are correctly deleted"""
        today = datetime.datetime.now().strftime("%Y%m%d")
        test1_id = f"{today}-test1"
        
        # Count files before deletion
        model_count_before = len(os.listdir(self.models_dir))
        checkpoint_count_before = len(os.listdir(self.checkpoints_dir))
        cm_history_count_before = len(os.listdir(self.cm_history_dir))
        visualization_count_before = len(os.listdir(self.cm_dir)) + len(os.listdir(self.loss_graphs_dir))
        
        # Delete artifacts for test1 (standard mode, not thorough)
        deleted_files = purge_experiments.delete_experiment_artifacts(test1_id, thorough=False, dry_run=False)
        
        # Should delete model, metadata, cm_history, and checkpoint (4 files)
        self.assertEqual(len(deleted_files), 4)
        
        # Count files after deletion
        model_count_after = len(os.listdir(self.models_dir))
        checkpoint_count_after = len(os.listdir(self.checkpoints_dir))
        cm_history_count_after = len(os.listdir(self.cm_history_dir))
        visualization_count_after = len(os.listdir(self.cm_dir)) + len(os.listdir(self.loss_graphs_dir))
        
        # Model count should decrease by 2 (model and metadata)
        self.assertEqual(model_count_before - model_count_after, 2)
        
        # Checkpoint count should decrease by 1
        self.assertEqual(checkpoint_count_before - checkpoint_count_after, 1)
        
        # CM history count should decrease by 1
        self.assertEqual(cm_history_count_before - cm_history_count_after, 1)
        
        # Visualization count should remain the same
        self.assertEqual(visualization_count_before, visualization_count_after)
    
    def test_delete_experiment_artifacts_thorough(self):
        """Test that artifacts are correctly deleted in thorough mode"""
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")
        test2_id = f"{yesterday}-test2"
        
        # Count files before deletion
        model_count_before = len(os.listdir(self.models_dir))
        checkpoint_count_before = len(os.listdir(self.checkpoints_dir))
        cm_history_count_before = len(os.listdir(self.cm_history_dir))
        visualization_count_before = len(os.listdir(self.cm_dir)) + len(os.listdir(self.loss_graphs_dir))
        
        # Delete artifacts for test2 (thorough mode)
        deleted_files = purge_experiments.delete_experiment_artifacts(test2_id, thorough=True, dry_run=False)
        
        # Should delete all 7 files
        self.assertEqual(len(deleted_files), 7)
        
        # Count files after deletion
        model_count_after = len(os.listdir(self.models_dir))
        checkpoint_count_after = len(os.listdir(self.checkpoints_dir))
        cm_history_count_after = len(os.listdir(self.cm_history_dir))
        visualization_count_after = len(os.listdir(self.cm_dir)) + len(os.listdir(self.loss_graphs_dir))
        
        # Model count should decrease by 2 (model and metadata)
        self.assertEqual(model_count_before - model_count_after, 2)
        
        # Checkpoint count should decrease by 1
        self.assertEqual(checkpoint_count_before - checkpoint_count_after, 1)
        
        # CM history count should decrease by 1
        self.assertEqual(cm_history_count_before - cm_history_count_after, 1)
        
        # Visualization count should decrease by 3 (gif, png, loss graph)
        self.assertEqual(visualization_count_before - visualization_count_after, 3)
    
    def test_today_mode(self):
        """Test that today's experiments are correctly identified"""
        # Mock the arguments
        args = mock.MagicMock()
        args.today = True
        args.dry_run = True
        args.no_confirm = True
        
        # Redirect stdout to capture output
        import io
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        with redirect_stdout(output):
            # Mock the input function to handle confirmation
            with mock.patch('builtins.input', return_value="1"):
                # Replace sys.argv temporarily
                old_argv = sys.argv
                sys.argv = ['purge_experiments.py', '--today']
                try:
                    with mock.patch('argparse.ArgumentParser.parse_args', return_value=args):
                        purge_experiments.main()
                finally:
                    sys.argv = old_argv
        
        # Check output to see if today's experiment was identified
        today = datetime.datetime.now().strftime("%Y%m%d")
        today_id = f"{today}-test1"
        
        output_str = output.getvalue()
        
        # Today mode should identify the experiment from today
        self.assertIn(today_id, output_str)
        
        # Should not identify yesterday's experiments
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")
        yesterday_id = f"{yesterday}-test2"
        
        # In dry run mode the ID should not be processed
        self.assertNotIn(f"Processing {yesterday_id}", output_str)
    
    def test_list_orphaned(self):
        """Test that orphaned artifacts are correctly identified"""
        # Create an orphaned experiment artifact
        orphan_id = "20250101-orphan"
        with open(os.path.join(self.models_dir, f"{orphan_id}.pt"), 'w') as f:
            f.write("dummy orphan model")
        
        # Find orphaned artifacts
        orphaned = purge_experiments.find_orphaned_artifacts()
        
        # Should find the orphaned artifact
        self.assertIn(orphan_id, orphaned)
        self.assertEqual(len(orphaned[orphan_id]), 1)

if __name__ == '__main__':
    unittest.main()