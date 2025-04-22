import unittest
from unittest import mock
import tempfile
import os
import json
import sys
import datetime

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
import manage_queue

class TestManageQueue(unittest.TestCase):
    """Test the manage_queue.py functionality"""
    
    def setUp(self):
        """Setup for each test"""
        # Create a temporary file for the pending experiments
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        # Mock the PENDING_EXPERIMENTS_FILE path
        self.original_path = manage_queue.PENDING_EXPERIMENTS_FILE
        manage_queue.PENDING_EXPERIMENTS_FILE = self.temp_file.name
        
        # Sample experiment data for testing
        self.sample_experiments = [
            {
                "experiment_id": "20250401-test1",
                "hyperparams": {"d_model": 128, "nhead": 4},
                "data_params": {"cipher_names": ["caesar", "vigenere"], "num_samples": 10000}
            },
            {
                "experiment_id": "20250401-test2",
                "hyperparams": {"d_model": 256, "nhead": 8},
                "data_params": {"cipher_names": ["caesar", "vigenere"], "num_samples": 10000}
            }
        ]
    
    def tearDown(self):
        """Clean up after each test"""
        # Restore the original path
        manage_queue.PENDING_EXPERIMENTS_FILE = self.original_path
        
        # Remove the temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_add_experiments_to_queue(self):
        """Test adding experiments to the queue"""
        # Start with an empty queue
        with open(self.temp_file.name, 'w') as f:
            json.dump([], f)
        
        # Generate some new experiments
        new_experiments = [
            {
                "experiment_id": "20250401-test3",
                "hyperparams": {"d_model": 512, "nhead": 8},
                "data_params": {"cipher_names": ["caesar", "vigenere"], "num_samples": 10000}
            }
        ]
        
        # Mock the generate_experiments function to return our test data
        with mock.patch('manage_queue.generate_experiments', return_value=new_experiments):
            # Call the function to test
            manage_queue.add_to_queue(new_experiments)
        
        # Read the queue file and check if the experiments were added
        with open(self.temp_file.name, 'r') as f:
            queue = json.load(f)
        
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["experiment_id"], "20250401-test3")
    
    def test_replace_queue(self):
        """Test replacing the queue with new experiments"""
        # Start with an existing queue
        with open(self.temp_file.name, 'w') as f:
            json.dump(self.sample_experiments, f)
        
        # Generate new experiments to replace the queue
        new_experiments = [
            {
                "experiment_id": "20250401-test3",
                "hyperparams": {"d_model": 512, "nhead": 8},
                "data_params": {"cipher_names": ["caesar", "vigenere"], "num_samples": 10000}
            }
        ]
        
        # Mock the generate_experiments function to return our test data
        with mock.patch('manage_queue.generate_experiments', return_value=new_experiments):
            # Call the function to test
            manage_queue.replace_queue(new_experiments)
        
        # Read the queue file and check if the queue was replaced
        with open(self.temp_file.name, 'r') as f:
            queue = json.load(f)
        
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["experiment_id"], "20250401-test3")
    
    def test_clear_queue(self):
        """Test clearing the queue"""
        # Start with an existing queue
        with open(self.temp_file.name, 'w') as f:
            json.dump(self.sample_experiments, f)
        
        # Call the function to test
        manage_queue.clear_queue()
        
        # Read the queue file and check if it's empty
        with open(self.temp_file.name, 'r') as f:
            queue = json.load(f)
        
        self.assertEqual(len(queue), 0)
    
    def test_parameter_grid_expansion(self):
        """Test that parameters are correctly expanded into a grid of experiments"""
        # Define test parameters
        params = {
            'cipher_names': [['caesar', 'vigenere']],
            'num_samples': [10000, 20000],
            'd_model': [128, 256],
            'nhead': [4, 8]
        }
        
        # Mock the date for consistent experiment IDs
        today = "20250401"
        with mock.patch('datetime.datetime') as mock_date:
            mock_date.now.return_value.strftime.return_value = today
            
            # Generate experiments
            experiments = manage_queue.generate_experiments(params)
        
        # Should generate 2×2×2 = 8 experiments
        self.assertEqual(len(experiments), 8)
        
        # Check that all combinations are present
        param_combinations = set()
        for exp in experiments:
            d_model = exp['hyperparams']['d_model']
            nhead = exp['hyperparams']['nhead']
            num_samples = exp['data_params']['num_samples']
            param_combinations.add((d_model, nhead, num_samples))
        
        expected_combinations = {
            (128, 4, 10000), (128, 4, 20000),
            (128, 8, 10000), (128, 8, 20000),
            (256, 4, 10000), (256, 4, 20000),
            (256, 8, 10000), (256, 8, 20000)
        }
        
        self.assertEqual(param_combinations, expected_combinations)

if __name__ == '__main__':
    unittest.main()