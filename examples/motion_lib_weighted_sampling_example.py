#!/usr/bin/env python3
"""
Example usage of the advanced motion sampling strategy with scoring.

This example demonstrates how to:
1. Initialize the motion library with weighted sampling enabled
2. Update motion scores based on training performance
3. Verify that weighted sampling is working correctly
"""

import torch
from holomotion.src.training.lmdb_motion_lib import LmdbMotionLib


def example_motion_scoring_workflow():
    """Example workflow showing how to use motion scoring."""
    
    # Example configuration (normally loaded from YAML)
    class MockConfig:
        def __init__(self):
            self.motion_file = "path/to/your/motion.lmdb"
            self.use_linear_weighted_sampling = True  # Enable weighted sampling
            self.sampling_base_probability = 0.1     # 10% base probability
            self.min_frame_length = 100
            self.step_dt = 1/50
            self.handpicked_motion_names = []
            self.excluded_motion_names = []
            # ... other config parameters
            
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize motion library
    motion_lib = LmdbMotionLib(config, device)
    print(f"Initialized motion library with {len(motion_lib.motion_ids)} motions")
    print(f"Weighted sampling enabled: {motion_lib.use_linear_weighted_sampling}")
    
    # Example 1: Update individual motion scores
    # Lower scores mean less likely to be sampled
    motion_lib.update_motion_score(motion_id=0, score=0.5)   # Reduce sampling of motion 0
    motion_lib.update_motion_score(motion_id=1, score=2.0)   # Increase sampling of motion 1
    
    # Example 2: Batch update scores based on training performance
    # Assume you have computed performance metrics for a batch of motions
    motion_ids = torch.tensor([10, 20, 30, 40, 50])
    # Higher tracking errors -> lower scores
    tracking_errors = torch.tensor([0.1, 0.5, 0.2, 0.8, 0.3])  
    scores = 1.0 / (1.0 + tracking_errors)  # Convert errors to scores
    motion_lib.update_motion_scores_batch(motion_ids, scores)
    
    # Example 3: Sample motions (will use weighted sampling)
    num_samples = 100
    sampled_motion_ids = motion_lib.resample_new_motions(num_samples)
    print(f"Sampled {len(sampled_motion_ids)} motion IDs using weighted sampling")
    
    # Example 4: Verify sampling behavior
    stats = motion_lib.verify_weighted_sampling(num_test_samples=10000)
    print(f"Sampling verification: {stats}")
    
    # Example 5: Get current scores
    all_scores = motion_lib.get_motion_scores()
    print(f"Current score statistics - Mean: {torch.mean(all_scores):.3f}, Std: {torch.std(all_scores):.3f}")


def example_integration_in_training_loop():
    """Example of how to integrate motion scoring in a training loop."""
    
    # Pseudo-code for training integration
    print("""
    # Example integration in training loop:
    
    for episode in range(num_episodes):
        # Sample motions for this episode
        motion_ids = motion_lib.resample_new_motions(num_envs)
        
        # Train on these motions...
        # ... training code ...
        
        # Compute performance metrics (e.g., tracking error, success rate, etc.)
        tracking_errors = compute_tracking_errors(motion_ids, ...)
        
        # Update motion scores based on performance
        # Lower error = higher score = more likely to be sampled
        new_scores = compute_scores_from_errors(tracking_errors)
        motion_lib.update_motion_scores_batch(motion_ids, new_scores)
        
        # Optional: Log sampling statistics
        if episode % log_interval == 0:
            stats = motion_lib.verify_weighted_sampling()
            logger.info(f"Sampling stats: {stats}")
    """)


if __name__ == "__main__":
    print("Motion Library Weighted Sampling Example")
    print("=" * 50)
    
    # Run the example (commented out as it requires actual LMDB file)
    # example_motion_scoring_workflow()
    
    # Show integration example
    example_integration_in_training_loop()
    
    print("\nConfiguration Example:")
    print("Add to your robot config file (*.yaml):")
    print("""
    robot:
      motion:
        use_linear_weighted_sampling: True   # Enable weighted sampling
        sampling_base_probability: 0.1       # Base probability for all clips
        # ... other motion config ...
    """)
