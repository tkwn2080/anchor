from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Basic training parameters
    epochs: int = 100
    batch_size: int = 64  # Increased batch size for simpler model
    learning_rate: float = 5e-5  # Reduced learning rate for better convergence
    patience: int = 20  # Increased patience
    min_delta: float = 1e-5  # More sensitive to improvements
    max_grad_norm: float = 1.0
    validation_split: float = 0.2
    use_cache: bool = True
    wandb_project: str = "club-recommender"
    model_name: str = "club_recommender"

    # Model architecture parameters
    hidden_size: int = 512  # Reduced for simpler model
    intermediate_size: int = 1024  # Reduced intermediate size
    num_attention_heads: int = 8  # Number of attention heads
    attention_dropout: float = 0.1  # Attention dropout rate
    
    # Loss function parameters
    diversity_weight: float = 0.1  # Reduced diversity weight to focus on matches
    min_location_distance: float = 0.2  # Increased minimum distance between predictions
    
    # Regularization parameters
    feature_dropout: float = 0.2
    
    # Learning rate scheduler parameters
    scheduler_div_factor: float = 10.0  # Reduced for gentler warmup
    scheduler_final_div_factor: float = 1e3  # Reduced for gentler cooldown
    scheduler_pct_start: float = 0.3  # Longer warmup period
    
    # Performance optimization
    use_amp: bool = True
    grad_clip_enabled: bool = True
    
    # Recommendation parameters
    num_locations: int = 3  # Number of embedding space locations to predict
    clubs_per_location: int = 3  # Number of clubs to sample per location
    total_recommendations: int = 10  # Total number of recommendations to generate
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f) 