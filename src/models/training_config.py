from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Basic training parameters
    epochs: int = 100
    batch_size: int = 32  # Reduced batch size for larger model
    learning_rate: float = 2e-5  # Slightly lower LR for stability
    patience: int = 20  # Increased patience for larger model
    min_delta: float = 1e-4
    max_grad_norm: float = 1.0
    validation_split: float = 0.2
    use_cache: bool = True
    wandb_project: str = "club-recommender"
    model_name: str = "club_recommender"

    # Model architecture parameters
    hidden_size: int = 1024  # Increased from 768
    num_hidden_layers: int = 4  # More transformer layers
    intermediate_size: int = 2048  # Larger intermediate representations
    num_attention_heads: int = 16  # More attention heads
    
    # Loss function parameters
    temperature: float = 0.1  # Lower temperature for sharper predictions
    label_smoothing: float = 0.1
    contrastive_weight: float = 0.2  # Increased contrastive learning
    
    # Regularization parameters
    categorical_dropout: float = 0.2  # Reduced dropout for larger model
    text_dropout: float = 0.2
    feature_dropout: float = 0.2
    attention_dropout: float = 0.1
    use_layer_norm: bool = True

    # Learning rate scheduler parameters
    scheduler_div_factor: float = 10.0  # Gentler LR schedule
    scheduler_final_div_factor: float = 1e2
    scheduler_pct_start: float = 0.1  # Shorter warmup for pretrained embeddings

    # Performance optimization
    use_amp: bool = True
    grad_clip_enabled: bool = True
    gradient_checkpointing: bool = True  # Enable for memory efficiency
    
    # Architecture enhancements
    use_club_metadata: bool = True
    attention_heads: int = 16  # Matched with num_attention_heads
    
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