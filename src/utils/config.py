from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Literal

@dataclass
class ClusterConfig:
    min_topic_size: int = 4  # Minimum size for meaningful clusters
    nr_topics: Union[str, int] = 8  # Target 8 topics for better interpretability
    similarity_threshold: float = 0.45  # Slightly lower threshold for better topic connections
    cache_dir: str = ".cache"

@dataclass
class DataConfig:
    num_samples: int = 1000  # Total number of samples to generate
    max_workers: int = 4
    primary_clubs_per_combo: int = 2
    secondary_clubs_per_combo: int = 1
    output_file: str = "synthetic_data.json"

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 5e-5
    patience: int = 15  # Early stopping patience
    min_delta: float = 1e-4  # Minimum improvement for early stopping
    use_cache: bool = True

@dataclass
class Config:
    # File paths
    clubs_file: str = "data/raw/club_list.csv"
    base_dir: Path = Path(".")
    
    # Mode of operation
    mode: Literal["cluster", "data", "train", "test", "pipeline"] = "pipeline"
    
    # Specific configurations
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.clubs_file = str(self.base_dir / self.clubs_file)
        self.data.output_file = str(self.base_dir / "data/processed" / self.data.output_file)
        self.cluster.cache_dir = str(self.base_dir / self.cluster.cache_dir)

# Default configuration
DEFAULT_CONFIG = Config()

# Example configurations for different scenarios
FAST_TEST_CONFIG = Config(
    mode="pipeline",
    data=DataConfig(
        num_samples=2000,
        max_workers=8
    ),
    train=TrainConfig(
        epochs=100,
        batch_size=32
    )
)

FULL_TRAINING_CONFIG = Config(
    mode="pipeline",
    data=DataConfig(
        num_samples=2000,
        max_workers=8,
        primary_clubs_per_combo=2,
        secondary_clubs_per_combo=2
    ),
    train=TrainConfig(
        epochs=100,
        batch_size=64,
        learning_rate=1e-4
    )
)

# You can create specific configurations for different scenarios
CLUSTER_ONLY_CONFIG = Config(
    mode="cluster",
    cluster=ClusterConfig(
        min_topic_size=4,
        nr_topics=8,  # Fixed number for our club dataset
        similarity_threshold=0.45
    )
)

DATA_ONLY_CONFIG = Config(
    mode="data",
    data=DataConfig(
        num_samples=2000,
        max_workers=6
    )
)

__all__ = [
    'Config', 'ClusterConfig', 'DataConfig', 'TrainConfig',
    'DEFAULT_CONFIG', 'FAST_TEST_CONFIG', 'FULL_TRAINING_CONFIG',
    'CLUSTER_ONLY_CONFIG', 'DATA_ONLY_CONFIG'
]