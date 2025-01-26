#!/usr/bin/env python3
"""
Find Your Anchor - Club Recommendation System
Main entry point for the application.
"""

import logging
import sys
from pathlib import Path

from src.utils.config import Config, FULL_TRAINING_CONFIG
from src.models.cluster import TopicBasedGenerator
from src.models.data import generate_dataset
from src.models.train import train_model
from src.utils.test import main as run_test
from src.models.embeddings import ClubEmbeddings
from src.models.training_config import TrainingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_clustering(config: Config) -> TopicBasedGenerator | None:
    """Initialize and fit the topic-based clustering"""
    try:
        logger.info("=== Starting Clustering Setup ===")
        logger.info("Loading club embeddings from %s...", config.clubs_file)
        club_embeddings = ClubEmbeddings(config.clubs_file)
        
        logger.info("Initializing topic generator with:")
        logger.info(f"  - Minimum topic size: {config.cluster.min_topic_size}")
        logger.info(f"  - Number of topics: {config.cluster.nr_topics}")
        logger.info(f"  - Similarity threshold: {config.cluster.similarity_threshold}")
        
        generator = TopicBasedGenerator(
            min_topic_size=config.cluster.min_topic_size,
            nr_topics=config.cluster.nr_topics,
            cache_dir=config.cluster.cache_dir,
            similarity_threshold=config.cluster.similarity_threshold
        )
        generator.fit(club_embeddings)
        
        logger.info("\n=== Clustering Setup Complete ===")
        return generator
    except Exception as e:
        logger.error(f"Error setting up clustering: {e}")
        return None

def run_cluster(config: Config) -> None:
    """Run just the clustering stage"""
    logger.info("Running clustering...")
    try:
        # Load embeddings and run clustering
        club_embeddings = ClubEmbeddings(config.clubs_file)
        generator = TopicBasedGenerator(
            min_topic_size=config.cluster.min_topic_size,
            nr_topics=config.cluster.nr_topics,
            cache_dir=config.cluster.cache_dir,
            similarity_threshold=config.cluster.similarity_threshold
        )
        generator.fit(club_embeddings)
        logger.info("Clustering complete!")
    except Exception as e:
        logger.error(f"Error in clustering: {e}")
        raise

def run_data(config: Config) -> None:
    """Run just the data generation stage"""
    logger.info("Running data generation...")
    try:
        # Generate data using loaded clusters
        generate_dataset(
            clubs_csv=config.clubs_file,
            num_samples=config.data.num_samples,
            output_file=config.data.output_file,
            cache_dir=config.cluster.cache_dir
        )
        logger.info("Data generation complete!")
    except Exception as e:
        logger.error(f"Error in data generation: {e}")
        raise

def run_train(config: Config) -> None:
    """Run just the training stage"""
    logger.info("Running model training...")
    
    # Create training config from the main config
    training_config = TrainingConfig(
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
        patience=config.train.patience,
        min_delta=config.train.min_delta,
        use_cache=config.train.use_cache,
        wandb_project="club-recommender",
        model_name="club_recommender"
    )
    
    train_model(
        synthetic_data=config.data.output_file,
        config=training_config,
        clubs_csv=config.clubs_file
    )
    logger.info("Training complete!")

def run_pipeline(config: Config) -> None:
    """Run the complete pipeline"""
    logger.info("\n=== Starting Full Pipeline ===")
    try:
        logger.info("\n=== Stage 1: Topic Clustering ===")
        run_cluster(config)
        
        logger.info("\n=== Stage 2: Synthetic Data Generation ===")
        run_data(config)
        
        logger.info("\n=== Stage 3: Model Training ===")
        run_train(config)
        
        logger.info("\n=== Pipeline Completed Successfully! ===")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def print_usage():
    print("\nUsage: python main.py [stage]")
    print("\nAvailable stages:")
    print("  cluster  - Run topic clustering")
    print("  data     - Generate synthetic data")
    print("  train    - Train the model")
    print("  pipeline - Run the complete pipeline")
    print("\nExample: python main.py data")

def main():
    # Use the full training configuration by default
    config = FULL_TRAINING_CONFIG
    
    # Ensure directories exist
    Path(config.cluster.cache_dir).mkdir(exist_ok=True)
    Path(config.data.output_file).parent.mkdir(exist_ok=True)
    
    # Get the stage from command line
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    
    stage = sys.argv[1].lower()
    
    if stage == 'cluster':
        run_cluster(config)
    elif stage == 'data':
        run_data(config)
    elif stage == 'train':
        run_train(config)
    elif stage == 'pipeline':
        run_pipeline(config)
    else:
        print(f"Unknown stage: {stage}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main() 