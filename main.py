#!/usr/bin/env python3
"""
Find Your Anchor - Club Recommendation System
Main entry point for the application.
"""

import logging
import sys
from pathlib import Path
import argparse
import torch
import numpy as np

from src.utils.config import Config, FULL_TRAINING_CONFIG
from src.models.cluster import TopicBasedGenerator
from src.models.data import generate_dataset
from src.models.train import (
    train_model, 
    get_latest_model,
    get_device,
    ClubRecommenderModel,
    RecommenderDataset,
    print_prediction_results,
    VALID_OPTIONS,
    load_model_from_checkpoint
)
from src.utils.test import run_model_evaluation, get_user_input, predict_clubs
from src.models.embeddings import ClubEmbeddings
from src.models.training_config import TrainingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

def run_train(config: dict) -> None:
    """Run model training"""
    logger.info("Running model training...")
    
    # Create training config
    train_config = TrainingConfig(
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 32),
        learning_rate=config.get('learning_rate', 1e-4),
        patience=config.get('patience', 10),
        min_delta=config.get('min_delta', 1e-4),
        use_cache=config.get('use_cache', True),
        wandb_project="club-recommender",
        model_name="club_recommender"
    )
    
    # Train model
    train_model(
        synthetic_data=config['synthetic_data'],
        config=train_config,
        clubs_csv=config['clubs_csv']
    )

def run_test(config):
    """Run model testing and evaluation"""
    logger.info("Running model testing...")
    
    # Load the latest trained model
    model_path = get_latest_model()
    model, device = load_model_from_checkpoint(model_path, device=get_device())
    
    # Get config values, handling both Config and dict objects
    if isinstance(config, dict):
        synthetic_data = config.get('synthetic_data', 'data/processed/synthetic_data.json')
        clubs_csv = config.get('clubs_csv', 'data/raw/club_list.csv')
    else:
        synthetic_data = config.data.output_file
        clubs_csv = config.clubs_file
    
    # Initialize dataset and embeddings
    dataset = RecommenderDataset(
        data_path=synthetic_data,
        clubs_csv=clubs_csv,
        batch_size=1,
        use_cache=True,
        augment=False
    )
    
    club_embeddings = ClubEmbeddings(clubs_csv)
    
    # Run evaluation
    print("\nRunning Model Evaluation")
    print("=" * 80)
    stats = run_model_evaluation(model, dataset, device)
    
    # Interactive mode
    while True:
        print("\nWould you like to:")
        print("1. Try the questionnaire")
        print("2. Exit")
        choice = input("\nEnter your choice (1 or 2): ")
        
        if choice == "2":
            break
        elif choice == "1":
            try:
                # Get user input
                profile = get_user_input()
                
                # Get predictions
                predictions = predict_clubs(profile, model, device, club_embeddings)
                
                # Print recommendations
                print("\nBased on your responses, here are your club recommendations:")
                print("=" * 80)
                
                for i, location_preds in enumerate(predictions, 1):
                    print(f"\nLocation {i} Recommendations:")
                    print("-" * 40)
                    for j, (similarity, club) in enumerate(location_preds, 1):
                        club_desc = club_embeddings.get_description(club)
                        print(f"{j}. {club}")
                        print(f"   Similarity: {similarity:.3f}")
                        if club_desc:
                            print(f"   {club_desc}")
                
            except ValueError as e:
                print(f"\nError: Invalid input - {e}")
                continue
        else:
            print("\nInvalid choice. Please try again.")

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
        
        logger.info("\n=== Stage 4: Model Testing ===")
        run_test(config)
        
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
    print("  test     - Test the model")
    print("  pipeline - Run the complete pipeline")
    print("\nExample: python main.py data")

def main():
    parser = argparse.ArgumentParser(description='Club Recommender System')
    
    # Add stage as positional argument for backward compatibility
    parser.add_argument('stage', nargs='?', type=str, 
                      choices=['cluster', 'data', 'train', 'test', 'pipeline'],
                      help='Stage to run (backward compatibility)')
    
    # Add new style arguments
    parser.add_argument('--mode', type=str, choices=['train', 'test'],
                      help='Mode to run: train or test')
    parser.add_argument('--synthetic_data', type=str, default='data/processed/synthetic_data.json',
                      help='Path to synthetic data file')
    parser.add_argument('--clubs_csv', type=str, default='data/raw/club_list.csv',
                      help='Path to clubs CSV file')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    
    args = parser.parse_args()
    
    # Handle old style command (positional argument)
    if args.stage:
        config = FULL_TRAINING_CONFIG
        # Ensure directories exist
        Path(config.cluster.cache_dir).mkdir(exist_ok=True)
        Path(config.data.output_file).parent.mkdir(exist_ok=True)
        
        stage = args.stage.lower()
        if stage == 'cluster':
            run_cluster(config)
        elif stage == 'data':
            run_data(config)
        elif stage == 'train':
            run_train(config)
        elif stage == 'test':
            run_test(config)
        elif stage == 'pipeline':
            run_pipeline(config)
        else:
            print(f"Unknown stage: {stage}")
            print_usage()
            sys.exit(1)
    
    # Handle new style command (--mode argument)
    elif args.mode:
        # Convert args to dict for easier handling
        config = vars(args)
        if args.mode == 'train':
            run_train(config)
        else:  # test mode
            run_test(config)
    
    else:
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main() 