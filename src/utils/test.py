import torch
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import random
from collections import defaultdict

from ..models.train import (
    ClubRecommenderModel,
    get_latest_model,
    RecommenderDataset,
    load_model_from_checkpoint,
    get_device
)
from .schema import UserProfile, VALID_OPTIONS
from ..models.embeddings import ClubEmbeddings, get_text_embeddings

# Disable sentence-transformers logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

def print_profile(profile: Dict) -> None:
    """Print a user profile in a readable format"""
    print("\nProfile:")
    print("-" * 40)
    print(f"Free Time: {profile['free_time']}")
    print(f"College Purpose: {profile['college_purpose']}")
    print(f"Self Description: {profile['self_description']}")
    print(f"Happiness Description: {profile['happiness_description']}")
    print(f"Hobbies: {profile['hobbies']}")
    print("\nActual Clubs:")
    print(f"Primary: {profile['target_club_name']}")
    print("Secondary:", ", ".join(profile['secondary_clubs']))
    print("-" * 40)

def calculate_prediction_metrics(predictions: List[Tuple[float, str]], actual_clubs: List[str]) -> Dict:
    """Calculate metrics for predictions vs actual clubs"""
    metrics = {
        'primary_correct': False,
        'any_correct': False,
        'top3_hits': 0,
        'best_similarity': 0.0,
        'mean_similarity': 0.0
    }
    
    # Check primary prediction
    if predictions[0][1] in actual_clubs:
        metrics['primary_correct'] = True
    
    # Check all predictions
    similarities = []
    for similarity, club in predictions:
        if club in actual_clubs:
            metrics['any_correct'] = True
            metrics['top3_hits'] += 1
        similarities.append(similarity)
    
    metrics['best_similarity'] = max(similarities)
    metrics['mean_similarity'] = sum(similarities) / len(similarities)
    
    return metrics

def run_model_evaluation(model: ClubRecommenderModel, dataset: RecommenderDataset, device: torch.device, num_samples: int = 10) -> Dict:
    """Run evaluation on random test samples and return statistics"""
    model.eval()
    
    # Track statistics
    stats = defaultdict(list)
    
    # Select random samples
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(sample_indices, 1):
        # Get sample data
        categorical, happiness, hobbies, happiness_match, hobbies_match, *club_embs = [
            b.unsqueeze(0).to(device) if isinstance(b, torch.Tensor) else torch.tensor(b).unsqueeze(0).to(device)
            for b in dataset[idx]
        ]
        club_embeddings = torch.stack(club_embs, dim=1)
        
        # Get actual profile and clubs
        profile = dataset.data[idx]
        actual_clubs = [profile['target_club_name']] + profile['secondary_clubs']
        
        print(f"\nTest Sample {i}")
        print_profile(profile)
        
        # Get model predictions
        with torch.no_grad():
            predictions = model(categorical, happiness, hobbies, happiness_match, hobbies_match, club_embeddings)
        
        # Convert predictions to club recommendations
        all_predictions = []
        for pred_idx in range(len(predictions)):
            pred_embedding = predictions[pred_idx][0].cpu().numpy()
            similarities = []
            for club_name, club_desc in dataset.club_embeddings.clubs_df[['Activity Name', 'Description']].values:
                if pd.isna(club_desc):
                    continue
                club_emb = dataset.club_embeddings.get_embedding(club_name)
                if club_emb is not None:
                    similarity = np.dot(pred_embedding, club_emb) / (np.linalg.norm(pred_embedding) * np.linalg.norm(club_emb))
                    similarities.append((similarity, club_name))
            
            top_predictions = sorted(similarities, reverse=True)[:3]
            all_predictions.append(top_predictions)
            
            # Print predictions for this location
            print(f"\nLocation {pred_idx + 1} Predictions:")
            for j, (similarity, club) in enumerate(top_predictions, 1):
                match = "✓" if club in actual_clubs else "✗"
                print(f"{j}. {match} {club:<40} (similarity: {similarity:.3f})")
        
        # Calculate metrics for this sample
        metrics = calculate_prediction_metrics(all_predictions[0], actual_clubs)  # Primary location metrics
        
        # Store metrics
        for key, value in metrics.items():
            stats[key].append(value)
    
    # Calculate final statistics
    final_stats = {
        'primary_accuracy': sum(stats['primary_correct']) / len(stats['primary_correct']),
        'any_correct_rate': sum(stats['any_correct']) / len(stats['any_correct']),
        'avg_top3_hits': sum(stats['top3_hits']) / len(stats['top3_hits']),
        'avg_best_similarity': sum(stats['best_similarity']) / len(stats['best_similarity']),
        'avg_mean_similarity': sum(stats['mean_similarity']) / len(stats['mean_similarity'])
    }
    
    print("\nEvaluation Statistics:")
    print("-" * 40)
    print(f"Primary Recommendation Accuracy: {final_stats['primary_accuracy']:.2%}")
    print(f"Any Correct Recommendation Rate: {final_stats['any_correct_rate']:.2%}")
    print(f"Average Top-3 Hits: {final_stats['avg_top3_hits']:.2f}")
    print(f"Average Best Similarity: {final_stats['avg_best_similarity']:.3f}")
    print(f"Average Mean Similarity: {final_stats['avg_mean_similarity']:.3f}")
    
    return final_stats

def get_user_input():
    """Get user input for all fields"""
    print("\nWelcome to the Club Recommender System!")
    
    # Categorical inputs
    print("\nPlease select from the following options:")
    
    # Get responses
    responses = {}
    
    print("\nHow do you spend your free time?")
    for i, option in enumerate(VALID_OPTIONS['free_time'], 1):
        print(f"{i}. {option}")
    idx = int(input("Enter the number of your choice: ")) - 1
    responses['free_time'] = VALID_OPTIONS['free_time'][idx]
    
    print("\nWhat should college be about?")
    for i, option in enumerate(VALID_OPTIONS['college_purpose'], 1):
        print(f"{i}. {option}")
    idx = int(input("Enter the number of your choice: ")) - 1
    responses['college_purpose'] = VALID_OPTIONS['college_purpose'][idx]
    
    print("\nHow would you describe yourself?")
    for i, option in enumerate(VALID_OPTIONS['self_description'], 1):
        print(f"{i}. {option}")
    idx = int(input("Enter the number of your choice: ")) - 1
    responses['self_description'] = VALID_OPTIONS['self_description'][idx]
    
    # Free text inputs
    print("\nDescribe environments that make you happy (1-2 sentences):")
    responses['happiness_description'] = input()
    
    print("\nDescribe your hobbies and interests:")
    responses['hobbies'] = input()
    
    # Validate responses using Pydantic
    return UserProfile(**responses)

def encode_categorical(profile: UserProfile):
    """One-hot encode categorical responses"""
    encoded = []
    
    # Encode each categorical field
    for field, options in VALID_OPTIONS.items():
        one_hot = [1 if getattr(profile, field) == val else 0 for val in options]
        encoded.extend(one_hot)
    
    return encoded

def load_latest_model(device: torch.device = None):
    """Load the latest trained model"""
    if device is None:
        device = get_device()
    
    # Get the latest model file
    model_path = get_latest_model()
    print(f"Loading model from: {model_path}")
    
    # Load the model using the new function
    model, device = load_model_from_checkpoint(model_path, device)
    
    return model, device

def predict_clubs(
    profile: UserProfile,
    model: ClubRecommenderModel,
    device: torch.device,
    club_embeddings: ClubEmbeddings
) -> List[List[Tuple[float, str]]]:
    """Predict clubs for a given profile"""
    
    # Encode categorical features
    categorical = encode_categorical(profile)
    
    # Get text embeddings
    happiness_embedding = get_text_embeddings(profile.happiness_description)
    hobbies_embedding = get_text_embeddings(profile.hobbies)
    
    # Get best matching embeddings using the existing find_closest method
    happiness_matches = club_embeddings.find_closest(happiness_embedding)
    hobbies_matches = club_embeddings.find_closest(hobbies_embedding)
    
    # Get the embeddings for the best matches
    happiness_match_embedding = club_embeddings.get_embedding(happiness_matches[0][0])
    hobbies_match_embedding = club_embeddings.get_embedding(hobbies_matches[0][0])
    
    # Convert to tensors and add batch dimension
    categorical = torch.tensor(categorical, dtype=torch.float32).unsqueeze(0).to(device)
    happiness_embedding = torch.tensor(happiness_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    hobbies_embedding = torch.tensor(hobbies_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    happiness_match_embedding = torch.tensor(happiness_match_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    hobbies_match_embedding = torch.tensor(hobbies_match_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Create dummy club embeddings (will be ignored in prediction)
    dummy_club_embeddings = torch.zeros((1, 3, happiness_embedding.shape[-1]), device=device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(
            categorical,
            happiness_embedding,
            hobbies_embedding,
            happiness_match_embedding,
            hobbies_match_embedding,
            dummy_club_embeddings
        )
    
    # Convert predictions to club recommendations
    all_predictions = []
    for pred_idx in range(len(predictions)):
        pred_embedding = predictions[pred_idx][0].cpu().numpy()
        similarities = []
        for club_name, club_desc in club_embeddings.clubs_df[['Activity Name', 'Description']].values:
            if pd.isna(club_desc):
                continue
            club_emb = club_embeddings.get_embedding(club_name)
            if club_emb is not None:
                similarity = np.dot(pred_embedding, club_emb) / (np.linalg.norm(pred_embedding) * np.linalg.norm(club_emb))
                similarities.append((similarity, club_name))
        
        top_predictions = sorted(similarities, reverse=True)[:3]
        all_predictions.append(top_predictions)
    
    return all_predictions

def main(synthetic_data: str = 'data/processed/synthetic_data.json', clubs_file: str = 'data/raw/club_list.csv'):
    """Run the club recommendation system"""
    
    # Load trained model and get device
    model, device = load_latest_model()
    
    # Initialize club embeddings
    club_embeddings = ClubEmbeddings(clubs_file)
    
    # Load dataset for evaluation
    dataset = RecommenderDataset(synthetic_data, clubs_csv=clubs_file, use_cache=True)
    
    # Run evaluation on test samples
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

if __name__ == "__main__":
    main() 