import torch
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import logging

from ..models.train import ClubRecommenderModel, get_latest_model
from .schema import UserProfile, VALID_OPTIONS
from ..models.embeddings import ClubEmbeddings, get_text_embeddings

# Disable sentence-transformers logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

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
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    # Get the latest model file
    model_path = get_latest_model()
    print(f"Loading model from: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create and load the model
    model = ClubRecommenderModel(
        categorical_size=checkpoint['categorical_size'],
        text_embedding_size=checkpoint['text_embedding_size']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model trained for {checkpoint['epoch']+1} epochs")
    print(f"Final training loss: {checkpoint['train_loss']:.6f}")
    print(f"Final validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Dataset size: {checkpoint['dataset_size']}")
    
    return model, device

def predict_club(
    free_time: str,
    college_purpose: str,
    happiness_description: str,
    self_description: str,
    hobbies: str,
    model: ClubRecommenderModel = None,
    device: torch.device = None
) -> torch.Tensor:
    """Predict club embedding for a given profile"""
    
    # Load model if not provided
    if model is None:
        model, device = load_latest_model(device)
    
    # Validate categorical inputs
    if free_time not in VALID_OPTIONS['free_time']:
        raise ValueError(f"Invalid free_time. Must be one of: {VALID_OPTIONS['free_time']}")
    if college_purpose not in VALID_OPTIONS['college_purpose']:
        raise ValueError(f"Invalid college_purpose. Must be one of: {VALID_OPTIONS['college_purpose']}")
    if self_description not in VALID_OPTIONS['self_description']:
        raise ValueError(f"Invalid self_description. Must be one of: {VALID_OPTIONS['self_description']}")
    
    # Encode categorical features
    categorical = []
    for field, options in VALID_OPTIONS.items():
        values = {
            'free_time': free_time,
            'college_purpose': college_purpose,
            'self_description': self_description
        }
        one_hot = [1 if values[field] == val else 0 for val in options]
        categorical.extend(one_hot)
    
    # Get text embeddings
    happiness_embedding = get_text_embeddings(happiness_description)
    hobbies_embedding = get_text_embeddings(hobbies)
    
    # Convert to tensors and add batch dimension
    categorical = torch.tensor(categorical, dtype=torch.float32).unsqueeze(0).to(device)
    happiness_embedding = torch.tensor(happiness_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    hobbies_embedding = torch.tensor(hobbies_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        prediction = model(categorical, happiness_embedding, hobbies_embedding)
    
    return prediction

def main(clubs_file: str = 'club_list.csv'):
    """Run the club recommendation questionnaire"""
    
    # Load trained model and get device
    model, device = load_latest_model()
    
    # Initialize club embeddings
    club_embeddings = ClubEmbeddings(clubs_file)
    
    # Get user input
    try:
        profile = get_user_input()
    except ValueError as e:
        print(f"\nError: Invalid input - {e}")
        return
    
    # Get predictions
    predicted_embeddings = predict_club(
        free_time=profile.free_time,
        college_purpose=profile.college_purpose,
        happiness_description=profile.happiness_description,
        self_description=profile.self_description,
        hobbies=profile.hobbies,
        model=model,
        device=device
    )
    
    # Get text for semantic matching
    interests = f"{profile.hobbies} {profile.happiness_description}"
    interests_embedding = get_text_embeddings(interests)
    
    # Print recommendations
    print("\nBased on your responses, here are your club recommendations:\n")
    
    # Find closest clubs for each prediction
    primary_embedding = predicted_embeddings[0][0].cpu().numpy()
    secondary_embeddings = [p[0].cpu().numpy() for p in predicted_embeddings[1:]]
    
    # Get primary recommendation
    primary_matches = club_embeddings.find_closest(primary_embedding, n=1)
    primary_club_name, primary_club_desc = primary_matches[0]
    
    print("Primary Recommendation:")
    print(f"1. {primary_club_name}")
    print(f"   {primary_club_desc}\n")
    
    # Get secondary recommendations
    print("You might also be interested in:")
    for i, embedding in enumerate(secondary_embeddings, 2):
        matches = club_embeddings.find_closest(embedding, n=1)
        club_name, description = matches[0]
        print(f"\n{i}. {club_name}")
        print(f"   {description}")

if __name__ == "__main__":
    main() 