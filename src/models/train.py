import torch
import torch.nn as nn
import torch.backends.mps
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import json
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
import pandas as pd
import logging
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..utils.schema import VALID_OPTIONS
from .embeddings import ClubEmbeddings, get_text_embeddings
from .training_config import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_epoch_metrics(epoch: int, epochs: int, train_loss: float, val_loss: float, lr: float):
    """Log epoch metrics in a clean, consistent format"""
    logger.info(
        f"Epoch [{epoch:3d}/{epochs}] "
        f"Train: {train_loss:.4f} "
        f"Val: {val_loss:.4f} "
        f"LR: {lr:.2e}"
    )

def print_prediction_results(profile: dict, actual_clubs: List[str], predictions: List[tuple]):
    """Print prediction results in a clean, consistent format"""
    print("\nProfile Summary:")
    print("-" * 40)
    print(f"Interests: {profile['hobbies'][:100]}...")
    print(f"Happiness: {profile['happiness_description'][:100]}...")
    
    print("\nRecommendations vs Actual:")
    print("-" * 40)
    
    # Group recommendations by location
    for loc_idx, loc_preds in enumerate(predictions[:3], 1):
        print(f"\nLocation {loc_idx} recommendations:")
        for i, (conf, club) in enumerate(loc_preds[:3], 1):
            status = "✓" if club in actual_clubs else "✗"
            print(f"{i}. {status} {club:<30} ({conf:.3f})")
    
    # Show the additional recommendation
    if len(predictions) > 3:
        print("\nAdditional recommendation:")
        conf, club = predictions[3][0]
        status = "✓" if club in actual_clubs else "✗"
        print(f"   {status} {club:<30} ({conf:.3f})")
    
    print("\nActual Clubs:")
    print("-" * 40)
    for i, club in enumerate(actual_clubs, 1):
        print(f"{i}. {club}")
    print("-" * 40)

class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.feature_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.feature_dropout)
    
    def forward(self, x):
        # Self attention with residual
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class ClubRecommenderModel(nn.Module):
    def __init__(self, categorical_size: int, text_embedding_size: int, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Project each input type to hidden size
        self.categorical_projection = nn.Linear(categorical_size, config.hidden_size)
        self.happiness_projection = nn.Linear(text_embedding_size, config.hidden_size)
        self.hobbies_projection = nn.Linear(text_embedding_size, config.hidden_size)
        self.happiness_match_projection = nn.Linear(text_embedding_size, config.hidden_size)
        self.hobbies_match_projection = nn.Linear(text_embedding_size, config.hidden_size)
        
        # Normalization layers for each input
        self.categorical_norm = nn.LayerNorm(config.hidden_size)
        self.happiness_norm = nn.LayerNorm(config.hidden_size)
        self.hobbies_norm = nn.LayerNorm(config.hidden_size)
        self.happiness_match_norm = nn.LayerNorm(config.hidden_size)
        self.hobbies_match_norm = nn.LayerNorm(config.hidden_size)
        
        # Input type embeddings to distinguish different inputs
        self.input_type_embeddings = nn.Parameter(
            torch.randn(5, config.hidden_size)  # 5 types: categorical, happiness, hobbies, happiness_match, hobbies_match
        )
        
        # Club embedding projection and normalization
        self.club_projection = nn.Linear(text_embedding_size, config.hidden_size)
        self.club_norm = nn.LayerNorm(config.hidden_size)
        
        # Transformer blocks for processing
        self.input_transformer = TransformerBlock(config)
        
        # Cross attention between inputs and clubs
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(config.hidden_size)
        
        # Final processing
        self.hidden_transformer = TransformerBlock(config)
        
        # Location predictors - one for each embedding space location
        self.location_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.LayerNorm(config.intermediate_size),
                nn.GELU(),
                nn.Dropout(config.feature_dropout),
                nn.Linear(config.intermediate_size, text_embedding_size),
                nn.LayerNorm(text_embedding_size)
            ) for _ in range(config.num_locations)
        ])
        
        # Match influence projection
        self.match_influence_projection = nn.Linear(config.hidden_size, text_embedding_size)
    
    def forward(self, categorical_input, text_happiness_input, text_hobbies_input, happiness_match_embedding, hobbies_match_embedding, club_embeddings):
        batch_size = categorical_input.shape[0]
        
        # Project and normalize each input
        categorical_features = self.categorical_norm(self.categorical_projection(categorical_input))  # [batch, hidden]
        happiness_features = self.happiness_norm(self.happiness_projection(text_happiness_input))  # [batch, hidden]
        hobbies_features = self.hobbies_norm(self.hobbies_projection(text_hobbies_input))  # [batch, hidden]
        happiness_match_features = self.happiness_match_norm(self.happiness_match_projection(happiness_match_embedding))  # [batch, hidden]
        hobbies_match_features = self.hobbies_match_norm(self.hobbies_match_projection(hobbies_match_embedding))  # [batch, hidden]
        
        # Add input type embeddings
        categorical_features = categorical_features + self.input_type_embeddings[0]
        happiness_features = happiness_features + self.input_type_embeddings[1]
        hobbies_features = hobbies_features + self.input_type_embeddings[2]
        happiness_match_features = happiness_match_features + self.input_type_embeddings[3]
        hobbies_match_features = hobbies_match_features + self.input_type_embeddings[4]
        
        # Stack features for attention [batch, 5, hidden_size]
        stacked_features = torch.stack([
            categorical_features,
            happiness_features,
            hobbies_features,
            happiness_match_features,
            hobbies_match_features
        ], dim=1)
        
        # Process inputs through first transformer
        features = self.input_transformer(stacked_features)
        
        # Project and normalize club embeddings
        club_features = self.club_norm(self.club_projection(club_embeddings))
        
        # Cross attention between processed inputs and clubs
        cross_attended, _ = self.cross_attention(
            features,  # query from processed inputs
            club_features,  # keys/values from clubs
            club_features
        )
        
        # Combine features
        features = self.cross_norm(features + cross_attended)
        
        # Final transformer processing
        features = self.hidden_transformer(features)
        
        # Pool features across sequence dimension with learned attention
        attention_weights = torch.softmax(
            features.mean(-1, keepdim=True),  # [batch, 5, 1]
            dim=1
        )
        pooled_features = (features * attention_weights).sum(dim=1)  # [batch, hidden_size]
        
        # Generate location embeddings
        locations = []
        for predictor in self.location_predictors:
            # Get base location embedding
            loc = predictor(pooled_features)  # [batch, text_embedding_size]
            
            # Get match influence in embedding space
            match_influence = self.match_influence_projection(
                (features * attention_weights).sum(dim=1)  # [batch, hidden_size]
            )  # [batch, text_embedding_size]
            
            # Combine base prediction with match influence
            loc = loc + 0.5 * match_influence
            
            # Normalize final embedding
            loc = loc / (loc.norm(dim=1, keepdim=True) + 1e-6)
            locations.append(loc)
        
        return locations

def club_recommendation_loss(predictions: List[torch.Tensor], targets: List[torch.Tensor], config: TrainingConfig) -> torch.Tensor:
    """Loss function for three location-based recommendations
    
    Args:
        predictions: List of three predicted location embeddings
        targets: List of three target club embeddings
        config: Training configuration
    
    Returns:
        Combined loss value
    """
    total_loss = 0.0
    batch_size = predictions[0].shape[0]
    
    # Normalize predictions and targets
    normalized_preds = [pred / (pred.norm(dim=1, keepdim=True) + 1e-6) for pred in predictions]
    normalized_targets = [target / (target.norm(dim=1, keepdim=True) + 1e-6) for target in targets]
    
    # Stack predictions and targets for easier computation
    stacked_preds = torch.stack(normalized_preds, dim=1)  # [batch, 3, embedding_dim]
    stacked_targets = torch.stack(normalized_targets, dim=1)  # [batch, 3, embedding_dim]
    
    # 1. Direct matching loss - encourage at least one prediction to strongly match each target
    direct_similarities = torch.bmm(
        stacked_preds,  # [batch, 3, embedding_dim]
        stacked_targets.transpose(1, 2)  # [batch, embedding_dim, 3]
    )  # [batch, 3, 3]
    
    # For each target, get the best matching prediction
    max_similarities, _ = direct_similarities.max(dim=1)  # [batch, 3]
    direct_loss = (1 - max_similarities).mean()
    
    # 2. Semantic coherence loss - predictions should be semantically related to targets
    semantic_loss = 0.0
    for pred in normalized_preds:
        # Get similarity to all targets
        target_sims = torch.stack([
            F.cosine_similarity(pred.unsqueeze(1), target.unsqueeze(1), dim=2)
            for target in normalized_targets
        ], dim=1)  # [batch, 3]
        # Take the best similarity for each prediction
        semantic_loss += (1 - target_sims.max(dim=1)[0]).mean()
    semantic_loss = semantic_loss / len(predictions)
    
    # 3. Diversity loss - encourage spread between predictions
    diversity_loss = 0.0
    if config.diversity_weight > 0:
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                sim = F.cosine_similarity(normalized_preds[i], normalized_preds[j], dim=1)
                diversity_loss += torch.clamp(sim - config.min_location_distance, min=0).mean()
    
    # Combine losses with weights
    total_loss = (
        direct_loss * 2.0 +  # Direct matches are most important
        semantic_loss * 1.0 +  # Semantic relationships matter
        diversity_loss * config.diversity_weight  # Keep some diversity
    )
    
    return total_loss

class RecommenderDataset(Dataset):
    def __init__(self, data_path: str, clubs_csv: str = 'club_list.csv', batch_size: int = 32, use_cache: bool = True, augment: bool = True, noise_std: float = 0.1):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.augment = augment
        self.noise_std = noise_std
        
        # Initialize club embeddings
        self.club_embeddings = ClubEmbeddings(clubs_csv)
        
        # Pre-compute best matching clubs and their embeddings for both happiness and hobbies
        print("\nComputing best matching clubs...")
        self.happiness_match_embeddings = []
        self.hobbies_match_embeddings = []
        club_names = self.club_embeddings.clubs_df['Activity Name'].tolist()
        
        for item in tqdm(self.data):
            # Get matches for happiness
            happiness_matches = self.get_keyword_matches(item['happiness_description'], club_names)
            happiness_best_idx = np.argmax(happiness_matches)
            happiness_best_club = club_names[happiness_best_idx]
            happiness_best_embedding = self.club_embeddings.get_embedding(happiness_best_club)
            self.happiness_match_embeddings.append(happiness_best_embedding)
            
            # Get matches for hobbies
            hobbies_matches = self.get_keyword_matches(item['hobbies'], club_names)
            hobbies_best_idx = np.argmax(hobbies_matches)
            hobbies_best_club = club_names[hobbies_best_idx]
            hobbies_best_embedding = self.club_embeddings.get_embedding(hobbies_best_club)
            self.hobbies_match_embeddings.append(hobbies_best_embedding)
        
        self.happiness_match_embeddings = np.array(self.happiness_match_embeddings)
        self.hobbies_match_embeddings = np.array(self.hobbies_match_embeddings)
        
        # Configure caching
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        dataset_size = len(self.data)
        self.cache_path = self.cache_dir / f"training_embeddings_n{dataset_size}.npz"
        
        if use_cache and self.cache_path.exists():
            print(f"\nLoading cached embeddings for {dataset_size} samples...")
            cached = np.load(str(self.cache_path))
            self.happiness_embeddings = cached['happiness_embeddings']
            self.hobbies_embeddings = cached['hobbies_embeddings']
            self.club_embeddings_list = [
                cached[f'club_embeddings_{i}'] for i in range(3)
            ]
            print("✓ Loaded cached embeddings")
        else:
            print(f"\nComputing embeddings for {dataset_size} samples...")
            # Extract all text fields
            happiness_texts = [item['happiness_description'] for item in self.data]
            hobbies_texts = [item['hobbies'] for item in self.data]
            
            # Get all club descriptions (now treating all clubs equally)
            club_descriptions = []
            print("\nProcessing clubs...")
            for item in self.data:
                clubs = [item['target_club_name']] + item['secondary_clubs']
                descriptions = []
                for club_name in clubs[:3]:  # Take first 3 clubs
                    desc = self.club_embeddings.get_description(club_name)
                    descriptions.append(desc if desc else "")
                club_descriptions.append(descriptions)
            
            # Compute embeddings with progress bars
            print("\nComputing happiness description embeddings...")
            self.happiness_embeddings = get_text_embeddings(happiness_texts, batch_size=batch_size, show_progress=True)
            
            print("\nComputing hobbies embeddings...")
            self.hobbies_embeddings = get_text_embeddings(hobbies_texts, batch_size=batch_size, show_progress=True)
            
            print("\nComputing club embeddings...")
            self.club_embeddings_list = []
            for i in range(3):
                club_texts = [desc[i] for desc in club_descriptions]
                embeddings = get_text_embeddings(club_texts, batch_size=batch_size, show_progress=True)
                self.club_embeddings_list.append(embeddings)
            
            if use_cache:
                print("\nCaching embeddings for future use...")
                save_dict = {
                    'happiness_embeddings': self.happiness_embeddings,
                    'hobbies_embeddings': self.hobbies_embeddings,
                }
                for i, emb in enumerate(self.club_embeddings_list):
                    save_dict[f'club_embeddings_{i}'] = emb
                np.savez(str(self.cache_path), **save_dict)
                print(f"✓ Saved embeddings cache for {dataset_size} samples")
    
    def augment_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Add controlled noise to embedding for augmentation"""
        if not self.augment:
            return embedding
        
        # Generate Gaussian noise
        noise = torch.randn_like(embedding) * self.noise_std
        
        # Add noise and normalize to maintain unit norm
        augmented = embedding + noise
        augmented = augmented / (augmented.norm(dim=-1, keepdim=True) + 1e-6)
        
        return augmented
    
    def get_keyword_matches(self, text: str, club_names: List[str]) -> np.ndarray:
        """Get keyword match scores between input text and club names"""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        scores = []
        
        for club in club_names:
            # Split club name into words and check for matches
            club_words = set(club.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(club_words & text_words)
            union = len(club_words | text_words)
            score = intersection / union if union > 0 else 0.0
            scores.append(score)
        
        return np.array(scores, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Encode categorical features
        item = self.data[idx]
        categorical = []
        for field, options in VALID_OPTIONS.items():
            one_hot = [1 if item[field] == val else 0 for val in options]
            categorical.extend(one_hot)
        
        # Get pre-computed embeddings and best match embeddings
        happiness_embedding = torch.tensor(self.happiness_embeddings[idx], dtype=torch.float32)
        hobbies_embedding = torch.tensor(self.hobbies_embeddings[idx], dtype=torch.float32)
        club_embeddings = [torch.tensor(emb[idx], dtype=torch.float32) for emb in self.club_embeddings_list]
        happiness_match_embedding = torch.tensor(self.happiness_match_embeddings[idx], dtype=torch.float32)
        hobbies_match_embedding = torch.tensor(self.hobbies_match_embeddings[idx], dtype=torch.float32)
        
        # Augment embeddings if enabled
        happiness_embedding = self.augment_embedding(happiness_embedding)
        hobbies_embedding = self.augment_embedding(hobbies_embedding)
        club_embeddings = [self.augment_embedding(emb) for emb in club_embeddings]
        happiness_match_embedding = self.augment_embedding(happiness_match_embedding)
        hobbies_match_embedding = self.augment_embedding(hobbies_match_embedding)
        
        return (
            torch.tensor(categorical, dtype=torch.float32),
            happiness_embedding,
            hobbies_embedding,
            happiness_match_embedding,
            hobbies_match_embedding,
            *club_embeddings
        )

def get_model_path(model_name: str = "club_recommender") -> Path:
    """Get the path for saving/loading models with timestamp"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return models_dir / f"{model_name}_{timestamp}.pt"

def get_latest_model(model_prefix: str = "club_recommender") -> Path:
    """Get the most recent model file"""
    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError("No models directory found")
    
    model_files = list(models_dir.glob(f"{model_prefix}_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found with prefix {model_prefix}")
    
    # Sort by modification time and return most recent
    return max(model_files, key=lambda p: p.stat().st_mtime)

def load_model_from_checkpoint(checkpoint_path: Path, device: Optional[torch.device] = None) -> Tuple[ClubRecommenderModel, torch.device]:
    """Load a model from a checkpoint file"""
    if device is None:
        device = get_device()
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Convert config dict to TrainingConfig if needed
    config = checkpoint['config']
    if isinstance(config, dict):
        config = TrainingConfig(**config)
    
    # Create and load the model
    model = ClubRecommenderModel(
        categorical_size=checkpoint['categorical_size'],
        text_embedding_size=checkpoint['text_embedding_size'],
        config=config
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def get_device() -> torch.device:
    """Get the best available device for training"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Ensure both MPS is available and PyTorch was built with MPS support
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_model(
    synthetic_data: str,
    config: Optional[TrainingConfig] = None,
    clubs_csv: str = 'club_list.csv'
) -> Path:
    """Train the club recommendation model"""
    
    # Load or create config
    if config is None:
        config = TrainingConfig()
    
    # Initialize wandb
    run = wandb.init(
        project=config.wandb_project,
        config=config.__dict__
    )
    
    logger.info(f"Starting training with config: {config.__dict__}")
    
    # Dataset setup with augmentation
    logger.info("Loading dataset...")
    dataset = RecommenderDataset(
        synthetic_data,
        clubs_csv=clubs_csv,
        batch_size=config.batch_size,
        use_cache=config.use_cache,
        augment=True,  # Enable augmentation
        noise_std=0.05  # Small noise for subtle variations
    )
    
    # Split indices for train/validation
    indices = list(range(len(dataset)))
    split = int(np.floor(config.validation_split * len(dataset)))
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create data loaders with proper configuration for device type
    device = get_device()
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'})")
    
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=0 if device.type == 'mps' else 4,  # MPS doesn't support multiprocessing
        pin_memory=device.type == "cuda"
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=0 if device.type == 'mps' else 4,
        pin_memory=device.type == "cuda"
    )
    
    # Model setup
    categorical_size = sum(len(options) for options in VALID_OPTIONS.values())
    text_embedding_size = dataset.happiness_embeddings.shape[1]
    
    model = ClubRecommenderModel(categorical_size, text_embedding_size, config).to(device)
    wandb.watch(model)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.scheduler_pct_start,
        div_factor=config.scheduler_div_factor,
        final_div_factor=config.scheduler_final_div_factor
    )
    
    # Initialize mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and device.type == 'cuda' else None
    
    logger.info(f"Starting training with {len(dataset)} samples")
    logger.info(f"Training: {len(train_indices)} samples, Validation: {len(val_indices)} samples")
    
    best_val_loss = float('inf')
    best_model_path = None
    epochs_without_improvement = 0
    
    # Track metrics for early stopping
    val_losses = []
    train_losses = []
    
    epoch_pbar = tqdm(range(config.epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        model.train()
        train_metrics = {'total_loss': 0, 'batches': 0}
        
        # Training loop
        for batch in train_loader:
            categorical, happiness, hobbies, happiness_match, hobbies_match, *club_embs = [b.to(device) for b in batch]
            # Stack club embeddings into a single tensor
            club_embeddings = torch.stack(club_embs, dim=1)  # [batch, num_clubs, embedding_dim]
            optimizer.zero_grad()
            
            # Use mixed precision training if enabled
            if config.use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    predictions = model(categorical, happiness, hobbies, happiness_match, hobbies_match, club_embeddings)
                    loss = club_recommendation_loss(predictions, club_embs, config)
                
                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                
                # Gradient clipping if enabled
                if config.grad_clip_enabled:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(categorical, happiness, hobbies, happiness_match, hobbies_match, club_embeddings)
                loss = club_recommendation_loss(predictions, club_embs, config)
                
                loss.backward()
                if config.grad_clip_enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            
            train_metrics['total_loss'] += loss.item()
            train_metrics['batches'] += 1
        
        # Validation loop
        model.eval()
        val_metrics = {'total_loss': 0, 'batches': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                categorical, happiness, hobbies, happiness_match, hobbies_match, *club_embs = [b.to(device) for b in batch]
                # Stack club embeddings into a single tensor
                club_embeddings = torch.stack(club_embs, dim=1)  # [batch, num_clubs, embedding_dim]
                
                if config.use_amp and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        predictions = model(categorical, happiness, hobbies, happiness_match, hobbies_match, club_embeddings)
                        val_loss = club_recommendation_loss(
                            predictions,
                            club_embs,
                            config
                        ).item()
                else:
                    predictions = model(categorical, happiness, hobbies, happiness_match, hobbies_match, club_embeddings)
                    val_loss = club_recommendation_loss(
                        predictions,
                        club_embs,
                        config
                    ).item()
                
                val_metrics['total_loss'] += val_loss
                val_metrics['batches'] += 1
        
        # Calculate metrics
        avg_train_loss = train_metrics['total_loss'] / train_metrics['batches']
        avg_val_loss = val_metrics['total_loss'] / val_metrics['batches']
        current_lr = scheduler.get_last_lr()[0]
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update progress bar and log metrics
        epoch_pbar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'val_loss': f"{avg_val_loss:.4f}",
            'lr': f"{current_lr:.2e}"
        })
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr
        })
        
        # Log to file
        log_epoch_metrics(epoch + 1, config.epochs, avg_train_loss, avg_val_loss, current_lr)
        
        # Save best model
        if avg_val_loss < best_val_loss - config.min_delta:
            best_val_loss = avg_val_loss
            best_model_path = get_model_path(config.model_name)
            
            # Save model with metadata
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config.__dict__,
                'categorical_size': categorical_size,
                'text_embedding_size': text_embedding_size,
                'dataset_size': len(dataset),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, best_model_path)
            
            logger.info(f"New best model saved (val_loss: {avg_val_loss:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= config.patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    # Log final metrics to wandb
    wandb.log({
        'best_val_loss': best_val_loss,
        'final_train_loss': avg_train_loss,
        'total_epochs': epoch + 1
    })
    
    # Save loss plots
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plot_path = Path('training_plot.png')
    plt.savefig(plot_path)
    wandb.log({"training_plot": wandb.Image(str(plot_path))})
    
    wandb.save(str(best_model_path))
    run.finish()
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")
    
    # Final evaluation
    logger.info("\nEvaluating final model on sample predictions...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Show sample predictions
    model.eval()
    with torch.no_grad():
        # Get a small batch of validation samples
        val_iter = iter(val_loader)
        sample_batch = next(val_iter)
        categorical, happiness, hobbies, happiness_match, hobbies_match, *club_embs = [b.to(device) for b in sample_batch]
        club_embeddings = torch.stack(club_embs, dim=1)  # [batch, num_clubs, embedding_dim]
        
        if config.use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                predictions = model(categorical, happiness, hobbies, happiness_match, hobbies_match, club_embeddings)
        else:
            predictions = model(categorical, happiness, hobbies, happiness_match, hobbies_match, club_embeddings)
        
        # Show predictions for first few samples
        for i in range(min(3, len(predictions[0]))):  # Use predictions[0] for batch size
            # Get the actual profile using the validation sampler index
            profile_idx = val_indices[i]
            profile = dataset.data[profile_idx]
            actual_clubs = [profile['target_club_name']] + profile['secondary_clubs'][:2]
            
            # Get predicted clubs
            pred_results = []
            for pred_idx in range(3):  # For each of the three predictions
                # Get the prediction for this sample and club position
                pred_embedding = predictions[pred_idx][i].cpu().numpy()
                similarities = []
                for club_name, club_desc in dataset.club_embeddings.clubs_df[['Activity Name', 'Description']].values:
                    if pd.isna(club_desc):
                        continue
                    club_emb = dataset.club_embeddings.get_embedding(club_name)
                    if club_emb is not None:
                        similarity = np.dot(pred_embedding, club_emb) / (np.linalg.norm(pred_embedding) * np.linalg.norm(club_emb))
                        similarities.append((similarity, club_name))
                
                pred_results.append(sorted(similarities, reverse=True)[:3])
            
            print_prediction_results(profile, actual_clubs, pred_results)
    
    return best_model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train club recommendation model')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--synthetic_data', type=str, default='synthetic_data.json',
                      help='Path to synthetic data file')
    parser.add_argument('--clubs_csv', type=str, default='club_list.csv',
                      help='Path to clubs CSV file')
    
    args = parser.parse_args()
    
    config = TrainingConfig.from_yaml(args.config) if args.config else TrainingConfig()
    train_model(args.synthetic_data, config, args.clubs_csv)