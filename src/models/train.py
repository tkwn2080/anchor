import torch
import torch.nn as nn
import torch.backends.mps
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import json
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import List, Optional
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

def print_prediction_results(profile: dict, actual_clubs: List[str], predictions: List[tuple], confidence_threshold: float = 0.5):
    """Print prediction results in a clean, consistent format"""
    print("\nProfile Summary:")
    print("-" * 40)
    print(f"Interests: {profile['hobbies'][:100]}...")
    print(f"Happiness: {profile['happiness_description'][:100]}...")
    
    print("\nPredictions vs Actual:")
    print("-" * 40)
    for i, (actual, pred) in enumerate(zip(actual_clubs, predictions)):
        confidence = pred[0][0]  # First prediction's confidence
        status = "✓" if pred[0][1] == actual else "✗"
        confidence_indicator = "!" if confidence > confidence_threshold else " "
        
        print(f"{i+1}. {status} Predicted: {pred[0][1]:<30} ({confidence:.3f}){confidence_indicator}")
        print(f"   Actual:    {actual:<30}")
        if pred[0][1] != actual:
            print(f"   Also considered: {pred[1][1]} ({pred[1][0]:.3f}), {pred[2][1]} ({pred[2][0]:.3f})")
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
        
        # Store config for use in forward pass
        self.config = config
        
        # Print sizes for debugging
        print(f"Model initialized with: categorical_size={categorical_size}, text_embedding_size={text_embedding_size}")
        
        # Categorical pathway with larger dimensions
        self.categorical_encoder = nn.Sequential(
            nn.Linear(categorical_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.categorical_dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.categorical_dropout)
        )
        
        # Text embedding projections
        self.happiness_projection = nn.Linear(text_embedding_size, config.hidden_size)
        self.hobbies_projection = nn.Linear(text_embedding_size, config.hidden_size)
        
        # Transformer layers for text processing
        self.text_transformer = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Feature fusion transformer
        self.fusion_transformer = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_hidden_layers // 2)
        ])
        
        # Club prediction heads
        self.club_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.LayerNorm(config.intermediate_size),
                nn.GELU(),
                nn.Dropout(config.feature_dropout),
                nn.Linear(config.intermediate_size, text_embedding_size)
            ) for _ in range(3)
        ])
        
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        for transformer in self.text_transformer:
            transformer.attention._use_gradient_checkpointing = True
        for transformer in self.fusion_transformer:
            transformer.attention._use_gradient_checkpointing = True
    
    def forward(self, categorical_input, text_happiness_input, text_hobbies_input):
        batch_size = categorical_input.shape[0]
        
        # Process categorical input
        categorical_encoded = self.categorical_encoder(categorical_input)
        categorical_encoded = categorical_encoded.unsqueeze(1)  # Add sequence dimension
        
        # Project text inputs to hidden size
        happiness_encoded = self.happiness_projection(text_happiness_input).unsqueeze(1)
        hobbies_encoded = self.hobbies_projection(text_hobbies_input).unsqueeze(1)
        
        # Combine text features
        text_sequence = torch.cat([happiness_encoded, hobbies_encoded], dim=1)
        
        # Process through text transformer layers
        for transformer in self.text_transformer:
            text_sequence = transformer(text_sequence)
        
        # Cross-modal attention between text and categorical
        cross_modal_features, _ = self.cross_modal_attention(
            categorical_encoded,
            text_sequence,
            text_sequence
        )
        
        # Combine features for fusion
        combined_features = torch.cat([
            categorical_encoded,
            cross_modal_features,
            text_sequence
        ], dim=1)
        
        # Process through fusion transformer
        for transformer in self.fusion_transformer:
            combined_features = transformer(combined_features)
        
        # Global average pooling
        pooled_features = combined_features.mean(dim=1)
        
        # Generate predictions
        predictions = []
        for predictor in self.club_predictors:
            pred = predictor(pooled_features)
            # Only normalize and apply temperature, no positional weights here
            pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-6)
            pred = pred / self.config.temperature
            predictions.append(pred)
        
        return predictions

def club_recommendation_loss(predictions: List[torch.Tensor], targets: List[torch.Tensor], config: TrainingConfig) -> torch.Tensor:
    """Loss function for three club recommendations
    
    Args:
        predictions: List of three predicted club embeddings
        targets: List of three target club embeddings
        config: Training configuration
    
    Returns:
        Combined loss value
    """
    total_loss = 0.0
    batch_size = predictions[0].shape[0]
    
    # Normalize targets if they aren't already
    normalized_targets = []
    for target in targets:
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-6)
        normalized_targets.append(target_norm)
    
    # Compute loss for each prediction
    for pred, target in zip(predictions, normalized_targets):
        # Apply label smoothing
        smoothed_target = (1 - config.label_smoothing) * target + \
                         config.label_smoothing * torch.mean(target, dim=1, keepdim=True)
        
        # Compute cosine similarity (values between -1 and 1)
        cos_sim = torch.sum(pred * smoothed_target, dim=1)
        
        # Convert to distance (0 to 2)
        distance = 1 - cos_sim
        total_loss = total_loss + distance.mean()
        
        # Add contrastive loss if enabled
        if config.contrastive_weight > 0:
            # Get negative samples (other targets)
            neg_targets = [t for t in normalized_targets if not torch.allclose(t, target)]
            for neg_target in neg_targets:
                neg_sim = torch.sum(pred * neg_target, dim=1)
                # Hinge loss: max(0, margin + pos_sim - neg_sim)
                contrastive = torch.clamp(
                    0.5 + cos_sim - neg_sim,  # margin of 0.5
                    min=0
                )
                total_loss = total_loss + (contrastive.mean() * config.contrastive_weight)
    
    # Average across all predictions
    avg_loss = total_loss / len(predictions)
    
    return avg_loss

class RecommenderDataset(Dataset):
    def __init__(self, data_path: str, clubs_csv: str = 'club_list.csv', batch_size: int = 32, use_cache: bool = True):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Initialize club embeddings
        self.club_embeddings = ClubEmbeddings(clubs_csv)
        
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Encode categorical features
        item = self.data[idx]
        categorical = []
        for field, options in VALID_OPTIONS.items():
            one_hot = [1 if item[field] == val else 0 for val in options]
            categorical.extend(one_hot)
        
        # Get pre-computed embeddings
        happiness_embedding = self.happiness_embeddings[idx]
        hobbies_embedding = self.hobbies_embeddings[idx]
        club_embeddings = [emb[idx] for emb in self.club_embeddings_list]
        
        return (
            torch.tensor(categorical, dtype=torch.float32),
            torch.tensor(happiness_embedding, dtype=torch.float32),
            torch.tensor(hobbies_embedding, dtype=torch.float32),
            *[torch.tensor(emb, dtype=torch.float32) for emb in club_embeddings]
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

def get_device() -> torch.device:
    """Get the best available device for training"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
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
    
    # Dataset setup
    logger.info("Loading dataset...")
    dataset = RecommenderDataset(
        synthetic_data,
        clubs_csv=clubs_csv,
        batch_size=config.batch_size,
        use_cache=config.use_cache
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
            categorical, happiness, hobbies, *club_embeddings = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            # Use mixed precision training if enabled
            if config.use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    predictions = model(categorical, happiness, hobbies)
                    loss = club_recommendation_loss(predictions, [club_embeddings[i] for i in range(3)], config)
                
                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                
                # Gradient clipping if enabled
                if config.grad_clip_enabled:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(categorical, happiness, hobbies)
                loss = club_recommendation_loss(predictions, [club_embeddings[i] for i in range(3)], config)
                
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
                categorical, happiness, hobbies, *club_embeddings = [b.to(device) for b in batch]
                
                if config.use_amp and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        predictions = model(categorical, happiness, hobbies)
                        val_loss = club_recommendation_loss(
                            predictions,
                            [club_embeddings[i] for i in range(3)],
                            config
                        ).item()
                else:
                    predictions = model(categorical, happiness, hobbies)
                    val_loss = club_recommendation_loss(
                        predictions,
                        [club_embeddings[i] for i in range(3)],
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
        categorical, happiness, hobbies, *club_embeddings = [b.to(device) for b in sample_batch]
        
        if config.use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                predictions = model(categorical, happiness, hobbies)
        else:
            predictions = model(categorical, happiness, hobbies)
        
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