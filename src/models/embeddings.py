import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
from typing import List, Dict
import torch
from torch.utils.data import DataLoader, Dataset
import hashlib
import json

# Disable sentence-transformers logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

class TextEmbeddingsCache:
    CACHE_VERSION = 1

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_path = self.cache_dir / f"text_embeddings_v{self.CACHE_VERSION}.npz"
        self.load_cache()

    def load_cache(self):
        if self.cache_path.exists():
            cached = np.load(str(self.cache_path), allow_pickle=True)
            self.text_keys = cached['keys'].tolist()
            self.embeddings = cached['embeddings']
            self.cache_dict = {k: i for i, k in enumerate(self.text_keys)}
        else:
            self.text_keys = []
            self.embeddings = np.array([])
            self.cache_dict = {}

    def save_cache(self):
        np.savez(
            str(self.cache_path),
            keys=np.array(self.text_keys, dtype=str),
            embeddings=self.embeddings
        )

    def get_text_key(self, text: str) -> str:
        """Generate a consistent key for a text string"""
        return hashlib.sha256(text.encode()).hexdigest()

    def get_embedding(self, text: str, compute_fn=None) -> np.ndarray:
        key = self.get_text_key(text)
        if key in self.cache_dict:
            return self.embeddings[self.cache_dict[key]]
        elif compute_fn is not None:
            embedding = compute_fn(text)
            self.add_embedding(text, embedding)
            return embedding
        else:
            raise KeyError("Text not in cache and no compute function provided")

    def add_embedding(self, text: str, embedding: np.ndarray):
        key = self.get_text_key(text)
        if key not in self.cache_dict:
            self.text_keys.append(key)
            self.cache_dict[key] = len(self.text_keys) - 1
            if len(self.embeddings) == 0:
                self.embeddings = embedding[np.newaxis, :]
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
            self.save_cache()

# Global cache instance
text_embeddings_cache = TextEmbeddingsCache()

def batch_encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts in batches with progress bar"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

class TextBatchDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

class ClubEmbeddings:
    CACHE_VERSION = 2  # Increment this when changing model
    
    def __init__(self, clubs_file: str, cache_dir: str = ".cache", batch_size: int = 32):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_path = self.cache_dir / f"club_embeddings_v{self.CACHE_VERSION}.npz"
        self.batch_size = batch_size
        self.s2s_prompt = "Instruct: Retrieve semantically similar text.\nQuery: "  # Same prompt as cluster.py
        
        self.clubs_df = pd.read_csv(clubs_file)
        self.load_or_compute_embeddings()
    
    def load_or_compute_embeddings(self):
        try:
            if self.cache_path.exists():
                print("Loading cached club embeddings...")
                cached = np.load(str(self.cache_path))
                names = cached['names']
                embeddings = cached['embeddings']
                self.embeddings = {name: emb for name, emb in zip(names, embeddings)}
                print(f"Loaded {len(self.embeddings)} club embeddings from cache")
            else:
                self._compute_and_cache_embeddings()
        except Exception as e:
            print(f"Cache load failed ({str(e)}), recomputing embeddings...")
            self._compute_and_cache_embeddings()
    
    def _compute_and_cache_embeddings(self):
        print("Computing club embeddings...")
        # Use the same model as cluster.py
        text_model = SentenceTransformer('NovaSearch/stella_en_1.5B_v5', trust_remote_code=True)
        
        # Get all descriptions
        descriptions = self.clubs_df['Description'].tolist()
        club_names = self.clubs_df['Activity Name'].tolist()
        
        # Add s2s prompt to descriptions
        descriptions = [self.s2s_prompt + desc for desc in descriptions]
        
        # Compute embeddings in batches
        embeddings = batch_encode_texts(text_model, descriptions, self.batch_size)
        
        # Create dictionary
        self.embeddings = {name: emb for name, emb in zip(club_names, embeddings)}
        
        # Cache embeddings
        names = np.array(club_names, dtype=str)
        embeddings_array = np.array(list(self.embeddings.values()), dtype=np.float32)
        
        print(f"Caching {len(self.embeddings)} club embeddings...")
        np.savez(
            str(self.cache_path),
            names=names,
            embeddings=embeddings_array
        )
    
    def find_closest(self, query_embedding: np.ndarray, n: int = 3):
        """Find closest clubs to the query embedding using vectorized operations"""
        # Convert embeddings to matrix
        club_embeddings = np.stack(list(self.embeddings.values()))
        club_names = list(self.embeddings.keys())
        
        # Compute distances using vectorized operations
        distances = np.linalg.norm(club_embeddings - query_embedding, axis=1)
        
        # Get top N closest
        closest_indices = np.argsort(distances)[:n]
        return [(club_names[i], self.clubs_df[self.clubs_df['Activity Name'] == club_names[i]]['Description'].iloc[0]) 
                for i in closest_indices]
    
    def get_embedding(self, club_name: str) -> np.ndarray:
        """Get embedding for a specific club"""
        return self.embeddings[club_name]
    
    def get_description(self, club_name: str) -> str:
        """Get description for a specific club"""
        return self.clubs_df[self.clubs_df['Activity Name'] == club_name]['Description'].iloc[0]

def get_text_embeddings(texts: str | List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """Get embeddings for arbitrary text using the same model as club embeddings"""
    model = SentenceTransformer('NovaSearch/stella_en_1.5B_v5', trust_remote_code=True)
    s2s_prompt = "Instruct: Retrieve semantically similar text.\nQuery: "
    
    def compute_embedding(text):
        return model.encode(s2s_prompt + text, show_progress_bar=False)
    
    # Handle single text
    if isinstance(texts, str):
        try:
            return text_embeddings_cache.get_embedding(texts, compute_fn=compute_embedding)
        except Exception as e:
            print(f"Cache error: {e}, computing directly")
            return compute_embedding(texts)
    
    # Handle list of texts
    results = []
    for text in (tqdm(texts) if show_progress else texts):
        try:
            embedding = text_embeddings_cache.get_embedding(text, compute_fn=compute_embedding)
        except Exception as e:
            print(f"Cache error for text: {e}, computing directly")
            embedding = compute_embedding(text)
        results.append(embedding)
    
    return np.array(results) 