from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm
import hashlib
from .embeddings import ClubEmbeddings
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import json

logger = logging.getLogger(__name__)

# Custom stopwords for club descriptions
CUSTOM_STOPWORDS = {
    # Generic club terms
    'club', 'group', 'organization', 'society', 'association', 'team',
    # Common activity words
    'meet', 'meeting', 'weekly', 'monthly', 'join', 'member', 'members',
    'participate', 'participating', 'welcome', 'welcomes', 'interested',
    # Generic descriptors
    'new', 'various', 'different', 'many', 'several', 'etc', 'also',
    'well', 'good', 'great', 'best', 'better', 'more', 'most',
    # Common verbs
    'provide', 'provides', 'providing', 'learn', 'learning', 'help',
    'helping', 'helps', 'make', 'making', 'get', 'getting',
    # Time-related
    'year', 'semester', 'quarter', 'time', 'day', 'week', 'month',
    # Generic purpose words
    'goal', 'goals', 'purpose', 'mission', 'vision', 'aim', 'aims',
    # Additional Rollins-specific terms
    'rollins', 'college', 'student', 'students', 'campus'
}

@dataclass
class TopicCluster:
    id: int
    topic_name: str
    clubs: List[str]
    embedding: np.ndarray
    similar_topics: List[int] = None  # Will store IDs of related topics
    
    def __post_init__(self):
        if self.similar_topics is None:
            self.similar_topics = []
    
    def print_details(self):
        """Print detailed information about the cluster"""
        print(f"\n{'='*80}")
        print(f"Topic {self.id}: {self.topic_name}")
        print(f"Number of clubs: {len(self.clubs)}")
        print(f"Similar topics: {self.similar_topics}")
        print("\nClubs in this topic:")
        for club in sorted(self.clubs):
            print(f"  • {club}")
        print(f"{'='*80}\n")

@dataclass
class ClubCombination:
    primary_clubs: List[str]
    secondary_clubs: List[str]
    primary_topic_id: int
    secondary_topic_id: Optional[int]
    topic_similarity: float

class TopicBasedGenerator:
    def __init__(
        self,
        min_topic_size: int = 4,
        nr_topics: str = "auto",
        top_n_words: int = 10,
        embedding_model: str = "NovaSearch/stella_en_1.5B_v5",  # Updated to stella model
        cache_dir: str = ".cache",
        similarity_threshold: float = 0.5
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create custom vectorizer with stopwords
        vectorizer = CountVectorizer(
            stop_words=list(CUSTOM_STOPWORDS),
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            ngram_range=(1, 2)  # Allow bigrams
        )
        
        # Initialize dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # Initialize clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Enhanced BERTopic configuration
        self.topic_model = BERTopic(
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            top_n_words=top_n_words,
            vectorizer_model=vectorizer,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=True
        )
        
        # Initialize stella model with s2s prompt for semantic similarity
        self.sentence_model = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.s2s_prompt = "Instruct: Retrieve semantically similar text.\nQuery: "
        
        self.clusters: List[TopicCluster] = []
        self.topic_similarity_threshold = similarity_threshold
        
        # Additional tracking
        self.topic_to_clubs: Dict[int, Set[str]] = {}
        self.club_to_topics: Dict[str, Set[int]] = {}
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using stella model with s2s prompt"""
        if isinstance(text, list):
            texts = [self.s2s_prompt + t for t in text]
        else:
            texts = self.s2s_prompt + text
        return self.sentence_model.encode(texts)

    def _get_cache_path(self, club_embeddings: ClubEmbeddings) -> Path:
        """Generate a cache file path based on input data and parameters"""
        clubs_df = club_embeddings.clubs_df
        data_str = clubs_df.to_json()
        params_str = f"{self.topic_model.min_topic_size}_{self.topic_model.nr_topics}_{self.topic_similarity_threshold}"
        
        hash_input = f"{data_str}_{params_str}".encode('utf-8')
        hash_val = hashlib.md5(hash_input).hexdigest()
        
        return self.cache_dir / f"topic_clusters_{hash_val}.npz"
    
    def _save_to_cache(self, cache_path: Path) -> None:
        """Save topic clustering results to cache"""
        cache_data = {
            'topic_ids': [c.id for c in self.clusters],
            'topic_names': [c.topic_name for c in self.clusters],
            'topic_clubs': json.dumps([c.clubs for c in self.clusters]),  # Use JSON for lists
            'topic_embeddings': np.array([c.embedding for c in self.clusters]),
            'topic_similarities': [','.join(map(str, c.similar_topics)) for c in self.clusters]
        }
        
        np.savez(cache_path, **cache_data)
        logger.info(f"Saved topic clusters to cache: {cache_path}")
    
    def _load_from_cache(self, cache_path: Path) -> bool:
        """Load topic clustering results from cache"""
        try:
            cached = np.load(cache_path, allow_pickle=True)
            
            # Reconstruct clusters
            self.clusters = []
            for i in range(len(cached['topic_ids'])):
                cluster = TopicCluster(
                    id=int(cached['topic_ids'][i]),
                    topic_name=str(cached['topic_names'][i]),
                    clubs=json.loads(cached['topic_clubs'])[i],  # Parse JSON array
                    embedding=cached['topic_embeddings'][i],
                    similar_topics=[int(x) for x in cached['topic_similarities'][i].split(',') if x]
                )
                self.clusters.append(cluster)
                
                # Rebuild lookup dictionaries
                self.topic_to_clubs[cluster.id] = set(cluster.clubs)
                for club in cluster.clubs:
                    if club not in self.club_to_topics:
                        self.club_to_topics[club] = set()
                    self.club_to_topics[club].add(cluster.id)
            
            logger.info(f"Loaded {len(self.clusters)} topic clusters from cache")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def _calculate_topic_similarities(self) -> None:
        """Calculate similarities between topics and store related topics"""
        topic_embeddings = np.array([c.embedding for c in self.clusters])
        similarity_matrix = cosine_similarity(topic_embeddings)
        
        # For each topic, find similar topics
        for i, cluster in enumerate(self.clusters):
            similarities = similarity_matrix[i]
            similarities[i] = 0  # Exclude self
            
            similar_topic_indices = np.where(similarities > self.topic_similarity_threshold)[0]
            cluster.similar_topics = [
                self.clusters[idx].id for idx in similar_topic_indices
            ]
    
    def _save_cluster_relationships(self, cache_path: Path) -> None:
        """Save cluster relationships in a format optimized for data generation"""
        relationships = []
        
        # For each cluster, store its clubs and related clusters with their clubs
        for cluster in self.clusters:
            cluster_data = {
                'id': int(cluster.id),  # Convert from np.int64 to int
                'name': cluster.topic_name,
                'primary_clubs': cluster.clubs,
                'related_clusters': []
            }
            
            # Add related clusters and their similarity scores
            for related_id in cluster.similar_topics:
                related_cluster = next(c for c in self.clusters if c.id == related_id)
                similarity = float(cosine_similarity(  # Convert to float
                    cluster.embedding.reshape(1, -1),
                    related_cluster.embedding.reshape(1, -1)
                )[0][0])
                
                cluster_data['related_clusters'].append({
                    'id': int(related_id),  # Convert from np.int64 to int
                    'name': related_cluster.topic_name,
                    'clubs': related_cluster.clubs,
                    'similarity': similarity
                })
            
            relationships.append(cluster_data)
        
        # Save to a JSON file for easy access
        relationship_file = cache_path.parent / 'cluster_relationships.json'
        with open(relationship_file, 'w') as f:
            json.dump(relationships, f, indent=2)
        
        logger.info(f"Saved cluster relationships to: {relationship_file}")

    def fit(self, club_embeddings: ClubEmbeddings) -> None:
        """Create topic clusters from club descriptions"""
        logger.info("Starting topic clustering process...")
        
        # Check cache first
        cache_path = self._get_cache_path(club_embeddings)
        if cache_path.exists():
            logger.info("Found existing cache, attempting to load...")
            if self._load_from_cache(cache_path):
                # Save relationships in the new format
                self._save_cluster_relationships(cache_path)
                self._print_cluster_summary()
                return
            logger.info("Cache load failed, proceeding with new clustering...")
        
        # Get club descriptions and names
        logger.info("Preparing club data for clustering...")
        clubs_df = club_embeddings.clubs_df
        descriptions = clubs_df['Description'].tolist()
        club_names = clubs_df['Activity Name'].tolist()
        logger.info(f"Processing {len(descriptions)} club descriptions...")
        
        # Fit BERTopic model
        logger.info("Fitting BERTopic model to club descriptions...")
        topics, probs = self.topic_model.fit_transform(descriptions)
        logger.info("BERTopic model fitting complete.")
        
        # Get topic information
        topic_info = self.topic_model.get_topic_info()
        logger.info(f"Identified {len(topic_info)-1} distinct topics (excluding outliers)")
        
        # Create topic clusters
        logger.info("Creating topic clusters and calculating embeddings...")
        for topic_id in tqdm(topic_info['Topic'].unique()):
            if topic_id == -1:  # Skip outlier topic
                continue
                
            # Get clubs for this topic
            topic_mask = topics == topic_id
            topic_clubs = [club_names[i] for i in range(len(topics)) if topic_mask[i]]
            
            # Get topic representation
            topic_words = self.topic_model.get_topic(topic_id)
            topic_name = " + ".join([word for word, _ in topic_words[:3]])
            
            # Get topic embedding using stella model
            topic_descriptions = [descriptions[i] for i in range(len(topics)) if topic_mask[i]]
            topic_embedding = np.mean(self._encode_text(topic_descriptions), axis=0)
            
            # Create cluster
            cluster = TopicCluster(
                id=topic_id,
                topic_name=topic_name,
                clubs=topic_clubs,
                embedding=topic_embedding
            )
            self.clusters.append(cluster)
            
            # Update lookups
            self.topic_to_clubs[topic_id] = set(topic_clubs)
            for club in topic_clubs:
                if club not in self.club_to_topics:
                    self.club_to_topics[club] = set()
                self.club_to_topics[club].add(topic_id)
        
        # Calculate topic similarities and store related topics
        logger.info("Calculating inter-topic similarities...")
        self._calculate_topic_similarities()
        
        # Save results to cache
        logger.info("Saving clustering results to cache...")
        self._save_to_cache(cache_path)
        
        # Save relationships in the new format
        self._save_cluster_relationships(cache_path)
        
        # Print cluster summary
        self._print_cluster_summary()
        
        logger.info(f"Clustering complete. Created {len(self.clusters)} topic clusters")
    
    def _print_cluster_summary(self):
        """Print detailed summary of all clusters"""
        print("\n" + "="*40 + " CLUSTER SUMMARY " + "="*40)
        for cluster in sorted(self.clusters, key=lambda x: len(x.clubs), reverse=True):
            cluster.print_details()
    
    def generate_club_combinations(
        self,
        n_combinations: int,
        primary_clubs_per_combo: int = 2,
        secondary_clubs_per_combo: int = 1,
        max_attempts: int = 100
    ) -> List[ClubCombination]:
        """Generate club combinations using topic clusters"""
        combinations = []
        attempts = 0
        
        while len(combinations) < n_combinations and attempts < max_attempts:
            attempts += 1
            
            # 1. Pick a random primary topic cluster
            primary_cluster = random.choice(self.clusters)
            
            if len(primary_cluster.clubs) < primary_clubs_per_combo:
                continue
            
            # 2. Pick clubs from primary topic
            primary_clubs = random.sample(
                primary_cluster.clubs,
                primary_clubs_per_combo
            )
            
            # 3. Pick a related topic and get secondary clubs
            secondary_clubs = []
            secondary_topic_id = None
            topic_similarity = 0.0
            
            if primary_cluster.similar_topics and secondary_clubs_per_combo > 0:
                secondary_topic_id = random.choice(primary_cluster.similar_topics)
                similar_cluster = next(
                    c for c in self.clusters if c.id == secondary_topic_id
                )
                
                # Calculate similarity
                similarity = cosine_similarity(
                    primary_cluster.embedding.reshape(1, -1),
                    similar_cluster.embedding.reshape(1, -1)
                )[0][0]
                
                available_secondary = [
                    c for c in similar_cluster.clubs 
                    if c not in primary_clubs
                ]
                
                if len(available_secondary) >= secondary_clubs_per_combo:
                    secondary_clubs = random.sample(
                        available_secondary,
                        secondary_clubs_per_combo
                    )
                    topic_similarity = similarity
            
            # Create combination if valid
            if len(secondary_clubs) == secondary_clubs_per_combo:
                combination = ClubCombination(
                    primary_clubs=primary_clubs,
                    secondary_clubs=secondary_clubs,
                    primary_topic_id=primary_cluster.id,
                    secondary_topic_id=secondary_topic_id,
                    topic_similarity=topic_similarity
                )
                combinations.append(combination)
                attempts = 0  # Reset attempts after successful generation
        
        if len(combinations) < n_combinations:
            logger.warning(
                f"Only generated {len(combinations)}/{n_combinations} combinations "
                f"after {max_attempts} attempts"
            )
        
        return combinations

    def get_topic_info(self, topic_id: int) -> Dict:
        """Get detailed information about a topic"""
        cluster = next(c for c in self.clusters if c.id == topic_id)
        return {
            'name': cluster.topic_name,
            'clubs': cluster.clubs,
            'similar_topics': [
                {
                    'id': similar_id,
                    'name': next(c.topic_name for c in self.clusters if c.id == similar_id)
                }
                for similar_id in cluster.similar_topics
            ]
        }