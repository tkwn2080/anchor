from typing import List, Dict, Optional
import boto3
from botocore.exceptions import ClientError
import json
import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import multiprocessing
from functools import partial
import ray

from ..utils.schema import SyntheticProfile, VALID_OPTIONS
from .embeddings import ClubEmbeddings, get_text_embeddings
from .cluster import TopicBasedGenerator, ClubCombination

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 2.0

# Initialize Ray only when needed
def init_ray():
    """Initialize Ray if not already initialized"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

@ray.remote
class ProfileGenerator:
    def __init__(self):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    def generate_profile(self, club_embeddings: ClubEmbeddings, combination: ClubCombination) -> Optional[dict]:
        """Generate a single profile using the combination"""
        profile = generate_profile_for_combination(self.client, club_embeddings, combination)
        if profile:
            return profile.model_dump()
        return None

def init_nova():
    """Initialize Nova client with AWS credentials"""
    try:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv('AWS_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Nova client: {e}")
        return None

def should_retry_error(error: Exception) -> bool:
    if isinstance(error, ClientError):
        error_code = error.response.get('Error', {}).get('Code', '')
        retryable_errors = {
            'ModelErrorException',
            'ThrottlingException',
            'ServiceUnavailableException',
            'InternalServerException'
        }
        return error_code in retryable_errors
    return False

def call_nova_with_retry(client, request_params: dict, retry_count: int = 0) -> dict:
    try:
        return client.converse(**request_params)
    except Exception as e:
        if retry_count < MAX_RETRIES and should_retry_error(e):
            delay = min(INITIAL_RETRY_DELAY * (2 ** retry_count), MAX_RETRY_DELAY)
            logger.warning(f"Retrying Nova call after error: {e}. Attempt {retry_count + 1} of {MAX_RETRIES}")
            time.sleep(delay)
            return call_nova_with_retry(client, request_params, retry_count + 1)
        raise

def validate_profile_data(profile_data: dict) -> tuple[bool, str]:
    """Validate profile data and return (is_valid, error_message)"""
    # Validate required fields are present
    required_fields = {'free_time', 'college_purpose', 'happiness_description', 
                      'self_description', 'hobbies'}
    missing_fields = required_fields - set(profile_data.keys())
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    # Validate categorical fields match exactly
    if profile_data['free_time'] not in VALID_OPTIONS['free_time']:
        return False, f"Invalid free_time value: '{profile_data['free_time']}'. Must be one of: {VALID_OPTIONS['free_time']}"
    if profile_data['college_purpose'] not in VALID_OPTIONS['college_purpose']:
        return False, f"Invalid college_purpose value: '{profile_data['college_purpose']}'. Must be one of: {VALID_OPTIONS['college_purpose']}"
    if profile_data['self_description'] not in VALID_OPTIONS['self_description']:
        return False, f"Invalid self_description value: '{profile_data['self_description']}'. Must be one of: {VALID_OPTIONS['self_description']}"
    
    return True, ""

def parse_llm_response(response: dict) -> Optional[dict]:
    """Parse and validate LLM response"""
    try:
        content = response.get('output', {}).get('message', {}).get('content', [])
        if not content or 'text' not in content[0]:
            logger.error(f"No valid content in response: {response}")
            return None
        
        response_text = content[0]['text'].strip()
        
        # Clean up markdown formatting if present
        if response_text.startswith('```'):
            response_text = '\n'.join(response_text.split('\n')[1:-1])
        
        # Ensure we have valid JSON
        if not response_text.startswith('{'):
            logger.error(f"Response is not JSON: {response_text}")
            return None
        
        return json.loads(response_text)
        
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return None

def generate_profile_for_combination(
    client,
    club_embeddings: ClubEmbeddings,
    combination: ClubCombination,
    max_retries: int = 3
) -> Optional[SyntheticProfile]:
    """Generate a realistic student profile for a specific club combination using Nova"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Build error guidance if needed
            error_guidance = ""
            if last_error:
                error_guidance = f"""
CORRECTION NEEDED: The previous attempt failed validation:
{last_error}

Please ensure you:
1. Use EXACTLY the provided values for categorical fields (no variations allowed)
2. Create interests that naturally align with ALL the specified clubs
3. Keep the JSON structure exactly as requested"""
            
            # Get descriptions for all clubs
            club_descriptions = {
                club: club_embeddings.get_description(club)
                for club in combination.primary_clubs + combination.secondary_clubs
            }
            
            # Build detailed prompt
            prompt = f"""Generate a realistic student profile for someone who would be interested in these college clubs:

Primary Interests:
{', '.join(combination.primary_clubs)}
Club Descriptions:
{chr(10).join(f'- {club}: {desc}' for club, desc in club_descriptions.items())}

Generate a profile that feels like it was written by a real college student. The profile should:
- Use casual language and potentially imperfect grammar/spelling
- Include short, natural responses (not necessarily complete sentences)
- Feel authentic and relatable
- Match the interests implied by their club choices
- Use EXACTLY the provided categorical options (no variations allowed)

{error_guidance}

Return ONLY a raw JSON object with these fields:
- free_time: EXACTLY one of: {json.dumps(VALID_OPTIONS['free_time'])}
- college_purpose: EXACTLY one of: {json.dumps(VALID_OPTIONS['college_purpose'])}
- happiness_description: A casual 1-2 sentence description of what makes them happy (written like a student)
- self_description: EXACTLY one of: {json.dumps(VALID_OPTIONS['self_description'])}
- hobbies: A realistic description of their interests (written like a student would write)"""

            # Call Nova
            response = call_nova_with_retry(client, {
                "modelId": "us.amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "system": [{"text": "You are an expert at creating realistic student profiles that feel authentic and natural."}],
                "inferenceConfig": {
                    "temperature": max(0.7, 0.9 - (attempt * 0.1)),  # Keep temperature high for variety
                    "maxTokens": 1024
                }
            })
            
            # Parse and validate response
            profile_data = parse_llm_response(response)
            if not profile_data:
                continue
            
            # Validate the core profile data
            is_valid, error_message = validate_profile_data(profile_data)
            if not is_valid:
                logger.warning(f"Validation failed (attempt {attempt + 1}/{max_retries}): {error_message}")
                last_error = error_message
                continue
            
            # Add club and topic information
            profile_data.update({
                'target_club_name': combination.primary_clubs[0],
                'target_club_description': club_descriptions[combination.primary_clubs[0]],
                'secondary_clubs': combination.primary_clubs[1:] + combination.secondary_clubs,
                'topic_info': {
                    'primary_topic_id': combination.primary_topic_id,
                    'secondary_topic_id': combination.secondary_topic_id,
                    'topic_similarity': combination.topic_similarity
                }
            })
            
            # Create and return the profile
            return SyntheticProfile(**profile_data)
            
        except Exception as e:
            logger.error(f"Error generating profile (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                return None
            continue
    
    return None

def load_cluster_relationships(cache_dir: str) -> List[Dict]:
    """Load pre-computed cluster relationships"""
    relationship_file = Path(cache_dir) / 'cluster_relationships.json'
    if not relationship_file.exists():
        raise FileNotFoundError(
            f"Cluster relationships not found at {relationship_file}. "
            "Please run clustering first with: python index.py cluster"
        )
    
    with open(relationship_file) as f:
        return json.load(f)

def generate_club_combination(relationships: List[Dict]) -> ClubCombination:
    """Generate a single club combination from the relationships"""
    # Pick a random primary cluster
    primary_cluster = random.choice(relationships)
    
    # Pick 2 random clubs from primary cluster
    if len(primary_cluster['primary_clubs']) < 2:
        return None
    
    primary_clubs = random.sample(primary_cluster['primary_clubs'], 2)
    
    # Pick a related cluster and 1 club from it
    if not primary_cluster['related_clusters']:
        return None
    
    # Weight the selection by similarity
    weights = [rc['similarity'] for rc in primary_cluster['related_clusters']]
    related_cluster = random.choices(
        primary_cluster['related_clusters'],
        weights=weights,
        k=1
    )[0]
    
    # Pick 1 club from related cluster
    available_secondary = [
        c for c in related_cluster['clubs']
        if c not in primary_clubs
    ]
    
    if not available_secondary:
        return None
    
    secondary_clubs = random.sample(available_secondary, 1)
    
    return ClubCombination(
        primary_clubs=primary_clubs,
        secondary_clubs=secondary_clubs,
        primary_topic_id=primary_cluster['id'],
        secondary_topic_id=related_cluster['id'],
        topic_similarity=related_cluster['similarity']
    )

def get_safe_description(club_embeddings: ClubEmbeddings, club_name: str) -> str:
    """Safely get a club description, returning empty string if not found"""
    try:
        return club_embeddings.get_description(club_name)
    except (IndexError, KeyError):
        logger.warning(f"Description not found for club: {club_name}")
        return ""

def worker_generate_profile(args):
    """Worker function for multiprocessing profile generation"""
    client, club_embeddings, combo = args
    return generate_profile_for_combination(client, club_embeddings, combo)

def generate_dataset(
    clubs_csv: str,
    num_samples: int,
    output_file: str,
    cache_dir: str,
    save_chunk_size: int = 50
) -> None:
    """Generate synthetic student profiles"""
    try:
        # Initialize Ray for parallel processing
        init_ray()
        
        # Load club embeddings and cluster relationships
        club_embeddings = ClubEmbeddings(clubs_csv)
        relationships = load_cluster_relationships(cache_dir)
        
        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize profile generators
        num_workers = min(multiprocessing.cpu_count(), 4)  # Limit to 4 workers
        generators = [ProfileGenerator.remote() for _ in range(num_workers)]
        
        # Load existing profiles if any
        try:
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    existing_profiles = json.load(f)
                logger.info(f"Loaded {len(existing_profiles)} existing profiles")
            else:
                existing_profiles = []
        except Exception as e:
            logger.warning(f"Error reading existing profiles: {e}")
            existing_profiles = []
        
        def save_profiles(profiles_to_save):
            """Helper to save profiles to file"""
            if not profiles_to_save:
                return
            
            all_profiles = existing_profiles + profiles_to_save
            temp_file = f"{output_file}.tmp"
            
            try:
                with open(temp_file, 'w') as f:
                    json.dump(all_profiles, f, indent=2)
                os.replace(temp_file, output_file)
                logger.info(f"Saved batch of {len(profiles_to_save)} profiles. Total: {len(all_profiles)}")
            except Exception as e:
                logger.error(f"Error saving profiles: {e}")
        
        # Generate profiles in chunks
        profiles_generated = 0
        logger.info(f"Generating {num_samples} profiles using {num_workers} Ray workers")
        
        with tqdm(total=num_samples, desc="Generating profiles") as pbar:
            while profiles_generated < num_samples:
                # Generate combinations for this chunk
                current_chunk_size = min(save_chunk_size, num_samples - profiles_generated)
                combinations = []
                while len(combinations) < current_chunk_size:
                    combo = generate_club_combination(relationships)
                    if combo:
                        combinations.append(combo)
                
                # Distribute work among generators
                futures = []
                for i, combo in enumerate(combinations):
                    generator = generators[i % num_workers]
                    futures.append(generator.generate_profile.remote(club_embeddings, combo))
                
                # Collect results as they complete
                new_profiles = []
                for profile in ray.get(futures):
                    if profile:
                        new_profiles.append(profile)
                        profiles_generated += 1
                        pbar.update(1)
                
                # Save this batch of profiles
                save_profiles(new_profiles)
                # Update existing_profiles for next batch
                existing_profiles.extend(new_profiles)
        
        # Shutdown Ray
        ray.shutdown()
        logger.info(f"Generation complete! Total profiles in dataset: {len(existing_profiles)}")
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")

if __name__ == "__main__":
    import sys
    clubs_csv = sys.argv[1] if len(sys.argv) > 1 else "clubs.tsv"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    output_file = sys.argv[3] if len(sys.argv) > 3 else "synthetic_data.json"
    cache_dir = sys.argv[4] if len(sys.argv) > 4 else ".cache"
    
    generate_dataset(
        clubs_csv=clubs_csv,
        num_samples=num_samples,
        output_file=output_file,
        cache_dir=cache_dir
    )