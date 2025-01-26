import json
from pathlib import Path

def clean_synthetic_data(input_file: str = "synthetic_data.json", output_file: str = None):
    """Remove embeddings from synthetic data file while preserving all other data"""
    if output_file is None:
        output_file = input_file
        # Create backup of original file
        backup_file = Path(input_file).with_suffix('.json.bak')
        Path(input_file).rename(backup_file)
        print(f"Created backup of original file at: {backup_file}")
    
    # Read the original data
    with open(backup_file if output_file == input_file else input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original file size: {Path(backup_file if output_file == input_file else input_file).stat().st_size / 1024:.2f} KB")
    print(f"Number of profiles: {len(data)}")
    
    # Remove embeddings while keeping all other data
    cleaned_data = []
    removed_count = 0
    for profile in data:
        # Remove the embedding field if it exists
        if 'target_club_embedding' in profile:
            del profile['target_club_embedding']
            removed_count += 1
        cleaned_data.append(profile)
    
    # Save the cleaned data
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"Cleaned file size: {Path(output_file).stat().st_size / 1024:.2f} KB")
    print(f"Removed embeddings from {removed_count} profiles")
    print(f"Saved cleaned data to: {output_file}")

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    clean_synthetic_data(input_file, output_file) 