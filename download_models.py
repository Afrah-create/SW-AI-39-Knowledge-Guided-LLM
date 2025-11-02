"""
Download trained models from cloud storage if not present locally.
This ensures models are available for cloud deployments.
"""

import os
import requests
from pathlib import Path

# Model URLs (you'll need to upload your models to cloud storage)
MODELS_CONFIG = {
    'best_model.pth': 'https://YOUR_STORAGE_URL/models/best_model.pth',
    'gcn_model.pth': 'https://YOUR_STORAGE_URL/models/gcn_model.pth',
    'complex_model.pth': 'https://YOUR_STORAGE_URL/models/complex_model.pth',
    'distmult_model.pth': 'https://YOUR_STORAGE_URL/models/distmult_model.pth',
    'graphsage_model.pth': 'https://YOUR_STORAGE_URL/models/graphsage_model.pth',
    'transe_model.pth': 'https://YOUR_STORAGE_URL/models/transe_model.pth',
}

def download_model(model_name, url, dest_path):
    """Download a model file from URL."""
    print(f"Downloading {model_name}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded {model_name}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False

def ensure_models_exist(models_dir='processed/trained_models'):
    """Ensure all required model files exist, download if missing."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    all_models_exist = True
    for model_name, url in MODELS_CONFIG.items():
        model_path = models_path / model_name
        
        if not model_path.exists():
            all_models_exist = False
            download_model(model_name, url, model_path)
        else:
            print(f"✓ {model_name} already exists")
    
    return all_models_exist

if __name__ == '__main__':
    print("Checking for trained models...")
    ensure_models_exist()
    print("Model check complete!")

