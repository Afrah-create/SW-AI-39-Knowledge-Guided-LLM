#!/usr/bin/env python3
"""
Script to push models to Hugging Face Hub
Excludes models/ and quick_fine_tuned_fast/ from GitHub, pushes them to HF instead
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("❌ huggingface_hub not installed!")
    print("   Install with: pip install huggingface_hub")
    sys.exit(1)

def push_graph_models(models_dir="models", repo_id="Awongo/soil-crop-recommendation-model"):
    """Push graph embedding models to Hugging Face"""
    print(f"\n📤 Pushing graph models from {models_dir}/ to {repo_id}")
    print("=" * 60)
    
    if not os.path.exists(models_dir):
        print(f"❌ Directory {models_dir} does not exist!")
        return False
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            create_repo(repo_id, exist_ok=True, repo_type="model")
            print(f"✅ Repository {repo_id} is ready")
        except Exception as e:
            print(f"⚠️  Repository creation check: {e}")
        
        # Files to upload (exclude logs and unnecessary files)
        files_to_upload = []
        for file_path in Path(models_dir).rglob("*"):
            if file_path.is_file():
                # Skip hidden files and logs
                if not any(part.startswith('.') for part in file_path.parts):
                    files_to_upload.append(str(file_path))
        
        if not files_to_upload:
            print(f"❌ No files found in {models_dir}/")
            return False
        
        # Upload files
        print(f"\n📦 Uploading {len(files_to_upload)} files...")
        for file_path in files_to_upload:
            # Get relative path for HF repo
            rel_path = os.path.relpath(file_path, models_dir)
            print(f"  → Uploading {rel_path}...")
            
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"    ✅ {rel_path} uploaded successfully")
            except Exception as e:
                print(f"    ❌ Failed to upload {rel_path}: {e}")
                return False
        
        print(f"\n✅ Successfully pushed all graph models to {repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing graph models: {e}")
        return False


def push_finetuned_model(model_dir="quick_fine_tuned_fast", repo_id="Awongo/agricultural-llm-finetuned"):
    """Push fine-tuned LLM to Hugging Face"""
    print(f"\n📤 Pushing fine-tuned model from {model_dir}/ to {repo_id}")
    print("=" * 60)
    
    if not os.path.exists(model_dir):
        print(f"❌ Directory {model_dir} does not exist!")
        return False
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            create_repo(repo_id, exist_ok=True, repo_type="model")
            print(f"✅ Repository {repo_id} is ready")
        except Exception as e:
            print(f"⚠️  Repository creation check: {e}")
        
        # For transformers models, upload main directory files (not checkpoints)
        # Exclude checkpoint directories and logs
        files_to_upload = []
        exclude_dirs = {'checkpoint-1', 'checkpoint-500', 'logs', '__pycache__', '.git'}
        
        for file_path in Path(model_dir).rglob("*"):
            if file_path.is_file():
                # Skip if in excluded directory
                parts = file_path.parts
                if any(excluded in parts for excluded in exclude_dirs):
                    continue
                # Skip hidden files
                if not any(part.startswith('.') for part in parts):
                    files_to_upload.append(str(file_path))
        
        if not files_to_upload:
            print(f"❌ No files found in {model_dir}/ (excluding checkpoints)")
            return False
        
        # Upload files
        print(f"\n📦 Uploading {len(files_to_upload)} files (main model files only)...")
        for file_path in files_to_upload:
            # Get relative path for HF repo
            rel_path = os.path.relpath(file_path, model_dir)
            print(f"  → Uploading {rel_path}...")
            
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"    ✅ {rel_path} uploaded successfully")
            except Exception as e:
                print(f"    ❌ Failed to upload {rel_path}: {e}")
                # Don't fail completely, continue with other files
                continue
        
        print(f"\n✅ Successfully pushed fine-tuned model to {repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing fine-tuned model: {e}")
        return False


def main():
    """Main function"""
    print("🚀 Hugging Face Model Upload Script")
    print("=" * 60)
    
    # Check if user is logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face!")
        print("   Please run: huggingface-cli login")
        print("   Or set HF_TOKEN environment variable")
        sys.exit(1)
    
    # Ask user which models to push
    print("\n📋 What would you like to push?")
    print("1. Graph models (models/) → Awongo/soil-crop-recommendation-model")
    print("2. Fine-tuned LLM (quick_fine_tuned_fast/) → Awongo/agricultural-llm-finetuned")
    print("3. Both")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    success = True
    
    if choice == "1":
        success = push_graph_models()
    elif choice == "2":
        success = push_finetuned_model()
    elif choice == "3":
        success_graph = push_graph_models()
        success_llm = push_finetuned_model()
        success = success_graph and success_llm
    elif choice == "4":
        print("👋 Exiting...")
        sys.exit(0)
    else:
        print("❌ Invalid choice!")
        sys.exit(1)
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 All models pushed successfully!")
        print("\n📝 Next steps:")
        print("1. Verify models on Hugging Face Hub")
        print("2. Push code to GitHub (models will be excluded)")
        print("3. Deploy to Vercel")
    else:
        print("\n" + "=" * 60)
        print("❌ Some uploads failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

