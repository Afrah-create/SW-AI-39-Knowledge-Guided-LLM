# Hugging Face Model Deployment Guide

This guide explains how to push your models to Hugging Face Hub, excluding them from GitHub while keeping them accessible for deployment.

## Overview

- **Graph Models** (`models/`): Push to `Awongo/soil-crop-recommendation-model`
- **Fine-tuned LLM** (`quick_fine_tuned_fast/`): Push to `Awongo/agricultural-llm-finetuned`

These directories are excluded from GitHub via `.gitignore` but will be pushed to Hugging Face for distribution.

## Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co
2. **Hugging Face Token**: Get your token from https://huggingface.co/settings/tokens
   - Create a token with "Write" permissions
3. **Hugging Face CLI**: Install and login

```bash
pip install huggingface_hub
huggingface-cli login
```

Or set environment variable:
```bash
export HF_TOKEN=your_token_here
```

## Method 1: Using the Automated Script (Recommended)

### Step 1: Run the Push Script

```bash
python push_to_huggingface.py
```

### Step 2: Select What to Push

The script will prompt you:
- **Option 1**: Push graph models only
- **Option 2**: Push fine-tuned LLM only  
- **Option 3**: Push both
- **Option 4**: Exit

### Step 3: Wait for Upload

The script will:
- Check if you're logged in
- Create repositories if they don't exist
- Upload all model files
- Show progress for each file

## Method 2: Manual Upload via Python

### Push Graph Models

```python
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = "Awongo/soil-crop-recommendation-model"

# Create repo if needed
create_repo(repo_id, exist_ok=True, repo_type="model")

# Upload files
api.upload_file(
    path_or_fileobj="models/best_model.pth",
    path_in_repo="best_model.pth",
    repo_id=repo_id,
    repo_type="model"
)

# Repeat for other files: model_metadata.json, gcn_model.pth, etc.
```

### Push Fine-tuned LLM

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "Awongo/agricultural-llm-finetuned"

# Upload main model files (not checkpoints)
files = [
    "quick_fine_tuned_fast/config.json",
    "quick_fine_tuned_fast/model.safetensors",
    "quick_fine_tuned_fast/tokenizer.json",
    "quick_fine_tuned_fast/README.md",
    # ... other files
]

for file_path in files:
    rel_path = os.path.basename(file_path)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=rel_path,
        repo_id=repo_id,
        repo_type="model"
    )
```

## Method 3: Using Hugging Face CLI

### Upload Graph Models

```bash
cd models
huggingface-cli upload Awongo/soil-crop-recommendation-model . --repo-type model
```

### Upload Fine-tuned Model

```bash
cd quick_fine_tuned_fast
# Upload only main files, exclude checkpoints
huggingface-cli upload Awongo/agricultural-llm-finetuned \
    config.json \
    model.safetensors \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    vocab.json \
    merges.txt \
    generation_config.json \
    README.md \
    --repo-type model
```

## Repository Structure

### Graph Models Repository (`Awongo/soil-crop-recommendation-model`)

```
models/
├── best_model.pth
├── best_model_info.json
├── model_metadata.json
├── gcn_model.pth
├── transe_model.pth
├── distmult_model.pth
├── complex_model.pth
├── graphsage_model.pth
└── README.md
```

### Fine-tuned LLM Repository (`Awongo/agricultural-llm-finetuned`)

```
quick_fine_tuned_fast/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.json
├── merges.txt
├── generation_config.json
└── README.md
```

**Note**: Checkpoint directories (`checkpoint-1/`, `checkpoint-500/`) are NOT uploaded to save space.

## Verification

After uploading, verify on Hugging Face:

1. Go to https://huggingface.co/Awongo/soil-crop-recommendation-model
2. Go to https://huggingface.co/Awongo/agricultural-llm-finetuned
3. Check that all files are present
4. Test model loading:

```python
from huggingface_hub import hf_hub_download

# Test graph model download
model_path = hf_hub_download(
    repo_id="Awongo/soil-crop-recommendation-model",
    filename="best_model.pth"
)
print(f"✅ Model downloaded to: {model_path}")

# Test LLM loading
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Awongo/agricultural-llm-finetuned")
model = AutoModelForCausalLM.from_pretrained("Awongo/agricultural-llm-finetuned")
print("✅ LLM loaded successfully")
```

## Integration with Code

Your application code already loads models from Hugging Face:

### Graph Models
```python
# In app/main.py - AgriculturalModelLoader class
HF_REPO_ID = "Awongo/soil-crop-recommendation-model"
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="best_model.pth")
```

### Fine-tuned LLM
```python
# In app/main.py - FineTunedLLM class
HF_REPO_ID = "Awongo/agricultural-llm-finetuned"
model = AutoModelForCausalLM.from_pretrained(HF_REPO_ID)
```

## GitHub Exclusion

The following are excluded from GitHub but pushed to Hugging Face:

- `models/` - Graph embedding models
- `quick_fine_tuned_fast/` - Fine-tuned language model

These are listed in `.gitignore` and `.vercelignore`.

## Troubleshooting

### "Repository not found"
- Create the repository first on https://huggingface.co/new
- Or use `create_repo()` with `exist_ok=True`

### "Authentication failed"
```bash
huggingface-cli login
# Enter your token
```

### "File too large"
- Hugging Face free tier has 10GB per repo
- Use Git LFS if needed: https://huggingface.co/docs/hub/repositories-git-lfs

### "Upload timeout"
- Upload large files individually
- Use `huggingface-cli` for better progress tracking

## Next Steps

After pushing models to Hugging Face:

1. ✅ Verify models are accessible
2. ✅ Push code to GitHub (models will be excluded)
3. ✅ Deploy to Vercel (models download at runtime)
4. ✅ Test deployment to ensure models load correctly

---

**Note**: Models are automatically downloaded from Hugging Face when your application runs on Vercel, so they don't need to be in your GitHub repository.

