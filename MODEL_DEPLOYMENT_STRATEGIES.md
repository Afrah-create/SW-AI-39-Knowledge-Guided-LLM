# Model Deployment Strategies - Agricultural API

## The Problem

Your trained models (*.pth files) are too large for GitHub (~100-500MB each). Here are 5 practical solutions:

---

## Strategy 1: Include Models in Deployment Folder (Simplest) ⭐

**Best for**: Railway.app, Heroku, AWS, Google Cloud Run

### How it works:
When deploying, upload the entire `deployment` folder (including models) directly to the cloud platform.

### Steps:
1. **Locally**: Models stay in `deployment/processed/trained_models/`
2. **Deploy**: Upload entire deployment folder
3. **Platform**: Models are included in the deployment package

### Pros:
- ✅ Easiest to set up
- ✅ No external dependencies
- ✅ Models available immediately
- ✅ Works on most platforms

### Cons:
- ❌ Initial upload is slow (large file)
- ❌ Uses storage on platform

### Commands:
```bash
# For local testing (models already there)
python app/main.py

# For deployment - just push/release the folder
# Models are included automatically
```

---

## Strategy 2: Cloud Storage + Auto-Download

**Best for**: When models are too large for deployment packages

### How it works:
Store models on cloud storage (Google Drive, Dropbox, S3) and download on first run.

### Implementation:

#### Step 1: Upload models to cloud storage
```
1. Create a folder on Google Drive/Dropbox
2. Upload: models/*.pth, models/*.json
3. Make it publicly accessible (or use API key)
4. Get shareable link
```

#### Step 2: Update `download_models.py`
```python
MODELS_CONFIG = {
    'best_model.pth': 'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID',
    'gcn_model.pth': 'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID',
    # ... etc
}
```

#### Step 3: Call before starting app
```python
# At the beginning of app/main.py
from download_models import ensure_models_exist
ensure_models_exist()
```

### Pros:
- ✅ Keeps repository small
- ✅ Models can be updated independently
- ✅ Works with any cloud storage

### Cons:
- ❌ Requires internet on first run
- ❌ Slightly more complex setup

---

## Strategy 3: Git LFS (Large File Storage)

**Best for**: When you want models in Git repository

### Setup:
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.pkl"

# Commit
git add .gitattributes
git add models/*.pth
git commit -m "Add models via LFS"
git push
```

### Pros:
- ✅ Models versioned with code
- ✅ Easy to track changes
- ✅ Professional approach

### Cons:
- ❌ GitHub LFS has size limits (1GB free)
- ❌ Requires Git LFS installation

### Cost:
- Free: 1 GB storage, 1 GB bandwidth/month
- Paid: $5/month for 50 GB storage

---

## Strategy 4: Platform-Specific Solutions

### Railway.app
```bash
# Models are included automatically in deployment
# No special steps needed
cd deployment
railway up
```

### Google Cloud Run
```python
# Use Cloud Storage Bucket
from google.cloud import storage

def download_model_from_gcs(bucket_name, model_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"models/{model_name}")
    blob.download_to_filename(f"processed/trained_models/{model_name}")
```

### AWS EC2
```bash
# Use S3 bucket
aws s3 sync s3://your-bucket/models processed/trained_models
```

---

## Strategy 5: Rebuild Models in CI/CD (Advanced)

**Best for**: When you want to train models fresh on each deployment

### GitHub Actions Example:
```yaml
name: Train Models on Deploy
on:
  push:
    branches: [main]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train models
        run: python 12_graph_embeddings_training_optimized.py
      - name: Deploy
        uses: deployment-action
```

### Pros:
- ✅ Always fresh models
- ✅ No storage issues
- ✅ Models match code version

### Cons:
- ❌ Slow deployments (training time)
- ❌ Requires compute resources
- ❌ More complex

---

## 🎯 Recommended Approach for Your Project

### For Local Development:
**Use Strategy 1**: Keep models in `deployment/processed/trained_models/`
- Already working ✅
- No changes needed ✅

### For Cloud Deployment (Railway/Heroku):
**Use Strategy 1 or 2**:
- **Option A**: Upload deployment folder with models (easiest)
- **Option B**: Use cloud storage + auto-download (if size limit issues)

### Setup Instructions:

#### Option A: Include Models (Recommended)
```bash
# 1. Your models are in deployment/processed/trained_models/
# 2. When deploying to Railway:
cd deployment
railway init
railway up  # Uploads everything including models
```

#### Option B: Cloud Storage Download
```bash
# 1. Upload models to Google Drive
# 2. Get shareable links
# 3. Update download_models.py with URLs
# 4. Deploy to Railway (without models)
# 5. Models download automatically on first run
```

---

## Quick Reference

| Strategy | Difficulty | Size Limit | Setup Time | Best For |
|----------|-----------|------------|------------|----------|
| **Include in Folder** | Easy | Platform dependent | 0 mins | Railway, local |
| **Cloud Storage** | Medium | None | 15 mins | Large models |
| **Git LFS** | Medium | 1GB free | 5 mins | Professional projects |
| **CI/CD Train** | Hard | None | 30 mins | Always fresh models |
| **Platform-Specific** | Hard | Depends | 20 mins | Enterprise |

---

## 📝 Step-by-Step: Deploy to Railway with Models

### Current Status:
- ✅ Models in `deployment/processed/trained_models/`
- ✅ App code ready
- ✅ Requirements installed

### Next Steps:
1. **Push to GitHub** (code only, no models)
2. **Deploy to Railway** (models uploaded separately OR included in initial upload)
3. **Set environment variables** (API keys, etc.)

### Command:
```bash
# In deployment folder
cd C:\Users\HP\Desktop\Final\deployment

# Connect to Railway (download Railway CLI or use web interface)
railway login
railway init
railway add  # Add models folder
railway deploy  # Deploy!
```

---

## 🔗 Model Storage Links

Once uploaded, add these to your deployment config:

### Google Drive (Example):
```python
MODEL_URLS = {
    'best_model.pth': 'https://drive.google.com/uc?export=download&id=YOUR_ID'
}
```

### Dropbox (Example):
```python
MODEL_URLS = {
    'best_model.pth': 'https://www.dropbox.com/s/YOUR_LINK/best_model.pth?dl=1'
}
```

---

**Bottom Line**: For now, just push your code without models to GitHub. When deploying to Railway/Cloud, include the models folder in that deployment. Models don't need to be in Git!

