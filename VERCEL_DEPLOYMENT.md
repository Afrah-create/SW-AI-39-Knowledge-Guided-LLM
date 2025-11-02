# Vercel Deployment Configuration

This document explains the Vercel deployment setup for the Agricultural AI application.

## Configuration Files

### `vercel.json`
- **Entry Point**: `api/index.py` (Vercel serverless function)
- **Routes**: All routes (`/`, `/api/*`) are handled by the Flask app
- **Memory**: 3008 MB (maximum for Vercel Pro)
- **Timeout**: 60 seconds
- **Models Excluded**: `models/` and `quick_fine_tuned_fast/` are excluded from deployment

### `.vercelignore`
Excludes large files from Vercel deployment:
- Model directories (`models/`, `quick_fine_tuned_fast/`)
- Large data files
- Cache and temporary files
- Logs and environment files

Models are downloaded from Hugging Face at runtime instead.

### `api/index.py`
Serverless function entry point that imports and exposes the Flask app from `app/main.py`.

## Deployment Steps

### 1. Push Models to Hugging Face First

```bash
# Login to Hugging Face
huggingface-cli login

# Push models
python push_to_huggingface.py
```

See `HUGGINGFACE_DEPLOYMENT.md` for detailed instructions.

### 2. Push Code to GitHub

```bash
# Ensure models are excluded (check .gitignore)
git status

# Add and commit code (models/ and quick_fine_tuned_fast/ should NOT be staged)
git add .
git commit -m "Deploy to Vercel - models excluded"
git push origin main
```

### 3. Deploy to Vercel

#### Option A: Via Vercel Dashboard
1. Go to https://vercel.com
2. Click "New Project"
3. Import your GitHub repository
4. Vercel will auto-detect `vercel.json`
5. Click "Deploy"

#### Option B: Via Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel

# For production
vercel --prod
```

## Environment Variables

Set these in Vercel Dashboard → Project Settings → Environment Variables:

```
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token (optional, for faster downloads)
PYTHONUNBUFFERED=1
```

## How It Works

1. **Code Deployment**: Vercel deploys your code from GitHub (without models)
2. **Model Loading**: When the app starts, it downloads models from Hugging Face:
   - Graph models: `Awongo/soil-crop-recommendation-model`
   - Fine-tuned LLM: `Awongo/agricultural-llm-finetuned`
3. **Caching**: Models are cached in Vercel's filesystem between requests
4. **Runtime**: Flask app handles all routes via serverless functions

## File Structure for Vercel

```
deployment/
├── api/
│   └── index.py          # Vercel serverless entry point
├── app/
│   └── main.py           # Flask application
├── vercel.json           # Vercel configuration
├── .vercelignore         # Files excluded from deployment
├── .gitignore            # Files excluded from GitHub
├── requirements.txt      # Python dependencies
└── [other code files]
```

## Model Download Strategy

Models are downloaded lazily on first request:

```python
# From app/main.py
class AgriculturalModelLoader:
    HF_REPO_ID = "Awongo/soil-crop-recommendation-model"
    
    def load_model(self):
        # Downloads from Hugging Face if not cached
        model_path = hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename="best_model.pth"
        )
```

## Troubleshooting

### Models Not Loading
- Check `HF_TOKEN` environment variable is set
- Verify models are uploaded to Hugging Face
- Check Vercel logs for download errors

### Timeout Issues
- First request may timeout while downloading models
- Consider using Vercel Pro for longer timeouts (60s max)
- Models are cached after first download

### Memory Issues
- Vercel Pro provides up to 3008 MB memory
- Ensure models fit in memory
- Consider using `DISABLE_FINETUNED_MODEL=true` to disable LLM if needed

## Testing Locally

Test Vercel deployment locally:

```bash
# Install Vercel CLI
npm i -g vercel

# Run development server
vercel dev
```

## Production Considerations

1. **Cold Starts**: First request may be slow (model download)
2. **Caching**: Models are cached, subsequent requests are faster
3. **Memory**: Ensure models fit within 3008 MB limit
4. **Timeout**: 60 seconds max for Pro plan
5. **Bandwidth**: Monitor Hugging Face download bandwidth

---

**Note**: Models are NOT included in the deployment. They must be available on Hugging Face Hub before deployment.

