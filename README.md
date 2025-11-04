# Agricultural AI Recommendation System

An AI-powered agricultural recommendation system for crop recommendations in Uganda, featuring Graph Neural Networks (GCN) and fine-tuned language models.

## 🚀 Features

- **Graph Neural Network Models**: Multiple GNN architectures (GCN, TransE, DistMult, ComplEx, GraphSAGE) for crop-soil-climate relationship modeling
- **Fine-tuned LLM**: Custom agricultural language model for intelligent recommendations
- **Hugging Face Integration**: Models hosted on Hugging Face Hub for easy deployment
- **Vercel Deployment**: Serverless deployment optimized for cloud platforms
- **Flask API**: RESTful API with web interface

## 📋 Prerequisites

- Python 3.11+ (or 3.12 for Vercel)
- Hugging Face account and token
- Google Gemini API key (optional, for enhanced LLM features)

## 🏗️ Project Structure

```
deployment/
├── api/
│   └── index.py              # Vercel serverless entry point
├── app/
│   └── main.py               # Flask application
├── models/                    # Graph models (excluded from Git, pushed to HF)
├── quick_fine_tuned_fast/     # Fine-tuned LLM (excluded from Git, pushed to HF)
├── processed/                 # Processed data files
├── vercel.json                # Vercel configuration
├── requirements.txt           # Python dependencies
├── push_to_huggingface.py     # Script to push models to Hugging Face
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Afrah-create/SW-AI-39-Knowledge-Guided-LLM.git
cd deployment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: PyTorch is installed from the CPU-only index. If you need GPU support, modify `requirements.txt` accordingly.

### 3. Set Up Environment Variables

Create a `.env` file (not committed to Git):

```bash
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token  # Optional, speeds up model downloads
DISABLE_FINETUNED_MODEL=true     # Optional, disable LLM to save memory
```

### 4. Push Models to Hugging Face

Before deploying, push your models to Hugging Face:

```bash
# Option 1: Interactive script
python push_to_huggingface.py

# Option 2: Push all models
python push_to_huggingface.py --all
```

Models will be uploaded to:
- Graph models: `Awongo/soil-crop-recommendation-model`
- Fine-tuned LLM: `Awongo/agricultural-llm-finetuned`

See [HUGGINGFACE_DEPLOYMENT.md](./HUGGINGFACE_DEPLOYMENT.md) for detailed instructions.

### 5. Run Locally

```bash
python app/main.py
```

Access the application at:
- Web Interface: http://localhost:5000
- API Endpoint: http://localhost:5000/api/recommend
- Health Check: http://localhost:5000/api/health

## ☁️ Deploy to Vercel

### Prerequisites

1. **Hugging Face Models**: Ensure models are pushed to Hugging Face Hub
2. **GitHub Repository**: Code should be pushed to GitHub
3. **Vercel Account**: Sign up at https://vercel.com

### Deployment Steps

#### Option 1: Via Vercel Dashboard (Recommended)

1. Go to https://vercel.com and click "New Project"
2. Import your GitHub repository: `Afrah-create/SW-AI-39-Knowledge-Guided-LLM`
3. Configure:
   - **Framework Preset**: Flask (auto-detected)
   - **Root Directory**: `./` (default)
   - **Build Command**: (auto-detected)
   - **Output Directory**: (auto-detected)
4. Set Environment Variables:
   - `GEMINI_API_KEY`: Your Gemini API key
   - `HF_TOKEN`: Your Hugging Face token (optional)
   - `DISABLE_FINETUNED_MODEL`: `true` (recommended for Hobby plan)
5. Click "Deploy"

#### Option 2: Via Vercel CLI

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

### Configuration

The `vercel.json` configuration includes:
- **Memory**: 2048 MB (Hobby plan limit)
- **Timeout**: 60 seconds
- **Routes**: All routes handled by Flask app
- **Models**: Excluded from deployment (downloaded from Hugging Face at runtime)

See [VERCEL_DEPLOYMENT.md](./VERCEL_DEPLOYMENT.md) for detailed deployment guide.

## 📦 Dependencies

Key dependencies:

- **flask==3.1.2** - Web framework
- **torch==2.2.0+cpu** - PyTorch (CPU-only version)
- **transformers==4.35.0** - Hugging Face transformers
- **huggingface_hub>=0.16.4,<0.18** - Hugging Face Hub integration
- **google-generativeai==0.3.2** - Google Gemini API
- **pandas==2.0.3** - Data manipulation
- **scikit-learn==1.3.0** - Machine learning utilities

See `requirements.txt` for complete list.

## 🔧 Configuration Files

### `vercel.json`
Vercel serverless function configuration:
- Entry point: `api/index.py`
- Routes: All requests handled by Flask app
- Memory: 2048 MB (Hobby plan)

### `.vercelignore` & `.gitignore`
Exclude large files from deployment:
- `models/` - Graph models (downloaded from HF)
- `quick_fine_tuned_fast/` - Fine-tuned LLM (downloaded from HF)
- Large data files and cache

### `api/index.py`
Serverless function entry point that imports and exposes the Flask app.

## 🤖 Model Loading

Models are loaded lazily from Hugging Face on first request:

```python
# Graph models
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="Awongo/soil-crop-recommendation-model",
    filename="best_model.pth"
)

# Fine-tuned LLM
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Awongo/agricultural-llm-finetuned")
```

## 🔑 Environment Variables

### Required
- `GEMINI_API_KEY` - Google Gemini API key for LLM features

### Optional
- `HF_TOKEN` - Hugging Face token (speeds up model downloads)
- `DISABLE_FINETUNED_MODEL` - Set to `true` to disable fine-tuned LLM (saves memory)
- `PYTHONUNBUFFERED` - Set to `1` for better logging

## 📚 API Endpoints

- `GET /` - Web interface
- `POST /api/recommend` - Get crop recommendations
- `GET /api/health` - Health check
- `POST /api/download_pdf` - Download recommendation PDF

## 🐳 Docker Deployment

```bash
# Build image
docker build -t agricultural-api .

# Run container
docker run -p 5000:5000 \
  -e GEMINI_API_KEY=your_key \
  -e HF_TOKEN=your_token \
  agricultural-api
```

## 🛠️ Troubleshooting

### Models Not Loading
- Verify models are uploaded to Hugging Face Hub
- Check `HF_TOKEN` environment variable is set
- Review Vercel logs for download errors

### Memory Issues on Vercel
- Ensure `DISABLE_FINETUNED_MODEL=true` (saves 328MB)
- Models are cached after first download
- Consider upgrading to Pro plan for more memory

### Build Failures
- Check Python version compatibility (3.11+)
- Verify all dependencies in `requirements.txt`
- Review Vercel build logs for specific errors

### PyTorch Installation Issues
- CPU-only version is used (`torch==2.2.0+cpu`)
- Installed from PyTorch CPU index
- For GPU support, modify `requirements.txt`

## 📖 Documentation

- [Hugging Face Deployment Guide](./HUGGINGFACE_DEPLOYMENT.md)
- [Vercel Deployment Guide](./VERCEL_DEPLOYMENT.md)
- [Model Deployment Strategies](./MODEL_DEPLOYMENT_STRATEGIES.md)

## 🔗 Links

- **Hugging Face Repositories**:
  - [Graph Models](https://huggingface.co/Awongo/soil-crop-recommendation-model)
  - [Fine-tuned LLM](https://huggingface.co/Awongo/agricultural-llm-finetuned)

## 📝 Notes

- Models are **NOT** included in the GitHub repository
- Models are downloaded from Hugging Face at runtime
- First request may be slower due to model download
- Subsequent requests use cached models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Push models to Hugging Face if needed
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

---

**Built with**: Flask, PyTorch, Transformers, Hugging Face Hub, Google Gemini API
