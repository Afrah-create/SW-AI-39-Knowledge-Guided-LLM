# Agricultural Recommendation System - Deployment Guide

## Quick Start

### 1. Local Deployment
```bash
cd deployment
pip install -r requirements.txt
python app/main.py
```

### 2. Docker Deployment
```bash
cd deployment
docker build -t agricultural-api .
docker run -p 5000:5000 agricultural-api
```

### 3. Access the Application
- Web Interface: http://localhost:5000
- API Endpoint: http://localhost:5000/api/recommend
- Health Check: http://localhost:5000/api/health

## Cloud Deployment Options

### AWS EC2
1. Launch Ubuntu 20.04 LTS instance
2. Install Docker: `sudo apt update && sudo apt install docker.io`
3. Copy your deployment folder
4. Run: `docker build -t agricultural-api . && docker run -p 80:5000 agricultural-api`

### Google Cloud Run
1. Build image: `gcloud builds submit --tag gcr.io/PROJECT-ID/agricultural-api`
2. Deploy: `gcloud run deploy agricultural-api --image gcr.io/PROJECT-ID/agricultural-api`

### Azure Container Instances
1. Create resource group: `az group create --name agricultural-rg --location eastus`
2. Deploy: `az container create --resource-group agricultural-rg --name agricultural-api --image your-image`

## Next Steps
1. Copy your trained models to the `models/` directory
2. Update the API code to load your actual models
3. Set up your API keys in environment variables
4. Deploy to your chosen cloud platform
