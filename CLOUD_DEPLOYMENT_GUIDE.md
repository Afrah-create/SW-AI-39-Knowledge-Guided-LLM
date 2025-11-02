# Cloud Deployment Guide - Agricultural Recommendation System

## Overview
This guide provides step-by-step instructions for deploying the Agricultural Recommendation System to various cloud platforms. All methods allow you to deploy without Docker Desktop on your local machine.

---

## Table of Contents
1. [AWS EC2 Deployment](#1-aws-ec2-deployment)
2. [Google Cloud Run](#2-google-cloud-run)
3. [Azure Container Instances](#3-azure-container-instances)
4. [Heroku Deployment](#4-heroku-deployment)
5. [Railway.app (Simplest Option)](#5-railwayapp---simplest-option)
6. [Pre-Deployment Checklist](#pre-deployment-checklist)

---

## 1. AWS EC2 Deployment

### Prerequisites
- AWS Account (Free tier available for 12 months)
- Credit card (for verification, not charged on free tier)

### Step-by-Step Instructions

#### Step 1: Create EC2 Instance
1. **Login to AWS Console**: https://console.aws.amazon.com
2. **Navigate to EC2 Dashboard**
3. **Click "Launch Instance"**
4. **Configure Instance**:
   - **Name**: agricultural-api
   - **AMI**: Ubuntu 22.04 LTS (Free tier eligible)
   - **Instance type**: t2.micro (Free tier)
   - **Key pair**: Create new key pair, name it "agricultural-api-key", download `.pem` file
   - **Network settings**: Create security group with these rules:
     - SSH (22) - Your IP
     - Custom TCP (5000) - Anywhere (0.0.0.0/0)
     - HTTPS (443) - Anywhere
     - HTTP (80) - Anywhere
5. **Launch Instance**

#### Step 2: Connect to Your Instance
```bash
# On Windows (using PowerShell or Git Bash)
cd C:\Users\HP\Desktop\Final\deployment
scp -i agricultural-api-key.pem -r . ubuntu@YOUR_EC2_IP:/home/ubuntu/app
```

Or connect via SSH:
```bash
ssh -i agricultural-api-key.pem ubuntu@YOUR_EC2_IP
```

#### Step 3: Install Dependencies on EC2
Once connected via SSH:
```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip -y

# Create virtual environment
cd ~/app
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 4: Run the Application
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the application
python app/main.py
```

#### Step 5: Access Your Application
- **Your API will be available at**: `http://YOUR_EC2_IP:5000`
- **Web Interface**: Navigate to the URL in your browser

#### Step 6: Keep Application Running (Optional - Production)
```bash
# Install systemd service
sudo nano /etc/systemd/system/agricultural-api.service
```

Add this content:
```ini
[Unit]
Description=Agricultural API Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/app
Environment="PATH=/home/ubuntu/app/venv/bin"
ExecStart=/home/ubuntu/app/venv/bin/python app/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable agricultural-api
sudo systemctl start agricultural-api
sudo systemctl status agricultural-api
```

**Estimated Cost**: Free for 12 months (t2.micro), then ~$10/month

---

## 2. Google Cloud Run

### Prerequisites
- Google Cloud Account (Free $300 credit for 90 days)
- Google Cloud SDK installed (optional)

### Step-by-Step Instructions

#### Step 1: Set Up Google Cloud
1. **Go to**: https://console.cloud.google.com
2. **Create a New Project**: Name it "agricultural-api"
3. **Note your Project ID**

#### Step 2: Prepare Your Files
You need these files in your deployment directory:

**Create `Dockerfile`** (already exists):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Set environment variable for Cloud Run
ENV PORT=8080

# Run the application
CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 app.main:app
```

**Create `.dockerignore`**:
```dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.env
.git
*.log
```

**Modify `app/main.py`** (if needed) to use environment variable for port:
```python
import os
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
```

#### Step 3: Deploy to Cloud Run

**Option A: Using Google Cloud Console (Easier)**
1. Go to Cloud Run in Google Cloud Console
2. Click "Create Service"
3. Upload your deployment folder as a ZIP
4. Set:
   - **Service name**: agricultural-api
   - **Region**: Choose closest to you
   - **CPU**: 1
   - **Memory**: 2GB
   - **Container port**: 8080
5. Click "Deploy"

**Option B: Using Command Line (if you have gcloud CLI)**
```bash
# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy agricultural-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1
```

#### Step 4: Access Your Application
After deployment, you'll get a URL like:
- `https://agricultural-api-xxxxx-uc.a.run.app`

Your API is now live at this URL!

**Estimated Cost**: Free tier: 2 million requests/month, then pay per use

---

## 3. Azure Container Instances

### Prerequisites
- Microsoft Azure Account (Free $200 credit for 30 days)

### Step-by-Step Instructions

#### Step 1: Build and Push Docker Image
Since Azure uses Docker, you'll need to build the image somewhere Docker is installed (like GitHub Codespaces or a friend's computer).

**Build the image**:
```bash
cd deployment
docker build -t agricultural-api .
docker tag agricultural-api YOUR_AZURE_REGISTRY.azurecr.io/agricultural-api
```

#### Step 2: Push to Azure Container Registry
1. Create Azure Container Registry:
```bash
az acr create --resource-group agricultural-rg --name YOUR_REGISTRY_NAME --sku Basic
az acr login --name YOUR_REGISTRY_NAME
docker push YOUR_REGISTRY_NAME.azurecr.io/agricultural-api
```

#### Step 3: Deploy to Azure Container Instances
```bash
az container create \
  --resource-group agricultural-rg \
  --name agricultural-api \
  --image YOUR_REGISTRY_NAME.azurecr.io/agricultural-api \
  --dns-name-label agricultural-api \
  --ports 5000 \
  --registry-login-server YOUR_REGISTRY_NAME.azurecr.io \
  --registry-username YOUR_REGISTRY_NAME \
  --registry-password YOUR_PASSWORD \
  --memory 2 --cpu 1
```

#### Step 4: Access Your Application
Get the FQDN:
```bash
az container show --resource-group agricultural-rg --name agricultural-api --query "{FQDN:ipAddress.fqdn,IP:ipAddress.ip,Port:containers[0].ports[0].port}" --out table
```

Access at: `http://YOUR_FQDN.azurecontainer.io:5000`

**Estimated Cost**: ~$20/month

---

## 4. Heroku Deployment

### Prerequisites
- Heroku Account (Free tier available with limitations)
- Heroku CLI (optional but recommended)

### Step-by-Step Instructions

#### Step 1: Create Heroku App
1. **Go to**: https://dashboard.heroku.com/new
2. **Create new app**: Name it `agricultural-api` (must be unique)
3. **Note your app name**

#### Step 2: Prepare Your Files
Add these files to your deployment directory:

**Create `Procfile`**:
```procfile
web: gunicorn app.main:app --bind 0.0.0.0:$PORT
```

**Modify `app/main.py`** to listen on Heroku's port:
```python
import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

**Create `runtime.txt`**:
```
python-3.11.0
```

#### Step 3: Deploy via Git
```bash
# Login to Heroku (if you have CLI)
heroku login

# In your deployment directory
git init
git add .
git commit -m "Initial commit"

# Create Heroku app
heroku create agricultural-api

# Push to Heroku
git push heroku main
```

#### Step 4: Access Your Application
Your app will be available at: `https://agricultural-api.herokuapp.com`

**Note**: Heroku free tier has been discontinued, but student accounts may have free dynos.

**Estimated Cost**: $7/month (basic dyno)

---

## 5. Railway.app - Simplest Option

### Prerequisites
- GitHub Account
- Railway Account (Connect via GitHub)

### Step-by-Step Instructions

#### Step 1: Push Your Code to GitHub
1. **Create a new repository** on GitHub (e.g., `agricultural-api`)
2. **Push your deployment folder**:
```bash
cd C:\Users\HP\Desktop\Final\deployment
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/agricultural-api.git
git push -u origin main
```

#### Step 2: Deploy on Railway
1. **Go to**: https://railway.app
2. **Login with GitHub**
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Select your `agricultural-api` repository**
6. **Railway will automatically**:
   - Detect it's a Python app
   - Install dependencies from `requirements.txt`
   - Run `python app/main.py`

#### Step 3: Access Your Application
Railway provides a unique URL: `https://agricultural-api-production.railway.app`

#### Step 4: Add Environment Variables (if needed)
Go to your project settings and add:
- `GEMINI_API_KEY` (if you're using Gemini API)

**Estimated Cost**: $5/month (hobby plan) or free tier available

---

## Pre-Deployment Checklist

Before deploying, ensure:

### Files Required
- [x] `app/main.py` - Main Flask application
- [x] `requirements.txt` - Python dependencies
- [x] All data files in `processed/` folder
- [x] All model files in `models/` folder
- [x] `Dockerfile` (for Docker-based deployments)
- [ ] `.gitignore` - Ignore sensitive files
- [ ] `README.md` - Deployment instructions

### Environment Variables to Set
- `PORT` - Server port (often set automatically)
- `GEMINI_API_KEY` - For Gemini API integration (optional)
- `FLASK_ENV` - Set to `production` for production

### Security Considerations
- [ ] Never commit API keys to Git
- [ ] Use environment variables for sensitive data
- [ ] Enable HTTPS (most platforms do this automatically)
- [ ] Set up proper CORS policies

### Testing Before Deployment
- [ ] Test API endpoints locally first
- [ ] Verify all data files are accessible
- [ ] Check model loading works correctly
- [ ] Test PDF generation feature
- [ ] Verify web interface loads properly

---

## Comparison Table

| Platform | Difficulty | Cost | Setup Time | Best For |
|----------|-----------|------|------------|----------|
| **AWS EC2** | Medium | Free → $10/mo | 2-3 hours | Full control, custom setup |
| **Google Cloud Run** | Easy | Free tier → Pay per use | 30 mins | Container-based, auto-scaling |
| **Azure** | Medium | Free tier → $20/mo | 1-2 hours | Enterprise integration |
| **Heroku** | Easy | $7/mo | 20 mins | Quick deployment (no free tier) |
| **Railway** | Easiest | Free → $5/mo | 15 mins | Modern, simple, GitHub integration |

---

## Recommendation

For your situation, I recommend **Railway.app** (Option 5) because:
- ✅ Easiest setup (just connect GitHub)
- ✅ No Docker required
- ✅ Automatic HTTPS
- ✅ Built-in environment variables
- ✅ Free tier available
- ✅ Automatic deployments from Git
- ✅ Modern interface

### Quick Start with Railway:
1. Push your `deployment` folder to GitHub
2. Sign up at railway.app with GitHub
3. Click "New Project" → Select your repo
4. Wait 5 minutes for deployment
5. Your API is live!

---

## Need Help?

If you encounter issues:
1. Check application logs on your chosen platform
2. Verify all file paths are correct
3. Ensure environment variables are set
4. Test locally first before deploying
5. Check platform-specific documentation

---

**Last Updated**: January 2025
**Version**: 1.0
