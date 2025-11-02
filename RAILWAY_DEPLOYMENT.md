# Railway Deployment Guide - Agricultural AI

This guide will help you deploy your Agricultural AI application to Railway in minutes.

## Prerequisites

1. **GitHub Repository**: Your code should be in the GitHub repository (https://github.com/Afrah-create/Agricultural-Ai-Models)
2. **Railway Account**: Sign up at https://railway.app (free tier available)
3. **Hugging Face Token** (optional): For faster model downloads

---

## Step-by-Step Deployment

### Step 1: Connect Railway to GitHub

1. **Go to Railway Dashboard**: https://railway.app/dashboard
2. **Click "New Project"**
3. **Select "Deploy from GitHub repo"**
4. **Connect your GitHub account** if not already connected
5. **Select Repository**: Choose `Agricultural-Ai-Models`
6. **Select Branch**: Choose `main`
7. **Set Root Directory**: 
   - Click on the three dots (settings)
   - Go to "Settings" → "Root Directory"
   - Set to: `deployment`
8. **Click "Deploy"**

### Step 2: Configure Build Settings

Railway should auto-detect your Dockerfile, but verify:

1. **Go to Project Settings** → **Settings**
2. **Ensure Build Command** is empty (uses Dockerfile)
3. **Start Command**: Should be empty (handled by Dockerfile)
4. **Root Directory**: `deployment`

### Step 3: Set Environment Variables

Go to your Railway project → **Variables** tab and add:

#### Required Environment Variables:

```env
PORT=8080
FLASK_ENV=production
PYTHONUNBUFFERED=1
```

#### Memory Optimization (Recommended for Railway Free Tier):

```env
DISABLE_FINETUNED_MODEL=true
```

**Important**: By default, the fine-tuned LLM is **disabled** to save memory (328MB model). The application will use a highly optimized structured fallback that provides excellent agricultural insights. Set this to `false` only if you have sufficient memory (>2GB available).

#### Optional Environment Variables (for enhanced features):

```env
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
```

**How to get these keys:**

1. **GEMINI_API_KEY** (optional, for LLM features):
   - Go to https://aistudio.google.com/app/apikey
   - Create a new API key
   - Copy and paste to Railway

2. **HF_TOKEN** (optional, for faster model downloads):
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Read" access
   - Copy and paste to Railway

**Note**: Without GEMINI_API_KEY, the app will use fallback AI analysis (still works perfectly!)

### Step 4: Deploy

1. **Railway will automatically start building** when you connect the repo
2. **Monitor the build logs** in the Railway dashboard
3. **Wait for deployment to complete** (first build takes 5-10 minutes)

### Step 5: Configure Domain (Optional)

1. Go to **Settings** → **Generate Domain**
2. Railway will create a domain like: `your-app-name.up.railway.app`
3. Or add a custom domain if you have one

---

## Deployment Configuration

### What Railway Will Do:

1. **Detect Dockerfile**: Automatically builds from `deployment/Dockerfile`
2. **Install Dependencies**: Installs all packages from `requirements.txt`
3. **Download Models**: Models will be downloaded from Hugging Face on first request:
   - Graph models: `Awongo/soil-crop-recommendation-model`
   - Fine-tuned LLM: `Awongo/agricultural-llm-finetuned`
4. **Start Application**: Runs Gunicorn server

### Resource Requirements:

- **Memory**: ~2GB (includes model loading)
- **CPU**: 1-2 cores recommended
- **Storage**: ~5GB for models and dependencies

---

## Verification Steps

### 1. Check Build Logs

After deployment, check the Railway logs for:

```
✅ AgriculturalAPI initialized (lazy model loading)
✅ Hugging Face Hub library imported successfully
✅ Transformers library imported successfully
```

### 2. Test the Application

1. **Open your Railway URL**: `https://your-app-name.up.railway.app`
2. **You should see the AgriAI interface**
3. **Fill in the form and test a recommendation**

### 3. Test API Endpoint

```bash
curl -X POST https://your-app-name.up.railway.app/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "soil_properties": {
      "pH": 6.5,
      "organic_matter": 3.0,
      "texture_class": "loam",
      "nitrogen": 60,
      "phosphorus": 30,
      "potassium": 150
    },
    "climate_conditions": {
      "temperature_mean": 25,
      "rainfall_mean": 1200
    },
    "farming_conditions": {
      "available_land": 5
    }
  }'
```

---

## Troubleshooting

### Issue 1: Build Fails

**Symptoms**: Deployment fails during build phase

**Solutions**:
- Check Railway logs for specific error
- Ensure `deployment/Dockerfile` exists
- Verify `deployment/requirements.txt` is correct
- Check that Python version in Dockerfile is compatible

### Issue 2: Application Crashes on Startup

**Symptoms**: Deployed but app doesn't respond

**Solutions**:
- Check Railway logs for Python errors
- Verify environment variables are set correctly
- Check that PORT environment variable is used
- Ensure Gunicorn is installed in requirements.txt

### Issue 3: Models Not Loading

**Symptoms**: App runs but models aren't loaded

**Solutions**:
- Check logs for Hugging Face download errors
- Verify `HF_TOKEN` is set if models are private
- Models load on first request (lazy loading), wait a bit
- Check internet connectivity (models download from Hugging Face)

### Issue 4: Memory Issues

**Symptoms**: App crashes or times out

**Solutions**:
- Upgrade Railway plan to have more RAM
- Models are loaded lazily (on first request) to save memory
- Consider removing unused dependencies

### Issue 5: Slow Response Times

**Symptoms**: API responses are slow

**Solutions**:
- First request is slow (downloads models from Hugging Face)
- Subsequent requests should be faster
- Consider upgrading Railway plan for better CPU

---

## Monitoring

### View Logs

1. Go to Railway dashboard
2. Click on your project
3. Click on the deployment
4. View "Deploy Logs" and "HTTP Logs"

### Health Check

Railway automatically monitors your app, but you can check:

```bash
curl https://your-app-name.up.railway.app
```

---

## Updating Your Deployment

### Automatic Updates

Railway automatically deploys when you push to the `main` branch of your connected GitHub repo.

### Manual Update

1. **Make changes** to your code
2. **Commit and push** to GitHub:
   ```bash
   git add .
   git commit -m "Update application"
   git push origin main
   ```
3. **Railway detects the push** and rebuilds automatically

---

## Cost Estimation

### Railway Free Tier:
- **$5 free credit monthly**
- **500 hours of runtime** (shared across projects)
- **Suitable for development/testing**

### Railway Pro ($5/month):
- **Better performance**
- **More resources**
- **Custom domains**
- **Better for production**

---

## Post-Deployment Checklist

- [ ] Application is accessible via Railway URL
- [ ] Web interface loads correctly
- [ ] Can submit form and get recommendations
- [ ] Models download successfully (check logs)
- [ ] PDF download works (test via UI)
- [ ] Environment variables are set (if using optional features)
- [ ] Domain is configured (optional)

---

## Support

### Railway Documentation
- Railway examples: https://github.com/railwayapp-starters
- Railway docs: https://docs.railway.app

### Your Repository
- GitHub: https://github.com/Afrah-create/Agricultural-Ai-Models
- Issues: Create an issue if you encounter problems

---

## Quick Reference

**Railway Dashboard**: https://railway.app/dashboard
**Your Repository**: https://github.com/Afrah-create/Agricultural-Ai-Models
**Hugging Face Models**: 
- https://huggingface.co/Awongo/soil-crop-recommendation-model
- https://huggingface.co/Awongo/agricultural-llm-finetuned

---

**Happy Deploying! 🚀**

