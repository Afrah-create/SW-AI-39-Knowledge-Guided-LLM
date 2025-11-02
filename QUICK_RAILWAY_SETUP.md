# Quick Railway Setup - 5 Minutes

## 🚀 Ultra-Quick Setup

### Step 1: Connect to Railway (2 min)
1. Go to https://railway.app/new
2. Click **"Deploy from GitHub repo"**
3. Select **`Agricultural-Ai-Models`** repository
4. Set **Root Directory** to: `deployment`
5. Click **"Deploy"**

### Step 2: Set Root Directory (Important!)
1. Click your project → **Settings**
2. Scroll to **"Root Directory"**
3. Enter: `deployment`
4. Save

### Step 3: Wait for Build (3-5 min)
- Railway will automatically build and deploy
- Watch the logs in Railway dashboard
- Look for: `✅ AgriculturalAPI initialized`

### Step 4: Test (1 min)
1. Click on your deployment
2. Copy the generated domain (e.g., `your-app.up.railway.app`)
3. Open in browser and test!

---

## ⚙️ Optional: Environment Variables

If you want enhanced features, add these in Railway → Variables:

```env
GEMINI_API_KEY=your_key_here        # Optional: For advanced LLM features
HF_TOKEN=your_token_here            # Optional: For faster model downloads
```

**Without these, the app still works perfectly!** It uses fallback analysis.

---

## ✅ Done!

Your app is now live at: `https://your-app.up.railway.app chances`

---

## 📚 Need More Details?

See **RAILWAY_DEPLOYMENT.md** for complete guide.

