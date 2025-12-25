# 🚀 Deployment Guide

Get your project live on the internet for FREE! This will give you a demo link to put on your resume.

## 📋 Overview

We'll deploy:
- **Frontend** (React): Vercel or Netlify (FREE)
- **Backend** (FastAPI): Render or Railway (FREE tier)

## 🎯 Option 1: Quick Deploy (Recommended)

### Frontend → Vercel

1. **Push to GitHub** (if you haven't)
   ```bash
   git add .
   git commit -m "Add professional README and deployment configs"
   git push origin main
   ```

2. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub
   - Click "Add New Project"
   - Select your `texasholdem` repository
   - Set root directory to: `react`
   - Build command: `npm run build`
   - Output directory: `dist`
   - Click "Deploy"

3. **Update API URL**
   - After backend is deployed, edit `react/src/App.tsx`
   - Change `const API_URL = 'http://localhost:8000'`
   - To `const API_URL = 'https://your-backend.onrender.com'`

### Backend → Render

1. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Create Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Settings:
     - **Name**: `texasholdem-api`
     - **Root Directory**: Leave blank (or `.`)
     - **Runtime**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
   - Click "Create Web Service"

3. **Wait for Deployment** (5-10 minutes)
   - You'll get a URL like: `https://texasholdem-api.onrender.com`
   - Copy this URL

4. **Update Frontend API URL**
   - Go back to your Vercel project
   - Settings → Environment Variables
   - Add: `VITE_API_URL = https://your-render-url.onrender.com`
   - Redeploy

## 🎯 Option 2: Railway (Alternative Backend Host)

Railway is faster but has a $5/month trial credit (free to start):

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Add environment variables if needed
6. Railway auto-detects Python and builds

## ⚙️ Configuration Files Needed

### For Render: Create `render.yaml` (optional, for easier deploys)

```yaml
services:
  - type: web
    name: texasholdem-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api_server:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
```

### For Frontend: Update CORS in `api_server.py`

Find this section:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
```

For production, update to:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://your-vercel-app.vercel.app",  # Add your Vercel URL
    ],
```

## 🔍 Testing Your Deployment

1. **Test Backend**
   - Visit `https://your-backend-url.onrender.com/docs`
   - You should see FastAPI's interactive docs
   - Try the `/state` endpoint

2. **Test Frontend**
   - Visit your Vercel URL
   - The poker game should load
   - Try playing a hand

## ⚠️ Common Issues

### Frontend can't connect to backend
- Check CORS settings in `api_server.py`
- Verify API_URL in `App.tsx` is correct
- Check browser console for errors

### Backend deployment fails
- Make sure `requirements.txt` includes all dependencies
- Check Render logs for specific error messages
- Verify Python version compatibility

### Free tier limitations
- **Render Free**: Backend sleeps after 15 min inactivity (takes 30s to wake up)
- **Vercel Free**: 100GB bandwidth/month (plenty for portfolio)
- **Railway**: $5 credit (runs out after ~1 month of continuous use)

## 📝 Update Your README

After deployment, update `PROJECT_README.md`:

Add this after the title:
```markdown
🎮 **[Live Demo](https://your-vercel-url.vercel.app)** | 📚 **[API Docs](https://your-backend-url.onrender.com/docs)**
```

## 🎓 Resume Bullet Point

Once deployed:
```
Texas Hold'em AI with Deep Q-Learning | Live Demo ↗ | GitHub ↗
- Deployed full-stack poker app (React + FastAPI) to Vercel and Render with CI/CD
- Implemented DQN reinforcement learning agent achieving 65%+ win rate
- Built real-time game state sync and analytics dashboard with PGN export
```

## 🆘 Need Help?

If deployment fails:
1. Check the deployment logs in Render/Vercel dashboard
2. Make sure all files are committed to GitHub
3. Verify requirements.txt has all dependencies
4. Try deploying backend first, then frontend

---

**Time Estimate**: 30-45 minutes for first-time deployment

**Cost**: $0 (completely free with free tiers)
