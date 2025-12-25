# 🚀 Setting Up Your Personal Repository

This guide will help you move this project to YOUR OWN GitHub repository (not the CS196 course repo).

## 📋 Prerequisites

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name: `poker-ai` (or `texas-holdem-ai`, your choice)
   - Visibility: **Public** (so recruiters can see it)
   - **DO NOT** check "Initialize with README"
   - Click "Create repository"

2. Copy your new repository URL. It will look like:
   ```
   https://github.com/YOUR_USERNAME/poker-ai.git
   ```

## 🔧 Option 1: Copy Project Files (Recommended)

This creates a clean repository with just your poker project:

### Step 1: Create a new directory for your project
```bash
# Go to your home directory or wherever you keep projects
cd ~
# or
cd C:\Users\yuhao\Projects

# Create new directory
mkdir poker-ai
cd poker-ai
```

### Step 2: Initialize git and copy files
```bash
# Initialize new git repository
git init

# Copy all files from your current project
# Replace the path with your actual path
cp -r "C:/Users/yuhao/FA25-Group18/Project/texasholdem/*" .

# Or on Windows, use:
xcopy "C:\Users\yuhao\FA25-Group18\Project\texasholdem" "C:\Users\yuhao\poker-ai" /E /I
```

### Step 3: Clean up files you don't need
```bash
# Remove cache files
rm -rf __pycache__ texasholdem/__pycache__
rm -rf react/node_modules
rm -rf .venv

# Remove temporary files
rm nul
rm train_state.pkl
rm replay.pkl

# Optional: Remove old documentation you don't need
rm ACTION_SPACE_FIX.md BUG_FIX_SUMMARY.md CALL_CHECK_ISSUE.md
rm REWARD_TUNING_LOG.md TESTING_RESULTS.md
```

### Step 4: Add to git and push
```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit: AI-powered Texas Hold'em poker game"

# Add your remote (replace with YOUR repository URL)
git remote add origin https://github.com/YOUR_USERNAME/poker-ai.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 🔧 Option 2: Keep Git History

If you want to keep the commit history from the course project:

```bash
# In your current directory
cd C:\Users\yuhao\FA25-Group18\Project\texasholdem

# Remove the old remote
git remote remove origin

# Add your new remote
git remote add origin https://github.com/YOUR_USERNAME/poker-ai.git

# Push
git push -u origin master
```

**Note**: This will include ALL commits from the course repo, which might not be ideal.

## 📝 After Pushing to GitHub

1. **Update README.md** with your actual information:
   - Replace `YOUR_USERNAME` with your GitHub username
   - Add your name, LinkedIn, and email
   - Update the repository URL

2. **Add a .gitignore file**:
```bash
# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/

# PyTorch
*.pt
*.pth
*.pkl
checkpoints/
!checkpoints/.gitkeep

# Node
node_modules/
dist/
.vite/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# PGN files (game history)
pgns/
!pgns/.gitkeep
EOF

git add .gitignore
git commit -m "Add .gitignore"
git push
```

3. **Add a nice description** on GitHub:
   - Go to your repository on GitHub
   - Click "About" (top right, near the green "Code" button)
   - Description: `AI-powered Texas Hold'em poker with DQN reinforcement learning`
   - Website: (leave blank for now, or add after deployment)
   - Topics: `poker`, `reinforcement-learning`, `pytorch`, `react`, `fastapi`, `deep-q-network`

## ✅ Verification

Your repository should now have:
- ✅ README.md with professional description
- ✅ All source code (backend + frontend)
- ✅ requirements.txt
- ✅ Training scripts
- ✅ Documentation files
- ✅ .gitignore

## 🎯 Next Steps

1. **Customize the README**:
   - Replace `YOUR_USERNAME` with your actual GitHub username
   - Add your name, LinkedIn, email

2. **Add screenshots** (makes it look professional):
   ```bash
   mkdir screenshots
   # Take screenshots of your app and add them
   git add screenshots/
   git commit -m "Add screenshots"
   git push
   ```

3. **Deploy your app** (follow DEPLOYMENT.md)

4. **Add to your resume** with the link!

## ❓ Troubleshooting

**Problem**: `git push` asks for username/password but rejects password
**Solution**: Use a Personal Access Token instead:
1. Go to GitHub Settings → Developer Settings → Personal Access Tokens
2. Generate new token (classic)
3. Select scopes: `repo`
4. Use the token as your password

**Problem**: Files are too large to push
**Solution**: Remove large checkpoint files:
```bash
git rm --cached *.pt *.pkl
git commit -m "Remove large model files"
git push
```

---

**Questions?** Just ask!
