@echo off
echo ========================================
echo Agricultural API - Cloud Deployment
echo ========================================
echo.

echo Adding all files...
git add .
echo.

echo Committing changes...
git commit -m "Deployment-ready Agricultural API with cloud deployment support"
echo.

echo ========================================
echo Next steps:
echo 1. Create a new repository on GitHub named: agricultural-api
echo 2. Copy the repository URL
echo 3. Run: git remote add origin YOUR_GITHUB_URL
echo 4. Run: git push -u origin main
echo 5. Deploy on Railway.app
echo ========================================
echo.
pause

