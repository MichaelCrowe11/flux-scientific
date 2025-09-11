@echo off
echo Setting up GitHub remote and pushing FLUX Scientific Computing Language...
echo.

REM Add your GitHub repository URL here
REM Replace YOUR_USERNAME with your actual GitHub username
set GITHUB_URL=https://github.com/YOUR_USERNAME/flux-scientific.git

echo Adding remote origin: %GITHUB_URL%
git remote add origin %GITHUB_URL%

echo.
echo Renaming branch to main...
git branch -M main

echo.
echo Pushing to GitHub...
git push -u origin main

echo.
echo =============================================
echo FLUX has been pushed to GitHub!
echo.
echo Next steps:
echo 1. Star your own repository
echo 2. Add topics: scientific-computing, pde, cfd, fem, gpu, cuda
echo 3. Create first issue: "Call for Contributors"
echo 4. Share on social media
echo =============================================

pause