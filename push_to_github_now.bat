@echo off
echo ========================================
echo FLUX Scientific Computing - GitHub Push
echo ========================================
echo.

echo STEP 1: First, create the repository on GitHub
echo ------------------------------------------------
echo 1. Go to: https://github.com/new
echo 2. Repository name: flux-scientific
echo 3. Set to PUBLIC
echo 4. DO NOT initialize with README, .gitignore, or license
echo 5. Click "Create repository"
echo.
echo Press any key once you've created the repository...
pause > nul

echo.
echo STEP 2: Pushing to GitHub...
echo ------------------------------------------------

REM Try with flux-scientific first
echo Trying repository: flux-scientific
git remote add origin https://github.com/MichaelCrowe11/flux-scientific.git
git branch -M main
git push -u origin main

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo First attempt failed. Trying flux-lang...
    git remote remove origin
    git remote add origin https://github.com/MichaelCrowe11/flux-lang.git
    git push -u origin main
)

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! FLUX is now on GitHub!
    echo ========================================
    echo.
    echo Your repository: https://github.com/MichaelCrowe11/flux-scientific
    echo.
    echo NEXT STEPS:
    echo 1. Add description: "Write PDEs like math, compile to GPU"
    echo 2. Add topics: scientific-computing, pde, cfd, gpu, cuda
    echo 3. Star your own repo
    echo 4. Create first issue: "Call for Contributors"
    echo.
    echo SHARE ON:
    echo - Hacker News: "Show HN: FLUX - Write ∂u/∂t = ∇²u, get GPU code"
    echo - Reddit r/CFD: "Built a DSL for scientific computing with GPU support"
    echo - Twitter/X: "Launching FLUX: PDEs to GPUs in one line"
    echo ========================================
) ELSE (
    echo.
    echo ERROR: Could not push to GitHub.
    echo Please ensure you've created the repository first.
)

pause