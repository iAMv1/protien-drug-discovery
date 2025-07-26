@echo off
echo 🚀 Setting up Git repository for Protein-Drug Discovery Platform

REM Add Git to PATH for this session
set PATH=%PATH%;C:\Program Files\Git\cmd;C:\Program Files\Git\bin

REM Check if Git is available
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git not found. Please install Git first:
    echo    Download from: https://git-scm.com/download/windows
    echo    Or run: winget install --id Git.Git -e --source winget
    pause
    exit /b 1
)

echo ✅ Git found!

REM Initialize Git repository
echo 📁 Initializing Git repository...
git init

REM Configure Git (you'll need to replace these)
echo ⚙️ Please configure Git with your details:
set /p username="Enter your Git username: "
set /p email="Enter your Git email: "

git config user.name "%username%"
git config user.email "%email%"

REM Add all files
echo 📝 Adding files to Git...
git add .

REM Create initial commit
echo 💾 Creating initial commit...
git commit -m "🧬 Initial commit: Protein-Drug Discovery Platform with DoubleSG-DTA integration"

echo.
echo ✅ Git repository initialized successfully!
echo.
echo 🔗 Next steps to push to GitHub:
echo 1. Create a new repository on GitHub
echo 2. Run: git remote add origin https://github.com/yourusername/protein-drug-discovery.git
echo 3. Run: git branch -M main
echo 4. Run: git push -u origin main
echo.
echo 📚 Repository includes:
echo - DoubleSG-DTA integration with ESM-2
echo - Complete training pipeline
echo - API and web interface
echo - Python environment setup specs
echo - Comprehensive documentation
echo - Demo scripts and tests

pause