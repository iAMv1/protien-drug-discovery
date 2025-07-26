# PowerShell script to set up Git repository
# Run this script after restarting PowerShell or adding Git to PATH

Write-Host "🚀 Setting up Git repository for Protein-Drug Discovery Platform" -ForegroundColor Green

# Check if Git is available
try {
    $gitVersion = git --version
    Write-Host "✅ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git not found. Please install Git first:" -ForegroundColor Red
    Write-Host "   Download from: https://git-scm.com/download/windows" -ForegroundColor Yellow
    Write-Host "   Or run: winget install --id Git.Git -e --source winget" -ForegroundColor Yellow
    exit 1
}

# Initialize Git repository
Write-Host "📁 Initializing Git repository..." -ForegroundColor Blue
git init

# Configure Git (replace with your details)
Write-Host "⚙️ Configuring Git..." -ForegroundColor Blue
$userName = Read-Host "Enter your Git username"
$userEmail = Read-Host "Enter your Git email"

git config user.name "$userName"
git config user.email "$userEmail"

# Add all files
Write-Host "📝 Adding files to Git..." -ForegroundColor Blue
git add .

# Create initial commit
Write-Host "💾 Creating initial commit..." -ForegroundColor Blue
git commit -m "🧬 Initial commit: Protein-Drug Discovery Platform with DoubleSG-DTA integration

Features:
- Enhanced DoubleSG-DTA integration with ESM-2
- Comprehensive training pipeline
- Real-time inference API
- Interactive web interface
- Authentication system
- Python environment migration specs
- Complete documentation and tests"

Write-Host "✅ Git repository initialized successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🔗 Next steps to push to GitHub:" -ForegroundColor Yellow
Write-Host "1. Create a new repository on GitHub"
Write-Host "2. Run: git remote add origin https://github.com/yourusername/protein-drug-discovery.git"
Write-Host "3. Run: git branch -M main"
Write-Host "4. Run: git push -u origin main"
Write-Host ""
Write-Host "📚 Repository contents:" -ForegroundColor Cyan
Write-Host "- DoubleSG-DTA integration (protein_drug_discovery/models/)"
Write-Host "- Training pipelines (protein_drug_discovery/training/)"
Write-Host "- API and web interface (protein_drug_discovery/api/, ui/)"
Write-Host "- Python environment setup specs (.kiro/specs/python-environment-setup/)"
Write-Host "- Comprehensive documentation (README.md, docs/)"
Write-Host "- Demo scripts and tests"