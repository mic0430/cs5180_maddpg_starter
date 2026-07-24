$ErrorActionPreference = "Stop"

Write-Host "Creating Python 3.10 virtual environment..."
py -3.10 -m venv .venv

Write-Host "Activating environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Updating packaging tools..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing project dependencies except PyTorch..."
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Base setup complete."
Write-Host "Next: install CUDA-enabled PyTorch using the official PyTorch Start Locally selector."
Write-Host "Then run:"
Write-Host "  python scripts\check_setup.py"
Write-Host "  python scripts\smoke_test_mpe.py"
Write-Host "  pytest"
