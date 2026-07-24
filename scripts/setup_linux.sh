#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Could not find $PYTHON_BIN."
  echo "Install Python 3.10 and python3.10-venv, or set PYTHON_BIN to another supported Python."
  exit 1
fi

echo "Creating virtual environment..."
"$PYTHON_BIN" -m venv .venv

echo "Activating environment..."
source .venv/bin/activate

echo "Updating packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing project dependencies except PyTorch..."
python -m pip install -r requirements.txt

echo
echo "Base setup complete."
echo "Next: install CUDA-enabled PyTorch using the official PyTorch Start Locally selector."
echo "Then run:"
echo "  python scripts/check_setup.py"
echo "  python scripts/smoke_test_mpe.py"
echo "  pytest"
