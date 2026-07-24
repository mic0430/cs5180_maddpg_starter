# CS5180 MADDPG Project — Teammate Setup Guide

Use this guide to set up the project on your own computer.

**Do not edit**:

```text
requirements.txt
requirements-windows-cuda-lock.txt
scripts/check_setup.py
scripts/smoke_test_mpe.py
```

The shared dependencies are already configured.

---

## 1. Open the project folder

```powershell
cd "C:\path\to\cs5180_maddpg_starter"
```

**What it does:** Opens the project root in PowerShell.

Check:

```powershell
Get-ChildItem
```

**Expected:** You should see:

```text
configs
docs
scripts
src
tests
README.md
requirements.txt
```

---

## 2. Check Python

```powershell
python --version
```

**Expected:**

```text
Python 3.10.x
```

The tested version is:

```text
Python 3.10.11
```

---

## 3. Create the virtual environment

```powershell
python -m venv .venv
```

**What it does:** Creates an isolated Python environment for this project.

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Expected:** The prompt starts with:

```text
(.venv)
```

Verify:

```powershell
python -c "import sys; print(sys.executable)"
```

**Expected:** The path ends with:

```text
.venv\Scripts\python.exe
```

---

## 4. Install shared dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Then:

```powershell
python -m pip install -r requirements.txt
```

**What it does:** Installs the packages shared by the whole team.

Check the file:

```powershell
Get-Content requirements.txt
```

It should already contain:

```text
pygame-ce>=2.5.7
```

Do not edit it.

---

## 5. Verify Pygame

```powershell
python -m pip show pygame
```

**Expected:**

```text
WARNING: Package(s) not found: pygame
```

Then:

```powershell
python -m pip show pygame-ce
```

**Expected:**

```text
Name: pygame-ce
Version: 2.5.7
Required-by: mpe2
```

Test the import:

```powershell
python -c "import pygame; print(pygame.version.ver)"
```

**Expected:**

```text
2.5.7
```

---

## 6. Install PyTorch for your computer

Choose only one option.

### Option A — NVIDIA GPU

First check:

```powershell
nvidia-smi
```

If an NVIDIA GPU appears, install:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Expected:**

```text
PyTorch: 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce ...
```

Run a GPU test:

```powershell
python -c "import torch; x=torch.randn(2000,2000,device='cuda'); y=x@x; print('Tensor device:', y.device); print('GPU test: PASS')"
```

**Expected:**

```text
Tensor device: cuda:0
GPU test: PASS
```

---

### Option B — Windows computer without NVIDIA

Install the CPU build:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device: CPU')"
```

**Expected:**

```text
PyTorch: 2.11.0+cpu
CUDA available: False
Device: CPU
```

This is correct for a CPU-only computer.

---

### Option C — Apple Silicon Mac

Install:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

Verify:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available()); print('Device:', 'mps' if torch.backends.mps.is_available() else 'cpu')"
```

**Expected on a supported Mac:**

```text
MPS available: True
Device: mps
```

---

## 7. Test shared imports

```powershell
python -c "import numpy, gymnasium, pettingzoo, mpe2, pandas, matplotlib, yaml, pytest, tqdm; print('Base imports: PASS')"
```

**Expected:**

```text
Base imports: PASS
```

---

## 8. Run the MPE2 smoke test

```powershell
python scripts\smoke_test_mpe.py
```

**Expected:** The output ends with:

```text
MPE2 Parallel API smoke test: PASS
```

---

## 9. Run the full setup check

```powershell
python scripts\check_setup.py
```

**Expected:**

```text
Setup check: PASS
```

On an NVIDIA computer, it should also show:

```text
CUDA available: True
CUDA tensor smoke test: PASS
```

On a CPU-only computer, `CUDA available: False` is normal.

---

## 10. Run the tests

```powershell
pytest
```

**Expected:**

```text
2 passed
```

---

## 11. Send your setup result to the team

Report:

```text
Operating system:
Python version:
PyTorch version:
Backend: CUDA / MPS / CPU
GPU model:
MPE2 smoke test: PASS / FAIL
Setup check: PASS / FAIL
pytest result:
Errors:
```

---

## Daily startup

```powershell
cd "C:\path\to\cs5180_maddpg_starter"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Expected:** The prompt begins with:

```text
(.venv)
```

When finished:

```powershell
deactivate
```

---

## Team rule

Teammates may use different hardware:

```text
NVIDIA GPU -> CUDA
Apple Silicon -> MPS
Other computers -> CPU
```

Everyone can write code, run tests, debug, and confirm training starts.

All official final seed runs will use the leader’s RTX 4050 CUDA machine so the final comparison uses one consistent hardware setup.
