# Stage 1 Run Guide

This guide gives the commands needed to verify the completed Stage 1 implementation. It checks the repository setup, selected compute device, automated tests, MADDPG smoke run, independent-DDPG smoke run, TensorBoard logs, and saved checkpoints.

The Stage 1 smoke runs only confirm that the training pipeline works end to end. They are not final experiments and should not be used to decide which algorithm performs better.

## 1. Update the repository

```powershell
git switch main
git pull origin main
```

## 2. Activate the virtual environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 3. Check the selected device

```powershell
python -c "from src.common.device import resolve_device; print('Selected device:', resolve_device('auto'))"
```

Expected on an NVIDIA GPU machine:

```text
Selected device: cuda
```

A teammate without an NVIDIA GPU may see `cpu` or `mps`.

## 4. Run all tests

```powershell
python -m pytest
```

Expected result:

```text
94 passed
```

## 5. Run the MADDPG smoke test

```powershell
python -m src.experiments.train_simple_spread `
  --config .\configs\smoke_maddpg.yaml
```

Expected summary:

```text
Algorithm: maddpg
Episodes: 6
Environment steps: 48
Updates: 41
```

## 6. Run the independent-DDPG smoke test

```powershell
python -m src.experiments.train_simple_spread `
  --config .\configs\smoke_independent_ddpg.yaml
```

Expected summary:

```text
Algorithm: independent_ddpg
Episodes: 6
Environment steps: 48
Updates: 41
```

## 7. Check the generated files

```powershell
Get-ChildItem .\runs -Recurse
Get-ChildItem .\checkpoints
```

Expected checkpoint files:

```text
checkpoints\smoke_maddpg.pt
checkpoints\smoke_independent_ddpg.pt
```

## Conclusion

Stage 1 is successfully verified when all tests pass, both smoke runs complete with optimizer updates, and the TensorBoard logs and checkpoint files are created.

After completing these checks, the repository is ready for Stage 2: longer Simple Spread training, evaluation across multiple seeds, learning-curve generation, and development of the custom cooperative-transport environment.
