from __future__ import annotations

import platform
import sys
from importlib import import_module


def version_of(module_name: str) -> str:
    module = import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    print("=== CS5180 setup check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    required = [
        "numpy",
        "gymnasium",
        "pettingzoo",
        "mpe2",
        "pygame",
        "matplotlib",
        "pandas",
        "yaml",
        "pytest",
        "tqdm",
    ]

    failed: list[str] = []
    for name in required:
        try:
            print(f"{name}: {version_of(name)}")
        except Exception as exc:
            failed.append(name)
            print(f"{name}: FAILED ({exc})")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA build: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x
            print(f"CUDA tensor smoke test: PASS ({tuple(y.shape)})")
        else:
            print("CUDA tensor smoke test: SKIPPED")
    except Exception as exc:
        failed.append("torch")
        print(f"torch: FAILED ({exc})")

    if failed:
        raise SystemExit(
            "\nSetup is incomplete. Missing or broken modules: " + ", ".join(failed)
        )

    print("\nSetup check: PASS")


if __name__ == "__main__":
    main()
