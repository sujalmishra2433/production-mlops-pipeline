"""Bootstrap script to initialize local environment and folders."""

import sys
from pathlib import Path


def bootstrap() -> int:
    print("=" * 60)
    print("Initializing MLOps Platform Environment...")
    print("=" * 60)

    # Required directories
    dirs = [
        "data/raw",
        "data/processed",
        "data/reference",
        "reports",
        "mlruns",
    ]
    for d in dirs:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Directory ready: {d}")

    # Check python version
    print(f"  ✓ Python version: {sys.version.split()[0]}")

    import importlib.util
    missing = []
    for pkg in ["mlflow", "pandera", "sklearn"]:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        print(f"  ✗ Missing dependencies: {missing}")
        return 1
    print("  ✓ Core dependencies verified (MLflow, Pandera, Scikit-learn).")

    print("\nEnvironment initialization complete! Ready to run pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(bootstrap())
