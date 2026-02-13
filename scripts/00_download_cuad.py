"""
00 – Download and extract the CUAD (Contract Understanding Atticus Dataset).
Run this script first (or notebooks will call it). Creates cuad/ and cuad/data/.

Usage (from project root):
    python scripts/00_download_cuad.py
"""

import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Tuple

CUAD_REPO_URL = "https://github.com/TheAtticusProject/cuad.git"


def get_project_root() -> Path:
    """Project root is the parent of the scripts/ folder."""
    return Path(__file__).resolve().parent.parent


def ensure_cuad_data(project_root: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Ensure the CUAD repo and data directory exist. Clone if needed; extract data.zip if needed.
    Returns (cuad_path, data_path).
    """
    root = project_root or get_project_root()
    cuad_path = root / "cuad"
    data_path = cuad_path / "data"

    if not cuad_path.exists():
        print("CUAD dataset not found. Cloning from GitHub...")
        subprocess.run(
            ["git", "clone", CUAD_REPO_URL, "cuad"],
            check=True,
            cwd=str(root),
        )
        print("Dataset cloned successfully!")
    else:
        print("CUAD dataset directory already exists.")

    if not data_path.exists() and (cuad_path / "data.zip").exists():
        print("\nExtracting data.zip...")
        with zipfile.ZipFile(cuad_path / "data.zip", "r") as zip_ref:
            zip_ref.extractall(cuad_path)
        print("Data extracted successfully!")

    if data_path.exists():
        print(f"\nDataset found at: {data_path.absolute()}")
        print("\nContents of data directory:")
        for item in sorted(data_path.iterdir()):
            if item.is_file():
                print(f"   {item.name}")
            elif item.is_dir():
                print(f"   {item.name}/")
    else:
        print(f"Data directory not found at {data_path}")
        print("Checking CUAD directory structure...")
        for item in sorted(cuad_path.iterdir()):
            prefix = "" if item.is_dir() else ""
            print(f"  {prefix}{item.name}")

    return cuad_path, data_path


if __name__ == "__main__":
    ensure_cuad_data()
