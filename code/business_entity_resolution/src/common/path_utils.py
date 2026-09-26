"""Utility functions for locating dataset and repository paths consistently."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
OUTPUT_ROOT = REPO_ROOT / "output"

def get_dataset_root() -> Path:
    """Find dataset root directory."""
    return REPO_ROOT / "student_resource" / "dataset"

