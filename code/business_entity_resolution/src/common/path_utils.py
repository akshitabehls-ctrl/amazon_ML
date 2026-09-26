"""Utility functions for locating dataset and repository paths consistently."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
OUTPUT_ROOT = REPO_ROOT / "output"

def get_dataset_root() -> Path:
    """Find dataset root directory, falling back to external extraction location if needed."""
    p1 = REPO_ROOT / "student_resource" / "dataset"
    if p1.exists():
        return p1
    p2 = Path("d:/6ab10eb3b23ba_student_resource/student_resource/dataset")
    if p2.exists():
        return p2
    return p1
