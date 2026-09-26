"""End-to-End Execution Pipeline for Business Entity Resolution.

Steps:
1. Normalization (run_normalize.py)
2. Candidate Blocking (block.py) & Blocking Recall Evaluation (eval_blocking.py)
3. Model Training & Threshold Tuning (tune.py)
4. Inference & Submission Output Generation (predict.py)
5. Output Validation (validate_submission.py)
"""
import os
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]
STUDENT_RESOURCE_DIR = REPO_ROOT / "student_resource"
VALIDATOR = STUDENT_RESOURCE_DIR / "utils" / "validate_submission.py"


def run_cmd(cmd: list[str]):
    print(f"\n[RUNNING]: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR.parent) + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd, check=True, env=env)
    return res.returncode


def ensure_dependencies():
    req_file = SRC_DIR.parent / "requirements.txt"
    if req_file.exists():
        print(f"Ensuring requirements from {req_file} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)], check=False)


def main():
    print("=" * 70)
    print("AMAZON ML CHALLENGE - BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    ensure_dependencies()

    # Step 1: Normalization
    print("\n--- STEP 1: Normalization ---")
    run_cmd([sys.executable, str(SRC_DIR / "normalize" / "run_normalize.py")])

    # Step 2: Blocking
    print("\n--- STEP 2: Candidate Blocking (Train & Test) ---")
    run_cmd([sys.executable, str(SRC_DIR / "blocking" / "block.py"), "train"])
    run_cmd([sys.executable, str(SRC_DIR / "blocking" / "block.py"), "test"])

    # Step 2b: Measure Blocking Recall
    print("\n--- STEP 2b: Blocking Recall Measurement ---")
    run_cmd([sys.executable, str(SRC_DIR / "blocking" / "eval_blocking.py")])

    # Step 3 & 4: Train LightGBM & Tune Thresholds
    print("\n--- STEP 3 & 4: Feature Engineering, Model Training & Decoding Tuning ---")
    run_cmd([sys.executable, str(SRC_DIR / "score" / "tune.py")])

    # Step 5: Inference & Output Generation
    print("\n--- STEP 5: Test Set Inference & Output Generation ---")
    run_cmd([sys.executable, str(SRC_DIR / "score" / "predict.py")])

    # Step 5b: Submission Validation
    print("\n--- STEP 5b: Submission Format Validation ---")
    dataset_dir = STUDENT_RESOURCE_DIR / "dataset" / "test"
    if not dataset_dir.exists():
        dataset_dir = Path("d:/6ab10eb3b23ba_student_resource/student_resource/dataset/test")

    run_cmd([
        sys.executable,
        str(VALIDATOR),
        "--matching",
        str(REPO_ROOT / "output" / "matching_results.tsv"),
        "--candidate",
        str(REPO_ROOT / "output" / "candidate_pairs.tsv"),
        "--test-dir",
        str(dataset_dir),
    ])

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
