import modal

app = modal.App("amazon-ml-entity-resolution")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
)

dataset = modal.Volume.from_name("amazon-ml-dataset")


@app.function(
    image=image,
    cpu=8,
    memory=64_000,
    timeout=60 * 60 * 6,
    volumes={
        "/mnt/dataset": dataset,
    },
)
def run():
    import subprocess
    import os
    import shutil

    repo = "/root/amazon_ML"

    # Clone the exact competition branch
    subprocess.run(
        [
            "git",
            "clone",
            "-b",
            "improve-entity-resolution",
            "https://github.com/akshitabehls-ctrl/amazon_ML.git",
            repo,
        ],
        check=True,
    )

    os.chdir(repo)

    # Install project dependencies
    subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "-r",
            "code/business_entity_resolution/requirements.txt",
        ],
        check=True,
    )

    # Put the private dataset where the existing pipeline expects it
    target = os.path.join(repo, "student_resource", "dataset")

    if os.path.exists(target):
        shutil.rmtree(target)

    shutil.copytree("/mnt/dataset", target)

    print("Dataset copied to:", target)

    # Run the complete pipeline
    subprocess.run(
        [
            "python",
            "code/business_entity_resolution/src/run_pipeline.py",
        ],
        check=True,
    )