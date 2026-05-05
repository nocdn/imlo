import os
from pathlib import Path

import modal

APP_NAME = "imlo-coursework"
VOLUME_NAME = "imlo-coursework"
VOLUME_MOUNT_PATH = Path("/mnt/imlo")
REMOTE_DATA_DIR = VOLUME_MOUNT_PATH / "data"
REMOTE_MODEL_FILE = VOLUME_MOUNT_PATH / "model.pth"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "pillow", "numpy")
    .add_local_python_source("coursework_data", "model", "transforms", "train", "test")
)


@app.function(
    image=image,
    gpu="A10",
    cpu=4.0,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def train_and_test_on_modal():
    os.environ["IMLO_DATA_DIR"] = str(REMOTE_DATA_DIR)
    os.environ["IMLO_MODEL_FILE"] = str(REMOTE_MODEL_FILE)
    os.environ["IMLO_NUM_WORKERS"] = "4"

    import torch

    import test
    import train

    print("remote device check:")
    print("  cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  gpu:", torch.cuda.get_device_name(0))

    if REMOTE_MODEL_FILE.exists():
        REMOTE_MODEL_FILE.unlink()

    train_results = train.main()
    volume.commit()

    test_results = test.main()
    volume.commit()

    return {
        "train": train_results,
        "test": test_results,
        "model_file": str(REMOTE_MODEL_FILE),
    }


@app.local_entrypoint()
def main():
    results = train_and_test_on_modal.remote()

    train_results = results["train"]
    test_results = results["test"]

    print("\nModal run complete")
    print(f"Model file: {results['model_file']}")
    print(f"Trainval accuracy: {train_results['trainval_accuracy'] * 100:.2f} %")
    if train_results["best_validation_accuracy"] is not None:
        print(f"Best validation epoch: {train_results['best_validation_epoch']}")
        print(
            f"Best validation accuracy: {train_results['best_validation_accuracy'] * 100:.2f} %"
        )
    print(f"Test accuracy: {test_results['test_accuracy'] * 100:.2f} %")
