import os
from pathlib import Path

import modal

app_name = "imlo-coursework"
volume_name = "imlo-coursework"
volume_mount_path = Path("/mnt/imlo")
remote_data_dir = volume_mount_path / "data"
remote_model_file = volume_mount_path / "model.pth"

app = modal.App(app_name)
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

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
    volumes={volume_mount_path: volume},
)
def train_and_test_on_modal():
    os.environ["IMLO_DATA_DIR"] = str(remote_data_dir)
    os.environ["IMLO_MODEL_FILE"] = str(remote_model_file)

    import torch

    import test
    import train

    print("remote device check:")
    print("  cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  gpu:", torch.cuda.get_device_name(0))

    if remote_model_file.exists():
        remote_model_file.unlink()

    train_results = train.main()
    volume.commit()

    test_results = test.main()
    volume.commit()

    return {
        "train": train_results,
        "test": test_results,
        "model_file": str(remote_model_file),
    }


@app.local_entrypoint()
def main():
    results = train_and_test_on_modal.remote()

    train_results = results["train"]
    test_results = results["test"]

    print("\nmodal run complete")
    print(f"model file: {results['model_file']}")
    print(f"trainval accuracy: {train_results['trainval_accuracy'] * 100:.2f} %")
    if train_results["best_validation_accuracy"] is not None:
        print(f"best validation epoch: {train_results['best_validation_epoch']}")
        print(
            f"best validation accuracy: {train_results['best_validation_accuracy'] * 100:.2f} %"
        )
    print(f"test accuracy: {test_results['test_accuracy'] * 100:.2f} %")
