from pathlib import Path
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from coursework_data import load_pet_split
from model import create_pet_breed_model
from transforms import build_evaluation_image_transforms

BATCH_SIZE = int(os.environ.get("IMLO_BATCH_SIZE", "64"))
MODEL_FILE_NAME = os.environ.get("IMLO_MODEL_FILE", "model.pth")
NUMBER_OF_DATA_WORKERS = int(os.environ.get("IMLO_NUM_WORKERS", "4"))


def choose_testing_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_test_set(dataloader, model, loss_fn, device):
    """evaluate the model on the official test images without augmentation."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            scores = model(images)
            loss = loss_fn(scores, labels)

            bsz = images.shape[0]
            total_loss += loss.item() * bsz
            total_correct += (scores.argmax(1) == labels).sum().item()
            total_images += bsz

    return total_loss / total_images, total_correct / total_images


def main():
    if not Path(MODEL_FILE_NAME).exists():
        raise FileNotFoundError("Run train.py first so that model.pth exists.")

    device = choose_testing_device()
    print("device:", device)

    test_dataset = load_pet_split(
        "test", download=True, transform=build_evaluation_image_transforms()
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUMBER_OF_DATA_WORKERS,
    )

    model = create_pet_breed_model().to(device)
    weights = torch.load(MODEL_FILE_NAME, map_location=device, weights_only=True)
    model.load_state_dict(weights)

    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    avg_loss, accuracy = evaluate_test_set(test_dataloader, model, loss_fn, device)
    elapsed = time.time() - t0

    print(f"Average test loss: {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f} %")
    print(f"Total testing time: {elapsed:.2f} seconds")

    return {
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "testing_seconds": elapsed,
        "model_file": MODEL_FILE_NAME,
    }


if __name__ == "__main__":
    main()
