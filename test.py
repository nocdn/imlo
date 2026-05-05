from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from coursework_data import load_pet_split
from model import create_pet_breed_model
from transforms import build_evaluation_image_transforms

BATCH_SIZE = 32
MODEL_FILE_NAME = "model.pth"


def choose_testing_device():
    # use a gpu if one is available
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def test_model_on_dataset(dataloader, model, loss_function, device):
    model.eval()

    # keep totals so the final test accuracy can be printed
    total_loss = 0
    total_correct_predictions = 0
    total_images = 0

    # gradients are not needed when testing the model
    with torch.no_grad():
        for images, labels in dataloader:
            # move the batch to the same device as the model
            images = images.to(device)
            labels = labels.to(device)

            # get the model predictions and compare them with the real labels
            prediction_scores = model(images)
            loss = loss_function(prediction_scores, labels)

            batch_size = images.shape[0]
            # choose the class with the highest score
            predicted_labels = prediction_scores.argmax(1)

            total_loss = total_loss + loss.item() * batch_size
            total_correct_predictions = (
                total_correct_predictions + (predicted_labels == labels).sum().item()
            )
            total_images = total_images + batch_size

    average_loss = total_loss / total_images
    test_accuracy = total_correct_predictions / total_images

    return average_loss, test_accuracy


def main():
    if not Path(MODEL_FILE_NAME).exists():
        raise FileNotFoundError("Run train.py first so that model.pth exists.")

    device = choose_testing_device()
    print("Using device:", device)

    test_dataset = load_pet_split(
        "test",
        download=True,
        transform=build_evaluation_image_transforms(),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = create_pet_breed_model()
    model = model.to(device)

    # load the saved weights from train.py
    saved_weights = torch.load(MODEL_FILE_NAME, map_location=device, weights_only=True)
    model.load_state_dict(saved_weights)

    loss_function = nn.CrossEntropyLoss()

    average_loss, test_accuracy = test_model_on_dataset(
        test_dataloader,
        model,
        loss_function,
        device,
    )

    print("Average test loss:", round(average_loss, 4))
    print("Test accuracy:", round(test_accuracy * 100, 2), "%")


if __name__ == "__main__":
    main()
