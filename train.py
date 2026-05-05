import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from coursework_data import load_pet_split
from model import create_pet_breed_model
from transforms import build_evaluation_image_transforms, build_training_image_transforms

BATCH_SIZE = 32
NUMBER_OF_EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_FILE_NAME = "model.pth"


def choose_training_device():
    # use a gpu if one is available
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def train_model_for_one_epoch(dataloader, model, loss_function, optimiser, device):
    model.train()

    # keep totals so the epoch accuracy can be printed
    total_loss = 0
    total_correct_predictions = 0
    total_images = 0

    for batch_number, (images, labels) in enumerate(dataloader):
        # move the batch to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # get the model predictions and compare them with the real labels
        prediction_scores = model(images)
        loss = loss_function(prediction_scores, labels)

        # reset old gradients before calculating new ones
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        batch_size = images.shape[0]
        # choose the class with the highest score
        predicted_labels = prediction_scores.argmax(1)

        total_loss = total_loss + loss.item() * batch_size
        total_correct_predictions = (
            total_correct_predictions + (predicted_labels == labels).sum().item()
        )
        total_images = total_images + batch_size

        if batch_number % 20 == 0:
            print("Batch", batch_number, "loss:", round(loss.item(), 4))

    average_loss = total_loss / total_images
    training_accuracy = total_correct_predictions / total_images

    return average_loss, training_accuracy


def evaluate_model_on_trainval_split(dataloader, model, loss_function, device):
    model.eval()

    # keep totals so the final trainval accuracy can be printed
    total_loss = 0
    total_correct_predictions = 0
    total_images = 0

    # gradients are not needed when checking accuracy
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
    trainval_accuracy = total_correct_predictions / total_images

    return average_loss, trainval_accuracy


def main():
    # use the same random seed each time this script starts
    torch.manual_seed(0)

    device = choose_training_device()
    print("Using device:", device)

    training_dataset = load_pet_split(
        "trainval",
        download=True,
        transform=build_training_image_transforms(),
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    trainval_evaluation_dataset = load_pet_split(
        "trainval",
        download=True,
        transform=build_evaluation_image_transforms(),
    )

    trainval_evaluation_dataloader = DataLoader(
        trainval_evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = create_pet_breed_model()
    model = model.to(device)

    # cross entropy is used for multi-class classification
    loss_function = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch_number in range(NUMBER_OF_EPOCHS):
        print("Epoch", epoch_number + 1, "of", NUMBER_OF_EPOCHS)

        # record how long this epoch takes
        epoch_start_time = time.time()

        average_loss, training_accuracy = train_model_for_one_epoch(
            training_dataloader,
            model,
            loss_function,
            optimiser,
            device,
        )

        epoch_end_time = time.time()
        epoch_duration_seconds = epoch_end_time - epoch_start_time

        print("Average training loss:", round(average_loss, 4))
        print("Training accuracy:", round(training_accuracy * 100, 2), "%")
        print("Epoch time:", round(epoch_duration_seconds, 2), "seconds")

    final_trainval_loss, final_trainval_accuracy = evaluate_model_on_trainval_split(
        trainval_evaluation_dataloader,
        model,
        loss_function,
        device,
    )

    print("Final trainval loss:", round(final_trainval_loss, 4))
    print("Final trainval accuracy:", round(final_trainval_accuracy * 100, 2), "%")

    # save only the learnt weights, not the whole model object
    torch.save(model.state_dict(), MODEL_FILE_NAME)
    print("Saved model to", MODEL_FILE_NAME)


if __name__ == "__main__":
    main()
