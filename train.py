import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from coursework_data import load_pet_split
from model import create_pet_breed_model
from transforms import (
    build_evaluation_image_transforms,
    build_training_image_transforms,
)

BATCH_SIZE = int(os.environ.get("IMLO_BATCH_SIZE", "64"))
NUMBER_OF_EPOCHS = int(os.environ.get("IMLO_EPOCHS", "30"))
LEARNING_RATE = float(os.environ.get("IMLO_LEARNING_RATE", "0.003"))
WEIGHT_DECAY = float(os.environ.get("IMLO_WEIGHT_DECAY", "5e-4"))
LABEL_SMOOTHING = float(os.environ.get("IMLO_LABEL_SMOOTHING", "0.05"))
WARMUP_EPOCHS = int(os.environ.get("IMLO_WARMUP_EPOCHS", "3"))
MIXUP_ALPHA = float(os.environ.get("IMLO_MIXUP_ALPHA", "0.0"))
USE_VALIDATION_SPLIT = os.environ.get("IMLO_USE_VALIDATION_SPLIT", "1") != "0"
VALIDATION_FRACTION = float(os.environ.get("IMLO_VALIDATION_FRACTION", "0.1"))
NUMBER_OF_DATA_WORKERS = int(os.environ.get("IMLO_NUM_WORKERS", "4"))
MODEL_FILE_NAME = os.environ.get("IMLO_MODEL_FILE", "model.pth")


def choose_training_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mixup_batch(images, labels, alpha):
    """combine each image with a randomly chosen partner from the same batch"""
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(images.shape[0], device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    return mixed_images, labels, labels[permutation], lam


def train_one_epoch(
    dataloader, model, loss_fn, optimiser, device, epoch_num, total_epochs
):
    model.train()
    total_loss = 0
    total_correct = 0
    total_images = 0
    n_batches = len(dataloader)
    heartbeat_at = max(1, n_batches // 2)

    for batch_index, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if MIXUP_ALPHA > 0:
            mixed_images, labels_a, labels_b, lam = mixup_batch(
                images, labels, MIXUP_ALPHA
            )
            scores = model(mixed_images)
            loss = lam * loss_fn(scores, labels_a) + (1.0 - lam) * loss_fn(
                scores, labels_b
            )
            # for accuracy reporting, score against the dominant label of each pair
            dominant_labels = labels_a if lam >= 0.5 else labels_b
            correct = (scores.argmax(1) == dominant_labels).sum().item()
        else:
            scores = model(images)
            loss = loss_fn(scores, labels)
            correct = (scores.argmax(1) == labels).sum().item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        bsz = images.shape[0]
        total_loss += loss.item() * bsz
        total_correct += correct
        total_images += bsz

        if batch_index + 1 == heartbeat_at:
            print(
                f"  e{epoch_num}/{total_epochs} b{batch_index + 1}/{n_batches} "
                f"acc {total_correct / total_images * 100:.1f}%",
                flush=True,
            )

    return total_loss / total_images, total_correct / total_images


def evaluate(dataloader, model, loss_fn, device):
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


def make_train_val_subsets(training_dataset, evaluation_dataset):
    n = len(training_dataset)
    n_val = int(n * VALIDATION_FRACTION)
    rng = torch.Generator().manual_seed(0)
    shuffled = torch.randperm(n, generator=rng).tolist()
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    return Subset(training_dataset, train_idx), Subset(evaluation_dataset, val_idx)


def build_lr_schedule(optimiser):
    """linear warmup for WARMUP_EPOCHS, then cosine to 0 over the epochs that are left"""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimiser,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=NUMBER_OF_EPOCHS - WARMUP_EPOCHS,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[warmup, cosine],
        milestones=[WARMUP_EPOCHS],
    )


def main():
    torch.manual_seed(0)

    device = choose_training_device()
    print("device:", device)

    train_ds_train_tf = load_pet_split(
        "trainval", download=True, transform=build_training_image_transforms()
    )
    train_ds_eval_tf = load_pet_split(
        "trainval", download=True, transform=build_evaluation_image_transforms()
    )

    if USE_VALIDATION_SPLIT:
        training_dataset, validation_dataset = make_train_val_subsets(
            train_ds_train_tf, train_ds_eval_tf
        )
    else:
        training_dataset = train_ds_train_tf
        validation_dataset = None

    pin = device == "cuda"

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUMBER_OF_DATA_WORKERS,
        pin_memory=pin,
        persistent_workers=NUMBER_OF_DATA_WORKERS > 0,
    )
    validation_dataloader = None
    if USE_VALIDATION_SPLIT:
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUMBER_OF_DATA_WORKERS,
            pin_memory=pin,
            persistent_workers=NUMBER_OF_DATA_WORKERS > 0,
        )
    full_trainval_dataloader = DataLoader(
        train_ds_eval_tf,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUMBER_OF_DATA_WORKERS,
        pin_memory=pin,
        persistent_workers=NUMBER_OF_DATA_WORKERS > 0,
    )

    print(
        "train images:",
        len(training_dataset),
        "val images:",
        len(validation_dataset) if USE_VALIDATION_SPLIT else 0,
    )

    model = create_pet_breed_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("model params:", n_params)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = build_lr_schedule(optimiser)

    best_val_acc = -1.0
    best_val_epoch = 0
    t_start = time.time()

    for epoch in range(NUMBER_OF_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            training_dataloader,
            model,
            loss_fn,
            optimiser,
            device,
            epoch + 1,
            NUMBER_OF_EPOCHS,
        )
        scheduler.step()
        secs = time.time() - t0

        if USE_VALIDATION_SPLIT:
            val_loss, val_acc = evaluate(validation_dataloader, model, loss_fn, device)
            saved = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch + 1
                torch.save(model.state_dict(), MODEL_FILE_NAME)
                saved = " *"
            print(
                f"E{epoch + 1:02d}/{NUMBER_OF_EPOCHS} "
                f"loss {train_loss:.3f} acc {train_acc * 100:.1f}% "
                f"| val loss {val_loss:.3f} acc {val_acc * 100:.1f}% "
                f"| {secs:.1f}s{saved}",
                flush=True,
            )
        else:
            print(
                f"E{epoch + 1:02d}/{NUMBER_OF_EPOCHS} "
                f"loss {train_loss:.3f} acc {train_acc * 100:.1f}% | {secs:.1f}s",
                flush=True,
            )

    total_secs = time.time() - t_start

    if USE_VALIDATION_SPLIT:
        weights = torch.load(MODEL_FILE_NAME, map_location=device, weights_only=True)
        model.load_state_dict(weights)
    else:
        torch.save(model.state_dict(), MODEL_FILE_NAME)

    final_loss, final_acc = evaluate(full_trainval_dataloader, model, loss_fn, device)
    print(f"Final trainval loss: {final_loss:.4f}")
    print(f"Final trainval accuracy: {final_acc * 100:.2f} %")
    if USE_VALIDATION_SPLIT:
        print(f"Best validation epoch: {best_val_epoch}")
        print(f"Best validation accuracy: {best_val_acc * 100:.2f} %")
    print(f"Total training time: {total_secs:.1f}s")
    print("Saved model to", MODEL_FILE_NAME)

    return {
        "trainval_loss": final_loss,
        "trainval_accuracy": final_acc,
        "best_validation_epoch": best_val_epoch if USE_VALIDATION_SPLIT else None,
        "best_validation_accuracy": best_val_acc if USE_VALIDATION_SPLIT else None,
        "training_seconds": total_secs,
        "model_file": MODEL_FILE_NAME,
    }


if __name__ == "__main__":
    main()
