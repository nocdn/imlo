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

batch_size = 32
number_of_epochs = 30
learning_rate = 0.002
weight_decay = 5e-4
label_smoothing = 0.05
warmup_epochs = 3
mixup_alpha = 0.0
use_validation_split = False
validation_fraction = 0.1
number_of_data_workers = 4
model_file_name = os.environ.get("IMLO_MODEL_FILE", "model.pth")
ema_decay = 0.0


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


def create_ema_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def update_ema_state(model, ema_state):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if torch.is_floating_point(value):
                ema_state[name].mul_(ema_decay).add_(
                    value.detach(), alpha=1.0 - ema_decay
                )
            else:
                ema_state[name].copy_(value)


def train_one_epoch(
    dataloader,
    model,
    loss_fn,
    optimiser,
    device,
    epoch_num,
    total_epochs,
    ema_state=None,
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

        if mixup_alpha > 0:
            mixed_images, labels_a, labels_b, lam = mixup_batch(
                images, labels, mixup_alpha
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
        if ema_state is not None:
            update_ema_state(model, ema_state)

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
    n_val = int(n * validation_fraction)
    rng = torch.Generator().manual_seed(0)
    shuffled = torch.randperm(n, generator=rng).tolist()
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    return Subset(training_dataset, train_idx), Subset(evaluation_dataset, val_idx)


def build_lr_schedule(optimiser):
    """linear warmup for warmup_epochs, then cosine to 0 over the epochs that are left"""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimiser,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=number_of_epochs - warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
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

    if use_validation_split:
        training_dataset, validation_dataset = make_train_val_subsets(
            train_ds_train_tf, train_ds_eval_tf
        )
    else:
        training_dataset = train_ds_train_tf
        validation_dataset = None

    pin = device == "cuda"

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=number_of_data_workers,
        pin_memory=pin,
        persistent_workers=number_of_data_workers > 0,
    )
    validation_dataloader = None
    if use_validation_split:
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=number_of_data_workers,
            pin_memory=pin,
            persistent_workers=number_of_data_workers > 0,
        )
    full_trainval_dataloader = DataLoader(
        train_ds_eval_tf,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_data_workers,
        pin_memory=pin,
        persistent_workers=number_of_data_workers > 0,
    )

    print(
        "train images:",
        len(training_dataset),
        "val images:",
        len(validation_dataset) if use_validation_split else 0,
    )

    model = create_pet_breed_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("model params:", n_params)
    ema_state = create_ema_state(model) if ema_decay > 0 else None
    if ema_state is not None:
        print("ema decay:", ema_decay)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = build_lr_schedule(optimiser)

    best_val_acc = -1.0
    best_val_epoch = 0
    t_start = time.time()

    for epoch in range(number_of_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            training_dataloader,
            model,
            loss_fn,
            optimiser,
            device,
            epoch + 1,
            number_of_epochs,
            ema_state,
        )
        scheduler.step()
        secs = time.time() - t0

        if use_validation_split:
            val_loss, val_acc = evaluate(validation_dataloader, model, loss_fn, device)
            saved = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch + 1
                torch.save(model.state_dict(), model_file_name)
                saved = " *"
            print(
                f"E{epoch + 1:02d}/{number_of_epochs} "
                f"loss {train_loss:.3f} acc {train_acc * 100:.1f}% "
                f"| val loss {val_loss:.3f} acc {val_acc * 100:.1f}% "
                f"| {secs:.1f}s{saved}",
                flush=True,
            )
        else:
            print(
                f"E{epoch + 1:02d}/{number_of_epochs} "
                f"loss {train_loss:.3f} acc {train_acc * 100:.1f}% | {secs:.1f}s",
                flush=True,
            )

    total_secs = time.time() - t_start

    if use_validation_split:
        weights = torch.load(model_file_name, map_location=device, weights_only=True)
        model.load_state_dict(weights)
    elif ema_state is not None:
        model.load_state_dict(ema_state)
        torch.save(model.state_dict(), model_file_name)
    else:
        torch.save(model.state_dict(), model_file_name)

    final_loss, final_acc = evaluate(full_trainval_dataloader, model, loss_fn, device)
    print(f"final trainval loss: {final_loss:.4f}")
    print(f"final trainval accuracy: {final_acc * 100:.2f} %")
    if use_validation_split:
        print(f"best validation epoch: {best_val_epoch}")
        print(f"best validation accuracy: {best_val_acc * 100:.2f} %")
    print(f"total training time: {total_secs:.1f}s")
    print("saved model to", model_file_name)

    return {
        "trainval_loss": final_loss,
        "trainval_accuracy": final_acc,
        "best_validation_epoch": best_val_epoch if use_validation_split else None,
        "best_validation_accuracy": best_val_acc if use_validation_split else None,
        "training_seconds": total_secs,
        "model_file": model_file_name,
    }


if __name__ == "__main__":
    main()
