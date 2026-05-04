from torchvision import transforms


IMAGE_SIZE = 160
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def build_train_transforms():
    """Basic augmentation for training images."""
    return transforms.Compose(
        [
            transforms.Resize((180, 180)),
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def build_eval_transforms():
    """Deterministic preprocessing for validation/test images."""
    return transforms.Compose(
        [
            transforms.Resize((180, 180)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
