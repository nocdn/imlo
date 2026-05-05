from torchvision import transforms


IMAGE_SIZE = 160
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def build_training_image_transforms():
    """make training images ready for the model"""
    return transforms.Compose(
        [
            transforms.Resize((200, 200)),
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.65, 1.0),
                ratio=(0.8, 1.25),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def build_evaluation_image_transforms():
    """make evaluation images ready for the model"""
    return transforms.Compose(
        [
            transforms.Resize((180, 180)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
