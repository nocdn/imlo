from torchvision import transforms


IMAGE_SIZE = 128


def build_training_image_transforms():
    """make training images ready for the model with moderate augmentation"""
    return transforms.Compose(
        [
            transforms.Resize((148, 148)),
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.7, 1.0),
                ratio=(0.85, 1.18),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    )


def build_evaluation_image_transforms():
    """make evaluation images ready for the model"""
    return transforms.Compose(
        [
            transforms.Resize((146, 146)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
