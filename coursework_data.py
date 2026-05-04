from pathlib import Path

from torchvision.datasets import OxfordIIITPet


DATA_DIR = Path("data")


def load_pet_split(split, download=True, transform=None):
    """Load an official Oxford-IIIT Pet dataset split."""
    if split not in {"trainval", "test"}:
        raise ValueError("split must be either 'trainval' or 'test'")

    return OxfordIIITPet(
        root=DATA_DIR,
        split=split,
        target_types="category",
        download=download,
        transform=transform,
    )


def download_pet_dataset():
    """Download and return the official trainval and test splits."""
    trainval_dataset = load_pet_split("trainval", download=True)
    test_dataset = load_pet_split("test", download=True)
    return trainval_dataset, test_dataset
