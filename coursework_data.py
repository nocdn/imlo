from pathlib import Path
import os

from torchvision.datasets import OxfordIIITPet

DATA_DIR = Path(os.environ.get("IMLO_DATA_DIR", "data"))
USE_TRIMAP_INPUT = os.environ.get("IMLO_USE_TRIMAP_INPUT", "1") != "0"


class PetBreedDatasetWithTrimap:
    def __init__(self, split, download=True, transform=None):
        self.dataset = OxfordIIITPet(
            root=DATA_DIR,
            split=split,
            target_types=("category", "segmentation"),
            download=download,
        )
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.root = self.dataset.root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        label, trimap = target
        if self.transform is not None:
            image = self.transform(image, trimap)
        return image, label


def load_pet_split(split, download=True, transform=None):
    """load one oxford-iiit pet dataset split"""
    if split not in {"trainval", "test"}:
        raise ValueError("split must be either 'trainval' or 'test'")

    if USE_TRIMAP_INPUT:
        return PetBreedDatasetWithTrimap(
            split=split,
            download=download,
            transform=transform,
        )

    return OxfordIIITPet(
        root=DATA_DIR,
        split=split,
        target_types="category",
        download=download,
        transform=transform,
    )


def download_pet_dataset():
    """download both dataset splits"""
    trainval_dataset = load_pet_split("trainval", download=True)
    test_dataset = load_pet_split("test", download=True)
    return trainval_dataset, test_dataset
