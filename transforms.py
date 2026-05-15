import random

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


image_size = 192


def _trimap_to_pet_channel(trimap_tensor):
    return (trimap_tensor != 2).float()


def _image_with_trimap_channel(image, trimap):
    image_tensor = TF.to_tensor(image)
    trimap_tensor = TF.pil_to_tensor(trimap)
    pet_channel = _trimap_to_pet_channel(trimap_tensor)
    return torch.cat([image_tensor, pet_channel], dim=0)


class TrainingImageTransforms:
    def __init__(self):
        self.erase = transforms.RandomErasing(p=0.15, scale=(0.02, 0.12))

    def __call__(self, image, trimap=None):
        if trimap is None:
            image = transforms.Resize((224, 224))(image)
            image = transforms.RandomResizedCrop(
                image_size,
                scale=(0.78, 1.0),
                ratio=(0.85, 1.18),
            )(image)
            image = transforms.RandomHorizontalFlip(p=0.5)(image)
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(image)
            image = TF.to_tensor(image)
            return self.erase(image)

        image = TF.resize(image, (224, 224), interpolation=InterpolationMode.BILINEAR)
        trimap = TF.resize(trimap, (224, 224), interpolation=InterpolationMode.NEAREST)

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image,
            scale=(0.78, 1.0),
            ratio=(0.85, 1.18),
        )
        image = TF.resized_crop(image, i, j, h, w, (image_size, image_size), InterpolationMode.BILINEAR)
        trimap = TF.resized_crop(trimap, i, j, h, w, (image_size, image_size), InterpolationMode.NEAREST)

        if random.random() < 0.5:
            image = TF.hflip(image)
            trimap = TF.hflip(trimap)

        angle = random.uniform(-8, 8)
        translate = (
            int(random.uniform(-0.04, 0.04) * image_size),
            int(random.uniform(-0.04, 0.04) * image_size),
        )
        scale = random.uniform(0.92, 1.08)
        shear = random.uniform(-4, 4)
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        trimap = TF.affine(
            trimap,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=2,
        )

        image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(image)
        image_tensor = self.erase(TF.to_tensor(image))
        trimap_tensor = TF.pil_to_tensor(trimap)
        pet_channel = _trimap_to_pet_channel(trimap_tensor)
        return torch.cat([image_tensor, pet_channel], dim=0)


class EvaluationImageTransforms:
    def __call__(self, image, trimap=None):
        if trimap is None:
            image = TF.resize(image, (216, 216), interpolation=InterpolationMode.BILINEAR)
            image = TF.center_crop(image, image_size)
            return TF.to_tensor(image)

        image = TF.resize(image, (216, 216), interpolation=InterpolationMode.BILINEAR)
        trimap = TF.resize(trimap, (216, 216), interpolation=InterpolationMode.NEAREST)
        image = TF.center_crop(image, image_size)
        trimap = TF.center_crop(trimap, image_size)
        return _image_with_trimap_channel(image, trimap)


def build_training_image_transforms():
    """make training images ready for the model with moderate augmentation"""
    return TrainingImageTransforms()


def build_evaluation_image_transforms():
    """make evaluation images ready for the model"""
    return EvaluationImageTransforms()
