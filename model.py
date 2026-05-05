import torch
from torch import nn


NUMBER_OF_CLASSES = 37
INPUT_IMAGE_SIZE = 160


class PetBreedConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # these layers learn image features from the input pixels
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # these layers turn the image features into class scores
        self.classifier_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, NUMBER_OF_CLASSES),
        )

    def forward(self, images):
        images = self.convolution_layers(images)
        output_scores = self.classifier_layers(images)
        return output_scores


def create_pet_breed_model():
    return PetBreedConvolutionalNetwork()


def check_model_output_shape():
    model = create_pet_breed_model()
    # this checks that the model gives one score for each class
    example_images = torch.randn(4, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    output_scores = model(example_images)
    print("Example output shape:", output_scores.shape)


if __name__ == "__main__":
    check_model_output_shape()
