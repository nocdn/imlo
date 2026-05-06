import torch
from torch import nn
from torch.nn import functional as F

NUMBER_OF_CLASSES = 37
INPUT_IMAGE_SIZE = 128
INPUT_CHANNELS = 4


class ResidualBlock(nn.Module):
    """a small residual block with two 3x3 convolutions"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv_a = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm_a = nn.BatchNorm2d(out_channels)

        self.conv_b = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm_b = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.skip_projection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip_projection = nn.Identity()

    def forward(self, features):
        residual = self.skip_projection(features)
        out = F.relu(self.norm_a(self.conv_a(features)), inplace=True)
        out = self.norm_b(self.conv_b(out))
        out = F.relu(out + residual, inplace=True)
        return out


class PetBreedConvolutionalNetwork(nn.Module):
    """residual cnn for the oxford-iiit pet dataset"""

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage_one = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
        )
        self.stage_two = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )
        self.stage_three = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
        )
        self.stage_four = nn.Sequential(
            ResidualBlock(256, 384, stride=2),
            ResidualBlock(384, 384, stride=1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(384, NUMBER_OF_CLASSES)

        self._initialise_weights()

    def _initialise_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, images):
        features = self.stem(images)
        features = self.stage_one(features)
        features = self.stage_two(features)
        features = self.stage_three(features)
        features = self.stage_four(features)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        return self.classifier(features)


def create_pet_breed_model():
    return PetBreedConvolutionalNetwork()


def check_model_output_shape():
    model = create_pet_breed_model()
    example_images = torch.randn(4, INPUT_CHANNELS, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    output_scores = model(example_images)
    total_params = sum(p.numel() for p in model.parameters())
    print("Output shape:", output_scores.shape, "Params:", total_params)


if __name__ == "__main__":
    check_model_output_shape()
