import torch.nn as nn


class TinyConvNet(nn.Module):
    """
    A simple convolutional neural network for image classification.
    """
    def __init__(self, input_channels=3, num_classes=2):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # HxW → HxW
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → H/2xW/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # H/2xW/2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → H/4xW/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # H/4xW/4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # always → 4x4 output
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),         # → 64*4*4 = 1024
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x
