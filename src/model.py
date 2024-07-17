import torch
import torch.nn as nn

class MyModel(nn.Module):
    """
    Custom CNN architecture.

    Args:
        num_classes (int): Number of classes in the classification task.
        dropout (float): Dropout probability to use in the model.

    Attributes:
        features (nn.Sequential): Sequential container for feature extraction layers.
        classifier (nn.Sequential): Sequential container for classification layers.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        # Define a CNN architecture
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 14 * 14, 128),  # Adjusted input size based on the architecture
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Feature extraction
        x = self.features(x)
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        # Classification
        x = self.classifier(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################

import pytest

@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=2)

def test_model_construction(data_loaders):
    """
    Test the construction of the MyModel class.

    Args:
        data_loaders (dict): Data loaders dictionary containing train and validation data loaders.
    """
    model = MyModel(num_classes=23, dropout=0.3)

    # Get a batch of data from the train data loader
    for images, labels in data_loaders["train"]:
        break

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
