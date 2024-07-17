import os

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as T
from .helpers import get_data_location


class Predictor(nn.Module):
    """
    A class used to encapsulate a model and apply necessary preprocessing
    transformations for prediction.
    
    Attributes
    ----------
    model : nn.Module
        The neural network model used for prediction.
    class_names : list
        A list of class names corresponding to the model's output classes.
    transforms : nn.Sequential
        A sequential container of image transformations.
    """

    def __init__(self, model, class_names, mean, std):
        """
        Initializes the Predictor with a model, class names, and normalization parameters.

        Parameters
        ----------
        model : nn.Module
            The neural network model used for prediction.
        class_names : list
            A list of class names corresponding to the model's output classes.
        mean : torch.Tensor
            The mean values used for normalization.
        std : torch.Tensor
            The standard deviation values used for normalization.
        """
        super().__init__()

        self.model = model.eval()
        self.class_names = class_names

        # Using nn.Sequential to ensure compatibility with torchscript
        self.transforms = nn.Sequential(
            T.Resize([256, ]),  # Using single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the predictor. Applies transformations, runs the model, and
        returns softmax probabilities.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The softmax probabilities for each class.
        """
        with torch.no_grad():
            # Apply transformations
            x = self.transforms(x)
            # Get the logits
            x = self.model(x)
            # Apply softmax across dim=1
            x = F.softmax(x, dim=1)

            return x


def predictor_test(test_dataloader, model_reloaded):
    """
    Tests the predictor on the test dataset.

    Parameters
    ----------
    test_dataloader : DataLoader
        The dataloader for the test dataset.
    model_reloaded : nn.Module
        The reloaded model for testing.

    Returns
    -------
    tuple
        The ground truth labels and the predicted labels.
    """
    folder = get_data_location()
    test_data = datasets.ImageFolder(os.path.join(folder, "test"), transform=T.ToTensor())

    pred = []
    truth = []
    for x in tqdm(test_data, total=len(test_dataloader.dataset), leave=True, ncols=80):
        softmax = model_reloaded(x[0].unsqueeze(dim=0))

        idx = softmax.squeeze().argmax()

        pred.append(int(x[1]))
        truth.append(int(idx))

    pred = np.array(pred)
    truth = np.array(truth)

    print(f"Accuracy: {(pred == truth).sum() / pred.shape[0]}")

    return truth, pred


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    """
    Fixture to provide data loaders for tests.

    Returns
    -------
    dict
        A dictionary containing train and test data loaders.
    """
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    """
    Tests the construction of the model and the Predictor class.

    Parameters
    ----------
    data_loaders : dict
        A dictionary containing train and test data loaders.
    """
    from .model import MyModel
    from .helpers import compute_mean_and_std

    mean, std = compute_mean_and_std()

    model = MyModel(num_classes=3, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)  # Use the next() function here

    predictor = Predictor(model, class_names=['a', 'b', 'c'], mean=mean, std=std)

    out = predictor(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 3]
    ), f"Expected an output tensor of size (2, 3), got {out.shape}"

    assert torch.isclose(
        out[0].sum(),
        torch.Tensor([1]).squeeze()
    ), "The output of the .forward method should be a softmax vector with sum = 1"
