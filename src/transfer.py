import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

def get_model_transfer_learning(model_name="resnet18", n_classes=50):
    """
    Loads a pre-trained model and modifies it for transfer learning by freezing
    its parameters and adjusting the final fully connected layer.

    Parameters
    ----------
    model_name : str, optional
        The name of the model architecture to load (default is "resnet18").
    n_classes : int, optional
        The number of output classes for the final fully connected layer (default is 50).

    Returns
    -------
    nn.Module
        The modified model ready for transfer learning.
    
    Raises
    ------
    ValueError
        If the specified model_name is not available in torchvision.models.
    """
    # Get the requested architecture
    if hasattr(models, model_name):
        # Get the appropriate weights
        model_cls = getattr(models, model_name)
        weights = models.get_model_weights(model_cls)
        model_transfer = model_cls(weights=weights.DEFAULT)
    else:
        # Get the major and minor version of torchvision
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model to prevent their updates during training
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer to match the number of output classes
    # Get the number of features from the last layer of the pre-trained model
    num_ftrs = model_transfer.fc.in_features

    # Create a new fully connected layer with the appropriate number of input features and output classes
    model_transfer.fc = nn.Linear(num_ftrs, n_classes)

    return model_transfer


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

def test_get_model_transfer_learning(data_loaders):
    """
    Tests the get_model_transfer_learning function to ensure it correctly modifies
    a pre-trained model for transfer learning.

    Parameters
    ----------
    data_loaders : dict
        A dictionary containing train and test data loaders.
    """
    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)  # Use the next() function to get a batch of data

    out = model(images)

    # Check if the output is a tensor of the expected shape
    assert isinstance(out, torch.Tensor), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    # Verify the shape of the output tensor
    assert out.shape == torch.Size([2, 23]), f"Expected an output tensor of size (2, 23), got {out.shape}"
