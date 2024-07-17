import torch
import torch.nn as nn
import torch.optim

def get_loss():
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True.
    """
    loss = nn.CrossEntropyLoss()
    return loss

def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
):
    """
    Returns an optimizer instance.

    :param model: the model to optimize
    :param optimizer: one of 'SGD' or 'Adam'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    """
    if optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt

######################################################################################
#                                     TESTS
######################################################################################

import pytest

@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)

def test_get_loss():
    """
    Test get_loss function
    """
    loss = get_loss()
    assert isinstance(loss, nn.CrossEntropyLoss), f"Expected cross entropy loss, found {type(loss)}"

def test_get_optimizer_type(fake_model):
    """
    Test get_optimizer function for returning correct optimizer type
    """
    opt = get_optimizer(fake_model)
    assert isinstance(opt, torch.optim.SGD), f"Expected SGD optimizer, got {type(opt)}"

def test_get_optimizer_is_linked_with_model(fake_model):
    """
    Test get_optimizer function to ensure it links optimizer with model correctly
    """
    opt = get_optimizer(fake_model)
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])

def test_get_optimizer_returns_adam(fake_model):
    """
    Test get_optimizer function to ensure it returns Adam optimizer
    """
    opt = get_optimizer(fake_model, optimizer="adam")
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
    assert isinstance(opt, torch.optim.Adam), f"Expected Adam optimizer, got {type(opt)}"

def test_get_optimizer_sets_learning_rate(fake_model):
    """
    Test get_optimizer function to ensure it sets learning rate correctly
    """
    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123)
    assert opt.param_groups[0]["lr"] == 0.123, "Learning rate not set correctly"

def test_get_optimizer_sets_momentum(fake_model):
    """
    Test get_optimizer function to ensure it sets momentum correctly
    """
    opt = get_optimizer(fake_model, optimizer="SGD", momentum=0.123)
    assert opt.param_groups[0]["momentum"] == 0.123, "Momentum not set correctly"

def test_get_optimizer_sets_weight_decay(fake_model):
    """
    Test get_optimizer function to ensure it sets weight decay correctly
    """
    opt = get_optimizer(fake_model, optimizer="SGD", weight_decay=0.123)
    assert opt.param_groups[0]["weight_decay"] == 0.123, "Weight decay not set correctly"
