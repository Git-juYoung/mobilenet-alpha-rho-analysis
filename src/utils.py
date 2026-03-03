import torch
import torch.nn as nn
import torch.optim as optim


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_criterion():
    return nn.CrossEntropyLoss()


def build_optimizer(model, lr, weight_decay=0.0):
    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )