import torch.nn as nn


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x ** 2


def get_activation(name: str):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'square':
        return Square()
    else:
        raise ValueError('Unknown activation!')
