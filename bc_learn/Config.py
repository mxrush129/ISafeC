import torch

from rl_train.Examples import Example


class Config:
    example: Example = None
    hidden_neurons = [10]
    activation = ['SKIP']
    batch_size = 100
    lr = 0.01
    loop = 100

    mul_hidden = []
    mul_activation = []
    margin = 1
    loss_weight = [1, 1, 1]
    R_b = 0.3
    device = torch.device('cpu')  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert len(self.hidden_neurons) == len(self.activation)
