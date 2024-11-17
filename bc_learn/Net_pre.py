import numpy as np
import sympy as sp
import torch
import torch.nn as nn
from torch.func import jacrev

from bc_learn.Activation import get_activation
from bc_learn.Config import Config


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x ** 2


class Net(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n = config.example.n_obs
        s = config.example.n_obs

        self.seq = nn.Sequential()
        for i, (h, act) in enumerate(zip(config.hidden_neurons, config.activation)):
            self.seq.add_module(f'layer{i + 1}', nn.Linear(s, h))
            self.seq.add_module(f'activation{i + 1}', get_activation(act))
            s = h

        self.seq.add_module(f'last_layer', nn.Linear(s, 1))

        # multiplier
        self.mul = nn.Sequential()
        self.mul.add_module('mul_layer', nn.Linear(config.example.n_obs, 5))
        self.mul.add_module('square', Square())
        self.mul.add_module('last', nn.Linear(5, 1))
        # self.mul = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.seq(x), self.mul(x)

    def get_mul(self):
        x = sp.symbols([[f'x{i + 1}' for i in range(self.n)]])
        w1 = self.mul[0].weight.detach().numpy()
        b1 = self.mul[0].bias.detach().numpy()
        w2 = self.mul[2].weight.detach().numpy()
        b2 = self.mul[2].bias.detach().numpy()
        y = (np.dot(x, w1.T) + b1) ** 2
        z = np.dot(y, w2.T) + b2
        z = sp.expand(z[0, 0])
        return z
        # return self.mul.detach().numpy()[0]


class Learner:
    def __init__(self, config: Config):
        self.net = Net(config)
        self.config = config

    def learn(self, optimizer, s, sdot):
        init, unsafe, domain = s
        print('Init samples:', len(init), 'Unsafe samples:', len(unsafe), 'Lie samples', len(domain))

        margin = self.config.margin
        relu6 = nn.ReLU6()
        slope = 1e-3
        loop = self.config.loop
        for i in range(loop):
            optimizer.zero_grad()

            bi, _ = self.net(init)
            bu, _ = self.net(unsafe)
            bd, lam = self.net(domain)
            bi, bu, bd = bi[:, 0], bu[:, 0], bd[:, 0]
            lam = lam[:, 0]

            acc_init = sum(bi < -margin / 4).item() * 100 / len(bi)
            acc_unsafe = sum(bu > margin / 4).item() * 100 / len(bu)

            loss1 = (torch.relu(bi + margin) - slope * relu6(-bi - margin)).mean()
            loss2 = (torch.relu(-bu + margin) - slope * relu6(bu - margin)).mean()

            # lie derivative
            jac_single = jacrev(self.net)
            jac = torch.vmap(jac_single)(domain)[0]
            lie_der = torch.sum(torch.mul(jac[:, 0, :], sdot), dim=1)
            lie = lie_der - lam * bd

            acc_lie = sum(lie < -margin / 4).item() * 100 / len(lie)

            loss3 = (torch.relu(lie + margin) - slope * relu6(-lie - margin)).mean()

            # print(self.net.get_mul())
            w1, w2, w3 = self.config.loss_weight
            loss = w1 * loss1 + w2 * loss2 + w3 * loss3
            loss.backward()
            optimizer.step()
            if i % (loop // 10) == 0 or (acc_init == 100 and acc_unsafe == 100 and acc_lie == 100):
                print(f'{i}->', end=' ')
                print(f'accuracy_init:{acc_init}, acc_unsafe:{acc_unsafe}, acc_lie:{acc_lie}', end=' ')
                print(f'loss:{loss}')

            if acc_init == 100 and acc_unsafe == 100 and acc_lie == 100:
                break


if __name__ == '__main__':
    from rl_train.Examples import examples

    ex = examples[1]
    opt = {
        'example': ex,
        'loop': 100,
        'batch_size': 500,
        'lr': 0.05,
        'margin': 1,
        'R_b': 0,
        'activation': ['relu'],
        'loss_weight': [1, 1, 1]
    }
    con = Config(**opt)

    learn = Learner(con)
    # print(learn.net.mul[0], learn.net.mul[1], learn.net.mul[2])
    print(learn.net.get_mul())
    # opt = torch.optim.AdamW(learn.net.parameters(), lr=1e-3)
    # from Generate_data import Data
    #
    # data = Data(con)
    # a, b, c, d = data.generate_data()
    # learn.learn(opt, (a, b, c), d)
