import numpy as np
import sympy as sp
import torch

from bc_learn.Config import Config
from rl_train.Env import Zones
from rl_train.Examples import Example


class Data:
    def __init__(self, config: Config, u):
        self.config = config
        self.ex: Example = config.example
        self.n = self.ex.n_obs
        self.u = u
        self.x = sp.symbols([f'x{i + 1}' for i in range(self.n)])
        self.u_fun = sp.lambdify(self.x, sp.sympify(self.u))

    def update(self, u):
        self.u = u
        self.u_fun = sp.lambdify(self.x, sp.sympify(self.u))

    def get_data(self, zone: Zones, batch_size):
        s = None
        if zone.shape == 'box':
            times = 1 / (1 - self.config.R_b)
            s = np.clip((np.random.rand(batch_size, self.n) - 0.5) * times, -0.5, 0.5)
            center = (zone.low + zone.up) / 2
            s = s * (zone.up - zone.low) + center

        # elif zone.shape == 'ball':
        #     s = np.random.randn(batch_size, self.n)
        #     s = np.array([e / np.sqrt(sum(e ** 2)) * np.sqrt(zone.r) for e in s])
        #     s = np.array(
        #         [e * np.random.random() ** (1 / self.n) if np.random.random() > self.config.C_b else e for e in s])
        #     s = s + zone.center

        # from matplotlib import pyplot as plt
        # plt.plot(s[:, :1], s[:, -1], '.')
        # plt.gca().set_aspect(1)
        # plt.show()
        # import time
        # time.sleep(1000)
        return torch.Tensor(s).to(self.config.device)

    def x2dotx(self, X, f):
        f_x = []
        for x in X:
            f_x.append([f[i](x, [self.u_fun(*x)]) for i in range(self.n)])
        return torch.Tensor(f_x).to(self.config.device)

    def generate_data(self):
        batch_size = self.config.batch_size
        domain = self.get_data(self.ex.D_zones, batch_size)
        init = self.get_data(self.ex.I_zones, batch_size)
        unsafe = self.get_data(self.ex.U_zones, batch_size)

        domain_dot = self.x2dotx(domain, self.ex.f)

        return init, unsafe, domain, domain_dot
