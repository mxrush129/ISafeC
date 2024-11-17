import numpy as np
import sympy as sp
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from bc_learn.Net_pre import Net
from rl_train.Env import Zones
from rl_train.Examples import Example


class Approximation:
    def __init__(self, net: Net, ex: Example):
        self.net = net
        self.domain: Zones = ex.D_zones
        self.n = ex.n_obs

    def fit(self, samples=100, deg=2):
        x_data, y_data = self.generate_data(samples)
        poly = PolynomialFeatures(deg, include_bias=True)
        x = poly.fit_transform(x_data)
        model = Ridge(fit_intercept=False)
        model.fit(x, y_data)
        y = model.predict(x)

        print('score:', model.score(x, y_data))
        print('error:', mean_squared_error(y_data, y))
        # print(model.intercept_, model.coef_)
        # print(poly.get_feature_names_out())

        s = ''
        for term, value in zip(poly.get_feature_names_out(), model.coef_[0]):
            term = term.replace(' ', '*')
            term = term.replace('^', '**')
            s += f'+ ({value}) * {term}'
        s = s[1:]
        x_pre = sp.symbols([f'x{i}' for i in range(self.n)])
        x = sp.symbols([f'x{i + 1}' for i in range(self.n)])
        s = sp.lambdify(x_pre, s)(*x)
        # print(s)
        return s

    def generate_data(self, nums):
        if self.domain.shape == 'box':
            s = np.random.rand(nums, self.n)
            center = (self.domain.low + self.domain.up) / 2
            s = s * (self.domain.up - self.domain.low) + center

        s_tensor = torch.Tensor(s)
        labels = self.net(s_tensor)[0].detach().numpy()
        return s, labels


if __name__ == '__main__':
    pass
    # from Examples_bc import examples
    # from bc_learn.Config import Config
    #
    # ex = examples[1]
    # opt = {
    #     'example': ex,
    #     'loop': 100,
    #     'batch_size': 500,
    #     'lr': 0.05,
    #     'margin': 1,
    #     'R_b': 0,
    #     'activation': ['relu'],
    #     'loss_weight': [1, 1, 1]
    # }
    # con = Config(**opt)
    #
    # net = Net(con)
    # approx = Approximation(net, ex)
    #
    # approx.fit(100, 2)
    # print(id(net) == id(approx.net))
