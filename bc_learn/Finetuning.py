import numpy as np
import sympy as sp
import torch
import torch.nn as nn

from rl_train.Examples import Example
from verify.kvh_verify import KVH


class FinetuneNet(nn.Module):
    def __init__(self, barr_fun, v, kvh: KVH, part_der, f, mul, device):
        super().__init__()
        self.device = device
        self.barr_fun = barr_fun
        self.v = nn.Parameter(torch.Tensor(v).reshape(-1, 1))
        self.lam = nn.Parameter(torch.Tensor([mul]))
        self.lam_sp = kvh.lam_sp_fine
        self.part_der = part_der
        self.f = f

    def forward(self, x):
        # ans = torch.Tensor([self.fun(item) for item in x])
        ans = self.fun_(x)

        res = torch.matmul(ans, self.v)

        # partial_ = [self.partial_fun(item) for item in x]
        partial = self.partial_fun_(x)

        # der_ = [self.der_fun(r, s) for r, s in zip(x, res)]
        der = self.der_fun_(x, res)

        y = torch.stack([torch.matmul(r, s) for r, s in zip(der, partial)])

        # b = torch.stack([self.barr_fun(*item) for item in x])
        b = self.get_barr_fun(x)

        result = y - self.lam * b
        return result

    def get_barr_fun(self, x):
        xt = x.T
        return self.barr_fun(*xt)

    def get_controller(self, n):
        x = sp.symbols([f'x{i + 1}' for i in range(n)])
        controller = np.dot(self.fun(x), self.v.detach().cpu().numpy())[0]
        return controller, self.lam.detach().cpu().numpy()[0]

    # def der_fun(self, x, u):
    #     return torch.stack([f(x, u) for f in self.f])

    def der_fun_(self, x, u):
        xt = x.T
        ut = u.T
        return torch.stack([f(xt, ut) for f in self.f], dim=1)

    # def partial_fun(self, x):
    #     return torch.stack([f(*x) for f in self.part_der])

    def partial_fun_(self, x):
        xt = x.T
        return torch.stack([f(*xt) for f in self.part_der], dim=1)

    def fun(self, x):
        return [f(*x) for f in self.lam_sp]

    def fun_(self, x):
        xt = x.T
        ans = [f(*xt) for f in self.lam_sp]
        ans[0] = torch.ones(len(xt[0])).to(self.device)
        return torch.stack(ans, dim=1)


class Finetuner:
    def __init__(self, barrier, ex, u, kvh: KVH, mul, device):
        self.device = device
        self.ex: Example = ex
        self.barrier = barrier
        self.u = u
        self.kvh = kvh
        self.v = self.get_vector()
        self.part_der = self.get_partial_derivative()
        self.fun_barrier = self.get_barrier()
        self.net = FinetuneNet(self.fun_barrier, self.v, self.kvh, self.part_der, self.ex.f, mul, self.device).to(
            self.device)

    def get_data(self, batch_size):
        zone = self.ex.D_zones
        s = None
        if zone.shape == 'box':
            times = 1 / (1 - 0.2)
            s = np.clip((np.random.rand(batch_size, self.ex.n_obs) - 0.5) * times, -0.5, 0.5)
            center = (zone.low + zone.up) / 2
            s = s * (zone.up - zone.low) + center

        return torch.Tensor(s)

    def learn(self, loop, margin=10, lr=0.1, bs=500, p=1):
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        data = self.get_data(bs).to(self.device)
        for i in range(loop):
            opt.zero_grad()
            y = self.net(data)
            accuracy = sum(y < -p / 4).item() * 100 / len(data)
            loss = torch.relu(self.net(data) + margin).mean()
            if i % 10 == 0:
                print(f'epoch:{i}, loss:{loss}, accuracy:{accuracy}')
            if accuracy == 100:
                break

            loss.backward()
            opt.step()

    def get_vector(self):
        if isinstance(self.u, str):
            self.u = sp.sympify(self.u)

        self.u = sp.expand(self.u)

        terms = self.u.as_ordered_terms()
        poly = {e: i for i, e in enumerate(self.kvh.poly_fine)}
        v = [0] * len(poly)
        for term in terms:
            item = term.as_coeff_Mul()
            v[poly[str(item[1])]] += item[0]
        return v

    def get_partial_derivative(self):
        if isinstance(self.barrier, str):
            self.barrier = sp.sympify(self.barrier)

        self.barrier = sp.expand(self.barrier)

        x = sp.symbols([f'x{i + 1}' for i in range(self.ex.n_obs)])
        partial_derivative = [sp.lambdify(x, sp.diff(self.barrier, x[i])) for i in range(self.ex.n_obs)]
        return partial_derivative

    def get_barrier(self):
        if isinstance(self.barrier, str):
            self.barrier = sp.sympify(self.barrier)
        self.barrier = sp.expand(self.barrier)
        x = sp.symbols([f'x{i + 1}' for i in range(self.ex.n_obs)])
        return sp.lambdify(x, self.barrier)
