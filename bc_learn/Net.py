import numpy as np
import sympy as sp
import torch
import torch.nn as nn

from bc_learn.Config import Config


class Net(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.layers1, self.layers2 = [], []
        self.mul_layer1, self.mul_layer2 = [], []
        self.input_size = input_size = config.example.n_obs
        k, bias = 1, True
        self.acts = config.activation
        n_prev = config.example.n_obs
        for n_hid, act in zip(config.hidden_neurons, config.activation):
            layer1 = nn.Linear(n_prev, n_hid, bias=bias)

            if act not in ['SKIP']:
                layer2 = nn.Linear(n_prev, n_hid, bias=bias)
            else:
                layer2 = nn.Linear(input_size, n_hid, bias=bias)

            self.register_parameter("W" + str(k), layer1.weight)
            self.register_parameter("W2" + str(k), layer2.weight)
            if bias:
                self.register_parameter("b" + str(k), layer1.bias)
                self.register_parameter("b2" + str(k), layer2.bias)
            self.layers1.append(layer1)
            self.layers2.append(layer2)

            n_prev = n_hid
            k = k + 1

        layer1 = nn.Linear(n_prev, 1, bias=False)
        self.register_parameter("W" + str(k), layer1.weight)
        self.layers1.append(layer1)

        if len(config.mul_activation) == 0:
            scalar = torch.nn.Parameter(torch.randn(1))
            self.register_parameter('scalar', scalar)
            self.mul_layer1.append(scalar)
        else:
            n_prev = config.example.n_obs
            j = 1
            for n_hid, act in zip(config.mul_hidden[:-1], config.mul_activation):
                mul1 = nn.Linear(n_prev, n_hid)
                if act == 'SKIP':
                    mul2 = nn.Linear(input_size, n_hid, bias=bias)
                else:
                    mul2 = nn.Linear(n_prev, n_hid, bias=bias)

                self.register_parameter("M1" + str(j), mul1.weight)
                self.register_parameter("M2" + str(j), mul2.weight)
                if bias:
                    self.register_parameter("mb1" + str(j), mul1.bias)
                    self.register_parameter("mb2" + str(j), mul2.bias)
                self.mul_layer1.append(mul1)
                self.mul_layer2.append(mul2)
                n_prev = n_hid
                j = j + 1

            mul1 = nn.Linear(n_prev, 1)
            self.mul_layer1.append(mul1)
            self.register_parameter("M1" + str(j), mul1.weight)
            self.register_parameter("mb1" + str(j), mul1.bias)

    def forward(self, x, xdot=None):
        yy = x
        if len(self.config.mul_activation) == 0:
            yy = x * 0 + self.mul_layer1[0]
        else:
            relu6 = nn.ReLU6()
            for idx, (mul1, mul2) in enumerate(zip(self.mul_layer1[:-1], self.mul_layer2)):
                if self.config.mul_activation[idx] == 'ReLU':
                    yy = relu6(mul1(yy))
                elif self.config.mul_activation[idx] == 'SQUARE':
                    yy = mul1(yy) ** 2
                elif self.config.mul_activation[idx] == 'MUL':
                    yy = mul1(yy) * mul2(yy)
                elif self.config.mul_activation[idx] == 'SKIP':
                    yy = mul1(yy) * mul2(x)
                elif self.config.mul_activation[idx] == 'LINEAR':
                    yy = mul1(yy)

            yy = self.multiplicators1[-1](yy)

        if xdot is not None:
            y = x
            jac = torch.diag_embed(torch.ones(x.shape[0], self.input_size)).to(self.config.device)
            for idx, (layer1, layer2) in enumerate(zip(self.layers1[:-1], self.layers2)):
                if self.acts[idx] in ['SQUARE']:
                    z = layer1(y)
                    y = z ** 2
                    jac = torch.matmul(torch.matmul(2 * torch.diag_embed(z), layer1.weight), jac)

                elif self.acts[idx] == 'MUL':
                    z1 = layer1(y)
                    z2 = layer2(y)
                    y = z1 * z2
                    grad = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(torch.diag_embed(z2),
                                                                                            layer1.weight)
                    jac = torch.matmul(grad, jac)

                elif self.acts[idx] == 'SKIP':
                    z1 = layer1(y)
                    z2 = layer2(x)
                    y = z1 * z2
                    jac = torch.matmul(torch.diag_embed(z1), layer2.weight) + torch.matmul(
                        torch.matmul(torch.diag_embed(z2), layer1.weight), jac)

                elif self.acts[idx] == 'LINEAR':
                    y = layer1(y)
                    jac = torch.matmul(layer1.weight, jac)

            numerical_b = torch.matmul(y, self.layers1[-1].weight.T)
            jac = torch.matmul(self.layers1[-1].weight, jac)
            numerical_bdot = torch.sum(torch.mul(jac[:, 0, :], xdot), dim=1)

            return numerical_b, numerical_bdot, y, yy
        else:
            y = x
            for idx, (layer1, layer2) in enumerate(zip(self.layers1[:-1], self.layers2)):
                if self.acts[idx] in ['SQUARE']:
                    z = layer1(y)
                    y = z ** 2

                elif self.acts[idx] == 'MUL':
                    z1 = layer1(y)
                    z2 = layer2(y)
                    y = z1 * z2

                elif self.acts[idx] == 'SKIP':
                    z1 = layer1(y)
                    z2 = layer2(x)
                    y = z1 * z2

                elif self.acts[idx] == 'LINEAR':
                    y = layer1(y)

            numerical_b = torch.matmul(y, self.layers1[-1].weight.T)

            return numerical_b, None, y, yy

    def get_barrier(self):
        x = sp.symbols([['x{}'.format(i + 1) for i in range(self.input_size)]])
        y = x
        for idx, (layer1, layer2) in enumerate(zip(self.layers1[:-1], self.layers2)):
            if self.acts[idx] == 'SQUARE':
                w1 = layer1.weight.detach().cpu().numpy()
                b1 = layer1.bias.detach().cpu().numpy()
                z = np.dot(y, w1.T) + b1
                y = z ** 2
            elif self.acts[idx] == 'MUL':
                w1 = layer1.weight.detach().cpu().numpy()
                b1 = layer1.bias.detach().cpu().numpy()
                z1 = np.dot(y, w1.T) + b1

                w2 = layer2.weight.detach().cpu().numpy()
                b2 = layer2.bias.detach().cpu().numpy()
                z2 = np.dot(y, w2.T) + b2

                y = np.multiply(z1, z2)
            elif self.acts[idx] == 'SKIP':
                w1 = layer1.weight.detach().cpu().numpy()
                b1 = layer1.bias.detach().cpu().numpy()
                z1 = np.dot(y, w1.T) + b1

                w2 = layer2.weight.detach().cpu().numpy()
                b2 = layer2.bias.detach().cpu().numpy()
                z2 = np.dot(x, w2.T) + b2
                y = np.multiply(z1, z2)

            elif self.acts[idx] == 'LINEAR':
                w1 = layer1.weight.detach().cpu().numpy()
                b1 = layer1.bias.detach().cpu().numpy()
                y = np.dot(y, w1.T) + b1

        w1 = self.layers1[-1].weight.detach().cpu().numpy()
        y = np.dot(y, w1.T)
        y = sp.expand(y[0, 0])
        return y

    def get_mul(self):
        if len(self.config.mul_activation) == 0:
            return self.mul_layer1[0].detach().cpu().numpy()[0]
        else:
            x = sp.symbols([['x{}'.format(i + 1) for i in range(self.input_size)]])
            y = x
            for idx, (layer1, layer2) in enumerate(zip(self.mul_layer1[:-1], self.mul_layer2)):
                if self.acts[idx] == 'SQUARE':
                    w1 = layer1.weight.detach().cpu().numpy()
                    b1 = layer1.bias.detach().cpu().numpy()
                    z = np.dot(y, w1.T) + b1
                    y = z ** 2
                elif self.acts[idx] == 'MUL':
                    w1 = layer1.weight.detach().cpu().numpy()
                    b1 = layer1.bias.detach().cpu().numpy()
                    z1 = np.dot(y, w1.T) + b1

                    w2 = layer2.weight.detach().cpu().numpy()
                    b2 = layer2.bias.detach().cpu().numpy()
                    z2 = np.dot(y, w2.T) + b2

                    y = np.multiply(z1, z2)
                elif self.acts[idx] == 'SKIP':
                    w1 = layer1.weight.detach().cpu().numpy()
                    b1 = layer1.bias.detach().cpu().numpy()
                    z1 = np.dot(y, w1.T) + b1

                    w2 = layer2.weight.detach().cpu().numpy()
                    b2 = layer2.bias.detach().cpu().numpy()
                    z2 = np.dot(x, w2.T) + b2
                    y = np.multiply(z1, z2)

            w1 = self.mul_layer1[-1].weight.detach().cpu().numpy()
            b1 = self.mul_layer1[-1].bias.detach().cpu().numpy()
            y = np.dot(y, w1.T) + b1
            y = sp.expand(y[0, 0])
            return y


class Learner(nn.Module):
    def __init__(self, config: Config):
        super(Learner, self).__init__()
        self.net = Net(config).to(config.device)
        self.loss_weight = config.loss_weight
        self.config = config

    def get_lie_by_hand(self, s, p):
        y = self.net.get_barrier()
        x = sp.symbols([f'x{i + 1}' for i in range(self.config.example.n_obs)])
        f = self.config.example.f
        controller = sp.sympify('0.0712195918155747*x1 - 0.713376341030575*x2')
        db = sum([sp.diff(y, x[i]) * f[i](x, [controller]) for i in range(self.config.example.n_obs)])
        db = sp.expand(db)
        lam_fun = sp.lambdify(x, db)
        res = torch.Tensor([lam_fun(*item) for item in s])
        print(res)
        for a, b in zip(res, p):
            print(a, b)

    def learn(self, optimizer, s, sdot):
        print('Init samples:', len(s[0]), 'Unsafe samples:', len(s[1]), 'Lie samples', len(s[2]))
        learn_loops = self.config.loop
        margin = self.config.margin
        slope = 1e-3
        relu6 = torch.nn.ReLU6()
        for t in range(learn_loops):
            optimizer.zero_grad()

            B_i, _, __, ___ = self.net(s[0])
            B_u, _, __, ___ = self.net(s[1])
            B_d, Bdot_d, __, yy = self.net(s[2], sdot)

            B_i = B_i[:, 0]
            B_u = B_u[:, 0]
            B_d = B_d[:, 0]
            yy = yy[:, 0]
            accuracy_init = sum(B_i < -margin / 2).item() * 100 / len(s[0])
            accuracy_unsafe = sum(B_u > margin / 2).item() * 100 / len(s[1])

            loss = self.loss_weight[0] * (torch.relu(B_i + margin) - slope * relu6(-B_i - margin)).mean()
            loss = loss + self.loss_weight[1] * (torch.relu(-B_u + margin) - slope * relu6(B_u - margin)).mean()

            dB_belt = Bdot_d - yy * B_d
            loss = loss + self.loss_weight[2] * (
                    torch.relu(dB_belt + margin) - slope * relu6(-dB_belt - margin)).mean()

            percent_belt = 100 * (sum(dB_belt <= -margin)).item() / dB_belt.shape[0]

            # print(Bdot_d.detach())
            # self.get_lie_by_hand(s[2], Bdot_d.detach())
            # import time
            # time.sleep(1000)

            if (t + 1) % (learn_loops // 10) == 0 or (
                    accuracy_init == 100 and percent_belt == 100 and accuracy_unsafe == 100):
                print(t + 1, "- loss:", loss.item(), '- accuracy init:', accuracy_init, 'accuracy unsafe:',
                      accuracy_unsafe, "- accuracy Lie:", percent_belt)

            if accuracy_init == 100 and percent_belt == 100 and accuracy_unsafe == 100:
                break
            loss.backward()
            optimizer.step()
