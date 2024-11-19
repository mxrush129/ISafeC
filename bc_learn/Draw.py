import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle

from rl_train.Examples import Example
# from utils.Config import CegisConfig
from rl_train.Examples import Zones as Zone


class Draw:
    def __init__(self, example: Example, b, controller):
        self.ex = example
        self.b1 = b
        self.controller = controller

    def draw(self):
        fig = plt.figure()
        ax = plt.gca()

        ax.add_patch(self.draw_zone(self.ex.D_zones, 'black', 'local_1'))
        ax.add_patch(self.draw_zone(self.ex.I_zones, 'g', 'init'))
        ax.add_patch(self.draw_zone(self.ex.U_zones, 'r', 'unsafe'))

        l1 = self.ex.D_zones

        x = sp.symbols([f'x{i + 1}' for i in range(self.ex.n_obs)])
        fun = [sp.lambdify(x, self.ex.f[k](x, [sp.sympify(self.controller)])) for k in range(self.ex.n_obs)]

        self.plot_vector_field(l1, fun)

        self.plot_barrier(l1, self.b1, 'b')

        # plt.xlim(l1.low[0] - 1, l1.up[0] + 1)
        # plt.ylim(l1.low[1] - 1, l1.up[1] + 1)
        ax.set_aspect(1)
        plt.legend()
        # plt.savefig(f'picture/{self.ex.name}_2d.png', dpi=1000, bbox_inches='tight')
        plt.show()

    def plot_barrier_3d(self, ax, zone, b, color):
        low, up = zone.low, zone.up
        x = np.linspace(low[0], up[0], 100)
        y = np.linspace(low[1], up[1], 100)
        X, Y = np.meshgrid(x, y)
        s_x = sp.symbols(['x1', 'x2'])
        lambda_b = sp.lambdify(s_x, b, 'numpy')
        plot_b = lambda_b(X, Y)
        ax.plot_surface(X, Y, plot_b, rstride=5, cstride=5, alpha=0.5, cmap='cool')

    def plot_barrier(self, zone, hx, color):
        low, up = zone.low, zone.up
        x = np.linspace(low[0], up[0], 100)
        y = np.linspace(low[1], up[1], 100)

        X, Y = np.meshgrid(x, y)

        s_x = sp.symbols(['x1', 'x2'])
        fun_hx = sp.lambdify(s_x, hx, 'numpy')
        value = fun_hx(X, Y)
        plt.contour(X, Y, value, 0, alpha=0.8, colors=color)

    def plot_vector_field(self, zone: Zone, f, color='grey'):
        low, up = zone.low, zone.up
        xv = np.linspace(low[0], up[0], 100)
        yv = np.linspace(low[1], up[1], 100)
        Xd, Yd = np.meshgrid(xv, yv)

        DX, DY = f[0](Xd, Yd), f[1](Xd, Yd)
        norm_x = np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
        norm_x = np.where(norm_x == 0, 1e-10, norm_x)
        DX = DX / norm_x

        norm_y = np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
        norm_y = np.where(norm_y == 0, 1e-10, norm_y)
        DY = DY / norm_y

        plt.streamplot(Xd, Yd, DX, DY, linewidth=0.3,
                       density=0.8, arrowstyle='-|>', arrowsize=1, color=color)

    def draw_zone(self, zone: Zone, color, label, fill=False):
        if zone.shape == 'ball':
            circle = Circle(zone.center, np.sqrt(zone.r), color=color, label=label, fill=fill, linewidth=1.5)
            return circle
        else:
            w = zone.up[0] - zone.low[0]
            h = zone.up[1] - zone.low[1]
            box = Rectangle(zone.low, w, h, color=color, label=label, fill=fill, linewidth=1.5)
            return box


if __name__ == '__main__':
    pass
