import numpy as np

from rl_train.Env import Zones, Example, Env

pi = np.pi

examples = {
    1: Example(
        n_obs=2,
        u_dim=1,
        D_zones=Zones(shape='box', low=[-2, -2], up=[2, 2]),
        I_zones=Zones(shape='box', low=[0, 1], up=[1, 2]),
        U_zones=Zones(shape='box', low=[-2, -0.75], up=[-0.5, 0.75]),
        f=[lambda x, u: x[1],
           lambda x, u: -10 * (0.005621 * x[0] ** 5 - 0.1551 * x[0] ** 3 + 0.9875 * x[0]) - 0.1 * x[1] + u[0]
           ],
        u=2,
        dense=4,
        units=20,
        dt=0.001,
        max_episode=2000,
        goal='avoid',
        name='Pendulum'),
    2: Example(
        n_obs=2,
        u_dim=1,
        D_zones=Zones('box', low=[-1, -0.2], up=[0.5, 0.8]),
        I_zones=Zones('box', low=[-0.51, 0.49], up=[-0.49, 0.51]),
        G_zones=Zones('box', low=[-0.05, -0.05], up=[0.05, 0.05]),
        U_zones=Zones('box', low=[-0.4, 0.2], up=[0.1, 0.35]),
        f=[lambda x, u: x[1],
           lambda x, u: (1 - x[0] ** 2) * x[1] - x[0] + u[0]
           ],
        u=3,
        dense=5,
        units=64,
        dt=0.005,
        max_episode=1500,
        goal='avoid',
        name='Oscillator'),
    3: Example(
        n_obs=3,
        u_dim=1,
        D_zones=Zones(shape='box', low=[0] * 3, up=[4] * 3),
        I_zones=Zones(shape='box', low=[0] * 3, up=[1.5] * 3),
        U_zones=Zones(shape='box', low=[2.5] * 3, up=[4] * 3),
        f=[lambda x, u: x[2] + 8 * x[1],
           lambda x, u: -x[1] + x[2],
           lambda x, u: -x[2] - x[0] ** 2 + u[0],
           ],
        u=3,
        dense=5,
        units=64,
        dt=0.005,
        max_episode=1500,
        goal='avoid',
        name='Academic_3D'
    ),  # Academic 3D
    4: Example(
        n_obs=4,
        u_dim=1,
        D_zones=Zones(shape='box', low=[-1] * 4, up=[4] * 4),
        I_zones=Zones(shape='box', low=[3] * 4, up=[4] * 4),
        U_zones=Zones(shape='box', low=[-1] * 4, up=[1] * 4),
        f=[lambda x, u: x[2],
           lambda x, u: x[3],
           lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u[0],
           lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2])],
        u=3,
        dense=5,
        units=64,
        dt=0.005,
        max_episode=1500,
        goal='avoid',
        name='C4'
    ),
    5: Example(
        n_obs=2,
        u_dim=1,
        D_zones=Zones('box', low=[0, 0], up=[4, 4]),
        I_zones=Zones('box', low=[0, 0], up=[1, 1]),
        U_zones=Zones('box', low=[2, 2], up=[4, 3]),
        f=[lambda x, u: -x[0] + x[1] - x[0] ** 2 - x[1] ** 3 + x[0] * u[0],
           lambda x, u: -2 * x[1] - x[0] ** 2 + u[0]],
        u=2,
        dense=5,
        units=64,
        dt=0.001,
        max_episode=1000,
        goal='avoid',
        name='C5'
    ),
    6: Example(
        n_obs=2,
        u_dim=1,
        D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
        I_zones=Zones('box', low=[-1, 1], up=[-0.9, 1.1]),
        U_zones=Zones('box', low=[-2.75, -2.25], up=[-1.75, -1.25]),
        f=[lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u[0],
           lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7)],
        u=2,
        dense=5,
        units=64,
        dt=0.005,
        max_episode=1500,
        goal='avoid',
        name='C6'
    ),
    7: Example(
        n_obs=3,
        u_dim=1,
        D_zones=Zones(shape='box', low=[-0.2, -0.2, -0.2], up=[0.2, 0.2, 0.2]),
        I_zones=Zones(shape='box', low=[-0.1, -0.1, -0.1], up=[0.1, 0.1, 0.1]),
        U_zones=Zones(shape='box', low=[-0.18, -0.18, -0.18], up=[-0.15, -0.15, -0.15]),
        f=[lambda x, u: x[1] + u[0],
           lambda x, u: -x[2],
           lambda x, u: -x[0] - 2 * x[1] - x[2] + x[0] ** 3,
           ],
        u=1,
        dense=5,
        units=64,
        dt=0.005,
        max_episode=1500,
        goal='avoid',
        name='C7'
    ),
    8: Example(
        n_obs=5,
        u_dim=1,
        D_zones=Zones('box', low=[-3, -3, -3, -3, -3], up=[3, 3, 3, 3, 3]),
        I_zones=Zones('box', low=[0.5] * 5, up=[1.5] * 5),
        U_zones=Zones('box', low=[-2.6] * 5, up=[-1.4] * 5),
        f=[lambda x, u: -0.1 * x[0] ** 2 - 0.4 * x[0] * x[3] - x[0] + x[1] + 3 * x[2] + 0.5 * x[3],
           lambda x, u: x[1] ** 2 - 0.5 * x[1] * x[4] + x[0] + x[2],
           lambda x, u: 0.5 * x[2] ** 2 + x[0] - x[1] + 2 * x[2] + 0.1 * x[3] - 0.5 * x[4],
           lambda x, u: x[1] + 2 * x[2] + 0.1 * x[3] - 0.2 * x[4],
           lambda x, u: x[2] - 0.1 * x[3] + u[0]
           ],
        u=5,
        dense=5,
        units=64,
        dt=0.005,
        max_episode=1500,
        goal='avoid',
        name='C8'
    ),
}


def get_example_by_id(id: int):
    return examples[id]


def get_example_by_name(name: str):
    for ex in examples.values():
        if ex.name == name:
            return ex
    raise ValueError('The example {} was not found.'.format(name))


if __name__ == '__main__':
    example = examples[1]
    env = Env(examples[1])
    env.reward_gaussian = False
    x, y, r = [], [], []
    s, info = env.reset(2024)
    print(s)
    x.append(s[0])
    y.append(s[1])
    done, truncated = False, False
    while not done and not truncated:
        action = np.array([1])
        observation, reward, terminated, truncated, info = env.step(action)
        x.append(observation[0])
        y.append(observation[1])
        r.append(reward)

    from rl_train.Plot import plot

    plot(env, x, y)
    print(sum(r))
