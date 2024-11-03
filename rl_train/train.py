import re

import matplotlib.pyplot as plt
import sympy as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from rl_train.Env import Env
from rl_train.Examples import get_example_by_name, Example
from rl_train.Plot import plot
from rl_train.ppo import PPO
from rl_train.share import *


def fit(env, agent, degree):
    x, y = [], []
    N = 100
    for i in range(N):
        state, info = env.reset()
        tot = 0
        while True:
            tot += 1
            # if tot >= 1000:
            #     print('第{}条轨迹'.format(i))
            #     break
            action = agent.take_action(state)
            x.append(state)
            y.append(action)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            if done or truncated:
                print('第{}条轨迹'.format(i))
                break

    P = PolynomialFeatures(degree, include_bias=False)
    x = P.fit_transform(x)
    model = Ridge(alpha=0.00, fit_intercept=False)
    model.fit(x, y)

    s = ''
    for k, v in zip(P.get_feature_names_out(), model.coef_[0]):
        k = re.sub(r' ', r'*', k)
        k = k.replace('^', '**')
        if v < 0:
            s += f'- {-v} * {k} '
        else:
            s += f'+ {v} * {k} '

    x = sp.symbols(['x0', 'x1'])
    x_ = sp.symbols(['x1', 'x2'])
    temp = sp.sympify(s[1:])
    u = sp.lambdify(x, temp)(*x_)
    print(f'controller:{u}')
    with open(f'../controller/{env.name}.txt', 'w', encoding='utf-8') as f:
        f.write(f'{u}')
    return u


def train_by_ppo(example: Example, deg=2):
    torch.manual_seed(2024)
    np.random.seed(2024)

    actor_lr = 1e-4
    critic_lr = 5e-3
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.95
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device('cpu')

    env = Env(example)
    env.reward_gaussian = False
    state_dim = env.n_obs
    action_dim = env.u_dim  # 连续动作空间
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    return_list = []
    for i_episode in range(num_episodes):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state_list = []
        state, info = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            state_list.append(state)
            action = agent.take_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        agent.update(transition_dict)

        print(f'episode:{i_episode + 1},reward:{episode_return},step:{len(state_list)}')
        if i_episode % 20 == 0:
            state_list = np.array(state_list)
            x = state_list[:, :1]
            y = state_list[:, 1:2]
            plot(env, x, y)
    fit(env, agent, deg)
    return return_list


if __name__ == '__main__':
    example = get_example_by_name('C11')
    return_list = train_by_ppo(example, 2)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO')
    plt.show()

    mv_return = moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO')
    plt.show()

    # simulation(env_name, agent)
