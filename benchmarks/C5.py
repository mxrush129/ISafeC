import sys

sys.path.append('../')
import numpy as np
import torch
import nni

from bc_learn.Config import Config
from bc_learn.main import main
from pprint import pprint
from rl_train.Examples import get_example_by_id, get_example_by_name


def solve(params):
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)
    np.random.seed(2025)

    ex = get_example_by_name('C5')
    with open(f'../controller/{ex.name}.txt', 'r', encoding='utf-8') as f:
        controller = f.readline()

    print(controller)
    opts = {
        'example': ex,
        'lr': params['lr'],
        'batch_size': params['batch_size'],
        'margin': params['margin'],
        'hidden_neurons': [params['hidden_neurons']],
        'activation': [params['activation']],
        'R_b': params['R_b'],
    }
    config = Config(**opts)
    res = main(config, controller, epoch=5, l=6, config_fine=(100, params['fine2'], 0.5, 500, params['fine5']),
               adaptive_margin=True)
    nni.report_final_result(res)

    if res:
        with open('C5_result.txt', 'a', encoding='utf-8') as file:
            pprint(params, stream=file)


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    solve(tuner_params)
