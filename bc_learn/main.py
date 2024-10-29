import time

import sympy as sp
import torch

from bc_learn.Config import Config
from bc_learn.Draw import Draw
from bc_learn.Finetuning import Finetuner
from bc_learn.Generate_data import Data
# from bc_learn.Net_pre import Net, Learner
from bc_learn.Net import Learner
from verify.kvh_verify import KVH


def print_success():
    heart = """
          *****         *****
       ***********   *************
     *************** ***************
    *********************************
     *******************************
      ****** SUCCESS SUCCESS ******
       ***************************
         ************************
           *********************
             *****************
               *************
                 *********
                   *****
                     *
    """

    print(heart)


def main(config: Config, controller, epoch=5, l=4, config_fine=(100, 10, 0.1, 500, 1), adaptive_margin=False):
    learner = Learner(config)
    opt = torch.optim.AdamW(learner.net.parameters(), lr=config.lr)
    data = Data(config, controller)

    kvh = KVH(config.example, config.example.n_obs, l)
    kvh.pos, vis = 1, False

    t_learn, t_kvh, t_finetune = 0, 0, 0

    for i in range(epoch):
        if adaptive_margin:
            config.margin = config.margin * 2
        print(f'Controller for epoch {i + 1}:{controller}')
        init, unsafe, domain, domain_dot = data.generate_data()
        t1 = time.time()
        print(f'---------------------------------------\nStart training--epoch {i + 1}\n'
              '---------------------------------------')
        learner.learn(opt, (init, unsafe, domain), domain_dot)

        t2 = time.time()
        t_learn += t2 - t1

        bc = learner.net.get_barrier()
        # print('bc:', bc)
        multiplier = learner.net.get_mul()
        # print('multiplier', multiplier)
        kvh.update_barrier(bc, multiplier, sp.sympify(controller))

        t3 = time.time()

        state = kvh.verify_all()

        t4 = time.time()
        t_kvh += t4 - t3
        if state:
            vis = True
            print_success()
            print(f'In the {i + 1} epoch, barrier certificate verification successful!')
            print(f'Barrier certificate:{bc}')
            print(f'Controller:{controller}')
            print(f'The time of learn:{t_learn}s')
            print(f'The time of verify:{t_kvh}s')
            print(f'The time of finetune:{t_finetune}s')
            if config.example.n_obs == 2:
                draw = Draw(config.example, bc, controller)
                draw.draw()
            break

        if vis:
            break

        print(f'Epoch {i + 1} failed verification barrier certificate:{bc}')
        print(f'---------------------------------------\nStart fine tuning--epoch {i + 1}\n'
              '---------------------------------------')

        finetuner = Finetuner(bc, config.example, controller, kvh, multiplier, config.device)

        t5 = time.time()

        finetuner.learn(*config_fine)

        t6 = time.time()
        t_finetune += t6 - t5
        controller, multiplier = finetuner.net.get_controller(config.example.n_obs)
        # print('controller:', controller)
        kvh.update_barrier(bc, multiplier, sp.sympify(controller))

        t3 = time.time()

        result = kvh.verify_all()

        t4 = time.time()
        t_kvh += t4 - t3
        if result:
            vis = True
            print_success()
            print(f'In the {i + 1} epoch, after the fine-tuning, barrier certificate verification successful!')
            print(f'Barrier certificate:{bc}')
            print(f'Controller:{controller}')
            print(f'The time of learn:{t_learn}s')
            print(f'The time of verify:{t_kvh}s')
            print(f'The time of finetune:{t_finetune}s')
            if config.example.n_obs == 2:
                draw = Draw(config.example, bc, controller)
                draw.draw()
            break
        else:
            data.update(controller)

    if not vis:
        print('No barrier certificate found!')
        return False
    return True
