You can run this tool by following the steps below:

1.First of all, the python environment we use is 3.10.14.

2.You need to install some packages for using it.
```python
pip install -r requirements.txt
```

3.Our tool relies on the Gurobi solver, for which you need to obtain a license. 
You can find it at: https://www.gurobi.com/solutions/gurobi-optimizer/

4.When you have all the environment ready, you can run the tool.

Let's take C1 as an example to illustrate its use.

<1>You need to add the definition of the dynamical system in `./rl_train/Examples.py`.

<2>You need to run `./rl_train/train.py` to train a controller for the current system using PPO.

<3>You need to adjust the parameters and run `./benchmarks/C1.py` to start the iteration to train a barrier certificate and verify it.

You can find the following code in `./benchmarks/C1.py`:

```python
import numpy as np
import torch

from bc_learn.Config import Config
from bc_learn.main import main
from rl_train.Examples import get_example_by_name

if __name__ == '__main__':
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)
    np.random.seed(2025)

    ex = get_example_by_name('Oscillator')
    with open(f'../controller/{ex.name}.txt', 'r', encoding='utf-8') as f:
        controller = f.readline()

    print(controller)
    opts = {
        'example': ex,
        'lr': 0.2,
        'batch_size': 300,
        'margin': 1,
        'hidden_neurons': [10],
        'activation': ['SKIP']
    }
    config = Config(**opts)
    main(config, controller, epoch=5, l=4, config_fine=(100, 10, 0.5, 500, 1))
```