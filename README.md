# JoyRL


**Attention!!!** : JoyRL is still under development, please switch to the branch [offline](https://github.com/datawhalechina/joyrl/tree/offline)

[![PyPI](https://img.shields.io/pypi/v/joyrl)](https://pypi.org/project/joyrl/)  [![GitHub issues](https://img.shields.io/github/issues/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/issues) [![GitHub stars](https://img.shields.io/github/stars/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/stargazers) [![GitHub forks](https://img.shields.io/github/forks/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/network) [![GitHub license](https://img.shields.io/github/license/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/blob/master/LICENSE)

`JoyRL` is a parallel reinforcement learning library based on PyTorch and Ray. Unlike existing RL libraries, `JoyRL` is helping users to release the burden of implementing algorithms with tough details, unfriendly APIs, and etc. JoyRL is designed for users to train and test RL algorithms with **only hyperparameters configuration**, which is mush easier for beginners to learn and use. Also, JoyRL supports plenties of state-of-art RL algorithms including **RLHF(core of ChatGPT)**(See algorithms below). JoyRL provides a **modularized framework** for users as well to customize their own algorithms and environments. 

## Install

```bash
# you need to install Anaconda first
conda create -n joyrl python=3.8
conda activate joyrl
pip install -U joyrl
```

Torch install:

```bash
# CPU only
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
# GPU 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# if network error, then GPU with mirrors
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
## Usage

the following presents a demo to use joyrl, you donot need to care about complicated details of code. All your need is just to set hyper parameters including `GeneralConfig()` and `AlgoConfig()`, which is also shown in [examples](./examples/) folder, and well trained results are shown in the [benchmarks](./benchmarks/) folder as well.
```python
import joyrl
class GeneralConfig():
    def __init__(self) -> None:
        self.env_name = "CartPole-v1" # name of environment
        self.algo_name = "DQN" # name of algorithm
        self.mode = "train" # train or test
        self.seed = 0 # random seed
        self.device = "cpu" # device to use
        self.train_eps = 100 # number of episodes for training
        self.test_eps = 20 # number of episodes for testing
        self.eval_eps = 10 # number of episodes for evaluation
        self.eval_per_episode = 5 # evaluation per episode
        self.max_steps = 200 # max steps for each episode
        self.load_checkpoint = False
        self.load_path = "tasks" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not

class AlgoConfig():
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay rate
        self.gamma = 0.95  # discount factor
        self.lr = 0.0001  # learning rate
        self.buffer_size = 100000  # size of replay buffer
        self.batch_size = 64  # batch size
        self.target_update = 4  # target network update frequency
        self.value_layers = [
            {'layer_type': 'linear', 'layer_dim': ['n_states', 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 'n_actions'],
             'activation': 'none'}]
if __name__ == "__main__":
    general_cfg = GeneralConfig()
    algo_cfg = AlgoConfig()
    joyrl.run(general_cfg,algo_cfg)
```
## Documentation

More tutorials and API documentation are hosted on [https://datawhalechina.github.io/joyrl/](https://datawhalechina.github.io/joyrl/)
## Algorithms

|       Name       |                          Reference                           |                    Author                     | Notes |
| :--------------: | :----------------------------------------------------------: | :-------------------------------------------: | :---: |
| DQN | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [johnjim0816](https://github.com/johnjim0816) |       |
