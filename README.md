# 424-project

This is the code for our MIE424 project, **Comparison of Various Optimization Methods for Deep Learning Image Classification** by Mustafa Arif, Anup Deb, Saad Jameel, Michael Mudrinic, and Yuan Hong Sun.

The implementation is based on the following papers and their code:
1. [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610) [code](https://github.com/michaelrzhang/lookahead)
2. [Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization](https://arxiv.org/abs/2009.13586) [code](https://github.com/XuezheMax/apollo)

Some things to note are:
1. The code for making plots is in ``Make_Plots.ipynb``
2. The code defining GoogLeNet is in ``apollo/classification/GoogLeNet.py`` and running/training it is in ``apollo/classification/run_GoogLeNet.py``. The code for ResNet and running/training ResNet is in ``apollo/classification/run_ResNet.py``
3. The Lookahead and Apollo optimizer codes are in ``apollo/optim/lookahead_pytorch.py`` and ``apollo/optim/apollo.py``, respectively.
4. The data from our runs are in ``results/``.
5. The plots from our runs are in ``plots/``. 

To train the optimizers, the following command structure can be used:

For ResNet, an example command is (with milestone decay and lookahead):
``
python 424-project/apollo/classification/run_ResNet.py --depth 110 --batch_size 128 --epochs 50 --opt [apollo|sgd|adamw] --lr 1.0 --opt_h1 0.9 --eps 1e-4 --weight_decay 2.5e-4 --weight_decay_type ['L2'|'decoupled'] --lr_decay milestone --milestone 80 120 --decay_rate 0.1 --warmup_updates 200 --init_lr 0.01 --dataset [emnist|cifar10] --data_path <data_path> --model_path <model_path> --run <run_num> --seed <seed> --lookahead
``

For GoogLeNet, an example command is (with cosine decay and without lookahead):
``
python 424-project/apollo/classification/run_GoogLeNet.py --depth 110 --batch_size 128 --epochs 50 --opt [apollo|sgd|adamw] --lr 1.0 --opt_h1 0.9 --eps 1e-4 --weight_decay 2.5e-4 --weight_decay_type ['L2'|'decoupled'] --lr_decay cosine --last_lr 0.001 --decay_rate 0.1 --warmup_updates 200 --init_lr 0.01 --dataset [emnist|cifar10] --data_path <data_path> --model_path <model_path> --run <run_num> --seed <seed>
``
