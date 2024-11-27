# File:                     config.py
# Created by:               Ivan Luiz De Moura Matos and Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2024/7/18
# Debiasing Surgeon : Fantastic weights and how to find them, ECCV 2024
# ==========================================================================


import argparse
from datetime import datetime
import os
import random
import numpy as np
import torch


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--batch-size-accumulation",
        type=int,
        default=1,
        metavar="N",
        help="batchsize accumulation (default: 1)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        metavar="N",
        help="number of epochs to train (default: 80)",
    ) 
    parser.add_argument("--lr",type=float,default=0.1,metavar="LR")
    parser.add_argument(
        "--lr_priv",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate for private head (default: 0.1)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--target_celeba", type=str, default="Blond_Hair")
    parser.add_argument("--gamma", type=float, default=10)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--dev", default="cuda:0") 
    parser.add_argument("--momentum-sgd", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0) 
    parser.add_argument("--datapath", default="../../debiasing_surgeon/data/")
    parser.add_argument("--dataset", default="Bmnist")
    parser.add_argument("--cifar10c_percent", default="0.5")
    parser.add_argument("--load_model", default=1, type=int)
    parser.add_argument("--load_PH", default=0, type=int)
    parser.add_argument("--load_gating_weights", default=0, type=int)
    parser.add_argument("--wandb", default=1, type=int)
    parser.add_argument("--max_PH_epochs", default=200, type=int)
    parser.add_argument("--PH_patience", default=5, type=int)
    parser.add_argument("--max_gate_epochs", default=200, type=int)
    parser.add_argument("--gate_patience", default=5, type=int)
    parser.add_argument("--stopping_criterion", default="min_temperature")
    parser.add_argument("--ver", default="work")
    args = parser.parse_args()
    if args.dataset == "MultiColorMNIST":
        args.nb_bias = 2
    else:
        args.nb_bias = 1
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.device != torch.device("cpu"):
        torch.cuda.set_device(args.device)
    return args
