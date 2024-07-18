# File:                     models.py
# Created by:               Ivan Luiz De Moura Matos and Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2024/7/18
# Debiasing Surgeon : Fantastic weights and how to find them, ECCV 2024
# ==========================================================================



from datasets.multi_color_mnist import MLP
from datasets.biased_mnist import SimpleConvNet
import torch
import torchvision
import torch.nn as nn
from irene.core import MI, Privacy_head
import copy

BMNIST_LIST = ["biased-mnist", "Bmnist"]
CELEBA_LIST = ["celeba", "celebA", "CelebA"]


from datasets.multi_color_mnist import MLP


def init_models(args, pretrained=False):

    # ====== MODELS ======

    if args.dataset in BMNIST_LIST:
        num_classes = 10
        num_features = 128
        model = SimpleConvNet(num_classes=num_classes).to(args.device)

    elif args.dataset in CELEBA_LIST:
        if pretrained:
            weights = "IMAGENET1K_V1"
        else:
            weights = None
        num_classes = 2
        num_features = 512
        model = torchvision.models.resnet18(weights=weights).to(args.device)
        model.fc = nn.Linear(
            in_features=num_features, out_features=num_classes, bias=True
        ).to(args.device)

    elif args.dataset == "MultiColorMNIST":
        num_classes = 10
        num_features = 100
        model = MLP(num_class=num_classes).to(args.device)

    elif args.dataset in ["cifar10c"]:
        if pretrained:
            weights = "IMAGENET1K_V1"
        else:
            weights = None
        num_classes = 10
        num_features = 512
        model = torchvision.models.resnet18(weights=weights).to(args.device)
        model.fc = nn.Linear(
            in_features=num_features, out_features=num_classes, bias=True
        ).to(args.device)

    if args.dataset != "MultiColorMNIST":
        model.avgpool = nn.Sequential(model.avgpool, nn.Identity().to(args.device))
    args.criterion = nn.CrossEntropyLoss(reduction="mean").to(args.device)

    # ====== Privacy Head ======

    # IVAN: We copy the original model, to obtain an "auxiliary model", just as a placeholder
    # to be used during the creation of the Privacy Head
    model_placeholder = copy.deepcopy(model)
    model_placeholder = model_placeholder.to(args.device)
    # If we attach the P.H. to 'model', we cannot perform a deepcopy of it, afterwards.
    # This is why I am creating a 'model_aux', which will be effectively used when training the P.H.

    args.PH = [
        Privacy_head(
            model_placeholder.avgpool,
            nn.Sequential(nn.Linear(num_features, num_classes)),
        ).to(args.device)
        for i in range(args.nb_bias)
    ]
    args.MI = [
        MI(device=args.device, privates=num_classes) for i in range(args.nb_bias)
    ]
    args.PH_optimizer = [
        torch.optim.SGD(
            args.PH[i].parameters(),
            lr=args.lr_priv,
            momentum=args.momentum_sgd,
            weight_decay=args.weight_decay,
        )
        for i in range(args.nb_bias)
    ]

    # ====== OPTIMIZERS ======

    if args.dataset in CELEBA_LIST:
        args.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum_sgd,
            weight_decay=args.weight_decay,
        )
        args.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            args.optimizer, mode="min", factor=0.1, patience=10, threshold=0, cooldown=5
        )
    elif args.dataset in BMNIST_LIST:
        args.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum_sgd,
            weight_decay=args.weight_decay,
        )
        args.sched = torch.optim.lr_scheduler.MultiStepLR(
            args.optimizer, milestones=[40, 60], gamma=0.1, verbose=True
        )
    elif args.dataset == "MultiColorMNIST":
        if args.epochs == 80:
            args.epochs = 100
        args.lr = 0.001
        args.wd = 1e-4
        args.batch_size = 256
        args.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
        )
        args.sched = torch.optim.lr_scheduler.MultiStepLR(
            args.optimizer, milestones=[args.epochs + 1], gamma=0.1, verbose=True
        )
    elif args.dataset == "cifar10c":
        if args.epochs == 80:
            args.epochs = 200
        args.lr = 0.001
        args.wd = 1e-4
        args.batch_size = 256
        args.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
        )
        args.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            args.optimizer, T_max=args.epochs
        )
    return model
