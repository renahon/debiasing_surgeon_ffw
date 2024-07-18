# File:                     dataloaders.py
# Created by:               Ivan Luiz De Moura Matos and Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2024/7/18
# Debiasing Surgeon : Fantastic weights and how to find them, ECCV 2024
# ==========================================================================



from __future__ import print_function
import torch
from torchvision import transforms
from datasets.biased_mnist import ColourBiasedMNIST
from datasets.celebA import *
from irene.utilities import *
from irene.core import *

from datasets.cifar10c import CorruptedCIFAR10
from datasets.multi_color_mnist import MultiColorMNIST


def get_dataloaders(
    dataset_name,
    seed,
    unbiased_proportions,
    batch_size,
    root_data_path,
    augment_data=True,
    suppl={},
):
    """
    Function that loads the different dataloader for any of the supported datasets (other than Biased-MNIST and CelebA)
    unbiased_proportions:
     - [train_prop, val_prop, test_prop] (list of floats) if validation unavailable (dataset in [multicolormnist, cifar10c])
    """
    dataloaders = {"unbiased_training": {}, "biased_training": {}}
    datasets, val_ds_exists = get_original_datasets(
        dataset_name, seed, augment_data, root_data_path, suppl
    )

    # Variation on the existence of a validation set in the original dataset
    if val_ds_exists:
        # If validation exists, we split it into unbiased train and unbiased val
        biased_training_splits = ["train", "test", "val"]

        nb_elts = len(datasets["biased_training"]["val"])
        train_prop = int(unbiased_proportions * nb_elts)

        datasets["unbiased_training"]["train"], datasets["unbiased_training"]["val"] = (
            torch.utils.data.random_split(
                datasets["biased_training"]["val"],
                [train_prop, nb_elts - train_prop],
            )
        )
        datasets["unbiased_training"]["test"] = datasets["biased_training"]["test"]

    else:
        # Otherwise we split the test set in 3 unbiased subsets
        biased_training_splits = ["train", "test"]
        nb_elts = len(datasets["biased_training"]["test"])
        train_prop = int(unbiased_proportions[0] * nb_elts)
        val_prop = int(unbiased_proportions[1] * nb_elts)
        test_prop = nb_elts - train_prop - val_prop
        (
            datasets["unbiased_training"]["train"],
            datasets["unbiased_training"]["test"],
            datasets["unbiased_training"]["val"],
        ) = torch.utils.data.random_split(
            datasets["biased_training"]["test"],
            [train_prop, test_prop, val_prop],
        )

    for split in biased_training_splits:
        dataloaders["biased_training"][split] = torch.utils.data.DataLoader(
            dataset=datasets["biased_training"][split],
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    for split in ["train", "val", "test"]:
        dataloaders["unbiased_training"][split] = torch.utils.data.DataLoader(
            dataset=datasets["unbiased_training"][split],
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    return dataloaders


def get_original_datasets(dataset_name, seed, augment_data, root_data_path, suppl={}):
    datasets = {"unbiased_training": {}, "biased_training": {}}
    val_ds_exists = False
    if dataset_name == "biased-mnist":
        pass
    elif dataset_name == "CelebA":
        pass
    elif dataset_name == "cifar10c":
        splits = ["train", "test"]
        if "percent" in suppl.keys():
            percent = suppl["percent"]
        else:
            percent = "0.5"
        if augment_data:
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            T = {
                "train": transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                ),
                "test": transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(mean, std)]
                ),
            }
        else:
            T = {"train": transforms.ToTensor(), "test": transforms.ToTensor()}

        for split in splits:
            datasets["biased_training"][split] = CorruptedCIFAR10(
                root=root_data_path,
                split=split,
                percent=f"{percent}pct",
                transform=T[split],
            )
    elif dataset_name == "MultiColorMNIST":
        for split in ["train", "test"]:
            datasets["biased_training"][split] = MultiColorMNIST(
                root=root_data_path,
                split=split,
                left_color_skew=0.01,
                right_color_skew=0.05,
                severity=4,
            )

    return datasets, val_ds_exists


def get_dataloaders_from_args(args):
    original_unbiased_val_dataloader = None
    if args.dataset == "Bmnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        biased_train_dataset = ColourBiasedMNIST(
            args.datapath + "MNIST/",
            train=True,
            download=True,
            data_label_correlation=args.rho,
            n_confusing_labels=9,
            transform=transform,
        )

        unbiased_dataset = ColourBiasedMNIST(
            args.datapath + "MNIST/",
            train=True,
            download=True,
            data_label_correlation=0.1,
            n_confusing_labels=9,
            transform=transform,
        )
        # Splitting the unbiased dataset into training data and validation data
        unbiased_train_dataset, unbiased_val_dataset, _ = torch.utils.data.random_split(
            unbiased_dataset, [10000, 10000, 40000]
        )  # "SMALL DATASET"

        unbiased_test_dataset = ColourBiasedMNIST(
            args.datapath + "MNIST/",
            train=False,
            download=True,
            data_label_correlation=0.1,
            n_confusing_labels=9,
            transform=transform,
        )

        biased_train_loader = torch.utils.data.DataLoader(
            dataset=biased_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        unbiased_train_loader = torch.utils.data.DataLoader(
            dataset=unbiased_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        unbiased_val_loader = torch.utils.data.DataLoader(
            dataset=unbiased_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        unbiased_test_loader = torch.utils.data.DataLoader(
            dataset=unbiased_test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        original_unbiased_test_loader = unbiased_test_loader

    elif args.dataset == "celebA":

        biased_train_dataset = CelebA(
            args.datapath,
            split="train",
            target=args.target_celeba,
            bias_attr="Male",
            unbiased=True,
        )

        biased_train_loader = torch.utils.data.DataLoader(
            dataset=biased_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        # unbiased_val_dataset = CelebA(args.datapath, split='valid', target=args.target_celeba, bias_attr='Male', unbiased=True)

        original_unbiased_val_dataset = CelebA(
            args.datapath,
            split="valid",
            target=args.target_celeba,
            bias_attr="Male",
            unbiased=True,
        )  # Used INTEGRALLY to validate the model during its training

        unbiased_train_dataset, unbiased_val_dataset = torch.utils.data.random_split(
            original_unbiased_val_dataset, [0.6, 0.4]
        )

        original_unbiased_val_dataloader = torch.utils.data.DataLoader(
            dataset=original_unbiased_val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        unbiased_train_loader = torch.utils.data.DataLoader(
            dataset=unbiased_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        unbiased_val_loader = torch.utils.data.DataLoader(
            dataset=unbiased_val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        original_unbiased_val_dataloader = unbiased_val_loader

        # unbiased_train_test_dataset = CelebA(args.datapath, split='test', target=args.target_celeba, bias_attr='Male', unbiased=True)
        unbiased_test_dataset = CelebA(
            args.datapath,
            split="test",
            target=args.target_celeba,
            bias_attr="Male",
            unbiased=True,
        )

        # unbiased_train_dataset, unbiased_test_dataset = torch.utils.data.random_split(unbiased_train_test_dataset, [0.6, 0.4])

        unbiased_test_loader = torch.utils.data.DataLoader(
            dataset=unbiased_test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        original_unbiased_test_loader = unbiased_test_loader

        # model = torchvision.models.resnet18(pretrained=False).to(args.device)
        # # model.avgpool = nn.Sequential(model.avgpool, torch.nn.Identity().to(args.device))   # we do it in order to use the hooks in the bottleneck
        # # args.PH = Privacy_head(model.avgpool, nn.Sequential(torch.nn.Linear(512, 2))).to(args.device)
        # # args.MI = MI(device = args.device, privates=2)
        # model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True).to(args.device)

    else:
        if args.dataset in ["MultiColorMNIST", "cifar10c"]:
            unbiased_proportions = [0.6, 0.2, 0.2]  # train_prop, val_prop, test_prop
            if args.dataset == "cifar10c":
                suppl = {"percent": args.cifar10c_percent}
            else:
                suppl = {}
        dataloaders = get_dataloaders(
            args.dataset,
            args.seed,
            unbiased_proportions,
            args.batch_size,
            args.datapath,
            suppl=suppl,
        )

        ##### IMPORTANT REMARKS (about terminology) #####
        # REMARK 1: Below, an "original" dataloader means "before (potential) split"
        # REMARK 2: In my naming for the dataloaders, "biased" and "unbiased" refers to the nature of
        # the data itself.

        biased_train_loader = dataloaders["biased_training"]["train"]
        original_unbiased_test_loader = dataloaders["biased_training"]["test"]
        if args.dataset == "waterbirds":
            original_unbiased_val_dataloader = dataloaders["biased_training"]["val"]

        unbiased_train_loader = dataloaders["unbiased_training"]["train"]
        unbiased_val_loader = dataloaders["unbiased_training"]["val"]
        unbiased_test_loader = dataloaders["unbiased_training"]["test"]
    return (
        biased_train_loader,
        original_unbiased_test_loader,
        unbiased_train_loader,
        unbiased_val_loader,
        unbiased_test_loader,
        original_unbiased_val_dataloader,
    )
