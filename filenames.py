# File:                     filenames.py
# Created by:               Ivan Luiz De Moura Matos and Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2024/7/18
# Debiasing Surgeon : Fantastic weights and how to find them, ECCV 2024
# ==========================================================================



import torch
import os


def get_dataset_info(args):
    if args.dataset == "Bmnist":
        suppl = f"rho{args.rho}"
    elif args.dataset == "celebA":
        suppl = args.target_celeba
    elif args.dataset == "cifar10c":
        suppl = f"percent{args.cifar10c_percent}"
    elif args.dataset == "MultiColorMNIST":
        suppl = "left_0.01_right_0.05_alpha_1.0"
    return suppl


def get_model_folder(args):
    path = (os.path.join(args.datapath, args.ver))
    if not os.path.exists(path):
        os.mkdir(path)
    full_path = os.path.join(
        path,
        "{0}_{1}_alpha{4}_gamma{5}_seed{6}".format(
            args.dataset,
            get_dataset_info(args),
            args.alpha,
            args.gamma,
            args.seed,))
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    return full_path

def get_wandb_name(args):
    wandb_name = "run_{0}_{1}_{2}_{3}_alpha{4}_gamma{5}_seed{6}".format(
        args.ver,
        args.dataset,
        get_dataset_info(args),
        args.alpha,
        args.gamma,
        args.seed,
    )
    return wandb_name


def get_model_name(args, epoch=None, biased=True):
    if epoch is None:
        epoch = args.epochs
    if biased:
        model_type = "BIASED"
    else:
        model_type = "PRUNED"
    model_name = "epoch{0}_{1}.pth".format(epoch, model_type)
    return os.path.join(get_model_folder(args),model_name)


def save_biased_model(model, args, epoch=None):
    torch.save(model.state_dict(), get_model_name(args, epoch, biased=True))


def get_path_to_gating_weights(args):
    return os.path.join(get_model_folder(args), "gating_weights")




def get_PH_file_name(args):
    PH_path = os.path.join(get_model_folder(args), "classifierPH")
    return [f"{PH_path}{i}.pt" for i in range(args.nb_bias)]


def save_PH_classifier(args):
    for i in range(args.nb_bias):
        torch.save(args.PH[i].classifier, get_PH_file_name(args)[i])

