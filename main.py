# File:                     main.py
# Created by:               Ivan Luiz De Moura Matos and Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2024/7/18
# Debiasing Surgeon : Fantastic weights and how to find them, ECCV 2024
# ==========================================================================


from __future__ import print_function
import torch
from config import init_config
from datasets.celebA import *
from gating_weights import train_gating_weights_based_on_MI_double
from irene.utilities import *
from irene.core import *

from datetime import datetime
import pickle  # see: https://pynative.com/python-save-dictionary-to-file/
import copy
import wandb
from models import init_models
from tools import *
from training_functions import * 
from dataloaders import get_dataloaders_from_args


args = init_config()
log_on_wandb = args.wandb == 1
load_model = args.load_model == 1
load_PH = args.load_PH == 1
load_gating_weights = args.load_gating_weights == 1

wandb_name = get_wandb_name(args)
wandb_project = f"runs_{args.ver}_{args.dataset}" 

if log_on_wandb:
    wandb.init(config=args,
        name=wandb_name + "_AT_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        project=wandb_project)
(
    biased_train_loader,
    original_unbiased_test_loader,
    unbiased_train_loader,
    unbiased_val_loader,
    unbiased_test_loader,
    original_unbiased_val_dataloader,
) = get_dataloaders_from_args(args)
dls = {
    "unbiased_training": unbiased_train_loader,
    "unbiased_validation": unbiased_val_loader,
    "unbiased_test": unbiased_test_loader,
    "biased_training": biased_train_loader}
model = init_models(args, pretrained=False)

path_to_model = get_model_name(args)
if args.load_model == 1 and os.path.exists(path_to_model):
    print("Loading model")
    model.load_state_dict(torch.load(path_to_model, map_location=args.device))
    model = model.to(args.device)
else:
    for epoch in range(1, args.epochs + 1):
        print("Epoch: {} / {}".format(epoch, args.epochs))

        train_vanilla_model(model, args, dls["biased_training"])

        with torch.no_grad():
            if args.dataset in ['Bmnist', 'MultiColorMNIST', 'cifar10c', 'waterbirds']:
                args.sched.step()
            elif args.dataset == "celebA":
                args.sched.step(test_vanilla(model, args, original_unbiased_val_dataloader))

        if args.dataset == "celebA":
            if args.optimizer.param_groups[0]["lr"] < 0.001:
                break

    save_biased_model(model, args)

model_aux = copy.deepcopy(model)
model_aux = model_aux.to(args.device)
print("Initializing Privacy Head")
init_PH(args, model_aux, dls, load_PH)

path_to_gating_weights = get_path_to_gating_weights(args)
if load_gating_weights and os.path.exists(path_to_gating_weights):
    with open(
        path_to_gating_weights,"rb",
    ) as file:
        gating_weights = pickle.load(file)
else:
    gating_weights, best_intermediate_gating_weights = (
        train_gating_weights_based_on_MI_double(
            model,args,dls,
            temp_sched_patience=args.gate_patience,))


prune_and_evaluate_model(model, gating_weights, 
                         args, model_type="Final pruned model", dls=dls)

if best_intermediate_gating_weights is not None:
    prune_and_evaluate_model(model, best_intermediate_gating_weights, 
                             args, model_type="Best pruned model", dls=dls)
    