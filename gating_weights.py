# File:                     gating_weights.py
# Created by:               Ivan Luiz De Moura Matos and Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2024/7/18
# Debiasing Surgeon : Fantastic weights and how to find them, ECCV 2024
# ==========================================================================


import copy
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from irene.core import Privacy_head

from irene.utilities import accuracy
from filenames import *
from tools import *
from gating_weights import *
import torch.nn as nn

from tools import (
    get_path_to_gating_weights,
    os,
    print_performance,
    torch,
)
from training_functions import *


def create_gating_weights(model, device):
    """
    Function to create and return a dictionary where the keys are the names of the original model's layers, and the values
    are the corresponding "gating weights" (which are initiliazed as tensors of zeroes).
    """
    gating_weights = {}

    for name, param_tensor in model.named_parameters():

        if ".weight" in name:
            layer_name = name.split(".weight")[0]
            if not issubclass(
                type(attrgetter(layer_name)(model)),
                torch.nn.modules.batchnorm._BatchNorm,
            ):
                gating_weights[layer_name] = torch.zeros(
                    param_tensor.size(), requires_grad=True, device=device
                )

    return gating_weights


def apply_gating(modif_model, orig_model, gating_weights, temperature=1.0):
    """
    Function to "apply the gating" to the auxiliary (modified) model, by reassigning its weights as a product between
    the original model's weights and the "clipped" 2*sigmoid() applied to the gating weights:
    modif_model's weights = (orig_model's weights) * [clipped 2*sigmoid(gating_weights)]
    """

    for layer_name in gating_weights.keys():

        modif_layer = attrgetter(layer_name)(modif_model)
        orig_layer = attrgetter(layer_name)(orig_model)
        # Procedure to make the weight parameter a non-leaf node, and ensure that the gradient w.r.t. the
        # gating weights are automatically computed and stored during backpropagation
        del modif_layer.weight

        modif_layer.weight = orig_layer.weight.data * TwoSigmoid.apply(
            gating_weights[layer_name] / temperature
        )


# ================= FOR STRUCTURED PRUNING =================
# Defining an "output gating" layer


class StructuredGate(nn.Module):
    def __init__(self, size, temperature=1.0):
        super().__init__()
        self.gating_weights = nn.Parameter(torch.zeros(size, requires_grad=True))
        self.temperature = temperature

    def forward(self, input):
        return input * TwoSigmoid.apply(self.gating_weights / self.temperature)


def init_gating_weights_training(orig_model, args):
    args.path_to_save_gating_weights = get_path_to_gating_weights(args)
    if not os.path.exists(args.path_to_save_gating_weights):
        os.mkdir(args.path_to_save_gating_weights)

    # Creating the model whose parameters will be modified, through the training of the gating weights
    modif_model = copy.deepcopy(orig_model)
    modif_model = modif_model.to(device=args.device)
    if args.dataset == "MultiColorMNIST":
        new_PHs = [
            Privacy_head(
                modif_model.extracter, copy.deepcopy(args.PH[i].classifier)
            ).to(args.device)
            for i in range(args.nb_bias)
        ]
    else:
        new_PHs = [
            Privacy_head(modif_model.avgpool, copy.deepcopy(args.PH[i].classifier)).to(
                args.device
            )
            for i in range(args.nb_bias)
        ]
    new_PHs_optimizers = [
        torch.optim.SGD(
            new_PHs[i].parameters(),
            lr=args.lr_priv,
            momentum=args.momentum_sgd,
            weight_decay=args.weight_decay,
        )
        for i in range(args.nb_bias)
    ]

    # Creating the gating weights: We will use a dictionary where the "keys" are the names of the layers we want to prune,
    # and the corresponding "values" are the gating weights tensors (initialized as tensors full of zeroes)
    gating_weights = create_gating_weights(modif_model, device=args.device)

    # The parameters (tensors) that will be updated after each optimizer step are the gating weights
    parameters_to_update = gating_weights.values()

    optimizer = torch.optim.Adam(parameters_to_update, weight_decay=args.weight_decay)
    return (
        modif_model,
        new_PHs,
        new_PHs_optimizers,
        optimizer,
        gating_weights,
    )

 


def train_gating_weights_based_on_MI_double(
    orig_model,
    args,
    dls,
    refine_PH=True,
    temp_sched_policy="based_on_MI",  # OPTIONS: 'based_on_MI' OR 'periodic'
    temp_sched_factor=0.1,
    temp_sched_patience=5,
    temp_sched_threshold=5e-2,
    n_epochs_max=400,
    min_temperature=1e-8,
):
    """
    Train and return gating weights associated to the original model. Those gating weights can be used afterwards to prune the model,
    by using them with the specific pruning function created.
    The gating weights are trained in order to minimize the mutual information (regarding the private task), as well as the
    loss of the target task.

    NOTE: the model passed as parameter (orig_model) is NOT modified in this function, since we perform a deepcopy of it when
    creating the auxiliary model (modif_model) which will be effectively modified ('gated') during the training of the gating weights.
    """
    (
        modif_model,
        new_PHs,
        new_PHs_optimizers,
        optimizer,
        gating_weights,
    ) = init_gating_weights_training(orig_model, args)

    print(
        "Method: learning gating weights (using a balanced training set) to guide the pruning"
    )
    print_ds(args)
    temperature = 1.0

    # Setting a "temperature scheduler"
    (
        temp_scheduler,
        stop_flag,
        BEST_private_acc_val_pruned,
        best_intermediate_gating_weights,
    ) = init_stopping_criterions(
        temp_sched_policy,
        temperature,
        temp_sched_factor,
        temp_sched_patience,
        temp_sched_threshold,
        args,
    )

    modif_model.train(False)

    # Evaluating the model (and associated P.H.) before training the gating weights
    print("***** Initial evaluation (before training the gating weights) *****")
    for subset in dls.keys():
        metrics = run_model(
            modif_model,
            args,
            new_PHs,
            dls[subset],
            device=args.device,
            prefix=f"gated/{subset}",
        )
        print_performance(metrics, model_type="gated", subset=subset, args=args)

    apply_gating(modif_model, orig_model, gating_weights, temperature)

    # Training loop: Learning the gating weights
    for epoch in range(args.max_gate_epochs):

        print(
            "=====================================================================================\n"
        )
        print('**** TRAINING GATING WEIGHTS AND P.H. (similarly to "IRENE") ****')

        # *********************************************************************************
        # ***** Loop to train the gating weights and the P.H. similarly as in "IRENE" *****
        train_gating_weights_epoch(
            modif_model,
            epoch,
            optimizer,
            new_PHs,
            dls,
            new_PHs_optimizers,
            orig_model,
            gating_weights,
            temperature,
            args,
        )

        # After each "IRENE" epoch, we refine (i.e. train more) the P.H., to adapt it to the current gated model
        print("**** REFINING THE PRIVACY HEAD ****")
        train_PH_double(
            model=modif_model,
            args=args,
            PHs=new_PHs,
            PHs_optimizers=new_PHs_optimizers,
            train_loader=dls["unbiased_training"],
            n_epochs_max=args.max_PH_epochs,
        )

        print("**** EVALUATION AFTER REFINING THE P.H. ****")
        metrics = {"gated": {}, "pruned": {}}
        for subset in dls.keys():
            metrics["gated"][subset] = run_model(
                modif_model,
                args,
                new_PHs,
                dls[subset],
                device=args.device,
                prefix=f"gated/{subset}",
            )

            model_type = "gated"
            print_performance(metrics[model_type][subset], model_type, subset, args)
        avg_MI = np.mean(
            [
                metrics["gated"]["unbiased_validation"]["MI_tot"][i].avg
                for i in range(args.nb_bias)
            ]
        )
        temperature, temp_update_flag = step_temperature(
            temp_sched_policy,
            temp_scheduler,
            avg_MI,
        )
        # ********************************************************

        # *** EACH TIME THE TEMPERATURE IS UPDATED: ***
        # - We evaluate an "intermediate pruned model", and store the corresponding gating weights, if improvement is verified.
        # - We check if the stopping criterion is satisfied. If not, we apply the NEW temperature on the gated model, refine the P.H.,
        #   and continue the training.

        if temp_update_flag:

            # ************************************************************************************
            # ********************* EVALUATION OF "INTERMEDIATE PRUNED MODEL" ********************

            ### We evaluate what would be the performance of the pruned model, if we stopped the method at this "intermediate" point.
            ### We store the "intermediate" gating weights, if the pruned model is associated to a lower private accuracy than the previous one.

            # Creating an auxiliary pruned model
            aux_pruned_model = copy.deepcopy(orig_model)
            aux_pruned_model = aux_pruned_model.to(device=args.device)
            prune_based_on_gating_weights(
                aux_pruned_model, gating_weights
            )  # Performing the pruning
            if args.dataset == "MultiColorMNIST":
                PHs_for_pruned_model = [
                    Privacy_head(
                        aux_pruned_model.extracter, copy.deepcopy(args.PH[i].classifier)
                    ).to(args.device)
                    for i in range(args.nb_bias)
                ]
            else:
                PHs_for_pruned_model = [
                    Privacy_head(
                        aux_pruned_model.avgpool, copy.deepcopy(args.PH[i].classifier)
                    ).to(args.device)
                    for i in range(args.nb_bias)
                ]

            PHs_optimizers_for_pruned_model = [
                torch.optim.SGD(
                    PHs_for_pruned_model[i].parameters(),
                    lr=args.lr_priv,
                    momentum=args.momentum_sgd,
                    weight_decay=args.weight_decay,
                )
                for i in range(len(PHs_for_pruned_model))
            ]

            ##### STORING the intermediate gating weights (added on v22) #####
            with open(
                f"{args.path_to_save_gating_weights}/epoch_{epoch}_ intermediate_gating_weights.pkl",
                "wb",
            ) as file:
                pickle.dump(gating_weights, file)

            print(
                f"SPARSITY OF INTERMEDIATE PRUNED MODEL:{get_sparsity(aux_pruned_model)}"
            )
            print("**** Training a P.H. for PRUNED model ****")
            train_PH_double(
                model=aux_pruned_model,
                args=args,
                PHs=PHs_for_pruned_model,
                PHs_optimizers=PHs_optimizers_for_pruned_model,
                train_loader=dls["unbiased_training"],
                n_epochs_max=args.max_PH_epochs,
            )

            for subset in ["unbiased_validation", "unbiased_test"]:
                metrics["pruned"][subset] = run_model(
                    aux_pruned_model,
                    args,
                    PHs_for_pruned_model,
                    dls[subset],
                    device=args.device,
                    prefix=f"gated/{subset}",
                )

            # Computing the task loss, task accuracy and private accuracy for the pruned model and for the non-pruned but modified model, using VALIDATION DATA

            print(
                "**** Comparing the PRUNED and the GATED models (on VALIDATION data) ****"
            )
            for model_type in ["gated", "pruned"]:
                subset = "unbiased_validation"
                print_performance(metrics[model_type][subset], model_type, subset, args)

            print("**** Evaluating the PRUNED model on TEST data ****")
            print_performance(
                metrics["pruned"]["unbiased_test"],
                model_type="pruned",
                subset="unbiased_test",
                args=args,
            )

            # Saving intermediate gating weights, if pruned model is better
            avg_private_acc_val_pruned = np.mean(
                [
                    metrics["pruned"]["unbiased_validation"]["private_acc"][i].avg
                    for i in range(args.nb_bias)
                ]
            )
            avg_BEST_private_acc_val_pruned = np.mean(BEST_private_acc_val_pruned)
            if avg_private_acc_val_pruned < avg_BEST_private_acc_val_pruned:
                print("\n ==========> BETTER PRUNED MODEL!")
                BEST_private_acc_val_pruned = [
                    metrics["pruned"]["unbiased_validation"]["private_acc"][i].avg
                    for i in range(args.nb_bias)
                ]
                best_intermediate_gating_weights = copy.deepcopy(gating_weights)

            if args.stopping_criterion == "min_temperature":

                if temperature < min_temperature:
                    print(
                        "\nTemperature dropped below the minimum specified. Stop training the gating weights.\n"
                    )

                    stop_flag = True
            del aux_pruned_model
            # If the stopping criterion wasn't satisfied yet: We update gated model, considering the NEW temperature
            apply_gating(modif_model, orig_model, gating_weights, temperature)

            # If we don't stop the training, we must refine again the P.H.,
            # to adapt it to the gated model with the NEW temperature
            if not stop_flag:

                print("**** REFINING THE PRIVACY HEAD after temperature update ****")
                train_PH_double(
                    model=modif_model,
                    args=args,
                    PHs=new_PHs,
                    PHs_optimizers=new_PHs_optimizers,
                    train_loader=dls["unbiased_training"],
                    n_epochs_max=args.max_PH_epochs,
                )

                print(
                    "**** EVALUATION AFTER REFINING THE P.H. (with updated temperature) ****"
                )
                for subset in [
                    "unbiased_training",
                    "unbiased_validation",
                    "unbiased_test",
                ]:
                    metrics["gated"][subset] = run_model(
                        modif_model,
                        args,
                        new_PHs,
                        dls[subset],
                        device=args.device,
                        prefix=f"gated/{subset}",
                    )
                    print_performance(metrics["gated"][subset], "gated", subset, args)

        if (
            stop_flag
        ):  # End training of gating weights, if stopping criterion is reached
            break

    del modif_model
    torch.cuda.empty_cache()
    # We return the final gating weights obtained ('gating_weights'), as well as the "best intermediate gating weights" ('best_intermediate_gating_weights')
    return gating_weights, best_intermediate_gating_weights


def train_gating_weights_epoch(
    modif_model,
    epoch,
    optimizer,
    new_PHs,
    dls,
    new_PHs_optimizers,
    orig_model,
    gating_weights,
    temperature,
    args,
):
    print(f"Training GW for epoch {epoch}")
    metrics = init_eval_metrics(args)
    batch_metrics = {}
    tk0 = tqdm(
        dls["unbiased_training"], total=int(len(dls["unbiased_training"])), leave=True
    )
    for _, (data, labels) in enumerate(tk0):

        data = data.to(args.device)
        target = labels[0].to(args.device)
        private_labels = [labels[1 + i].to(args.device) for i in range(args.nb_bias)]

        # Applying the gating weights to the original parameters of the model   (original weights * gating weights)
        # apply_gating(modif_model, orig_model, gating_weights, temperature)

        # Forward propagation: pass the samples through the network
        output = modif_model(data)

        # Output of Privacy Head (P.H.)
        output_private = [new_PHs[i]() for i in range(args.nb_bias)]

        # Calculate the task loss
        batch_metrics = {
            "loss_task": args.criterion(output, target),
            "loss_private": [
                args.criterion(output_private[i], private_labels[i])
                for i in range(args.nb_bias)
            ],
            "MI": [
                args.MI[i](new_PHs[i], private_labels[i]) for i in range(args.nb_bias)
            ],
            "acc1": accuracy(output, target, topk=(1,)),
        }

        # Defining the function we want to MINIMIZE (composed by the task loss and the M.I.)
        objective_function = args.alpha * batch_metrics[
            "loss_task"
        ] + args.gamma * torch.mean(
            torch.stack(batch_metrics["MI"])
        )

        ### Updating the gating weights, as well as the P.H. (in parallel!) similarly to the "IRENE" approach ###

        # Backpropagation of the objective function
        objective_function.backward()

        # We zero the gradients in the P.H., since we don't want the M.I. to be used to update the PH! <-- This step is crucial.
        for i in range(args.nb_bias):
            new_PHs_optimizers[i].zero_grad()

        # Backpropagation of the private loss
        for i in range(args.nb_bias):
            batch_metrics["loss_private"][i].backward()


        # Updating the gating weights
        optimizer.step()
        optimizer.zero_grad()

        # Updating the Privacy Head
        for i in range(args.nb_bias):
            new_PHs_optimizers[i].step()
            new_PHs_optimizers[i].zero_grad()

        # Updating the "gated model", by using the updated gating weights
        apply_gating(modif_model, orig_model, gating_weights, temperature)
        batch_metrics["private_acc"] = [
            accuracy(output_private[i], private_labels[i], topk=(1,))[0]
            for i in range(args.nb_bias)
        ]

        metrics = update_eval_metrics(
            metrics, batch_metrics, args, update_type=["task", "private"]
        )
        
        tk0.set_postfix(
            p_loss=metrics["private_acc"][i].avg, t_loss=metrics["top1"].avg
        )

    print_performance(metrics, model_type="gated", subset="unbiased_train", args=args)
