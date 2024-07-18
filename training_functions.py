# File:                     training_functions.py
# Created by:               Ivan Luiz De Moura Matos and Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2024/7/18
# Debiasing Surgeon : Fantastic weights and how to find them, ECCV 2024
# ==========================================================================

import copy
import os
import numpy as np
import torch
from tqdm import tqdm
import wandb
from irene.core import Privacy_head
from irene.utilities import AverageMeter, Hook, accuracy
from filenames import *
from tools import *
import torch.nn.utils.prune as prune



def init_PH(args, model_aux, dls, load_PH):
    for i in range(args.nb_bias):
        if args.dataset == "MultiColorMNIST":
            args.PH[i].bottleneck = Hook(model_aux.extracter, backward=False)
        else:
            args.PH[i].bottleneck = Hook(model_aux.avgpool, backward=False)
    path_to_PH = get_PH_file_name(args)
    if load_PH and os.path.exists(path_to_PH):
        classifierPH = torch.load(path_to_PH).to(args.device)
        args.PH.classifier = classifierPH
    else:
        train_PH_double(
            model=model_aux,
            args=args,
            PHs=args.PH,
            PHs_optimizers=args.PH_optimizer,
            train_loader=dls["unbiased_training"],
            n_epochs_max=args.max_PH_epochs,
        )
        save_PH_classifier(args=args)


def train_vanilla_model(model, args, train_loader):
    model.train(True)
    loss_task_tot = AverageMeter("Loss:.4e")
    if args.dataset == "MultiColorMNIST":
        top1 = {
            "L_R": AverageMeter("Acc@1", ":6.2f"),
            "L_notR": AverageMeter("Acc@1", ":6.2f"),
            "notL_R": AverageMeter("Acc@1", ":6.2f"),
            "notL_notR": AverageMeter("Acc@1", ":6.2f"),
        }
        elements = {
            "L_R": 0,
            "L_notR": 0,
            "notL_R": 0,
            "notL_notR": 0,
        }
    else:
        top1 = AverageMeter("Acc@1", ":6.2f")
    tk0 = tqdm(
        train_loader, total=int(len(train_loader)), leave=True, dynamic_ncols=True
    )

    for _, (data, labels) in enumerate(tk0):
        data = data.to(args.device)
        target = labels[0].to(args.device)

        output = model(data)

        loss_task = args.criterion(output, target)
        loss_task_tot.update(loss_task.item(), data.size(0))

        loss_task.backward()

        args.optimizer.step()
        args.optimizer.zero_grad()

        if args.dataset == "MultiColorMNIST":
            bias1 = labels[1].to(args.device)
            bias2 = labels[2].to(args.device)

            top1 = update_accuracy_multicolormnist(
                output,
                target,
                bias1,
                bias2,
                top1,
            )
            elements["L_R"] += torch.sum((bias1 == target) * (bias1 == target))
            elements["L_notR"] += torch.sum(
                (bias1 == target) * (torch.logical_not(bias2 == target))
            )
            elements["notL_R"] += torch.sum(
                (torch.logical_not(bias1 == target)) * (bias2 == target)
            )
            elements["notL_notR"] += torch.sum(
                (torch.logical_not(bias1 == target))
                * (torch.logical_not(bias2 == target))
            )

            tk0.set_postfix(
                loss_task=loss_task_tot.avg,
                top1=(
                    np.round(top1["L_R"].avg, 2),
                    np.round(top1["L_notR"].avg, 2),
                    np.round(top1["notL_R"].avg, 2),
                    np.round(top1["notL_notR"].avg, 2),
                ),
                elts=(
                    elements["L_R"].item(),
                    elements["L_notR"].item(),
                    elements["notL_R"].item(),
                    elements["notL_notR"].item(),
                ),
            )

        else:
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0], data.size(0))
            tk0.set_postfix(loss_task=loss_task_tot.avg, top1=top1.avg.item())
    if args.wandb == 1:
        wandb.log(
            {
                "vanilla/train_acc": top1.avg.item(),
                "vanilla/train_loss": loss_task_tot.avg,
            }
        )


def test_vanilla(model, args, val_loader):
    model.eval()
    loss_task_tot = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    tk0 = tqdm(val_loader, total=int(len(val_loader)), leave=True)
    for _, (data, labels) in enumerate(tk0):
        data = data.to(args.device)
        target = labels[0].to(args.device)
        # private_label = private_label.to(args.device)
        output = model(data)
        # output_private = args.PH()
        loss_task = args.criterion(output, target)
        loss_task_tot.update(loss_task.item(), data.size(0))
        acc1 = accuracy(output, target, topk=(1,))
        # private_acc = accuracy(output_private, private_label, topk=(1,))
        top1.update(acc1[0], data.size(0))
        # private_acc.update(private_acc[0], data.size(0))
        # tk0.set_postfix(loss_task = loss_task_tot.avg, top1 = top1.avg.item(), top1_private=private_acc.avg.item())
        tk0.set_postfix(loss_task=loss_task_tot.avg, top1=top1.avg.item())
    if args.wandb == 1:
        wandb.log(
            {
                "vanilla/test_acc": top1.avg.item(),
                "vanilla/test_loss": loss_task_tot.avg,
            }
        )

    return loss_task_tot.avg



def run_model(
    model, args, PH, dataloader, device, epoch=None, nb_epoch_max=None, prefix=""
):
    """
    Alternative function to run (once) the given model through the data associated to the given dataloader.
    Returns the task final loss, the task accuracy and the private accuracy (differently from the run_model function).
    """
    print(prefix)
    model.train(False)

    metrics = init_eval_metrics(args)
    batch_metrics = {}
    tk0 = tqdm(dataloader, total=int(len(dataloader)), leave=True, dynamic_ncols=True)
    for batch, (data, labels) in enumerate(tk0):

        data = data.to(device)
        target = labels[0].to(device)
        private_labels = [labels[i].to(device) for i in range(args.nb_bias)]
        output = model(data)
        output_private = [PH[i]() for i in range(args.nb_bias)]

        batch_metrics["loss_task"] = args.criterion(output, target)
        batch_metrics["acc1"] = accuracy(output, target, topk=(1,))
        batch_metrics["MI"] = [
            args.MI[i](PH[i], private_labels[i]) for i in range(args.nb_bias)
        ]

        batch_metrics["private_acc"] = [
            accuracy(output_private[i], private_labels[i], topk=(1,))
            for i in range(args.nb_bias)
        ]
        batch_metrics["loss_private"] = [
            args.criterion(output_private[i], private_labels[i])
            for i in range(args.nb_bias)
        ]

        acc_to_print = metrics["top1"].avg

        metrics = update_eval_metrics(
            args=args,
            metrics=metrics,
            batch_metrics=batch_metrics,
        )
        if epoch is not None:
            tk0.set_postfix(
                e=f"{epoch}/{nb_epoch_max}",
                MI=" ".join(
                    list(
                        "{:.3f}".format(metrics["MI_tot"][i].avg)
                        for i in range(args.nb_bias)
                    )
                ),
                top1=acc_to_print,
                top1_private=" ".join(
                    list(
                        "{:.3f}".format(metrics["private_acc"][i].avg)
                        for i in range(args.nb_bias)
                    )
                ),
            )
        else:
            tk0.set_postfix(
                MI=" ".join(
                    list(
                        "{:.3f}".format(metrics["MI_tot"][i].avg)
                        for i in range(args.nb_bias)
                    )
                ),
                t_acc="{:.2f}".format(acc_to_print),
                p_acc=" ".join(
                    list(
                        "{:.2f}".format(metrics["private_acc"][i].avg)
                        for i in range(args.nb_bias)
                    )
                ),
            )
        if (
            int(len(dataloader)) <= 10
            or batch % int(np.ceil(len(dataloader) / 10)) == 1
        ):
            if args.wandb == 1:
                to_log = {
                    f"{prefix}/MI_{i}": metrics["MI_tot"][i].avg
                    for i in range(args.nb_bias)
                }
                to_log.update(
                    {
                        f"{prefix}/private_acc_{i}": metrics["private_acc"][i].avg
                        for i in range(args.nb_bias)
                    }
                )
                to_log.update(
                    {
                        f"{prefix}/task_acc": metrics["top1"].avg,
                    }
                )
                wandb.log(to_log)

    return metrics


def train_PH_double(
    model,
    args,
    PHs,
    PHs_optimizers=None,
    train_loader=None,
    n_epochs_max=10,
    plateau_patience=5,
    plateau_threshold=0.001,
    verbose=True,
):
    """Train a Privacy Head until convergence (i.e. until a "plateau" is reached)
    or until a maximum number of epochs.
    """

    model.train(False)
    if PHs_optimizers is None:
        PHs_optimizers = [
            torch.optim.SGD(
                PH.parameters(),
                lr=args.lr_priv,
                momentum=args.momentum_sgd,
                weight_decay=args.weight_decay,
            ) for PH in PHs
        ]
   
    plateau_detector = PlateauDetector(
        patience=plateau_patience, threshold=plateau_threshold
    )

    for epoch in range(1, args.max_PH_epochs + 1):
        metrics = init_eval_metrics(args)
        batch_metrics = {}

        tk0 = tqdm(train_loader, total=int(len(train_loader)), leave=True)
        for _, (data, labels) in enumerate(tk0):
            data = data.to(args.device)
            private_labels = [
                labels[1 + i].to(args.device) for i in range(args.nb_bias)
            ]
            _ = model(data)
            output_private = [PHs[i]() for i in range(args.nb_bias)]

            batch_metrics["loss_private"] = [
                args.criterion(output_private[i], private_labels[i])
                for i in range(args.nb_bias)
            ]
            
            batch_metrics["MI"] = [
                args.MI[i](PHs[i], private_labels[i]) for i in range(args.nb_bias)
            ]
            batch_metrics["private_acc"] = [
                accuracy(output_private[i], private_labels[i], topk=(1,))
                for i in range(args.nb_bias)
            ]
           
            for i in range(args.nb_bias):
                batch_metrics["loss_private"][i].backward()
                PHs_optimizers[i].step()
                PHs_optimizers[i].zero_grad()

            metrics = update_eval_metrics(
                metrics, batch_metrics, args, update_type=["private"]
            )
            tk0.set_postfix(
                e=f"{epoch}/{n_epochs_max}",
                p_loss=" ".join(
                    list(
                        "{:.3f}".format(np.round(metrics["loss_private_tot"][i].avg, 3))
                        for i in range(args.nb_bias)
                    )
                ),
                MI=" ".join(
                    list(
                        "{:.3f}".format(np.round(metrics["MI_tot"][i].avg, 3))
                        for i in range(args.nb_bias)
                    )
                ),
                p_acc=" ".join(
                    list(
                        "{:.2f}".format(np.round(metrics["private_acc"][i].avg, 2))
                        for i in range(args.nb_bias)
                    )
                ),
            )

        if plateau_detector.step(
            np.mean([metrics["loss_private_tot"][i].avg for i in range(args.nb_bias)])
        ):
            break
        if args.wandb == 1 and (epoch // 10 == 0):
            to_log = {
                f"PH/MI_{i}": metrics["MI_tot"][i].avg for i in range(args.nb_bias)
            }
            to_log.update(
                {
                    f"PH/private_acc_{i}": metrics["private_acc"][i].avg
                    for i in range(args.nb_bias)
                }
            )
            wandb.log(to_log)


def init_stopping_criterions(
    temp_sched_policy,
    temperature,
    temp_sched_factor,
    temp_sched_patience,
    temp_sched_threshold,
    args,
):
    if temp_sched_policy == "based_on_MI":
        temp_scheduler = ReduceTempOnPlateau(
            initial_temperature=temperature,
            factor=temp_sched_factor,
            patience=temp_sched_patience,
            threshold=temp_sched_threshold,
        )
    else:
        temp_scheduler = PeriodicTempUpdate(
            initial_temperature=temperature,
            factor=temp_sched_factor,
            patience=temp_sched_patience,
        )
    stop_flag = False
    best_intermediate_gating_weights = None
    BEST_private_acc_val_pruned = [float("inf") for i in range(args.nb_bias)]
    return (
        temp_scheduler,
        stop_flag,
        BEST_private_acc_val_pruned,
        best_intermediate_gating_weights,
    )


def step_temperature(temp_sched_policy, temp_scheduler, MI_unbiased_val):
    if temp_sched_policy == "based_on_MI":
        temperature, temp_update_flag = temp_scheduler.step(MI_unbiased_val)
        print(
            "Temperature = {}   |   Best metric = {}   |   Count = {}\n".format(
                temperature, temp_scheduler.best, temp_scheduler.count
            )
        )
    else:
        temperature, temp_update_flag = temp_scheduler.step()
        print(f"Temperature = {temperature}   |   Count = {temp_scheduler.count}\n")
    return temperature, temp_update_flag


def evaluate_pruned_model(
    pruned_model,
    dls,
    args,
    model_type="final pruned model",
):
    PH_for_pruned_model = [
        Privacy_head(pruned_model.extracter, copy.deepcopy(args.PH1.classifier)).to(
            args.device
        )
        for i in range(args.nb_bias)
    ]
    PH_optimizer_for_pruned_model = [
        torch.optim.SGD(
            PH_for_pruned_model[i].parameters(),
            lr=args.lr_priv,
            momentum=args.momentum_sgd,
            weight_decay=args.weight_decay,
        )
        for i in range(args.nb_bias)
    ]
    train_PH_double(
        model=pruned_model,
        args=args,
        PHs=[PH_for_pruned_model],
        PHs_optimizers=[PH_optimizer_for_pruned_model],
        train_loader=dls["unbiased_training"],
        n_epochs_max=args.max_PH_epochs,
    )
    wandb_log = {
        "unbiased_training_pruned": 0,
        "unbiased_val_pruned": 0,
        "unbiased_test_pruned": 0,
        "biased_training_pruned": 0,
        "sparsity": get_sparsity(pruned_model),
    }
    ##### Evaluating on unbiased training data #####
    for subset in dls.keys():
        metrics = run_model(
            pruned_model,
            args,
            PH_for_pruned_model,
            PH_optimizer_for_pruned_model,
            dls[subset],
            device=args.device,
            prefix=f"gated/{subset}",
        )
        print(f"Running the {model_type} and its privacy head on {subset} data:")
        print_performance(metrics, model_type="pruned", subset=subset, args=args)
        if args.wandb == 1:
            wandb_log[f"{subset}/pruned"] = metrics
    if args.wandb == 1:
        wandb.log(wandb_log)


def evaluate_best_pruned_model(
    best_interm_pruned_model,
    PH_for_best_interm_pruned_model,
    dls,
    args,
):
    evaluate_pruned_model(
        best_interm_pruned_model,
        PH_for_best_interm_pruned_model,
        dls,
        args,
        model_type="best pruned model",
    )



def prune_based_on_gating_weights(model_to_prune, gating_weights):
    """
    Perform the pruning of the model, based on the trained gating weights. The criterion is the following: we prune the weights that
    are associated to (strictly) negative values of the gating weights.

    NOTE: We don't prune the biases of the network, since they exist in much lesser amount than the weights, and because
    their effect on the layers' outputs is usually very significant. For that reason, we are more interested in pruning
    the weights of the network.
    """

    pruning_masks = {
        layer_name: weights >= 0 for layer_name, weights in gating_weights.items()
    }

    for layer_name in gating_weights.keys():
        layer = attrgetter(layer_name)(model_to_prune)

        _ = prune.custom_from_mask(layer, name="weight", mask=pruning_masks[layer_name])


def prune_and_evaluate_model(model, gating_weights, args, model_type, dls):
    pruned_model = copy.deepcopy(model)
    pruned_model = pruned_model.to(args.device)
    prune_based_on_gating_weights(pruned_model, gating_weights)
    print(f"Evaluating the sparsity of the {model_type}:\n")
    print(f"\n**** REFINING THE PRIVACY HEAD FOR THE {model_type} ****")
    evaluate_pruned_model(pruned_model, dls, args, model_type=model_type)
