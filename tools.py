# File:                     tools.py
# Created by:               Ivan Luiz De Moura Matos and Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2024/7/18
# Debiasing Surgeon : Fantastic weights and how to find them, ECCV 2024
# ==========================================================================



import numpy as np
import torch
from operator import attrgetter
import numpy as np
from irene.utilities import AverageMeter, accuracy
from filenames import *
import torch.nn.functional as F


class TwoSigmoid(torch.autograd.Function):
    """
    Implementation of a modified version of the function 2 * sigmoid().
    In the forward pass, the function works as 2 * sigmoid() saturated on 1.0.
    In the backward pass, the derivative of the function is equal to the derivative of
    the non-saturated 2 * sigmoid().
    """

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        return torch.clamp(2 * F.sigmoid(input), max=1.0)

    @staticmethod
    def backward(ctx, grad_output):

        (input,) = ctx.saved_tensors
        return grad_output * 2 * F.sigmoid(input) * (1 - F.sigmoid(input))


class ReduceTempOnPlateau:
    """Reduce the temperature when a metric (in our case, the mutual information)
    has stopped improving.
    If no significant improvement is verified for a 'patience' amount of epochs,
    the temperature is reduced, by multiplying it by a positive factor < 1.
    NOTE: The 'threshold' parameter is RELATIVE, and not absolute.
    """

    def __init__(
        self, initial_temperature, factor=0.1, patience=10, threshold=1e-2
    ):  #### Changed in v12 ####: default tol is now 1e-2

        self.temperature = initial_temperature
        self.factor = factor
        self.patience = patience
        self.threshold = threshold

        self.reset = True
        self.count = 0
        self.best = torch.inf  ######## ADDED IN v12 ########

    def step(self, metric_value):

        if self.reset:
            self.best = metric_value  ######## ADDED IN v12 ########
            self.reset = False
        else:
            ######## ADDED IN v12 ########
            if (metric_value >= (1.0 - self.threshold) * self.best) or (
                self.best < 0
            ):  # Comparison is relative, not absolute
                self.count += 1
            else:
                self.best = metric_value
                self.count = 0
            ##############################

        if (
            self.count > self.patience
        ):  #### Changed in v12 ####: ">" (strictly greater) instead of ">="
            self.count = 0
            self.temperature *= self.factor
            print("*** UPDATED TEMPERATURE: {} ***".format(self.temperature))
            return self.temperature, True  # <=== Notebook v15_2
        else:
            return self.temperature, False  # <=== Notebook v15_2


class PeriodicTempUpdate:
    """Reduce the temperature periodically: every 'patience' epochs, the temperature is reduced,
    by multiplying it by a positive factor < 1.
    """

    def __init__(self, initial_temperature, factor=0.1, patience=10):

        self.temperature = initial_temperature
        self.factor = factor
        self.patience = patience

        self.reset = True
        self.count = 0

    def step(self):

        if self.reset:
            self.reset = False
        else:
            self.count += 1

        if self.count > self.patience:
            self.count = 0
            self.temperature *= self.factor
            print("*** UPDATED TEMPERATURE: {} ***".format(self.temperature))
            return self.temperature, True
        else:
            return self.temperature, False


class PlateauDetector:
    """Supervises if a metric has stopped improving.
    If no significant improvement is verified for a 'patience' amount of epochs,
    we assume that a "plateau" was reached, and we indicate it by returning "True"
    through the 'step' method. We return "False" otherwise.
    This boolean return value can be used to "break" a training loop, for example.
    NOTE: The 'threshold' parameter is RELATIVE, and not absolute.
    """

    def __init__(self, patience=10, threshold=1e-2):

        self.patience = patience
        self.threshold = threshold

        # self.reset = True
        self.count = 0  # number of "bad epochs"
        self.num_epochs = (
            0  # total number of epochs (= number of calls to the 'step' method)
        )
        self.best = float("inf")  # best measurement "until now"

    def step(self, metric_value):

        self.num_epochs += 1

        if metric_value > (1.0 - self.threshold) * self.best:
            self.count += 1
        else:
            self.best = metric_value
            self.count = 0

        # print(
        #     "  |  Count: ", self.count, " , Best value: ", float(np.round(self.best, 2))
        # )

        if self.count > self.patience:
            print("(Plateau reached after {} epochs)".format(self.num_epochs))

            return True
        else:
            return False


def vector_to_class(x):
    # Obtain the most "probable" class from the prediction
    y = torch.argmax(F.softmax(x, dim=1), dim=1)
    return y


def my_accuracy(predict, labels):
    # Calculates the accuracy of the results
    accuracy = np.sum(predict == labels) / len(predict)
    return accuracy


def get_sparsity(model):
    """
    Computes and prints the sparsity of each layer in the model, as well as the global sparsity.
    Returns the global sparsity.
    """

    # if prune.is_pruned(model):
    #     print("The model is pruned.\n")
    # else:
    #     print("The model is NOT pruned.\n")

    total_num_params = 0  # total number of weights in the network
    num_zero_params = 0  # total number of pruned weights in the network

    for name, _ in model.named_parameters():

        if (
            ".weight" in name
        ):  # Note: Due to the pruning, we now observe ".weight_orig" instead of ".model" when using .named_parameters

            layer_name = name.split(".weight")[0]

            if not issubclass(
                type(attrgetter(layer_name)(model)),
                torch.nn.modules.batchnorm._BatchNorm,
            ):  # We don't prune batchnorm layers

                parameter = attrgetter(layer_name + ".weight")(model)

                print(
                    "Sparsity in '{}': {:.2f}%".format(
                        layer_name,
                        100.0 * float(torch.sum(parameter == 0) / parameter.nelement()),
                    )
                )

                total_num_params += parameter.nelement()
                num_zero_params += torch.sum(parameter == 0)

    global_sparsity = float(num_zero_params / total_num_params)

    print("Global sparsity: {:.2f}%".format(100.0 * global_sparsity))

    return global_sparsity


def print_performance(metrics, model_type, subset, args):
    print_loss = "{:2f}".format(np.round(metrics['loss_task_tot'].avg,2))
    print_MI = " ".join(list("{:3f}".format(np.round(metrics['MI_tot'][i].avg,3)) for i in range(args.nb_bias)))
    print_priv_acc = " ".join(list("{:2f}".format(np.round(metrics['private_acc'][i].avg,2)) for i in range(args.nb_bias)))
    if args.dataset == "MultiColorMNIST":
        print_task_acc = f"({'{:2f}'.format(np.round(metrics['accuracies_tot']['L_R'].avg,2))}, {'{:2f}'.format(np.round(metrics['accuracies_tot']['L_notR'].avg,2))}, {'{:2f}'.format(np.round(metrics['accuracies_tot']['notL_R'].avg,2))}, {'{:2f}'.format(np.round(metrics['accuracies_tot']['notL_notR'].avg,2))}"
    else : 
        print_task_acc = "{:2f}".format(np.round(metrics['top1'].avg,2))
    print(f"{model_type} - {subset}"
        + f" - Task Loss = {print_loss}"
        + f" | MI = {print_MI}"
        + f" | Priv. Acc = {print_priv_acc}"
        + f" | Task Acc : {print_task_acc}"
    )
    


def update_accuracy_multicolormnist(output, target, bias1, bias2, accuracies_tot):
    mask_L_aligned = target == bias1
    mask_R_aligned = target == bias2
    masks = {}
    masks["L_R"] = mask_L_aligned * mask_R_aligned
    masks["L_notR"] = mask_L_aligned * torch.logical_not(mask_R_aligned)
    masks["notL_R"] = torch.logical_not(mask_L_aligned) * mask_R_aligned
    masks["notL_notR"] = torch.logical_not(mask_L_aligned) * torch.logical_not(
        mask_R_aligned
    )
    top1 = 0
    for split in ["L_R", "L_notR", "notL_R", "notL_notR"]:
        if torch.sum(masks[split]) > 0:
            # print(f"{split}:{torch.sum(masks[split]).item()} elts")
            acc = accuracy(
                output[masks[split]],
                target[masks[split]],
                topk=(1,),
            )[0].item()
            accuracies_tot[split].update(
                acc,
                torch.sum(masks[split]).item(),
            )
            top1 += acc

    accuracies_tot["top1"] = top1 / 4
    return accuracies_tot


def init_eval_metrics(args):
    metrics = {}
    if args.dataset == "MultiColorMNIST":
        metrics["accuracies_tot"] = {
            "L_R": AverageMeter("Acc@1", ":6.2f"),
            "notL_R": AverageMeter("Acc@1", ":6.2f"),
            "L_notR": AverageMeter("Acc@1", ":6.2f"),
            "notL_notR": AverageMeter("Acc@1", ":6.2f"),
        }
    metrics["top1"] = AverageMeter("Acc@1", ":6.2f")
    metrics["loss_task_tot"] = AverageMeter("Loss", ":.4e")
    metrics["loss_private_tot"] = [AverageMeter("Loss", ":.4e") for i in range(args.nb_bias)]
    metrics["MI_tot"] = [AverageMeter("Regu", ":.4e") for i in range(args.nb_bias)]
    metrics["private_acc"] = [AverageMeter("Acc@1", ":6.2f") for i in range(args.nb_bias)]
    return metrics


def update_eval_metrics(metrics, batch_metrics, args, update_type=["task", "private"]):
    if "private" in update_type:
        for i in range(args.nb_bias):
            metrics["MI_tot"][i].update(batch_metrics["MI"][i].item(), args.batch_size)
            metrics["private_acc"][i].update(
                batch_metrics["private_acc"][i][0].item(), args.batch_size
            )
            metrics["loss_private_tot"][i].update(
                batch_metrics["loss_private"][i].item(), args.batch_size
            )
    if "task" in update_type:
        metrics["loss_task_tot"].update(
            batch_metrics["loss_task"].item(), args.batch_size
        )
        if args.dataset != "MultiColorMNIST":
            metrics["top1"].update(batch_metrics["acc1"][0].item(), args.batch_size)

    
    return metrics


def check_weights_equality(model1, model2, gating_weights):
    for layer_name in gating_weights.keys():

        modif_layer = attrgetter(layer_name)(
            model1
        )  # corresponding layer of the auxiliary model, used during training of the gating weights
        orig_layer = attrgetter(layer_name)(
            model2
        )  # corresponding layer of the original model

        print(
            f"Layer {layer_name} {torch.abs(orig_layer.weight.data-modif_layer.weight.data)<1e-5}"
        )


def print_ds(args):
    print(
        "Method: learning gating weights (using a balanced training set) to guide the pruning"
    )
    if args.dataset == "Bmnist":
        print(
            "Biased MNIST -- rho = {:.3f} (alpha = {:.2f}, gamma = {:.2f}, seed = {})\n".format(
                args.rho, args.alpha, args.gamma, args.seed
            )
        )
    elif args.dataset == "celebA":
        print(
            "CelebA -- Target {} (alpha = {:.2f}, gamma = {:.2f}, seed = {})\n".format(
                args.target_celeba, args.alpha, args.gamma, args.seed
            )
        )
    elif args.dataset == "cifar10c":
        print(
            "CIFAR10-C -- percent {} (alpha = {:.2f}, gamma = {:.2f}, seed = {})\n".format(
                args.cifar10c_percent, args.alpha, args.gamma, args.seed
            )
        )
    elif args.dataset == "waterbirds":
        print(
            "WaterBirds (alpha = {:.2f}, gamma = {:.2f}, seed = {})\n".format(
                args.alpha, args.gamma, args.seed
            )
        )
    elif args.dataset == "MultiColorMNIST":
        print(
            "MultiColor MNIST -- left skew 0.01, right skew 0.05 (alpha = {:.2f}, gamma = {:.2f}, seed = {})\n".format(
                args.alpha, args.gamma, args.seed
            )
        )


### GradCam
