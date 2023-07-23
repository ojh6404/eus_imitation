"""
This file contains some PyTorch utilities.
"""
import os
import numpy as np
import torch
import torch.optim as optim


def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.copy_(target_param * (1.0 - tau) + param * tau)


def hard_update(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.copy_(param)


def get_torch_device(try_to_use_cuda):
    if try_to_use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def reparameterize(mu, logvar):
    # logvar = \log(\sigma^2) = 2 * \log(\sigma)
    # \sigma = \exp(0.5 * logvar)

    # clamped for numerical stability
    logstd = (0.5 * logvar).clamp(-4, 15)
    std = torch.exp(logstd)

    # Sample \epsilon from normal distribution
    # use std to create a new tensor, so we don't have to care
    # about running on GPU or not
    eps = std.new(std.size()).normal_()

    # Then multiply with the standard deviation and add the mean
    z = eps.mul(std).add_(mu)

    return z


def optimizer_from_optim_params(net_optim_params, net):
    optimizer_type = net_optim_params.get("optimizer_type", "adam")
    lr = net_optim_params["learning_rate"]["initial"]

    if optimizer_type == "adam":
        return optim.Adam(
            params=net.parameters(),
            lr=lr,
            weight_decay=net_optim_params["regularization"]["L2"],
        )
    elif optimizer_type == "adamw":
        return optim.AdamW(
            params=net.parameters(),
            lr=lr,
            weight_decay=net_optim_params["regularization"]["L2"],
        )


def lr_scheduler_from_optim_params(net_optim_params, net, optimizer):
    lr_scheduler_type = net_optim_params["learning_rate"].get(
        "scheduler_type", "multistep"
    )
    epoch_schedule = net_optim_params["learning_rate"]["epoch_schedule"]

    lr_scheduler = None
    if len(epoch_schedule) > 0:
        if lr_scheduler_type == "linear":
            assert len(epoch_schedule) == 1
            end_epoch = epoch_schedule[0]

            return optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=net_optim_params["learning_rate"]["decay_factor"],
                total_iters=end_epoch,
            )
        elif lr_scheduler_type == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=epoch_schedule,
                gamma=net_optim_params["learning_rate"]["decay_factor"],
            )
        else:
            raise ValueError("Invalid LR scheduler type: {}".format(lr_scheduler_type))

    return lr_scheduler


def backprop_for_loss(net, optim, loss, max_grad_norm=None, retain_graph=False):
    # backprop
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)

    # gradient clipping
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

    # compute grad norms
    grad_norms = 0.0
    for p in net.parameters():
        # only clip gradients for parameters for which requires_grad is True
        if p.grad is not None:
            grad_norms += p.grad.data.norm(2).pow(2).item()

    # step
    optim.step()

    return grad_norms


def load_dict_from_checkpoint(ckpt_path):
    """
    Load checkpoint dictionary from a checkpoint file.

    Args:
        ckpt_path (str): Path to checkpoint file.

    Returns:
        ckpt_dict (dict): Loaded checkpoint dictionary.
    """
    ckpt_path = os.path.expanduser(ckpt_path)
    if not torch.cuda.is_available():
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    else:
        ckpt_dict = torch.load(ckpt_path)
    return ckpt_dict


def maybe_dict_from_checkpoint(ckpt_path=None, ckpt_dict=None):
    """
    Utility function for the common use case where either an ckpt path
    or a ckpt_dict is provided. This is a no-op if ckpt_dict is not
    None, otherwise it loads the model dict from the ckpt path.

    Args:
        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

    Returns:
        ckpt_dict (dict): Loaded checkpoint dictionary.
    """
    assert (ckpt_path is not None) or (ckpt_dict is not None)
    if ckpt_dict is None:
        ckpt_dict = load_dict_from_checkpoint(ckpt_path)
    return ckpt_dict


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def maybe_no_grad(no_grad):
    return torch.no_grad() if no_grad else dummy_context_mgr()
