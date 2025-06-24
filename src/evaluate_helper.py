import torch
import numpy as np
import sys
import os
sys.path.append("..")
from plot_helper import plot_RTM, plot_HeatRate, plot_flux_and_hr

class MetricTracker(object):
    """
    track and compute running averages of metrics over multiple batches
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        resets the stored value and count
        """
        self.value = 0.0
        self.count = 0

    def update(self, value, count):
        """
        Adds a new value to the running total, weighted by the number of observations 
        """
        self.count += count
        self.value += value * count

    def getmean(self):
        """ 
        current average 
        """
        return self.value / self.count

    def getsqrtmean(self):
        """
        square root of the current average
        """
        return np.sqrt(self.getmean())


def mse_all(pred, true):
    """
    Mean Squared Error
    """
    return pred.numel(), torch.mean((pred - true) ** 2)

def mbe_all(pred, true):
    """
    Mean Bias Error
    """
    return pred.numel(), torch.mean(pred - true)

def mae_all(pred, true):
    """
    Mean Absolute Error
    """
    return pred.numel(), torch.mean(torch.abs(pred - true))

def nmae_all(pred, true):
    """
    """
    mae = torch.mean(torch.abs(pred - true))
    norm = torch.mean(torch.abs(true)) + 1e-8
    nmae = mae / norm
    return pred.numel(), nmae

def nmse_all(pred, true):
    """
    Normalized Mean Squared Error (NMSE)
    """
    mse = torch.mean((pred - true) ** 2)
    norm = torch.mean(true ** 2) + 1e-8
    nmse = mse / norm
    return pred.numel(), nmse

def mare_all(pred, true):
    """
    """
    relative_error = torch.abs(pred - true) / (torch.abs(true) + 1e-8)
    mare = torch.mean(relative_error)
    return pred.numel(), mare

def gmrae_all(pred, true):
    """
    """
    eps = 1e-8
    relative_errors = torch.abs(pred - true) / (torch.abs(true) + eps)
    log_rel_errors = torch.log(relative_errors + eps)
    gmrae = torch.exp(torch.mean(log_rel_errors))
    return pred.numel(), gmrae

def unnorm_mpas(pred, targ, norm, idxmap):
    """
    Unnormalize MPAS predictions and targets using the provided normalization mapping.

    Parameters:
        pred (torch.Tensor): Normalized predictions (batch, features, time).
        targ (torch.Tensor): Normalized targets (batch, features, time).
        norm (dict): Maps variable names to their 'mean' and 'std'.
        idxmap (dict): Maps feature indices to variable names.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Unnormalized predictions and targets.
    """
    device = pred.device
    upred = torch.zeros_like(pred, device=device)
    utarg = torch.zeros_like(targ, device=device)

    for i, name in idxmap.items():
        std = norm[name]["std"]
        mean = norm[name]["mean"]
        upred[:, i, :] = pred[:, i, :] * std + mean
        utarg[:, i, :] = targ[:, i, :] * std + mean

    return upred, utarg

def calc_abs(pred, targ, p=None):
    """
    """
    abs12_pred = calc_hr(pred[:, 0:1, :], pred[:, 1:2, :], p)
    abs12_targ = calc_hr(targ[:, 0:1, :], targ[:, 1:2, :], p)
    abs34_pred = calc_hr(pred[:, 2:3, :], pred[:, 3:4, :], p)
    abs34_targ = calc_hr(targ[:, 2:3, :], targ[:, 3:4, :], p)

    return abs12_pred, abs12_targ, abs34_pred, abs34_targ

def calc_hr(up, down, p=None):
    """
    """
    net = up - down
    dnet = net - torch.roll(net, 1, 2)

    if p is not None:
        g = 9.8066
        r = 287.0
        cp = 7.0 * r / 2.0
        fac = g * 8.64e4 / (cp * 100)

        dp = p - torch.roll(p, 1, 2)
        return dnet[:, :, 1:] / dp[:, :, 1:] * fac
    else:
        return -dnet[:, :, 1:]

def check_accuracy_evaluate_lsm(loader, model, norm_mapping, index_mapping, device, args, epoch):
    model.eval()

    metric_suffix = args.loss_type.lower()
    assert metric_suffix in ['mse', 'mae', 'nmae', 'nmse'], \
        "Invalid loss_type (should be one of 'mse', 'mae', 'nmae', 'nmse')"

    if metric_suffix == "mse":
        func = mse_all
    elif metric_suffix == "mae":
        func = mae_all
    elif metric_suffix == "nmae":
        func = nmae_all
    elif metric_suffix == "nmse":
        func = nmse_all
    else:
        raise ValueError(f"Unsupported loss type: {metric_suffix}")

    main_key = f"{metric_suffix}"
    abs12_key = f"abs12_{metric_suffix}"
    abs34_key = f"abs34_{metric_suffix}"

    valid_metrics = {
        main_key: MetricTracker(),
        abs12_key: MetricTracker(),
        abs34_key: MetricTracker()
    }

    valid_loss = MetricTracker()

    with torch.no_grad():
        for batch_idx, (feature, targets) in enumerate(loader):

            feature_shape = feature.shape
            target_shape = targets.shape
            inner_batch_size = feature_shape[0] * feature_shape[1]
            feature = feature.reshape(inner_batch_size, feature_shape[2], feature_shape[3]).to(device=device)
            targets = targets.reshape(inner_batch_size, target_shape[2], target_shape[3]).to(device=device)

            predicts = model(feature)

            predicts_unnorm, targets_unnorm = unnorm_mpas(predicts, targets, norm_mapping, index_mapping)
            abs12_predict, abs12_target, abs34_predict, abs34_target = calc_abs(predicts_unnorm, targets_unnorm)
            #abs12_predict, abs12_target, abs34_predict, abs34_target = calc_abs(predicts, targets)
            
            metric_values = {
                    main_key: func(predicts_unnorm, targets_unnorm),
                    abs12_key: func(abs12_predict, abs12_target),
                    abs34_key: func(abs34_predict, abs34_target)
                    }

            for key, (count, value) in metric_values.items():
                valid_metrics[key].update(value.item(), count)

            main_count, main_val = metric_values[main_key]
            abs12_count, abs12_val = metric_values[abs12_key]
            abs34_count, abs34_val = metric_values[abs34_key]

            weighted_loss = (1.0 - args.beta) * main_val * main_count + args.beta * (abs12_val * abs12_count + abs34_val * abs34_count)
            total_count = (1.0 - args.beta) * main_count + args.beta * (abs12_count + abs34_count)
            total_loss = weighted_loss / total_count

            valid_loss.update(total_loss.item(), 1)


            if epoch==args.num_epochs-1 and batch_idx < 50:
                print("making plot", batch_idx)
                base_dir = os.path.join("results", args.main_folder, args.sub_folder)
                plot_RTM(predicts, targets, os.path.join(base_dir, f"Flux{batch_idx}_{args.test_year}.png"), sample_index=0)
                plot_flux_and_hr(
                        predicts, targets,
                        abs12_predict=abs12_predict, abs12_target=abs12_target, abs34_predict=abs34_predict, abs34_target=abs34_target,
                        filename=os.path.join(base_dir, f"flux_abs_hexbin_{batch_idx}_{args.test_year}.png")
                        )

    return valid_loss.getmean(), {
            k: (tracker.getsqrtmean() if k.endswith('mse') else tracker.getmean())
            for k, tracker in valid_metrics.items()
            }
