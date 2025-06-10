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

def get_hr(pred, targ, p):
    """
    Compute shortwave and longwave heating rates for predictions and targets.

    Parameters:
        pred (torch.Tensor): Predicted fluxes (batch, 4, levels).
        targ (torch.Tensor): Target fluxes (batch, 4, levels).
        p (torch.Tensor): Pressure levels (batch, 1, levels).

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: SW HR pred, SW HR targ, LW HR pred, LW HR targ
    """
    sw_pred = calc_hr(pred[:, 0:1, :], pred[:, 1:2, :], p)
    sw_targ = calc_hr(targ[:, 0:1, :], targ[:, 1:2, :], p)
    sw_pred[:, :, -1] = 0.0
    sw_targ[:, :, -1] = 0.0

    lw_pred = calc_hr(pred[:, 2:3, :], pred[:, 3:4, :], p)
    lw_targ = calc_hr(targ[:, 2:3, :], targ[:, 3:4, :], p)

    return sw_pred, sw_targ, lw_pred, lw_targ


def calc_hr(up, down, p):
    """
    Compute heating rate from upward and downward fluxes.

    Parameters:
        up (Tensor): Upward flux (batch, 1, levels).
        down (Tensor): Downward flux (batch, 1, levels).
        p (Tensor): Pressure levels (batch, 1, levels).

    Returns:
        Tensor: Heating rate (batch, 1, levels - 1)
    """
    g = 9.8066
    r = 287.0
    cp = 7.0 * r / 2.0
    fac = g * 8.64e4 / (cp * 100)

    net = up - down
    dnet = net - torch.roll(net, 1, 2)
    dp = p - torch.roll(p, 1, 2)
    
    return dnet[:, :, 1:] / dp[:, :, 1:] * fac

def check_accuracy_evaluate_lsm(loader, model, norm_mapping, index_mapping, device, args, beta, epoch):
    model.eval()

    valid_metrics = {
            'rmse': MetricTracker(),
            'mae': MetricTracker()
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

            valid_len, valid_val = mse_all(predicts, targets)
            valid_metrics['rmse'].update(valid_val.item(), valid_len)
            total_loss = (1.0 - beta) * valid_val

            valid_len, valid_val = mae_all(predicts, targets)
            valid_metrics['mae'].update(valid_val.item(), valid_len)

            valid_loss.update(total_loss.item(), valid_len)

            if epoch==args.num_epochs-1 and batch_idx < 50:
                print("making plot", batch_idx)
                base_dir = os.path.join("results", args.main_folder, args.sub_folder)
                plot_RTM(predicts_unnorm, targets_unnorm, os.path.join(base_dir, f"Flux{batch_idx}.png"), sample_index=0)

    return valid_loss.getmean(), {
            k: (tracker.getsqrtmean() if 'rmse' in k else tracker.getmean()) for k, tracker in valid_metrics.items()
            }

def check_accuracy_evaluate(loader, model, norm_mapping, index_mapping, device, args, beta, epoch):
    model.eval()
    
    valid_metrics = {
            'rmse': MetricTracker(),
            'mae': MetricTracker(),
            'swhr_rmse': MetricTracker(),
            'lwhr_rmse': MetricTracker(),
            'swhr_mae': MetricTracker(),
            'lwhr_mae': MetricTracker()
            }

    valid_loss = MetricTracker()

    with torch.no_grad():
        for batch_idx, (feature, targets, auxis) in enumerate(loader):

            feature_shape = feature.shape
            target_shape = targets.shape
            auxis_shape = auxis.shape
            inner_batch_size = feature_shape[0] * feature_shape[1]
            feature = feature.reshape(inner_batch_size, feature_shape[2], feature_shape[3]).to(device=device)
            targets = targets.reshape(inner_batch_size, target_shape[2], target_shape[3]).to(device=device)
            auxis = auxis.reshape(inner_batch_size, auxis_shape[2], auxis_shape[3]).to(device=device)

            predicts = model(feature)

            predicts_unnorm, targets_unnorm = unnorm_mpas(predicts, targets, norm_mapping, index_mapping)
            swhr_predict, swhr_target, lwhr_predict, lwhr_target = get_hr(predicts_unnorm, targets_unnorm, auxis)

            valid_len, valid_val = mse_all(predicts, targets)
            valid_metrics['rmse'].update(valid_val.item(), valid_len)
            total_loss = (1.0 - beta) * valid_val

            valid_len, valid_val = mae_all(predicts, targets)
            valid_metrics['mae'].update(valid_val.item(), valid_len)
            
            valid_len, valid_val = mse_all(swhr_predict, swhr_target)
            valid_metrics['swhr_rmse'].update(valid_val.item(), valid_len)
            total_loss += beta * valid_val

            valid_len, valid_val = mae_all(swhr_predict, swhr_target)
            valid_metrics['swhr_mae'].update(valid_val.item(), valid_len)

            valid_len, valid_val = mse_all(lwhr_predict, lwhr_target)
            valid_metrics['lwhr_rmse'].update(valid_val.item(), valid_len)
            total_loss += beta * valid_val

            valid_len, valid_val = mae_all(lwhr_predict, lwhr_target)
            valid_metrics['lwhr_mae'].update(valid_val.item(), valid_len)

            valid_loss.update(total_loss.item(), valid_len)

            if epoch==args.num_epochs-1 and batch_idx < 50:
                print("making plot", batch_idx)
                base_dir = os.path.join("results", args.main_folder, args.sub_folder)
                plot_RTM(predicts_unnorm, targets_unnorm, os.path.join(base_dir, f"Flux{batch_idx}.png"), sample_index=0)
                plot_HeatRate(swhr_predict, swhr_target, lwhr_predict, lwhr_target, os.path.join(base_dir, f"HR{batch_idx}.png"), sample_index=0)
                plot_flux_and_hr(predicts_unnorm, targets_unnorm, swhr_predict, swhr_target, lwhr_predict, lwhr_target, os.path.join(base_dir, f"flux_hr_hexbin_{batch_idx}.png"))

    return valid_loss.getmean(), {
            k: (tracker.getsqrtmean() if 'rmse' in k else tracker.getmean()) for k, tracker in valid_metrics.items()
            }
