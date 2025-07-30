import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append("..")
from plot_helper import plot_flux_and_abs, plot_flux_and_abs_lines


class NMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(NMSELoss, self).__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        norm = torch.mean(target**2) + self.eps
        return mse / norm


class NMAELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(NMAELoss, self).__init__()
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        mae = self.l1(pred, target)
        norm = torch.mean(torch.abs(target)) + self.eps
        return mae / norm


class CombinedMSEMAELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedMSEMAELoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_mae = self.mae(pred, target)
        combined_loss = self.alpha * loss_mse + (1.0 - self.alpha) * loss_mae
        return combined_loss


class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target + 1e-8)))


class WMSELoss(nn.Module):
    def __init__(self, epsilon=1e-4, gamma=2):
        super().__init__()
        self.eps = epsilon
        self.gamma = gamma

    def forward(self, pred, target):
        weight = torch.log1p((1 / (target + self.eps)) ** self.gamma)
        loss = weight * (pred - target) ** 2
        return torch.mean(loss)


class MetricTracker(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.count = 0

    def update(self, value, count):
        self.count += count
        self.value += value * count

    def getmean(self):
        return self.value / self.count

    def getsqrtmean(self):
        return np.sqrt(self.getmean())


def get_loss_function(loss_type, args):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "nmae":
        return NMAELoss()
    elif loss_type == "nmse":
        return NMSELoss()
    elif loss_type == "wmse":
        return WMSELoss()
    elif loss_type == "logcosh":
        return LogCoshLoss()
    elif loss_type in ["smoothl1", "huber"]:
        if not hasattr(args, "beta_delta"):
            raise ValueError(f"{loss_type.capitalize()}Loss requires --beta_delta")
        return (
            nn.SmoothL1Loss(beta=args.beta_delta)
            if loss_type == "smoothl1"
            else nn.HuberLoss(delta=args.beta_delta)
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


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


def r2_all(pred, true):
    count = pred.numel()
    mse = torch.mean((pred - true) ** 2)
    var = torch.var(true)

    if var == 0:
        r2 = torch.tensor(1.0 if mse == 0 else 0.0, device=pred.device)
    else:
        r2 = 1 - mse / var
    return count, r2


def nmae_all(pred, true):
    """ """
    mae = torch.mean(torch.abs(pred - true))
    norm = torch.mean(torch.abs(true)) + 1e-8
    nmae = mae / norm
    return pred.numel(), nmae


def nmse_all(pred, true):
    """
    Normalized Mean Squared Error (NMSE)
    """
    mse = torch.mean((pred - true) ** 2)
    norm = torch.mean(true**2) + 1e-8
    nmse = mse / norm
    return pred.numel(), nmse


def mare_all(pred, true):
    """ """
    relative_error = torch.abs(pred - true) / (torch.abs(true) + 1e-8)
    mare = torch.mean(relative_error)
    return pred.numel(), mare


def gmrae_all(pred, true):
    """ """
    eps = 1e-8
    relative_errors = torch.abs(pred - true) / (torch.abs(true) + eps)
    log_rel_errors = torch.log(relative_errors + eps)
    gmrae = torch.exp(torch.mean(log_rel_errors))
    return pred.numel(), gmrae


def unnorm_mpas(pred, targ, norm_mapping, normalization_type, idxmap):
    """ """
    device = pred.device
    upred = torch.zeros_like(pred, device=device)
    utarg = torch.zeros_like(targ, device=device)

    for i, var_name in idxmap.items():
        norm_type = normalization_type.get(var_name, "log1p_minmax")
        norm = norm_mapping[var_name]

        if norm_type == "standard":
            mean = norm["vmean"]
            std = norm["vstd"]
            upred[:, i, :] = pred[:, i, :] * std + mean
            utarg[:, i, :] = targ[:, i, :] * std + mean

        elif norm_type == "minmax":
            vmin = norm["vmin"]
            vmax = norm["vmax"]
            upred[:, i, :] = pred[:, i, :] * (vmax - vmin) + vmin
            utarg[:, i, :] = targ[:, i, :] * (vmax - vmin) + vmin

        elif norm_type == "robust":
            median = norm["median"]
            iqr = norm["iqr"]
            upred[:, i, :] = pred[:, i, :] * iqr + median
            utarg[:, i, :] = targ[:, i, :] * iqr + median

        elif norm_type == "log1p_minmax":
            log_min = norm["log_min"]
            log_max = norm["log_max"]
            unnorm_pred = pred[:, i, :] * (log_max - log_min) + log_min
            unnorm_targ = targ[:, i, :] * (log_max - log_min) + log_min
            upred[:, i, :] = torch.expm1(unnorm_pred)
            utarg[:, i, :] = torch.expm1(unnorm_targ)

        elif norm_type == "log1p_standard":
            mean = norm["log_mean"]
            std = norm["log_std"]
            unnorm_pred = pred[:, i, :] * std + mean
            unnorm_targ = targ[:, i, :] * std + mean
            upred[:, i, :] = torch.expm1(unnorm_pred)
            utarg[:, i, :] = torch.expm1(unnorm_targ)

        elif norm_type == "log1p_robust":
            median = norm["log_median"]
            iqr = norm["log_iqr"]
            unnorm_pred = pred[:, i, :] * iqr + median
            unnorm_targ = targ[:, i, :] * iqr + median
            upred[:, i, :] = torch.expm1(unnorm_pred)
            utarg[:, i, :] = torch.expm1(unnorm_targ)

        elif norm_type == "sqrt_minmax":
            sqrt_min = norm["sqrt_min"]
            sqrt_max = norm["sqrt_max"]
            unnorm_pred = pred[:, i, :] * (sqrt_max - sqrt_min) + sqrt_min
            unnorm_targ = targ[:, i, :] * (sqrt_max - sqrt_min) + sqrt_min
            upred[:, i, :] = unnorm_pred**2
            utarg[:, i, :] = unnorm_targ**2

        elif norm_type == "sqrt_standard":
            mean = norm["sqrt_mean"]
            std = norm["sqrt_std"]
            unnorm_pred = pred[:, i, :] * std + mean
            unnorm_targ = targ[:, i, :] * std + mean
            upred[:, i, :] = unnorm_pred**2
            utarg[:, i, :] = unnorm_targ**2

        elif norm_type == "sqrt_robust":
            median = norm["sqrt_median"]
            iqr = norm["sqrt_iqr"]
            unnorm_pred = pred[:, i, :] * iqr + median
            unnorm_targ = targ[:, i, :] * iqr + median
            upred[:, i, :] = unnorm_pred**2
            utarg[:, i, :] = unnorm_targ**2
        else:
            raise ValueError(
                f"Unsupported normalization type '{norm_type}' for variable '{var_name}'"
            )

    return upred, utarg


def calc_abs(pred, targ, p=None):
    """ """
    abs12_pred = calc_hr(pred[:, 0:1, :], pred[:, 1:2, :], p)
    abs12_targ = calc_hr(targ[:, 0:1, :], targ[:, 1:2, :], p)
    abs34_pred = calc_hr(pred[:, 2:3, :], pred[:, 3:4, :], p)
    abs34_targ = calc_hr(targ[:, 2:3, :], targ[:, 3:4, :], p)

    return abs12_pred, abs12_targ, abs34_pred, abs34_targ


def calc_hr(up, down, p=None):
    """ """
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


def check_accuracy_evaluate_lsm(
    loader, model, norm_mapping, normalization_type, index_mapping, device, args, epoch
):
    model.eval()

    valid_loss_types = [
        "mse",
        "mae",
        "nmae",
        "nmse",
        "wmse",
        "logcosh",
        "smoothl1",
        "huber",
    ]
    loss_type = args.loss_type.lower()
    assert (
        loss_type in valid_loss_types
    ), f"Invalid loss_type (should be one of {valid_loss_types})"
    func = get_loss_function(loss_type, args)

    metric_names = ["NMAE", "NMSE", "R2"]
    metric_funcs = {"NMAE": nmae_all, "NMSE": nmse_all, "R2": r2_all}
    output_keys = ["fluxes", "abs12", "abs34"]
    valid_metrics = {
        f"{k}_{m}": MetricTracker() for k in output_keys for m in metric_names
    }

    valid_loss = MetricTracker()

    with torch.no_grad():
        for batch_idx, (feature, targets) in enumerate(loader):
            feature_shape = feature.shape
            target_shape = targets.shape
            inner_batch_size = feature_shape[0] * feature_shape[1]
            feature = feature.reshape(
                inner_batch_size, feature_shape[2], feature_shape[3]
            ).to(device=device)
            targets = targets.reshape(
                inner_batch_size, target_shape[2], target_shape[3]
            ).to(device=device)

            predicts = model(feature)

            predicts_unnorm, targets_unnorm = unnorm_mpas(
                predicts, targets, norm_mapping, normalization_type, index_mapping
            )
            abs12_predict, abs12_target, abs34_predict, abs34_target = calc_abs(
                predicts_unnorm, targets_unnorm
            )

            output_dict = {
                "fluxes": (predicts, targets),
                "abs12": (abs12_predict, abs12_target),
                "abs34": (abs34_predict, abs34_target),
            }

            for key in output_keys:
                pred, tgt = output_dict[key]
                for metric in metric_names:
                    metric_key = f"{key}_{metric}"
                    if metric_key not in valid_metrics:
                        raise KeyError(
                            f"Metric key '{metric_key}' not found in valid_metrics"
                        )
                    count, value = metric_funcs[metric](pred, tgt)
                    valid_metrics[metric_key].update(value.item(), count)

            main_count, main_val = predicts.numel(), func(predicts, targets)
            abs12_count, abs12_val = (
                abs12_predict.numel(),
                func(abs12_predict, abs12_target),
            )
            abs34_count, abs34_val = (
                abs34_predict.numel(),
                func(abs34_predict, abs34_target),
            )

            weighted_loss = (1.0 - args.beta) * main_val * main_count + args.beta * (
                abs12_val * abs12_count + abs34_val * abs34_count
            )
            total_count = (1.0 - args.beta) * main_count + args.beta * (
                abs12_count + abs34_count
            )
            total_loss = weighted_loss / total_count

            valid_loss.update(total_loss.item(), 1)

            if epoch == args.num_epochs - 1:
                print("making plot", batch_idx)
                base_dir = os.path.join("results", args.main_folder, args.sub_folder)
                # plot_RTM(predicts_unnorm, targets_unnorm, os.path.join(base_dir, f"Flux{batch_idx}_{args.test_year}.png"))
                # plot_HeatRate(abs12_predict, abs12_target, abs34_predict, abs34_target, os.path.join(base_dir, f"Abs{batch_idx}_{args.test_year}.png"))
                plot_flux_and_abs_lines(
                    predicts_unnorm,
                    targets_unnorm,
                    abs12_predict=abs12_predict,
                    abs12_target=abs12_target,
                    abs34_predict=abs34_predict,
                    abs34_target=abs34_target,
                    filename=os.path.join(
                        base_dir, f"Lineplot_Flux_Abs{batch_idx}_{args.test_year}.png"
                    ),
                )
                plot_flux_and_abs(
                    predicts_unnorm,
                    targets_unnorm,
                    abs12_predict=abs12_predict,
                    abs12_target=abs12_target,
                    abs34_predict=abs34_predict,
                    abs34_target=abs34_target,
                    filename=os.path.join(
                        base_dir, f"flux_abs_hexbin_{batch_idx}_{args.test_year}.png"
                    ),
                )

    return valid_loss.getmean(), {
        k: (tracker.getsqrtmean() if k.lower().endswith("mse") else tracker.getmean())
        for k, tracker in valid_metrics.items()
    }
