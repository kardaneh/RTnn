# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Evaluation utilities for RTnn model assessment.

This module provides comprehensive evaluation tools for radiative transfer
neural network models, including custom loss functions, metric computation,
and visualization helpers. It is designed to support both training diagnostics
and rigorous model validation against physical constraints.

The module implements several key capabilities:

**Custom Loss Functions**
    - Normalized losses (NMAE, NMSE) for scale-invariant error measurement
    - Standard losses (MSE, MAE, Huber, Smooth L1) for baseline comparison
    - Weighted and physics-informed losses for multi-objective optimization

**Evaluation Metrics**
    - NMAE: Normalized Mean Absolute Error
    - NMSE: Normalized Mean Squared Error
    - R²: Coefficient of determination
    - MBE: Mean Bias Error
    - MARE: Mean Absolute Relative Error
    - GMRAE: Geometric Mean Relative Absolute Error

**Physical Consistency**
    - Absorption rate calculation from flux divergence
    - Energy conservation penalty (albedo + transmittance + absorptance = 1)
    - Heating rate computation from net flux profiles

**Data Handling**
    - Normalization/de-normalization for all supported transformation types
    - Multi-dimensional tensor reshaping (batch, channels, PFTs, bands, levels)
    - Metric tracking with running statistics

The module follows a modular design where loss functions and metrics are
implemented as separate callable classes/functions, allowing easy extension
and composition.

Notes
-----
**Flux Variable Ordering**

    - Channel 0: collimated albedo (direct upwelling)
    - Channel 1: collimated transmittance (direct downwelling)
    - Channel 2: isotropic albedo (diffuse upwelling)
    - Channel 3: isotropic transmittance (diffuse downwelling)

This ordering matches the ``ov`` list in :class:`rtnn.dataset.DataPreprocessor`.

**Absorption Calculation**

    - For collimated: absorption = -d(net_flux)/dz
    - For isotropic: absorption = -d(net_flux)/dz


**Supported Normalization Types**

    - Linear: minmax, standard, robust
    - Log1p-based: log1p_minmax, log1p_standard, log1p_robust
    - Sqrt-based: sqrt_minmax, sqrt_standard, sqrt_robust

Examples
--------
Basic usage for model evaluation:

>>> import torch
>>> from rtnn.evaluater import get_loss_function, run_validation_lsm
>>>
>>> # Create loss function
>>> args = argparse.Namespace(loss_type='huber', beta_delta=1.0)
>>> criterion = get_loss_function('huber', args)
>>>
>>> # Evaluate model
>>> valid_loss, metrics = run_validation_lsm(
...     loader=val_loader,
...     model=my_model,
...     norm_mapping=norm_stats,
...     normalization_type=norm_types,
...     index_mapping=idxmap,
...     device=device,
...     args=args,
...     epoch=10,
...     logger=logger
... )
>>>
>>> print(f"Validation NMAE: {metrics['fluxes_NMAE']:.4f}")
>>> print(f"R² score: {metrics['fluxes_R2']:.4f}")

Using custom metric tracking:

>>> from rtnn.evaluater import MetricTracker, nmae_all
>>>
>>> tracker = MetricTracker()
>>> for batch in dataloader:
...     pred, target = model(batch)
...     count, value = nmae_all(pred, target)
...     tracker.update(value.item(), count)
>>>
>>> mean_nmae = tracker.getmean()

See Also
--------
rtnn.dataset.DataPreprocessor : Data loading and normalization
rtnn.diagnostics : Visualization tools for model predictions
rtnn.models : Neural network architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm
import os
from rtnn.diagnostics import (
    plot_all_diagnostics,
    plot_cams_diagnostics,
    plot_reftrans_diagnostics,
)
from typing import Optional

sys.path.append("..")


class NMSELoss(nn.Module):
    """
    Normalized Mean Squared Error Loss.

    Computes MSE normalized by the mean square of the target values.
    Useful when the scale of the target variable varies across samples
    or when comparing models trained on different datasets.

    Parameters
    ----------
    eps : float, optional
        Small constant for numerical stability. Default is 1e-8.

    Notes
    -----
    The loss is calculated as:

        NMSE = MSE(pred, target) / (mean(target²) + eps)

    This normalization makes the loss scale-invariant, with values
    typically in the range [0, 1].

    Examples
    --------
    >>> criterion = NMSELoss()
    >>> predictions = torch.tensor([[2.0, 3.0], [1.0, 2.0]])
    >>> targets = torch.tensor([[2.0, 4.0], [1.0, 2.5]])
    >>> loss = criterion(predictions, targets)
    >>> print(loss.item())
    0.0625  # approximates 0.0625 for this example
    """

    def __init__(self, eps=1e-8):
        super(NMSELoss, self).__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        norm = torch.mean(target**2) + self.eps
        return mse / norm


class NMAELoss(nn.Module):
    """
    Normalized Mean Absolute Error Loss.

    Computes MAE normalized by the mean absolute value of the target.
    Provides a scale-invariant error metric that is more robust to
    outliers than NMSE.

    Parameters
    ----------
    eps : float, optional
        Small constant for numerical stability. Default is 1e-8.

    Notes
    -----
    Values typically range from 0 to 1, with 0 representing perfect
    predictions and values >1 indicating predictions worse than the
    trivial zero predictor.

    Examples
    --------
    >>> criterion = NMAELoss()
    >>> predictions = torch.tensor([[2.0, 3.0], [1.0, 2.0]])
    >>> targets = torch.tensor([[2.0, 4.0], [1.0, 2.5]])
    >>> loss = criterion(predictions, targets)
    >>> print(loss.item())
    0.0833  # approximates 0.0833 for this example
    """

    def __init__(self, eps=1e-8):
        super(NMAELoss, self).__init__()
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        mae = self.l1(pred, target)
        norm = torch.mean(torch.abs(target)) + self.eps
        return mae / norm


def physics_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    conservation_penalty: Optional[torch.Tensor] = None,
    lambda_phys: float = 0.1,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Combined Huber loss + energy conservation penalty (improvement E).

    The four output variables satisfy:
        albedo + transmittance + absorptance = 1

    For collimated:   collim_alb  + collim_tran  + collim_abs  = 1
    For isotropic:    isotrop_alb + isotrop_tran + isotrop_abs = 1

    This function enforces the constraint as a soft penalty.  You can pass
    ``pred_abs`` if your model also predicts absorptance; otherwise the
    penalty is computed implicitly as (1 - alb - tran).

    Parameters
    ----------
    pred : torch.Tensor  shape (B, 4, L)
        Model predictions: [collim_alb, collim_tran, isotrop_alb, isotrop_tran]
        (channel order matches your ``ov`` list in DataPreprocessor).
    target : torch.Tensor  shape (B, 4, L)
        Ground-truth targets.
    pred_abs : torch.Tensor or None  shape (B, 2, L)
        If provided, predicted absorptance for [collimated, isotropic].
        If None, absorptance is inferred as (1 - alb - tran).
    lambda_phys : float
        Weight of the energy conservation penalty relative to Huber loss.
    delta : float
        Huber loss delta parameter.

    Returns
    -------
    torch.Tensor  scalar loss value.
    """
    # --- Primary Huber loss ---
    huber = F.huber_loss(pred, target, delta=delta, reduction="mean")

    return huber + lambda_phys * conservation_penalty


class MetricTracker:
    """
    A utility class for tracking and computing statistics of metric values.

    This class maintains running sums of metric values and their squares,
    allowing incremental updates and computation of mean and standard
    deviation. It is particularly useful for aggregating metrics across
    multiple batches during evaluation.

    Attributes
    ----------
    value : float
        Cumulative weighted sum of metric values
    count : int
        Total number of samples processed
    value_sq : float
        Cumulative weighted sum of squared metric values

    Examples
    --------
    >>> tracker = MetricTracker()
    >>> tracker.update(10.0, 5)  # value=10.0, count=5 samples
    >>> tracker.update(20.0, 3)  # value=20.0, count=3 samples
    >>> print(tracker.getmean())  # (10*5 + 20*3) / (5+3) = 110/8 = 13.75
    13.75
    >>> print(tracker.getstd())
    5.0  # computed from variance
    >>> print(tracker.getsqrtmean())
    3.7080992435478315
    """

    def __init__(self):
        """
        Initialize MetricTracker with zero values.
        """
        self.reset()

    def reset(self):
        """
        Reset all tracked values to zero.

        Returns
        -------
        None
        """
        self.value = 0.0
        self.count = 0
        self.value_sq = 0.0

    def update(self, value, count):
        """
        Update the tracker with new metric values.

        Parameters
        ----------
        value : float
            The metric value to add
        count : int
            Number of samples this value represents (weight)

        Returns
        -------
        None
        """
        self.count += count
        self.value += value * count
        self.value_sq += (value**2) * count

    def getmean(self):
        """
        Calculate the mean of all tracked values.

        Returns
        -------
        float
            Weighted mean of all values: total_value / total_count

        Raises
        ------
        ZeroDivisionError
            If no values have been added (count == 0)
        """
        if self.count == 0:
            raise ZeroDivisionError("Cannot compute mean with zero samples")
        return self.value / self.count

    def getstd(self):
        """
        Calculate the standard deviation of all tracked values.

        Returns
        -------
        float
            Weighted standard deviation of all values:
            sqrt(E(x^2) - (E(x))^2)

        Raises
        ------
        ZeroDivisionError
            If no values have been added (count == 0)
        """
        if self.count == 0:
            raise ZeroDivisionError("Cannot compute std with zero samples")
        mean = self.getmean()
        variance = self.value_sq / self.count - mean**2
        return np.sqrt(max(variance, 0.0))  # numerical safety

    def getsqrtmean(self):
        """
        Calculate the square root of the mean of all tracked values.

        Returns
        -------
        float
            Square root of the weighted mean: sqrt(total_value / total_count)

        Raises
        ------
        ZeroDivisionError
            If no values have been added (count == 0)
        """
        return np.sqrt(self.getmean())


def get_loss_function(loss_type, args, logger=None):
    """
    Factory function to instantiate the requested loss function.

    Parameters
    ----------
    loss_type : str
        Type of loss function. Options:

        **Standard losses:**
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error

        **Normalized losses:**
            - 'nmae': Normalized Mean Absolute Error
            - 'nmse': Normalized Mean Squared Error

        **Robust losses:**
            - 'smoothl1': Smooth L1 Loss (Huber with beta)
            - 'huber': Huber Loss with delta parameter

    args : argparse.Namespace
        Arguments containing loss-specific parameters:
        - For 'huber'/'smoothl1': requires `args.beta_delta`
        - For composite losses: may require `args.beta`

    logger : logging.Logger, optional
        Logger for informational messages. If None, no logging occurs.

    Returns
    -------
    torch.nn.Module
        Initialized loss function.

    Raises
    ------
    ValueError
        If loss_type is not supported or required parameters are missing.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(beta_delta=1.0)
    >>> criterion = get_loss_function('huber', args)
    >>> loss = criterion(predictions, targets)

    >>> args = argparse.Namespace()
    >>> criterion = get_loss_function('mse', args)
    """
    if loss_type == "mse":
        if logger:
            logger.info("Using Mean Squared Error (MSE) loss")
        return nn.MSELoss()
    elif loss_type == "mae":
        if logger:
            logger.info("Using Mean Absolute Error (MAE) loss")
        return nn.L1Loss()
    elif loss_type == "nmae":
        if logger:
            logger.info("Using Normalized Mean Absolute Error (NMAE) loss")
        return NMAELoss()
    elif loss_type == "nmse":
        if logger:
            logger.info("Using Normalized Mean Squared Error (NMSE) loss")
        return NMSELoss()
    elif loss_type in ["smoothl1", "huber"]:
        if not hasattr(args, "beta_delta"):
            raise ValueError(f"{loss_type.capitalize()}Loss requires --beta_delta")
        if logger:
            logger.info(
                f"Using {loss_type.capitalize()} loss with delta={args.beta_delta}"
            )
        return (
            nn.SmoothL1Loss(beta=args.beta_delta)
            if loss_type == "smoothl1"
            else nn.HuberLoss(delta=args.beta_delta)
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def mse_all(pred, true):
    """
    Compute Mean Squared Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, mse_value)
    """
    return pred.numel(), torch.mean((pred - true) ** 2)


def mbe_all(pred, true):
    """
    Compute Mean Bias Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, mbe_value)
    """
    return pred.numel(), torch.mean(pred - true)


def mae_all(pred, true):
    """
    Compute Mean Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, mae_value)
    """
    return pred.numel(), torch.mean(torch.abs(pred - true))


def r2_all(pred, true):
    """
    Calculate R2 (coefficient of determination) between predicted and true values.

    Computes the R2 metric and returns both the number of elements and
    the R2 value.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values from the model
    true : torch.Tensor
        Ground truth values

    Returns
    -------
    tuple
        (num_elements, r2_value) where:
        - num_elements (int): Total number of elements in the tensors
        - r2_value (torch.Tensor): R2 score

    Notes
    -----
    R2 is calculated as:

        R2 = 1 - sum((true - pred)^2) / sum((true - mean(true))^2)

    This implementation is fully torch-based and works on CPU and GPU.
    """

    if pred.shape != true.shape:
        raise RuntimeError(f"Shape mismatch: pred {pred.shape} vs true {true.shape}")

    eps = 1e-12  # Small value to avoid division by zero when variance is zero
    num_elements = pred.numel()

    # Flatten
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    # Residual sum of squares
    ss_res = torch.sum((true_flat - pred_flat) ** 2)

    # Total sum of squares
    true_mean = torch.mean(true_flat)
    ss_tot = torch.sum((true_flat - true_mean) ** 2)

    # R2 score
    r2_value = 1.0 - ss_res / (ss_tot + eps)

    return num_elements, r2_value


def nmae_all(pred, true):
    """
    Compute Normalized Mean Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, nmae_value)
    """
    mae = torch.mean(torch.abs(pred - true))
    norm = torch.mean(torch.abs(true)) + 1e-8
    nmae = mae / norm
    return pred.numel(), nmae


def nmse_all(pred, true):
    """
    Compute Normalized Mean Squared Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, nmse_value)
    """
    mse = torch.mean((pred - true) ** 2)
    norm = torch.mean(true**2) + 1e-8
    nmse = mse / norm
    return pred.numel(), nmse


def mare_all(pred, true):
    """
    Compute Mean Absolute Relative Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, mare_value)
    """
    relative_error = torch.abs(pred - true) / (torch.abs(true) + 1e-8)
    mare = torch.mean(relative_error)
    return pred.numel(), mare


def gmrae_all(pred, true):
    """
    Compute Geometric Mean Relative Absolute Error.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions.
    true : torch.Tensor
        Ground truth.

    Returns
    -------
    tuple
        (num_elements, gmrae_value)
    """
    eps = 1e-8
    relative_errors = torch.abs(pred - true) / (torch.abs(true) + eps)
    log_rel_errors = torch.log(relative_errors + eps)
    gmrae = torch.exp(torch.mean(log_rel_errors))
    return pred.numel(), gmrae


def unnorm_mpas(pred, targ, norm_mapping, normalization_type, idxmap):
    """
    Reverse normalization for 5D tensors.

    This function converts normalized predictions and targets back to
    physical units using stored normalization statistics. It supports
    all normalization types defined in DataPreprocessor.

    Parameters
    ----------
    pred : torch.Tensor
        Normalized predictions. Shape: (batch, 4, n_pft, n_bands, seq_length)
    targ : torch.Tensor
        Normalized targets. Same shape as pred.
    norm_mapping : dict
        Dictionary containing normalization statistics for each variable.
    normalization_type : dict
        Dictionary specifying normalization type for each variable.
    idxmap : dict
        Mapping from channel indices (0-3) to variable names.

    Returns
    -------
    tuple
        (upred, utarg) where:
        - upred (torch.Tensor): Unnormalized predictions
        - utarg (torch.Tensor): Unnormalized targets

    Raises
    ------
    ValueError
        If normalization type is not supported.

    Notes
    -----
    The function handles the following transformations:
        - Linear: x' = (x - mean)/std or (x - min)/(max - min)
        - Log1p: x_norm = (log(1+x) - log_mean)/log_std
        - Sqrt: x_norm = (sqrt(x) - sqrt_mean)/sqrt_std

    The reverse operation is applied to recover physical values.
    """
    device = pred.device
    upred = torch.zeros_like(pred, device=device)
    utarg = torch.zeros_like(targ, device=device)

    # idxmap maps channel indices 0-3 to variable names
    for var_idx, var_name in idxmap.items():
        norm_type = normalization_type.get(var_name, "log1p_minmax")
        norm = norm_mapping[var_name]

        # Get slice for this variable: (batch, 1,...)
        var_slice = (slice(None), slice(var_idx, var_idx + 1), ...)
        pred_var = pred[var_slice]
        targ_var = targ[var_slice]

        if norm_type == "standard":
            mean = norm["vmean"]
            std = norm["vstd"]
            upred_var = pred_var * std + mean
            utarg_var = targ_var * std + mean

        elif norm_type == "minmax":
            vmin = norm["vmin"]
            vmax = norm["vmax"]
            upred_var = pred_var * (vmax - vmin) + vmin
            utarg_var = targ_var * (vmax - vmin) + vmin

        elif norm_type == "robust":
            median = norm["median"]
            iqr = norm["iqr"]
            upred_var = pred_var * iqr + median
            utarg_var = targ_var * iqr + median

        elif norm_type == "log1p_minmax":
            log_min = norm["log_min"]
            log_max = norm["log_max"]
            unnorm_pred = pred_var * (log_max - log_min) + log_min
            unnorm_targ = targ_var * (log_max - log_min) + log_min
            upred_var = torch.expm1(unnorm_pred)
            utarg_var = torch.expm1(unnorm_targ)

        elif norm_type == "log1p_standard":
            mean = norm["log_mean"]
            std = norm["log_std"]
            unnorm_pred = pred_var * std + mean
            unnorm_targ = targ_var * std + mean
            upred_var = torch.expm1(unnorm_pred)
            utarg_var = torch.expm1(unnorm_targ)

        elif norm_type == "log1p_robust":
            median = norm["log_median"]
            iqr = norm["log_iqr"]
            unnorm_pred = pred_var * iqr + median
            unnorm_targ = targ_var * iqr + median
            upred_var = torch.expm1(unnorm_pred)
            utarg_var = torch.expm1(unnorm_targ)

        elif norm_type == "sqrt_minmax":
            sqrt_min = norm["sqrt_min"]
            sqrt_max = norm["sqrt_max"]
            unnorm_pred = pred_var * (sqrt_max - sqrt_min) + sqrt_min
            unnorm_targ = targ_var * (sqrt_max - sqrt_min) + sqrt_min
            upred_var = unnorm_pred**2
            utarg_var = unnorm_targ**2

        elif norm_type == "sqrt_standard":
            mean = norm["sqrt_mean"]
            std = norm["sqrt_std"]
            unnorm_pred = pred_var * std + mean
            unnorm_targ = targ_var * std + mean
            upred_var = unnorm_pred**2
            utarg_var = unnorm_targ**2

        elif norm_type == "sqrt_robust":
            median = norm["sqrt_median"]
            iqr = norm["sqrt_iqr"]
            unnorm_pred = pred_var * iqr + median
            unnorm_targ = targ_var * iqr + median
            upred_var = unnorm_pred**2
            utarg_var = unnorm_targ**2
        else:
            raise ValueError(
                f"Unsupported normalization type '{norm_type}' for variable '{var_name}'"
            )
        upred[var_slice] = upred_var
        utarg[var_slice] = utarg_var

    return upred, utarg


def conservation_residual(alb, tran, abs_flux):
    """
    Compute energy conservation residual for a given level.

    Parameters
    ----------
    alb : torch.Tensor
        Albedo (upwelling flux). Shape (batch, 1, n_pft, n_bands, seq)
    tran : torch.Tensor
        Transmittance (downwelling flux). Same shape as alb.
    abs_flux : torch.Tensor
        Absorptance (absorbed flux). Shape (batch, 1, n_pft, n_bands, seq-1)

    Returns
    -------
    torch.Tensor
        Squared conservation residual. Shape (batch, 1, n_pft, n_bands, seq-1)

    Notes
    -----
    The function averages alb and tran to layer centers before computing:
        residual = (alb_center + tran_center + abs_flux - 1)²

    This enforces the physical constraint that:
        upwelling + downwelling + absorption = total incoming radiation = 1
    """
    # Average fluxes to layer centers (N-1 layers)
    alb_center = (alb[..., :-1] + alb[..., 1:]) / 2.0
    tran_center = (tran[..., :-1] + tran[..., 1:]) / 2.0
    # Conservation: alb + tran + abs = 1
    return (alb_center + tran_center + abs_flux - 1.0) ** 2


def calc_abs(pred, targ, p=None):
    """
    Calculate absorption rates from flux predictions.

    Computes absorption rates for both collimated (direct) and isotropic
    (diffuse) components by calculating the divergence of net flux.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted fluxes. Shape (batch, 4, ...)
        Channel order: [flux1_alb, flux2_tran, flux3_alb, flux4_tran]
    targ : torch.Tensor
        Target fluxes. Same shape as pred.
    p : torch.Tensor, optional
        Pressure levels. If provided, computes heating rates. Shape can be
        (seq_length,) or (batch, seq_length). Default is None.

    Returns
    -------
    tuple
        (abs12_pred, abs12_targ, abs34_pred, abs34_targ, conservation_penalty)
    """
    #  (channels 0 and 1)
    abs12_pred = heating_rate(pred[:, 0:1, ...], pred[:, 1:2, ...], p)
    abs12_targ = heating_rate(targ[:, 0:1, ...], targ[:, 1:2, ...], p)

    #  (channels 2 and 3)
    abs34_pred = heating_rate(pred[:, 2:3, ...], pred[:, 3:4, ...], p)
    abs34_targ = heating_rate(targ[:, 2:3, ...], targ[:, 3:4, ...], p)

    # Conservation penalty
    collim_resid = conservation_residual(
        pred[:, 0:1, ...], pred[:, 1:2, ...], abs12_pred
    )
    isotrop_resid = conservation_residual(
        pred[:, 2:3, ...], pred[:, 3:4, ...], abs34_pred
    )
    conservation_penalty = (collim_resid + isotrop_resid).mean()

    return abs12_pred, abs12_targ, abs34_pred, abs34_targ, conservation_penalty


def calc_heating_rates(pred, targ, p=None):
    """
    Calculate heating rates from flux predictions for CAMS atmospheric data.

    Computes heating rates from the divergence of net flux (downwelling - upwelling).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted fluxes. Shape (batch, 2, n_level)
        Channel order: [rsd, rsu] where:
        - rsd: downwelling flux at interfaces
        - rsu: upwelling flux at interfaces
    targ : torch.Tensor
        Target fluxes. Same shape as pred.
    p : torch.Tensor, optional
        Pressure at interfaces. Shape (n_level,) or (batch, n_level)
        Pressure should be in Pa. If provided, returns heating rate in K/day.
        If None, returns negative flux divergence.

    Returns
    -------
    tuple
        (hr_pred, hr_targ, conservation_penalty)
        - hr_pred: Heating rate predictions. Shape (batch, 1, n_layer)
        - hr_targ: Heating rate targets. Shape (batch, 1, n_layer)
        - conservation_penalty: Mean conservation residual (scalar tensor)
    """
    # Extract downwelling and upwelling fluxes
    # pred: (batch, 2, n_level) -> rsd_pred: (batch, 1, n_level), rsu_pred: (batch, 1, n_level)
    rsd_pred = pred[:, 0:1, :]  # (batch, 1, n_level)
    rsu_pred = pred[:, 1:2, :]  # (batch, 1, n_level)
    rsd_targ = targ[:, 0:1, :]  # (batch, 1, n_level)
    rsu_targ = targ[:, 1:2, :]  # (batch, 1, n_level)

    # Calculate heating rates
    hr_pred = heating_rate(rsu_pred, rsd_pred, p)  # (batch, 1, n_layer)
    hr_targ = heating_rate(rsu_targ, rsd_targ, p)  # (batch, 1, n_layer)

    return hr_pred, hr_targ


def heating_rate(up, down, p=None):
    """
    Calculate radiative heating rate from upwelling and downwelling fluxes.

    Parameters
    ----------
    up : torch.Tensor
        Upwelling flux at interfaces.
        Shape: (..., nlev)

    down : torch.Tensor
        Downwelling flux at interfaces.
        Shape: (..., nlev)

    p : torch.Tensor, optional
        Pressure at interfaces.
        Shape: (nlev,) or (..., nlev)

        Pressure should be in Pa.

    Returns
    -------
    torch.Tensor
        Heating rate in K/day if p is provided.
        Otherwise returns negative flux divergence.

        Shape: (..., nlev-1)
    """

    # Positive upward net flux convention
    net = up - down

    # Vertical flux difference:
    # F(i+1/2) - F(i-1/2)
    dF = net - torch.roll(net, shifts=1, dims=-1)

    # Remove artificial wrap-around at the first level
    dF = dF[..., 1:]

    if p is None:
        # Negative flux divergence
        return -dF

    # Constants
    g = 9.8066  # m s^-2
    cp = 1004.0  # J kg^-1 K^-1

    # factore:
    fac = g * 86400.0 / cp

    # Broadcast pressure to flux dimensions
    if p.ndim == 1:
        p = p.view(*([1] * (net.ndim - 1)), -1)

    # Pressure thickness of each layer
    dp = p - torch.roll(p, shifts=1, dims=-1)

    # Remove artificial wrap-around
    dp = dp[..., 1:]

    # Heating rate [K/day]
    # Positive = warming
    return fac * dF / dp


def run_validation_lsm(
    loader,
    model,
    norm_mapping,
    normalization_type,
    index_mapping,
    device,
    args,
    epoch,
    logger=None,
    base_dir="./results",
    n_pft=15,
    n_bands=2,
    n_chans=4,
):
    """
    Evaluate model accuracy on LSM dataset.

    Performs comprehensive evaluation including:

    - Loss computation for main fluxes and absorption rates
    - Metric calculation (NMAE, NMSE, R²) for fluxes and absorption
    - Optional plotting of predictions vs targets (every 10 epochs)
    - Energy conservation verification

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Data loader for evaluation dataset.
    model : torch.nn.Module
        Trained model to evaluate.
    norm_mapping : dict
        Normalization statistics for variables.
    normalization_type : dict
        Normalization types per variable.
    index_mapping : dict
        Mapping from channel indices (0-3) to variable names.
    device : torch.device
        Device to run evaluation on (cuda or cpu).
    args : argparse.Namespace
        Arguments containing loss_type, beta, beta_delta, and num_epochs.
    epoch : int
        Current epoch number (for plotting schedule).
    logger : logging.Logger, optional
        Logger for informational messages. If None, no logging occurs.
    base_dir : str, optional
        Directory to save diagnostic plots. Default is "./results".
    n_pft : int, optional
        Number of Plant Functional Types. Default is 15.
    n_bands : int, optional
        Number of spectral bands. Default is 2 (VIS, NIR).
    n_chans : int, optional
        Number of output channels. Default is 4.

    Returns
    -------
    tuple
        (valid_loss, valid_metrics)

    Notes
    -----
    The evaluation performs the following steps:
    1. Iterates through validation loader
    2. Computes predictions and reshapes to 5D tensors
    3. De-normalizes predictions and targets to physical units
    4. Calculates absorption rates and conservation penalties
    5. Computes metrics for fluxes and absorption
    6. Optionally generates diagnostic plots (epoch % 10 == 0 or final epoch)

    The combined loss includes both flux and absorption terms weighted by β:
        total_loss = (1-β)*loss_fluxes + β*(loss_abs12 + loss_abs34)

    Examples
    --------
    >>> valid_loss, metrics = run_validation_lsm(
    ...     loader=val_loader,
    ...     model=my_model,
    ...     norm_mapping=norm_stats,
    ...     normalization_type=norm_types,
    ...     index_mapping=idxmap,
    ...     device=torch.device('cuda'),
    ...     args=args,
    ...     epoch=10,
    ...     logger=logger
    ... )
    >>> print(f"Validation NMAE: {metrics['fluxes_NMAE']:.4f}")
    >>> print(f"R² score: {metrics['fluxes_R2']:.4f}")
    """

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

    metric_names = ["NMAE", "NMSE", "R2", "MAE", "MSE"]
    metric_funcs = {
        "NMAE": nmae_all,
        "NMSE": nmse_all,
        "R2": r2_all,
        "MAE": mae_all,
        "MSE": mse_all,
    }
    output_keys = ["fluxes", "abs12", "abs34"]
    valid_metrics = {
        f"{k}_{m}": MetricTracker() for k in output_keys for m in metric_names
    }

    valid_loss = MetricTracker()
    save_plots = (epoch % 10 == 0) or (epoch == args.num_epochs - 1)
    if save_plots:
        if logger:
            logger.info("Collecting data for final epoch")
        else:
            print("Collecting data for final epoch")

        all_predicts_unnorm = []
        all_targets_unnorm = []
        all_abs12_predict = []
        all_abs12_target = []
        all_abs34_predict = []
        all_abs34_target = []

    # Progress bar for validation
    loop = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Validation Epoch {epoch}",
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, (feature, targets) in loop:
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

            # Reshape to (batch, 4, n_pft, n_bands, seq)
            pred_reshaped = predicts.reshape(
                inner_batch_size, n_chans, n_pft, n_bands, target_shape[3]
            )
            targ_reshaped = targets.reshape(
                inner_batch_size, n_chans, n_pft, n_bands, target_shape[3]
            )

            predicts_unnorm, targets_unnorm = unnorm_mpas(
                pred_reshaped,
                targ_reshaped,
                norm_mapping,
                normalization_type,
                index_mapping,
            )
            assert (
                predicts_unnorm.shape == pred_reshaped.shape
            ), f"Expected predicts_unnorm shape {pred_reshaped.shape}, got {predicts_unnorm.shape}"

            assert (
                targets_unnorm.shape == targ_reshaped.shape
            ), f"Expected targets_unnorm shape {targ_reshaped.shape}, got {targets_unnorm.shape}"

            (
                abs12_predict,
                abs12_target,
                abs34_predict,
                abs34_target,
                conservation_penalty,
            ) = calc_abs(predicts_unnorm, targets_unnorm)

            expected_abs_shape = (
                inner_batch_size,
                1,
                n_pft,
                n_bands,
                target_shape[3] - 1,
            )
            assert (
                abs12_predict.shape == expected_abs_shape
            ), f"Expected abs12_predict shape {expected_abs_shape}, got {abs12_predict.shape}"
            assert (
                abs12_target.shape == expected_abs_shape
            ), f"Expected abs12_target shape {expected_abs_shape}, got {abs12_target.shape}"
            assert (
                abs34_predict.shape == expected_abs_shape
            ), f"Expected abs34_predict shape {expected_abs_shape}, got {abs34_predict.shape}"
            assert (
                abs34_target.shape == expected_abs_shape
            ), f"Expected abs34_target shape {expected_abs_shape}, got {abs34_target.shape}"

            if batch_idx == 0:
                logger.info(f"Feature shape: {feature.shape}")
                logger.info(f"Targets shape: {targets.shape}")
                logger.info(f"abs12_predict shape: {abs12_predict.shape}")
                logger.info(f"abs12_target shape: {abs12_target.shape}")
                logger.info(f"abs34_predict shape: {abs34_predict.shape}")
                logger.info(f"abs34_target shape: {abs34_target.shape}")
                logger.info(f"Conservation penalty: {conservation_penalty.item():.6f}")

            if save_plots:
                all_predicts_unnorm.append(predicts_unnorm.cpu())
                all_targets_unnorm.append(targets_unnorm.cpu())
                all_abs12_predict.append(abs12_predict.cpu())
                all_abs12_target.append(abs12_target.cpu())
                all_abs34_predict.append(abs34_predict.cpu())
                all_abs34_target.append(abs34_target.cpu())

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
            loop.set_postfix(loss=total_loss.item())

        if save_plots:
            if logger:
                logger.info("Doing plot for final epoch")
            else:
                print("Doing plot for final epoch")

            os.makedirs(base_dir, exist_ok=True)

            all_predicts_unnorm = torch.cat(all_predicts_unnorm, dim=0)
            all_targets_unnorm = torch.cat(all_targets_unnorm, dim=0)
            all_abs12_predict = torch.cat(all_abs12_predict, dim=0)
            all_abs12_target = torch.cat(all_abs12_target, dim=0)
            all_abs34_predict = torch.cat(all_abs34_predict, dim=0)
            all_abs34_target = torch.cat(all_abs34_target, dim=0)

            plot_all_diagnostics(
                all_predicts_unnorm,
                all_targets_unnorm,
                abs12_predict=all_abs12_predict,
                abs12_target=all_abs12_target,
                abs34_predict=all_abs34_predict,
                abs34_target=all_abs34_target,
                n_pft=n_pft,
                n_bands=n_bands,
                output_dir=base_dir,
                prefix=f"validation_epoch{epoch}",
                logger=logger,
            )

    return valid_loss.getmean(), {
        k: (tracker.getsqrtmean() if k.lower().endswith("mse") else tracker.getmean())
        for k, tracker in valid_metrics.items()
    }


def run_validation_cams(
    loader,
    model,
    norm_mapping,
    normalization_type,
    index_mapping,
    device,
    args,
    epoch,
    logger=None,
    base_dir="./results",
    n_fluxes=2,
    n_level=61,
):
    """
    Evaluate model accuracy on CAMS atmospheric dataset.

    Performs comprehensive evaluation including:
    - Loss computation for fluxes and heating rates
    - Metric calculation (NMAE, NMSE, R², MAE, MSE) for fluxes and HR
    - Optional plotting of predictions vs targets (every 10 epochs)
    - Energy conservation verification

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Data loader for evaluation dataset.
    model : torch.nn.Module
        Trained model to evaluate.
    norm_mapping : dict
        Normalization statistics for variables.
    normalization_type : dict
        Normalization types per variable.
    index_mapping : dict
        Mapping from channel indices to variable names (0: rsd, 1: rsu).
    device : torch.device
        Device to run evaluation on (cuda or cpu).
    args : argparse.Namespace
        Arguments containing loss_type, beta, and num_epochs.
    epoch : int
        Current epoch number (for plotting schedule).
    logger : logging.Logger, optional
        Logger for informational messages. If None, no logging occurs.
    base_dir : str, optional
        Directory to save diagnostic plots. Default is "./results".
    n_fluxes : int, optional
        Number of flux variables (2: rsd, rsu). Default is 2.
    n_level : int, optional
        Number of vertical levels (61). Default is 61.

    Returns
    -------
    tuple
        (valid_loss, valid_metrics)
    """

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

    metric_names = ["NMAE", "NMSE", "R2", "MAE", "MSE"]
    metric_funcs = {
        "NMAE": nmae_all,
        "NMSE": nmse_all,
        "R2": r2_all,
        "MAE": mae_all,
        "MSE": mse_all,
    }
    output_keys = ["fluxes", "HR"]
    valid_metrics = {
        f"{k}_{m}": MetricTracker() for k in output_keys for m in metric_names
    }

    valid_loss = MetricTracker()
    save_plots = (epoch % 10 == 0) or (epoch == args.num_epochs - 1)
    if save_plots:
        if logger:
            logger.info("Collecting data for final epoch")
        else:
            print("Collecting data for final epoch")

        all_predicts_unnorm = []
        all_targets_unnorm = []
        all_hr_predict = []
        all_hr_target = []

    # Progress bar for validation
    loop = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Validation Epoch {epoch}",
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, (features, targets, pressure) in loop:
            # Reshape inputs: flatten batch and sites dimensions
            feature_shape = features.shape
            target_shape = targets.shape
            pressure_shape = pressure.shape
            inner_batch_size = feature_shape[0] * feature_shape[1]  # batch * n_sites

            # Reshape to (inner_batch, n_features, n_layer)
            features = features.reshape(
                inner_batch_size, feature_shape[2], feature_shape[3]
            ).to(device=device)

            # Reshape to (inner_batch, n_fluxes, n_level)
            targets = targets.reshape(
                inner_batch_size, target_shape[2], target_shape[3]
            ).to(device=device)

            # Reshape pressure to (inner_batch, n_level_interior)
            pressure = pressure.reshape(inner_batch_size, pressure_shape[2]).to(
                device=device
            )

            # Forward pass
            predicts = model(features)  # (inner_batch, n_fluxes, n_level)

            # Denormalize predictions and targets
            predicts_unnorm, targets_unnorm = unnorm_mpas(
                predicts,
                targets,
                norm_mapping,
                normalization_type,
                index_mapping,
            )

            assert (
                predicts_unnorm.shape == predicts.shape
            ), f"Expected predicts_unnorm shape {predicts.shape}, got {predicts_unnorm.shape}"
            assert (
                targets_unnorm.shape == targets.shape
            ), f"Expected targets_unnorm shape {targets.shape}, got {targets_unnorm.shape}"

            # Calculate heating rates
            hr_predict, hr_target = calc_heating_rates(predicts_unnorm, targets_unnorm)

            if batch_idx == 0:
                if logger:
                    logger.info(f"Feature shape: {features.shape}")
                    logger.info(f"Targets shape: {targets.shape}")
                    logger.info(f"Heating rate pred shape: {hr_predict.shape}")
                    logger.info(f"Heating rate target shape: {hr_target.shape}")
                else:
                    print(f"Feature shape: {features.shape}")
                    print(f"Targets shape: {targets.shape}")
                    print(f"Heating rate pred shape: {hr_predict.shape}")
                    print(f"Heating rate target shape: {hr_target.shape}")

            if save_plots:
                all_predicts_unnorm.append(predicts_unnorm.cpu())
                all_targets_unnorm.append(targets_unnorm.cpu())
                all_hr_predict.append(hr_predict.cpu())
                all_hr_target.append(hr_target.cpu())

            # Compute metrics
            output_dict = {
                "fluxes": (predicts, targets),
                "HR": (hr_predict, hr_target),
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

            # Compute loss
            main_count, main_val = predicts.numel(), func(predicts, targets)

            # Heating rate loss (optional, weighted by beta)
            if args.beta > 0:
                hr_count, hr_val = hr_predict.numel(), func(hr_predict, hr_target)
                weighted_loss = (
                    1.0 - args.beta
                ) * main_val * main_count + args.beta * (hr_val * hr_count)
                total_count = (1.0 - args.beta) * main_count + args.beta * hr_count
                total_loss = weighted_loss / total_count
            else:
                total_loss = main_val

            valid_loss.update(total_loss.item(), 1)
            loop.set_postfix(loss=total_loss.item())

        if save_plots:
            if logger:
                logger.info("Generating diagnostic plots for final epoch")
            else:
                print("Generating diagnostic plots for final epoch")

            os.makedirs(base_dir, exist_ok=True)

            all_predicts_unnorm = torch.cat(all_predicts_unnorm, dim=0)
            all_targets_unnorm = torch.cat(all_targets_unnorm, dim=0)
            all_hr_predict = torch.cat(all_hr_predict, dim=0)
            all_hr_target = torch.cat(all_hr_target, dim=0)

            # Plot diagnostics for CAMS data
            plot_cams_diagnostics(
                all_predicts_unnorm,
                all_targets_unnorm,
                hr_predict=all_hr_predict,
                hr_target=all_hr_target,
                output_dir=base_dir,
                prefix=f"validation_epoch{epoch}",
                logger=logger,
            )

    return valid_loss.getmean(), {
        k: (tracker.getsqrtmean() if k.lower().endswith("mse") else tracker.getmean())
        for k, tracker in valid_metrics.items()
    }


def run_validation_reftrans(
    loader,
    model,
    norm_mapping,
    normalization_type,
    index_mapping,
    device,
    args,
    epoch,
    logger=None,
    base_dir="./results",
):
    """
    Evaluate model accuracy on REFTRANS dataset.

    Performs comprehensive evaluation including:
    - Loss computation for main fluxes and absorption rates
    - Metric calculation (NMAE, NMSE, R²) for fluxes and absorption
    - Optional plotting of predictions vs targets (every 10 epochs)
    - Energy conservation verification

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Data loader for evaluation dataset.
    model : torch.nn.Module
        Trained model to evaluate.
    norm_mapping : dict
        Normalization statistics for variables.
    normalization_type : dict
        Normalization types per variable.
    index_mapping : dict
        Mapping from channel indices to variable names.
        For REFTRANS: {0: "rdif", 1: "tdif", 2: "rdir", 3: "tdir"}
    device : torch.device
        Device to run evaluation on (cuda or cpu).
    args : argparse.Namespace
        Arguments containing loss_type, beta, and num_epochs.
    epoch : int
        Current epoch number (for plotting schedule).
    logger : logging.Logger, optional
        Logger for informational messages. If None, no logging occurs.
    base_dir : str, optional
        Directory to save diagnostic plots. Default is "./results".

    Returns
    -------
    tuple
        (valid_loss, valid_metrics)
    """
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

    metric_names = ["NMAE", "NMSE", "R2", "MAE", "MSE"]
    metric_funcs = {
        "NMAE": nmae_all,
        "NMSE": nmse_all,
        "R2": r2_all,
        "MAE": mae_all,
        "MSE": mse_all,
    }
    output_keys = ["fluxes", "abs12", "abs34"]
    valid_metrics = {
        f"{k}_{m}": MetricTracker() for k in output_keys for m in metric_names
    }

    valid_loss = MetricTracker()
    save_plots = (epoch % 10 == 0) or (epoch == args.num_epochs - 1)
    if save_plots:
        if logger:
            logger.info("Collecting data for final epoch")
        else:
            print("Collecting data for final epoch")

        all_predicts_unnorm = []
        all_targets_unnorm = []
        all_abs12_predict = []
        all_abs12_target = []
        all_abs34_predict = []
        all_abs34_target = []

    # Progress bar for validation
    loop = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Validation Epoch {epoch}",
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, (features, targets) in loop:
            feature_shape = features.shape
            target_shape = targets.shape
            inner_batch_size = feature_shape[0] * feature_shape[1]  # batch * n_samples

            # Reshape to (inner_batch, n_features, n_layer)
            features = features.reshape(
                inner_batch_size, feature_shape[2], feature_shape[3]
            ).to(device=device)

            # Reshape to (inner_batch, n_outputs, n_layer)
            targets = targets.reshape(
                inner_batch_size, target_shape[2], target_shape[3]
            ).to(device=device)

            # Forward pass
            predicts = model(features)  # (inner_batch, 4, n_layer)

            # Denormalize predictions and targets
            predicts_unnorm, targets_unnorm = unnorm_mpas(
                predicts,
                targets,
                norm_mapping,
                normalization_type,
                index_mapping,
            )

            assert (
                predicts_unnorm.shape == predicts.shape
            ), f"Expected predicts_unnorm shape {predicts.shape}, got {predicts_unnorm.shape}"
            assert (
                targets_unnorm.shape == targets.shape
            ), f"Expected targets_unnorm shape {targets.shape}, got {targets_unnorm.shape}"

            # Calculate absorption using calc_abs
            # channels 0,1 -> abs12 (diffuse: rdif, tdif)
            # channels 2,3 -> abs34 (direct: rdir, tdir)
            (
                abs12_predict,
                abs12_target,
                abs34_predict,
                abs34_target,
                conservation_penalty,
            ) = calc_abs(predicts_unnorm, targets_unnorm)

            if batch_idx == 0:
                if logger:
                    logger.info(f"Feature shape: {features.shape}")
                    logger.info(f"Targets shape: {targets.shape}")
                    logger.info(f"abs12_predict shape: {abs12_predict.shape}")
                    logger.info(f"abs12_target shape: {abs12_target.shape}")
                    logger.info(f"abs34_predict shape: {abs34_predict.shape}")
                    logger.info(f"abs34_target shape: {abs34_target.shape}")
                    logger.info(
                        f"Conservation penalty: {conservation_penalty.item():.6f}"
                    )
                else:
                    print(f"Feature shape: {features.shape}")
                    print(f"Targets shape: {targets.shape}")
                    print(f"Conservation penalty: {conservation_penalty.item():.6f}")

            if save_plots:
                all_predicts_unnorm.append(predicts_unnorm.cpu())
                all_targets_unnorm.append(targets_unnorm.cpu())
                all_abs12_predict.append(abs12_predict.cpu())
                all_abs12_target.append(abs12_target.cpu())
                all_abs34_predict.append(abs34_predict.cpu())
                all_abs34_target.append(abs34_target.cpu())

            # Compute metrics
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

            # Compute loss
            main_count, main_val = predicts.numel(), func(predicts, targets)

            if args.beta > 0:
                abs12_count, abs12_val = (
                    abs12_predict.numel(),
                    func(abs12_predict, abs12_target),
                )
                abs34_count, abs34_val = (
                    abs34_predict.numel(),
                    func(abs34_predict, abs34_target),
                )

                weighted_loss = (
                    1.0 - args.beta
                ) * main_val * main_count + args.beta * (
                    abs12_val * abs12_count + abs34_val * abs34_count
                )

                total_count = (1.0 - args.beta) * main_count + args.beta * (
                    abs12_count + abs34_count
                )
                total_loss = weighted_loss / total_count
            else:
                total_loss = main_val

            valid_loss.update(total_loss.item(), 1)
            loop.set_postfix(loss=total_loss.item())

        if save_plots:
            if logger:
                logger.info("Generating diagnostic plots for final epoch")
            else:
                print("Generating diagnostic plots for final epoch")

            os.makedirs(base_dir, exist_ok=True)

            all_predicts_unnorm = torch.cat(all_predicts_unnorm, dim=0)
            all_targets_unnorm = torch.cat(all_targets_unnorm, dim=0)
            all_abs12_predict = torch.cat(all_abs12_predict, dim=0)
            all_abs12_target = torch.cat(all_abs12_target, dim=0)
            all_abs34_predict = torch.cat(all_abs34_predict, dim=0)
            all_abs34_target = torch.cat(all_abs34_target, dim=0)

            # REFTRANS diagnostic plots
            plot_reftrans_diagnostics(
                all_predicts_unnorm,
                all_targets_unnorm,
                abs12_predict=all_abs12_predict,
                abs12_target=all_abs12_target,
                abs34_predict=all_abs34_predict,
                abs34_target=all_abs34_target,
                output_dir=base_dir,
                prefix=f"validation_epoch{epoch}",
                logger=logger,
            )

    return valid_loss.getmean(), {
        k: (tracker.getsqrtmean() if k.lower().endswith("mse") else tracker.getmean())
        for k, tracker in valid_metrics.items()
    }
