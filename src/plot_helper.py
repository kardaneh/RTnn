import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import mpltex
import math
from matplotlib import rcParams as mpl
from sklearn.metrics import r2_score
from scipy.stats import iqr
import matplotlib.ticker as ticker
import random
from matplotlib.colors import LogNorm

params = {
    'font.family': 'DejaVu Sans',
#    'figure.dpi': 300,
#    'savefig.dpi': 300,
    'lines.linewidth': 2,
    'lines.dashed_pattern': [4, 2],
    'lines.dashdot_pattern': [6, 3, 2, 3],
    'lines.dotted_pattern': [2, 3],
    'mathtext.rm': 'arial',
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'legend.fontsize': 15,
    'legend.loc': 'best',
    'legend.frameon': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out'
}
mpl.update(params)

import xarray as xr
import os
import collections

def stats(file_list, logger, output_dir, norm_mapping=None):
    """
    """
    variable_data = collections.defaultdict(list)

    if norm_mapping is None:
        norm_mapping = {}

    logger.info("Reading and collecting variables across files...")
    for fpath in file_list:
        try:
            ds = xr.open_dataset(fpath, engine="netcdf4")
            logger.info(f"Processing file: {fpath}")
            for var_name in ds.data_vars:
                data = ds[var_name].values
                variable_data[var_name].append(data.flatten())
            ds.close()
        except Exception as e:
            logger.warning(f"Skipping file {fpath} due to error: {e}")
            continue

    for var_name, arrays in variable_data.items():
        full_data = np.concatenate(arrays)
        if full_data.size == 0:
            logger.warning(f"{var_name} is empty after filtering, skipping.")
            continue

        vmin = float(np.min(full_data))
        vmax = float(np.max(full_data))
        vmean = float(np.mean(full_data))
        vstd = float(np.std(full_data))
        q1 = float(np.percentile(full_data, 25))
        q3 = float(np.percentile(full_data, 75))
        iqr = q3 - q1 if q3 != q1 else 1.0
        median = float(np.median(full_data))

        log_data = np.log1p(np.clip(full_data, a_min=0, a_max=None))
        log_min = float(log_data.min())
        log_max = float(log_data.max())
        log_mean = float(log_data.mean())
        log_std = float(log_data.std())
        log_q1 = float(np.percentile(log_data, 25))
        log_q3 = float(np.percentile(log_data, 75))
        log_iqr = log_q3 - log_q1 if log_q3 != log_q1 else 1.0
        log_median = float(np.median(log_data))

        sqrt_data = np.sqrt(np.clip(full_data, a_min=0, a_max=None))
        sqrt_min = float(sqrt_data.min())
        sqrt_max = float(sqrt_data.max())
        sqrt_mean = float(sqrt_data.mean())
        sqrt_std = float(sqrt_data.std())
        sqrt_q1 = float(np.percentile(sqrt_data, 25))
        sqrt_q3 = float(np.percentile(sqrt_data, 75))
        sqrt_iqr = sqrt_q3 - sqrt_q1 if sqrt_q3 != sqrt_q1 else 1.0
        sqrt_median = float(np.median(sqrt_data))

        norm_mapping[var_name] = {
        'vmin': vmin,
        'vmax': vmax,
        'vmean': vmean,
        'vstd': vstd,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'median': median,

        'log_min': log_min,
        'log_max': log_max,
        'log_mean': log_mean,
        'log_std': log_std,
        'log_q1': log_q1,
        'log_q3': log_q3,
        'log_iqr': log_iqr,
        'log_median': log_median,

        'sqrt_min': sqrt_min,
        'sqrt_max': sqrt_max,
        'sqrt_mean': sqrt_mean,
        'sqrt_std': sqrt_std,
        'sqrt_q1': sqrt_q1,
        'sqrt_q3': sqrt_q3,
        'sqrt_iqr': sqrt_iqr,
        'sqrt_median': sqrt_median
        }
        norm_label = ""
        file_suffix = "_histogram.png"

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.hist(full_data, bins=200)
        ax.set_yscale("log")
        ax.set_title(f"Histogram of {var_name}{norm_label}")
        ax.set_xlabel(var_name + norm_label)
        ax.set_ylabel("Log Count")
        ax.grid(True)
        out_path = os.path.join(output_dir, f"{var_name}{file_suffix}")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    return norm_mapping

def plot_RTM(predicts, targets, filename):
    predicts = predicts.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    num_samples = predicts.shape[0]
    sample_indices = random.sample(range(num_samples), 7)

    fig = plt.figure(tight_layout=True, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.2, wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
    gs = gridspec.GridSpec(2, 2)

    name_dict = {
        (0, 0): {"plotname": r"$\mathrm{Flux_1}$"},
        (0, 1): {"plotname": r"$\mathrm{Flux_2}$"},
        (1, 0): {"plotname": r"$\mathrm{Flux_3}$"},
        (1, 1): {"plotname": r"$\mathrm{Flux_4}$"}
    }

    index_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
    legend_lines = []
    legend_labels = []

    for idx, (i, j) in enumerate(index_map):
        ax = fig.add_subplot(gs[i, j])
        linestyles = mpltex.linestyle_generator()
        for sample_index in sample_indices:
            pred_line, = ax.plot(
                    predicts[sample_index, idx, :],
                    label=f"Predict",
                    **next(linestyles)
                    )
            true_line, = ax.plot(
                    targets[sample_index, idx, :],
                    label=f"True",
                    **next(linestyles)
                    )
            if idx == 0:
                legend_lines.extend([pred_line, true_line])
                legend_labels.extend([pred_line.get_label(), true_line.get_label()])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.set_ylabel(name_dict[(i, j)]["plotname"])
        ax.set_xlabel(r"$\mathrm{Level}$")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, predicts.shape[-1] - 1)
        fig.legend(
                handles=legend_lines,
                labels=legend_labels,
                loc='center right',
                bbox_to_anchor=(1.2, 0.5),
                borderaxespad=0.5,
                frameon=False,
                ncol=1
                )
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_HeatRate(abs12_predict, abs12_target, abs34_predict, abs34_target, filename):
    abs12_predict = abs12_predict.cpu().detach().numpy()
    abs12_target = abs12_target.cpu().detach().numpy()
    abs34_predict = abs34_predict.cpu().detach().numpy()
    abs34_target = abs34_target.cpu().detach().numpy()

    num_samples = abs12_predict.shape[0]
    sample_indices = random.sample(range(num_samples), 7)

    fig = plt.figure(tight_layout=False, figsize=(6, 12))
    plt.subplots_adjust(hspace=0.2, wspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
    gs = gridspec.GridSpec(2, 1, figure=fig)

    legend_lines = []
    legend_labels = []

    ax1 = fig.add_subplot(gs[0, 0])
    linestyles = mpltex.linestyle_generator()

    for sample_index in sample_indices:
        pred_line, = ax1.plot(
            abs12_predict[sample_index, 0, :],
            label="Predict",
            **next(linestyles)
        )
        true_line, = ax1.plot(
            abs12_target[sample_index, 0, :],
            label="True",
            **next(linestyles)
        )

        legend_lines.extend([pred_line, true_line])
        legend_labels.extend([pred_line.get_label(), true_line.get_label()])

    ax1.set_ylabel(r"$\mathrm{Absorption_{12}}$")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax1.set_xlim(0, abs12_predict.shape[-1] - 1)
    ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(gs[1, 0])
    linestyles = mpltex.linestyle_generator()

    for sample_index in sample_indices:
        ax2.plot(
            abs34_predict[sample_index, 0, :],
            label="Predict",
            **next(linestyles)
        )
        ax2.plot(
            abs34_target[sample_index, 0, :],
            label="True",
            **next(linestyles)
        )

    ax2.set_ylabel(r"$\mathrm{Absorption_{34}}$")
    ax2.set_xlabel(r"$\mathrm{Level}$")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax2.set_xlim(0, abs34_predict.shape[-1] - 1)
    ax2.set_ylim(0, 1)

    fig.legend(
        handles=legend_lines,
        labels=legend_labels,
        loc='center right',
        bbox_to_anchor=(1.3, 0.5),
        borderaxespad=0.5,
        frameon=False,
        ncol=1
    )

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_flux_and_abs_lines(
        predicts,
        targets,
        abs12_predict=None,
        abs12_target=None,
        abs34_predict=None,
        abs34_target=None,
        filename="output_lines.png"
        ):
    
    predicts = predicts.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    include_abs12 = abs12_predict is not None and abs12_target is not None
    include_abs34 = abs34_predict is not None and abs34_target is not None
    include_abs = include_abs12 and include_abs34

    if include_abs:
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)

    num_samples = predicts.shape[0]
    sample_indices = random.sample(range(num_samples), 7)

    index_map = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    name_dict = {
        (0, 0): {"plotname": r"$\mathrm{Flux_{1u}}$"},
        (0, 1): {"plotname": r"$\mathrm{Flux_{1d}}$"},
        (1, 0): {"plotname": r"$\mathrm{Flux_{2u}}$"},
        (1, 1): {"plotname": r"$\mathrm{Flux_{2d}}$"}
    }

    legend_lines = []
    legend_labels = []

    for (r, c), props in name_dict.items():
        flux_idx = index_map[(r, c)]
        ax = axes[r, c]
        linestyles = mpltex.linestyle_generator()

        for sample_index in sample_indices:
            pred_line, = ax.plot(
                predicts[sample_index, flux_idx, :],
                label="Predict",
                **next(linestyles)
            )
            true_line, = ax.plot(
                targets[sample_index, flux_idx, :],
                label="True",
                **next(linestyles)
            )
            if (r, c) == (0, 0):
                legend_lines.extend([pred_line, true_line])
                legend_labels.extend([pred_line.get_label(), true_line.get_label()])

        ax.set_xlabel(r"$\mathrm{Level}$")
        ax.set_ylabel(props["plotname"])
        ax.set_xlim(0, predicts.shape[-1] - 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    if include_abs:
        ax = axes[2, 0]
        linestyles = mpltex.linestyle_generator()
        for sample_index in sample_indices:
            pred_line, = ax.plot(
                abs12_predict[sample_index, 0, :].cpu().detach().numpy(),
                label="Predict)",
                **next(linestyles)
            )
            true_line, = ax.plot(
                abs12_target[sample_index, 0, :].cpu().detach().numpy(),
                label="True",
                **next(linestyles)
            )

        ax.set_ylabel(r"$\mathrm{Absorption_{1}}$")
        ax.set_xlabel(r"$\mathrm{Level}$")
        ax.set_xlim(0, abs12_predict.shape[-1] - 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

        ax = axes[2, 1]
        linestyles = mpltex.linestyle_generator()
        for sample_index in sample_indices:
            pred_line, = ax.plot(
                abs34_predict[sample_index, 0, :].cpu().detach().numpy(),
                label="Predict",
                **next(linestyles)
            )
            true_line, = ax.plot(
                abs34_target[sample_index, 0, :].cpu().detach().numpy(),
                label="True",
                **next(linestyles)
            )

        ax.set_ylabel(r"$\mathrm{Absorption_{2}}$")
        ax.set_xlabel(r"$\mathrm{Level}$")
        ax.set_xlim(0, abs34_predict.shape[-1] - 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    fig.legend(
        handles=legend_lines,
        labels=legend_labels,
        loc='center right',
        bbox_to_anchor=(1.1, 0.5),
        borderaxespad=0.5,
        frameon=False,
        ncol=1
    )

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_flux_and_abs(
    predicts,
    targets,
    abs12_predict=None,
    abs12_target=None,
    abs34_predict=None,
    abs34_target=None,
    filename="output.png"
    ): 
    
    include_abs12 = abs12_predict is not None and abs12_target is not None
    include_abs34 = abs34_predict is not None and abs34_target is not None
    include_abs = include_abs12 and include_abs34

    if include_abs:
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)

    index_map = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    name_dict = {
        (0, 0): {"name": "Flux1u", "plotname": r"$\mathrm{Flux_{1u}}$"},
        (0, 1): {"name": "Flux1d", "plotname": r"$\mathrm{Flux_{1d}}$"},
        (1, 0): {"name": "Flux2u", "plotname": r"$\mathrm{Flux_{2u}}$"},
        (1, 1): {"name": "Flux2d", "plotname": r"$\mathrm{Flux_{2d}}$"}
    }

    num_ticks = 4

    for (r, c), props in name_dict.items():
        flux_idx = index_map[(r, c)]
        y_pred = predicts[:, flux_idx, :].reshape(-1).detach().cpu().numpy()
        y_true = targets[:, flux_idx, :].reshape(-1).detach().cpu().numpy()
        ax = axes[r, c]

        hb = ax.hexbin(y_true, y_pred, gridsize=100, cmap='jet', bins='log', vmin=1, vmax=1e6)
        #ax.set_xlim(y_true.min(), y_true.max())
        #ax.set_ylim(y_true.min(), y_true.max())
        ax.xaxis.set_major_locator(ticker.MultipleLocator((y_true.max() - y_true.min()) / 4))
        ax.yaxis.set_major_locator(ticker.MultipleLocator((y_true.max() - y_true.min()) / 4))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r:', linewidth=0.5)
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"$R^2$: {r2:.5f}", transform=ax.transAxes, fontsize=10)
        flux_name = props["plotname"]
        ax.set_title(rf"{flux_name}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

    if include_abs:
        ax = axes[2, 0]
        y_pred = abs12_predict.reshape(-1).detach().cpu().numpy()
        y_true = abs12_target.reshape(-1).detach().cpu().numpy()
        hb = ax.hexbin(y_true, y_pred, gridsize=100, cmap='jet', bins='log', vmin=1, vmax=1e6)
        #ax.set_xlim(y_true.min(), y_true.max())
        #ax.set_ylim(y_true.min(), y_true.max())
        ax.xaxis.set_major_locator(ticker.MultipleLocator((y_true.max() - y_true.min()) / 4))
        ax.yaxis.set_major_locator(ticker.MultipleLocator((y_true.max() - y_true.min()) / 4))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r:', linewidth=0.5)
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"$R^2$: {r2:.5f}", transform=ax.transAxes, fontsize=10)
        ax.set_title(r"$\mathrm{Abs_{1}}$")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

        ax = axes[2, 1]
        y_pred = abs34_predict.reshape(-1).detach().cpu().numpy()
        y_true = abs34_target.reshape(-1).detach().cpu().numpy()
        hb = ax.hexbin(y_true, y_pred, gridsize=100, cmap='jet', bins='log', vmin=1, vmax=1e6)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r:', linewidth=0.5)
        #ax.set_xlim(y_true.min(), y_true.max())
        #ax.set_ylim(y_true.min(), y_true.max())
        ax.xaxis.set_major_locator(ticker.MultipleLocator((y_true.max() - y_true.min()) / 4))
        ax.yaxis.set_major_locator(ticker.MultipleLocator((y_true.max() - y_true.min()) / 4))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"$R^2$: {r2:.5f}", transform=ax.transAxes, fontsize=10)
        ax.set_title(r"$\mathrm{Abs_{2}}$")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
    fig.colorbar(hb, cax=cbar_ax, label=r'$\mathrm{\log_{10}[Count]}$')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_metric_histories(train_history, valid_history, filename="training_validation_metrics.png"):
    """
    Plots training and validation histories for all tracked metrics.

    Parameters:
    -----------
    train_history : dict
        Dictionary containing training metrics history.
    valid_history : dict
        Dictionary containing validation metrics history.
    filename : str
        Output image file name for the plot.
    """
    num_metrics = len(train_history)
    cols = 3
    rows = math.ceil(num_metrics / cols)

    fig = plt.figure(tight_layout=True, figsize=(5 * cols, 4 * rows))
    gs = gridspec.GridSpec(rows, cols)

    for idx, key in enumerate(train_history):
        row, col = divmod(idx, cols)
        ax = fig.add_subplot(gs[row, col])
        linestyles = mpltex.linestyle_generator(markers=[])

        ax.plot(train_history[key], label="train", **next(linestyles))
        ax.plot(valid_history[key], label="valid", **next(linestyles))
        ax.set_yscale("log")
        #ax.set_title(key.replace('_', ' ').upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace('_', ' ').upper())
        ax.legend()
        ax.grid(True)

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_loss_histories(train_loss, valid_loss, filename="training_validation_loss.png"):
    """
    Plots training and validation loss in a single panel.

    Parameters:
    -----------
    train_loss : list or array
        History of training loss values.
    valid_loss : list or array
        History of validation loss values.
    filename : str
        Output image file name for the plot.
    """
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    linestyles = mpltex.linestyle_generator(markers=[])
    ax.plot(train_loss, label="train", **next(linestyles))
    ax.plot(valid_loss, label="valid", **next(linestyles))
    ax.set_yscale("log")
    ax.set_title("LOSS")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.legend()
    ax.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
