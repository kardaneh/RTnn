import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import mpltex
import math
from matplotlib import rcParams as mpl
from sklearn.metrics import r2_score

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

def plot_RTM(predicts, targets, filename, sample_index):
    predicts = predicts.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    fig = plt.figure(tight_layout=True, figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2)

    name_dict = {
        (0, 0): {"name": "Flux1", "plotname": r"$\mathrm{Flux_1}$"},
        (0, 1): {"name": "Flux2", "plotname": r"$\mathrm{Flux_2}$"},
        (1, 0): {"name": "Flux3", "plotname": r"$\mathrm{Flux_3}$"},
        (1, 1): {"name": "Flux4", "plotname": r"$\mathrm{Flux_4}$"}
    }

    index_map = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (i, j) in enumerate(index_map):
        ax = fig.add_subplot(gs[i, j])
        linestyles = mpltex.linestyle_generator(markers=[])
        ax.plot(predicts[sample_index, idx, :], label="predict", **next(linestyles))
        ax.plot(targets[sample_index, idx, :], label="true", **next(linestyles))
        ax.set_title(name_dict[(i, j)]["plotname"], fontsize=10)
        ax.set_xticks(np.arange(0, 56, 25))
        ax.set_ylabel(r"$\mathrm{Flux\ RMSE\ [W\ m^{-2}]}$")
        ax.legend()

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_HeatRate(abs12_predict, abs12_target, abs34_predict, abs34_target, filename, sample_index):
    abs12_predict = abs12_predict.cpu().detach().numpy()
    abs12_target = abs12_target.cpu().detach().numpy()
    abs34_predict = abs34_predict.cpu().detach().numpy()
    abs34_target = abs34_target.cpu().detach().numpy()

    fig = plt.figure(tight_layout=True, figsize=(5, 10))
    gs = gridspec.GridSpec(2, 1)

    linestyles = mpltex.linestyle_generator(markers=[])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(abs12_predict[sample_index, 0, :], label="predict", **next(linestyles))
    ax.plot(abs12_target[sample_index, 0, :], label="true", **next(linestyles))
    ax.set_title(r"$\mathrm{SW\ Heat\ Rate}$", fontsize=10)
    ax.set_ylabel(r"$\mathrm{Heat\ Rate\ [K\ d^{-1}]}$")
    ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    linestyles = mpltex.linestyle_generator()
    ax.plot(abs34_predict[sample_index, 0, :], label="predict", **next(linestyles))
    ax.plot(abs34_target[sample_index, 0, :], label="true", **next(linestyles))
    ax.set_title(r"$\mathrm{LW\ Heat\ Rate}$", fontsize=10)
    ax.set_ylabel(r"$\mathrm{Heat\ Rate\ [K\ d^{-1}]}$")
    ax.legend()

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_flux_and_hr(
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
    include_hr = include_abs12 and include_abs34

    if include_hr:
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Plot labels for each flux
    index_map = {
        (0, 0): 0,  # Flux1
        (0, 1): 1,  # Flux2
        (1, 0): 2,  # Flux3
        (1, 1): 3   # Flux4
    }

    name_dict = {
        (0, 0): {"name": "Flux1", "plotname": r"$\mathrm{Flux_1}$"},
        (0, 1): {"name": "Flux2", "plotname": r"$\mathrm{Flux_2}$"},
        (1, 0): {"name": "Flux3", "plotname": r"$\mathrm{Flux_3}$"},
        (1, 1): {"name": "Flux4", "plotname": r"$\mathrm{Flux_4}$"}
    }

    for (r, c), props in name_dict.items():
        flux_idx = index_map[(r, c)]
        y_pred = predicts[:, flux_idx, :].reshape(-1).detach().cpu().numpy()
        y_true = targets[:, flux_idx, :].reshape(-1).detach().cpu().numpy()
        ax = axes[r, c]

        hb = ax.hexbin(y_true, y_pred, gridsize=200, cmap='jet', bins='log')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1)

        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"$R^2$: {r2:.5f}", transform=ax.transAxes, fontsize=10)
        ax.set_title(props["plotname"])
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

    if include_hr:
        ax = axes[2, 0]
        y_pred = abs12_predict.reshape(-1).detach().cpu().numpy()
        y_true = abs12_target.reshape(-1).detach().cpu().numpy()
        hb = ax.hexbin(y_true, y_pred, gridsize=200, cmap='jet', bins='log')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1)
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"$R^2$: {r2:.5f}", transform=ax.transAxes, fontsize=10)
        ax.set_title(r"$\mathrm{Abs_{12}}$")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

        ax = axes[2, 1]
        y_pred = abs34_predict.reshape(-1).detach().cpu().numpy()
        y_true = abs34_target.reshape(-1).detach().cpu().numpy()
        hb = ax.hexbin(y_true, y_pred, gridsize=200, cmap='jet', bins='log')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1)
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"$R^2$: {r2:.5f}", transform=ax.transAxes, fontsize=10)
        ax.set_title(r"$\mathrm{Abs_{34}}$")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(hb, cax=cbar_ax, label='log10(density)')
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
    cols = 2
    rows = math.ceil(num_metrics / cols)

    fig = plt.figure(tight_layout=True, figsize=(10, 4 * rows))
    gs = gridspec.GridSpec(rows, cols)

    for idx, key in enumerate(train_history):
        row, col = divmod(idx, cols)
        ax = fig.add_subplot(gs[row, col])
        linestyles = mpltex.linestyle_generator(markers=[])

        ax.plot(train_history[key], label="train", **next(linestyles))
        ax.plot(valid_history[key], label="valid", **next(linestyles))
        ax.set_yscale("log")
        ax.set_title(key.replace('_', ' ').upper(), fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
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
