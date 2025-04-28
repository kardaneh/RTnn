import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import mpltex
import math
from matplotlib import rcParams as mpl

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
        (0, 0): {"name": "swuflx", "plotname": r"$\mathrm{Shortwave\ Up}$"},
        (0, 1): {"name": "swdflx", "plotname": r"$\mathrm{Shortwave\ Down}$"},
        (1, 0): {"name": "lwuflx", "plotname": r"$\mathrm{Longwave\ Up}$"},
        (1, 1): {"name": "lwdflx", "plotname": r"$\mathrm{Longwave\ Down}$"}
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

def plot_HeatRate(swhr_predict, swhr_target, lwhr_predict, lwhr_target, filename, sample_index):
    swhr_predict = swhr_predict.cpu().detach().numpy()
    swhr_target = swhr_target.cpu().detach().numpy()
    lwhr_predict = lwhr_predict.cpu().detach().numpy()
    lwhr_target = lwhr_target.cpu().detach().numpy()

    fig = plt.figure(tight_layout=True, figsize=(5, 10))
    gs = gridspec.GridSpec(2, 1)

    linestyles = mpltex.linestyle_generator(markers=[])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(swhr_predict[sample_index, 0, :], label="predict", **next(linestyles))
    ax.plot(swhr_target[sample_index, 0, :], label="true", **next(linestyles))
    ax.set_title(r"$\mathrm{SW\ Heat\ Rate}$", fontsize=10)
    ax.set_ylabel(r"$\mathrm{Heat\ Rate\ [K\ d^{-1}]}$")
    ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    linestyles = mpltex.linestyle_generator()
    ax.plot(lwhr_predict[sample_index, 0, :], label="predict", **next(linestyles))
    ax.plot(lwhr_target[sample_index, 0, :], label="true", **next(linestyles))
    ax.set_title(r"$\mathrm{LW\ Heat\ Rate}$", fontsize=10)
    ax.set_ylabel(r"$\mathrm{Heat\ Rate\ [K\ d^{-1}]}$")
    ax.legend()

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
