import pathlib

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import algorithm
import data

plt.rcParams['font.size'] = 9.5


PLOT_PATH = pathlib.Path("plots")

PLOT_KWARGS = {
    "palette": "colorblind",
    "markersize": 10,
    "markeredgewidth": 1.5,
    "linewidth": 1.5,
}


def save(fig, filename, close=True):
    path = PLOT_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path)
    print(f"[+] Saved figure: '{path}'")

    if close:
        plt.close(fig)


def plot_interpolation(experiment_group, **axis_kwargs):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    sns.lineplot(
        experiment_group.load_df_interpol(), x="out_channels", y="interpolated",
        ax=ax, marker="x", markersize=10, markeredgewidth=1.5, linewidth=1.5
    )

    ax.set_xlabel(f"first layer {'neurons' if experiment_group.method_name == 'linear' else 'filters'} $K$")
    ax.set_ylabel(None)
    ax.set_ylim(0.0, 1.0)
    ax.set(**axis_kwargs)
    ax.grid()

    fig.tight_layout()

    save(fig, f"{experiment_group.namespace}/{experiment_group.dataset_name}_{experiment_group.method_name}_interpolation.pdf")


def plot_parameters_on_axis(ax, df, enable_legend=True):
    sns.lineplot(
        df, x="out_channels", y="n_required_total_parameters",
        hue="f_covered", style="f_covered",
        errorbar="pi",
        ax=ax, **PLOT_KWARGS, markers = ["o", "^", "X"],
        legend=True if enable_legend else None
    )


def plot_components_on_axis(ax, df, enable_legend=True):
    sns.lineplot(
        df, x="out_channels", y="n_required_components",
        hue="f_covered", style="f_covered",
        errorbar="pi",
        ax=ax, **PLOT_KWARGS, markers = ["o", "^", "X"],
        legend=True if enable_legend else None,
    )


def plot_components(conv_experiment_group, linear_experiment_group, ylim_comp=None):
    fig, axs = plt.subplots(ncols=2, figsize=(5.5, 2.5), layout="constrained", gridspec_kw={'hspace': 0.05, 'wspace': 0.1})
    ax_conv, ax_linear = axs

    df_conv = conv_experiment_group.load_df()
    df_linear = linear_experiment_group.load_df()

    plot_components_on_axis(ax_conv, df_conv, enable_legend=False)
    ax_conv.set_title("convolutional network")
    ax_conv.set_xlabel(r"filters in first layer $K_{\text{ConvNet}}$")
    ax_conv.set_ylabel("filters in second layer")
    if ylim_comp is not None: ax_conv.set_ylim(*ylim_comp)
    ax_conv.grid()

    plot_components_on_axis(ax_linear, df_linear)
    ax_linear.set_title("fully-connected network")
    ax_linear.set_xlabel(r"neurons in first layer $K_{\text{FC}}$")
    ax_linear.set_ylabel("neurons in second layer")
    if ylim_comp is not None: ax_linear.set_ylim(*ylim_comp)
    ax_linear.grid()
    ax_linear.legend(reverse=True)

    handles, labels = ax_linear.get_legend_handles_labels()
    new_labels = [f"{float(l)*100:.0f}%" for l in labels]
    ax_linear.legend(handles[::-1], new_labels[::-1])

    save(fig, f"{conv_experiment_group.namespace}/{conv_experiment_group.dataset_name}_comp.pdf")


def plot_parameters(conv_experiment_group, linear_experiment_group, ylim=None):
    fig, axs = plt.subplots(ncols=2, figsize=(5.5, 2.5), layout="constrained", gridspec_kw={'hspace': 0.05, 'wspace': 0.1})
    ax_conv, ax_linear = axs

    df_conv = conv_experiment_group.load_df()
    df_linear = linear_experiment_group.load_df()

    plot_parameters_on_axis(ax_conv, df_conv, enable_legend=False)
    ax_conv.set_title("convolutional network")
    ax_conv.set_xlabel(r"filters in first layer $K_{\text{ConvNet}}$")
    ax_conv.set_ylabel("total parameters")
    if ylim is not None: ax_conv.set_ylim(*ylim)
    ax_conv.grid()

    plot_parameters_on_axis(ax_linear, df_linear)
    ax_linear.set_title("fully-connected network")
    ax_linear.set_xlabel(r"neurons in first layer $K_{\text{FC}}$")
    ax_linear.set_ylabel("total parameters")
    if ylim is not None: ax_linear.set_ylim(*ylim)
    ax_linear.grid()
    ax_linear.legend(reverse=True)

    handles, labels = ax_linear.get_legend_handles_labels()
    new_labels = [f"{float(l)*100:.0f}%" for l in labels]
    ax_linear.legend(handles[::-1], new_labels[::-1])

    save(fig, f"{conv_experiment_group.namespace}/{conv_experiment_group.dataset_name}_param.pdf")


def plot_pruning(*experiments_groups, include_legend=True):
    assert all(experiments_groups[0].namespace == e.namespace for e in experiments_groups)

    fig, ax = plt.subplots(figsize=(3.0, 2.6))

    fractions = [i / 10 for i in range(0, 11)]

    dfs = (experiment_group.load_df(fractions) for experiment_group in experiments_groups)
    dfs = (df.loc[df["out_channels"] == df["out_channels"].max()] for df in dfs)
    df = pd.concat(dfs)

    sns.lineplot(
        df, x="f_covered", y="n_required_components",
        hue="displayname", style="displayname",
        ax=ax, **PLOT_KWARGS,
        markers=["o", "^"],
        errorbar="pi",
        legend=include_legend
    )

    samples, _ = experiments_groups[0].load_dataset()
    n_samples = samples.shape[0]

    M = df["n_required_components"].max()
    ax.plot([0.0, M / n_samples], [0, M], linestyle="dashed", color="gray", label="trivial")

    ax.set_xlabel("fraction of interpolated samples $\\alpha$")
    ax.set_ylabel(None)
    if include_legend: ax.legend(reverse=True)
    ax.grid()

    width = 0.6
    plt.tight_layout()
    plt.subplots_adjust(left=(1 - width) / 2, right=(1 + width) / 2)

    save(fig, f"{experiments_groups[0].namespace}/{experiments_groups[0].dataset_name}_pruning.pdf")
