import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import algorithm
import utils
import plot


def generate_samples(n_samples, noise=0.35):
    """Generate two dimensional XOR data with Gaussian noise."""

    assert n_samples % 2 == 0, "n_samples must be even"
    
    samples_positive = torch.empty((n_samples // 2, 2))
    samples_positive[:, 0] = 1.0
    samples_positive[:, 1] = 1.0

    samples_negative = torch.empty((n_samples // 2, 2))
    samples_negative[:, 0] = -1.0
    samples_negative[:, 1] = 1.0

    mask = torch.empty(n_samples // 2, dtype=torch.bool)
    samples_positive[mask.bernoulli_()] *= -1
    samples_negative[mask.bernoulli_()] *= -1

    shift = torch.empty((n_samples // 2, 2))
    samples_positive += shift.normal_(0, noise)
    samples_negative += shift.normal_(0, noise)

    samples = torch.cat((samples_positive, samples_negative))
    mask_positive = torch.zeros(n_samples, dtype=torch.bool)
    mask_positive[:n_samples//2] = True
    return samples, mask_positive


def calculate_weights(samples, mask_positive, indices, kernel):
    """Recover the weights and biases from the accepted candidate samples."""

    u = (samples[indices] <= 0).to(samples.dtype)
    m = kernel(samples[~mask_positive], u).amin(0)
    return u, m


def generate_invariant_network(samples, mask_positive):
    """Execute the algorithm and construct the invariant network."""

    tessellation = algorithm.random_tessellation(2, 1_000, 5.0)

    def transform(inputs):
        inputs = torch.stack((inputs, -inputs), dim=1)
        inputs = tessellation(inputs)
        return inputs.transpose_(-1, -2)

    samples = transform(samples)

    indices, covering = algorithm.random_covering(
        samples, mask_positive, algorithm.kernel_conv1d_circular)

    u, m = calculate_weights(
        samples, mask_positive, indices, algorithm.kernel_conv1d_circular)

    def network(inputs):
        out = transform(inputs)
        out = algorithm.kernel_conv1d_circular(out, u) < m
        return algorithm.threshold(out)

    return indices, network


def generate_noninvariant_network(samples, mask_positive):
    """Execute the algorithm and construct the noninvariant network."""

    transform = algorithm.random_tessellation(2, 1_000, 5.0)

    samples = transform(samples)

    indices, covering = algorithm.random_covering(
        samples, mask_positive, algorithm.kernel_linear)

    u, m = calculate_weights(
        samples, mask_positive, indices, algorithm.kernel_linear)

    def network(inputs):
        out = transform(inputs)
        out = algorithm.kernel_linear(out, u) < m
        return algorithm.threshold(out)

    return indices, network


def plot_decision_boundary(network, ax, steps=100):
    """Plot the decision boundary (for each neuron) for a network."""

    xs, ys = np.meshgrid(
        np.linspace(*ax.get_xlim(), steps),
        np.linspace(*ax.get_ylim(), steps)
    )

    grid = np.vstack([xs.ravel(), ys.ravel()]).T
    grid = torch.tensor(grid, dtype=torch.float32)
    out = network(grid).numpy()

    colors = (plt.get_cmap("tab10")(i) for i in range(len(out.T)))
    for o, color in zip(out.T, colors):
        zs = np.reshape(o, xs.shape)

        color = mpl.colors.to_rgba(color, alpha=0.5)
        cmap = mpl.colors.ListedColormap(["#00000000", color])
        ax.contourf(xs, ys, zs, levels=1, cmap=cmap, zorder=2.0)

        color = mpl.colors.to_rgba(color, alpha=1.0)
        cmap = mpl.colors.ListedColormap(["#00000000", color])
        ax.contour(xs, ys, zs, levels=1, linewidths=2, cmap=cmap, zorder=2.0)


def make_plot(samples, mask_positive, generate_network_fn):
    """Plot the network decision boundary on top of the data."""

    idx, network = generate_network_fn(samples, mask_positive)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")

    ax.grid()

    plot_decision_boundary(network, ax)

    kwargs = {"marker": "o", "edgecolors": "black", "zorder": 3.0}
    ax.scatter(samples[mask_positive, 0], samples[mask_positive, 1], c="red", **kwargs)
    ax.scatter(samples[~mask_positive, 0], samples[~mask_positive, 1], c="blue", **kwargs)

    kwargs = {"marker": "*", "edgecolors": "white", "linewidths": 1.5, "zorder": 4.0}
    colors = [plt.get_cmap("tab10")(i) for i in range(len(idx))]
    ax.scatter(samples[idx, 0], samples[idx, 1], c=colors, s=250, **kwargs)

    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])

    return fig


def main():
    utils.manual_seed(2)
    samples, mask_positive = generate_samples(500)
    fig = make_plot(samples, mask_positive, generate_invariant_network)
    plot.save(fig, "2d/invariant.pdf")

    utils.manual_seed(2)
    fig = make_plot(samples, mask_positive, generate_noninvariant_network)
    plot.save(fig, "2d/noninvariant.pdf")


if __name__ == "__main__":
    main()
