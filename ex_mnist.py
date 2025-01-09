import torch

from experiment import ExperimentGroup, LinearExperiment, Conv2DExperiment
from plot import plot_pruning, plot_components


def main():
    EXPERIMENT_KWARGS = {
        "namespace": "mnist",
        "dataset_name": "mnist",
        "n_trials": 5,
        "dtype": torch.bfloat16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    d_input = 28
    stride = 1
    maximal_bias = 0.0
    d_hat = (d_input // stride) ** 2

    out_channels = [20, 40, 60, 80, 100]
    n_hyperplanes = [i * d_hat for i in out_channels]

    linear_experiment_group = ExperimentGroup(
        experiment_type=LinearExperiment,
        parameter_grid={
            "n_hyperplanes": n_hyperplanes,
            "maximal_bias": [maximal_bias]
        },
        **EXPERIMENT_KWARGS,
    )
    linear_experiment_group.run()

    conv2d_experiment_group = ExperimentGroup(
        experiment_type=Conv2DExperiment,
        parameter_grid={
            "out_channels": out_channels,
            "kernel_size": [d_input],
            "stride": [stride],
            "maximal_bias": [maximal_bias]
        },
        **EXPERIMENT_KWARGS,
    )
    conv2d_experiment_group.run()

    plot_components(conv2d_experiment_group, linear_experiment_group, ylim_comp=(1_750, 6_500))
    plot_pruning(linear_experiment_group, conv2d_experiment_group)


if __name__ == "__main__":
    main()
