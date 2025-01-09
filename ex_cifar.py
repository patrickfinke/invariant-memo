import torch

from experiment import ExperimentGroup, LinearExperiment, Conv2DExperiment
from plot import plot_pruning, plot_components


def main():
    EXPERIMENT_KWARGS = {
        "namespace": "cifar",
        "dataset_name": "cifar",
        "n_trials": 5,
        "dtype": torch.bfloat16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    d_input = 32
    stride = 2
    maximal_bias = 0.0
    d_hat = (d_input // stride) ** 2

    out_channels = [50, 100, 150, 200, 250, 300]
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

    plot_components(conv2d_experiment_group, linear_experiment_group, ylim_comp=(33_000, 41_000))
    plot_pruning(linear_experiment_group, conv2d_experiment_group, include_legend=False)


if __name__ == "__main__":
    main()
