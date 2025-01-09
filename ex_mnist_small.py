import torch

from experiment import ExperimentGroup, LinearExperiment, Conv2DExperiment
from plot import plot_interpolation, plot_parameters


def main():
    EXPERIMENT_KWARGS = {
        "namespace": "mnist_small",
        "dataset_name": "mnist",
        "n_trials": 10,
        "dtype": torch.bfloat16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    stride = 1
    maximal_bias = 0.0

    linear_experiment_group = ExperimentGroup(
        experiment_type=LinearExperiment,
        parameter_grid={
            "n_hyperplanes": [250, 300, 350, 400, 450, 500],
            "maximal_bias": [maximal_bias]
        },
        **EXPERIMENT_KWARGS,
    )
    linear_experiment_group.run()

    conv2d_experiment_group = ExperimentGroup(
        experiment_type=Conv2DExperiment,
        parameter_grid={
            "out_channels": [6, 8, 10, 12, 14, 16],
            "kernel_size": [28],
            "stride": [stride],
            "maximal_bias": [maximal_bias]
        },
        **EXPERIMENT_KWARGS,
    )
    conv2d_experiment_group.run()

    plot_interpolation(linear_experiment_group)
    plot_interpolation(conv2d_experiment_group)
    plot_parameters(conv2d_experiment_group, linear_experiment_group)


if __name__ == "__main__":
    main()
