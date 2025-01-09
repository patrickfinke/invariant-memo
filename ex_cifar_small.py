import torch

from experiment import ExperimentGroup, LinearExperiment, Conv2DExperiment
from plot import plot_interpolation, plot_parameters


def main():
    EXPERIMENT_KWARGS = {
        "namespace": "cifar_small",
        "dataset_name": "cifar",
        "n_trials": 10,
        "dtype": torch.bfloat16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    stride = 2
    maximal_bias = 0.0

    linear_experiment_group = ExperimentGroup(
        experiment_type=LinearExperiment,
        parameter_grid={
            "n_hyperplanes": [200, 250, 300, 350, 400, 450],
            "maximal_bias": [maximal_bias]
        },
        **EXPERIMENT_KWARGS,
    )
    linear_experiment_group.run()

    conv2d_experiment_group = ExperimentGroup(
        experiment_type=Conv2DExperiment,
        parameter_grid={
            "out_channels": [20, 25, 30, 35, 40, 45],
            "kernel_size": [32],
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
