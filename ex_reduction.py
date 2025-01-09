import torch

from experiment import ExperimentGroup, LinearExperiment, RotationExperiment, ShiftExperiment


def calculate_mean_reduction(experiment_group_linear, experiment_group_invariant):
    fractions = [1.0]

    df_linear = experiment_group_linear.load_df(fractions)
    df_linear = df_linear.loc[df_linear["n_hyperplanes"] == df_linear["n_hyperplanes"].max()]

    df_invariant = experiment_group_invariant.load_df(fractions)
    df_invariant = df_invariant.loc[df_invariant["out_channels"] == df_invariant["out_channels"].max()]

    f_reduction = df_linear["n_required_components"].mean() / df_invariant["n_required_components"].mean()
    return f_reduction


def run_once_rotation(base_dataset, rotinv_dataset):
    EXPERIMENT_KWARGS = {
        "namespace": "xor_reduction",
        "n_trials": 50,
        "device": "cpu",
    }

    out_channels = 5_000
    n_hyperplanes = 2 * out_channels

    experiment_group_linear = ExperimentGroup(
        experiment_type=LinearExperiment,
        dataset_name=rotinv_dataset,
        parameter_grid={
            "n_hyperplanes": [n_hyperplanes],
            "maximal_bias": [2.0]
        },
        **EXPERIMENT_KWARGS,
    )
    experiment_group_linear.run()

    experiment_group_rotation = ExperimentGroup(
        experiment_type=RotationExperiment,
        dataset_name=base_dataset,
        parameter_grid={
            "n_hyperplanes": [out_channels],
            "maximal_bias": [2.0]
        },
        **EXPERIMENT_KWARGS,
    )
    experiment_group_rotation.run()

    f_reduction = calculate_mean_reduction(experiment_group_linear, experiment_group_rotation)
    return f_reduction


def run_once_shift(base_dataset, shiftinv_dataset, shifts):
    EXPERIMENT_KWARGS = {
        "namespace": f"{base_dataset}_reduction",
        "n_trials": 5,
        "dtype": torch.bfloat16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    out_channels = 5_000
    n_hyperplanes = len(shifts) * out_channels

    linear_experiment_group = ExperimentGroup(
        experiment_type=LinearExperiment,
        dataset_name=shiftinv_dataset,
        parameter_grid={
            "n_hyperplanes": [n_hyperplanes],
            "maximal_bias": [0.0]
        },
        **EXPERIMENT_KWARGS,
    )
    linear_experiment_group.run()

    shift_base_experiment_group = ExperimentGroup(
        experiment_type=ShiftExperiment,
        dataset_name=base_dataset,
        parameter_grid={
            "n_hyperplanes": [out_channels],
            "maximal_bias": [0.0],
            "shifts": [shifts]
        },
        **EXPERIMENT_KWARGS,
    )
    shift_base_experiment_group.run()

    f_reduction = calculate_mean_reduction(linear_experiment_group, shift_base_experiment_group)
    return f_reduction


def main():
    xor = run_once_rotation("xor", "xor-rotinv")
    xor_star = run_once_rotation("xor", "xor")
    mnist = run_once_shift("mnist", "mnist-shiftinv", [(0, i) for i in (0, 7, 14, 21)])
    cifar = run_once_shift("cifar", "cifar-shiftinv", [(0, i) for i in (0, 8, 16, 24)])

    print("== RESULTS ==")
    print(f"XOR      {xor:.2f}")
    print(f"MNIST    {mnist:.2f}")
    print(f"CIFAR    {cifar:.2f}")
    print(f"XOR*     {xor_star:.2f}")
    print("=============")


if __name__ == "__main__":
    main()
