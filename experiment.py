import time
import pathlib
import itertools

import torch
import pandas as pd

import algorithm
import data
import utils
import plot


def calculate_required_components(ns_covered, ns_target_covered):
    ns_covered_total = ns_covered.sum(0) # Sum over multi-class.
    ns_covered_total = ns_covered_total.flip(0) # Accending order.

    n_required_components = 0
    ns_required_components = []
    for n_target_covered in ns_target_covered:
        try:
            # Skip ahead until the target is satisfied.
            while ns_covered_total[n_required_components] < n_target_covered:
                n_required_components += 1
        except IndexError:
            break # No more components left, cannot reach target count.

        ns_required_components.append(n_required_components)

    # Returned tensor does not include values for target counts that could not be reached.
    return torch.tensor(ns_required_components)


class ExperimentGroup():
    """A group of experiments with some shared configuration."""

    def __init__(self, namespace, dataset_name, experiment_type, parameter_grid, n_trials=3, dtype=torch.float32, device=None):
        self.namespace = namespace
        self.dataset_name = dataset_name
        self.experiment_type = experiment_type
        self.parameter_grid = parameter_grid
        self.n_trials = n_trials
        self.dtype = dtype
        self.device = device

        self._dataset = None

    @property
    def method_name(self):
        return self.experiment_type.method_name

    @property
    def displayname(self):
        return self.experiment_type.displayname

    def filename(self, experiment, i_trial):
        """Filename for the results of the i-th trial of an experiment."""

        return pathlib.Path(f"{self.namespace}/{self.dataset_name}_{experiment.name}.{i_trial}.pt")

    def load_dataset(self):
        if self._dataset is None:
            samples, labels = data.load(self.dataset_name)
            samples = samples.to(dtype=self.dtype, device=self.device)
            labels = labels.to(device=self.device)
            self._dataset = samples, labels

        return self._dataset

    def get_experiments(self):
        parameter_names = self.parameter_grid.keys()
        parameter_values_grid = self.parameter_grid.values()

        for parameter_values in itertools.product(*parameter_values_grid):
            kwargs = dict(zip(parameter_names, parameter_values))
            yield self.experiment_type(**kwargs)

    def run_experiment_trial_reproducible(self, experiment, i_trial, samples, labels):
        start_time = time.perf_counter()

        seed = utils.stable_hash((self.dataset_name, experiment.name, i_trial))
        utils.manual_seed(seed)
        results = experiment.trial(samples, labels, device=self.device)

        end_time = time.perf_counter()
        results["total_time_elapsed"] = end_time - start_time

        results["seed"] = seed

        return results

    def run(self, overwrite=False):
        print(f">> Load dataset {self.dataset_name} ...")
        samples, labels = self.load_dataset()
        print("   Done.")

        for experiment in self.get_experiments():
            print(f">> Begin {experiment.name} ...")

            for i_trial in range(1, self.n_trials + 1):
                print(f"  >> Trial {i_trial} ...")

                filename = self.filename(experiment, i_trial)
                if not overwrite and utils.exists(filename):
                    print("     Skipped.")
                    continue

                results = self.run_experiment_trial_reproducible(experiment, i_trial, samples, labels)
                utils.save(results, filename)

                print(f"     Done. (took {(results["total_time_elapsed"]) / 60:.1f} min)")

            print("   Done.")

    def _load_single_df(self, experiment, fractions=(0.95, 0.99, 1.0)):
        """Generate DataFrame for plotting."""

        fractions, _ = torch.tensor(fractions).sort()

        samples, _ = self.load_dataset()
        n_samples, d_samples = samples.flatten(1).shape

        results = [utils.load(self.filename(experiment, i_trial)) for i_trial in range(1, self.n_trials + 1)]
        ns_covered = [result["ns_covered"] for result in results]

        ns_target_covered = (fractions * n_samples).to(torch.int64)
        ns_required_components = [calculate_required_components(ns, ns_target_covered) for ns in ns_covered]

        ns_required_components = [ns for ns in ns_required_components if len(ns) == len(fractions)]
        if len(ns_required_components) < len(ns_covered): # Only plot if all restarts interpolated.
            return None
        
        # First layer parameters.
        n_parameters = experiment.n_hyperplanes * d_samples # weights
        n_parameters += experiment.n_hyperplanes # biases

        u_shape = results[0]["u_shape"]
        per_neuron_parameters = utils.prod(u_shape) + 1 # weights + bias

        df = pd.DataFrame({
            "f_covered": torch.cat([fractions[:len(r)] for r in ns_required_components]),
            "n_target_covered": torch.cat([ns_target_covered[:len(r)] for r in ns_required_components]),
            "n_required_components": torch.cat(ns_required_components),
            "n_required_parameters": torch.cat(ns_required_components) * per_neuron_parameters # Second layer parameters.
        })

        df["n_hyperplanes"] = experiment.n_hyperplanes
        df["out_channels"] = experiment.n_hyperplanes # Alias for easy plotting.
        df["n_parameters"] = n_parameters
        df["n_required_total_parameters"] = n_parameters + df["n_required_parameters"]
        df["method"] = experiment.method_name
        df["displayname"] = experiment.displayname

        return df

    def load_df(self, fractions=(0.95, 0.99, 1.0)):
        dfs = (self._load_single_df(e, fractions) for e in self.get_experiments())
        dfs = (df for df in dfs if df is not None)
        return pd.concat(dfs)

    def load_df_interpol(self):
        samples, _ = self.load_dataset()
        n_samples = samples.shape[0]

        records = []
        for experiment in self.get_experiments():
            results = [utils.load(self.filename(experiment, i_trial)) for i_trial in range(1, self.n_trials + 1)]
            ns_covered_total = [result["ns_covered"].sum(0) for result in results]
            has_interpolated = [ns[0].item() == n_samples for ns in ns_covered_total]
            
            for i in has_interpolated:
                records.append({
                    "n_hyperplanes": experiment.n_hyperplanes,
                    "out_channels": experiment.n_hyperplanes, # Alias for easy plotting.
                    "interpolated": i,
                })

        df = pd.DataFrame.from_records(records)
        return df


class Experiment():
    @property
    def name(self):
        raise NotImplementedError()

    def trial(self, samples, labels, device=None):
        raise NotImplementedError()


class LinearExperiment(Experiment):
    method_name = "linear"
    displayname = "FC"

    def __init__(self, n_hyperplanes, maximal_bias):
        self.n_hyperplanes = n_hyperplanes
        self.maximal_bias = maximal_bias

    @property
    def name(self):
        return f"{self.method_name}_{self.n_hyperplanes}_{int(self.maximal_bias*100)}"

    def trial(self, samples, labels, device=None):
        samples = samples.flatten(1)

        tessellation = algorithm.random_tessellation(
            samples.shape[1], self.n_hyperplanes, self.maximal_bias, dtype=samples.dtype, device=device)
        samples = tessellation(samples)

        results = algorithm.random_covering_multiclass(
            samples, labels, algorithm.kernel_linear, device=device)

        results["u_shape"] = samples.shape[1:]

        return results


class RotationExperiment(Experiment):
    method_name = "rotation"
    displayname = "RotInv"

    def __init__(self, n_hyperplanes, maximal_bias):
        self.n_hyperplanes = n_hyperplanes
        self.maximal_bias = maximal_bias

    @property
    def name(self):
        return f"{self.method_name}_{self.n_hyperplanes}_{int(self.maximal_bias*100)}"

    def trial(self, samples, labels, device=None):
        transformed_samples = [samples, -samples]
        samples = torch.stack(transformed_samples, dim=1).flatten(2)

        tessellation = algorithm.random_tessellation(
            samples.shape[-1], self.n_hyperplanes, self.maximal_bias, dtype=samples.dtype, device=device)
        samples = tessellation(samples).transpose_(-1, -2)

        results = algorithm.random_covering_multiclass(
            samples, labels, algorithm.kernel_conv1d_circular, device=device)

        results["u_shape"] = samples.shape[1:]

        return results


class ShiftExperiment(Experiment):
    method_name = "shift"
    displayname = "ShiftInv"

    def __init__(self, n_hyperplanes, maximal_bias, shifts):
        self.n_hyperplanes = n_hyperplanes
        self.maximal_bias = maximal_bias
        self.shifts = shifts

    @property
    def name(self):
        return f"{self.method_name}_{self.n_hyperplanes}_{int(self.maximal_bias*100)}_{'_'.join(map(str, self.shifts))}"

    def trial(self, samples, labels, device=None):
        transformed_samples = [samples.roll(shift, dims=(-2, -1)) for shift in self.shifts]
        samples = torch.stack(transformed_samples, dim=1).flatten(2)

        tessellation = algorithm.random_tessellation(
            samples.shape[-1], self.n_hyperplanes, self.maximal_bias, dtype=samples.dtype, device=device)
        samples = tessellation(samples).transpose_(-1, -2)

        results = algorithm.random_covering_multiclass(
            samples, labels, algorithm.kernel_conv1d_circular, device=device)

        results["u_shape"] = samples.shape[1:]

        return results


class Conv2DExperiment(Experiment):
    method_name = "conv2d"
    displayname = "ConvNet"

    def __init__(self, out_channels, kernel_size, stride, maximal_bias):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.maximal_bias = maximal_bias

        self.n_hyperplanes = out_channels # Alias for easy plotting.

    @property
    def name(self):
        return f"{self.method_name}_{self.out_channels}x{self.kernel_size}x{self.kernel_size}_{self.stride}_{int(self.maximal_bias*100)}"

    def trial(self, samples, labels, device=None):
        tessellation = algorithm.random_conv2d_circular(
            samples.shape[1],
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            maximal_bias=self.maximal_bias,
            dtype=samples.dtype,
            device=device
        )
        samples = tessellation(samples)

        results = algorithm.random_covering_multiclass(
            samples, labels, algorithm.kernel_conv2d_circular, device=device)

        results["u_shape"] = samples.shape[1:]

        return results
