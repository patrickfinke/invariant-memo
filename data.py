from dataclasses import dataclass
import urllib
import pathlib

import torch
from torchvision import datasets

import utils


DATA_PATH = pathlib.Path("data")


def load(name):
    return torch.load(DATA_PATH / f"{name}.pt")


def save(dataset, name):
    torch.save(dataset, DATA_PATH / f"{name}.pt")


def normalize_radius(samples):
    """Normalize Euclidean radius of data set in-place."""

    samples_flat = samples.reshape(samples.shape[0], -1)
    radius = torch.linalg.norm(samples_flat, ord=2, dim=1).max()
    samples /= radius

    return samples


def generate_xor(n_samples=500, noise=0.35, seed=None):
    """Generate two dimensional XOR data with Gaussian noise."""
    
    if seed is not None:
        utils.manual_seed(seed)

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
    labels = torch.cat((torch.ones(n_samples // 2, dtype=torch.uint8), torch.zeros(n_samples // 2, dtype=torch.uint8)))
    return samples, labels


def load_mnist():
    """Load the MNIST data set."""

    dataset = datasets.MNIST(DATA_PATH, train=True, download=True) # shape (height, width)

    samples, labels = dataset.data, dataset.targets
    samples.unsqueeze_(1) # shape (color, height, width)
    samples = samples.to(torch.float32) / 128 - 1 # map {0..255} to [-1, 1]
    labels = labels.to(torch.int32)

    return samples, labels


def load_cifar10():
    """Load the CIFAR-10 data set."""

    dataset = datasets.CIFAR10(DATA_PATH, train=True, download=True) # shape (height, width, color)

    samples, labels = torch.from_numpy(dataset.data), torch.tensor(dataset.targets)
    samples = samples.transpose_(-1, -2).transpose_(-2, -3) # shape (color, height, width)
    samples = samples.to(torch.float32) / 128 - 1 # map {0..255} to [-1, 1]
    labels = labels.to(torch.int32)

    return samples, labels


def make_shiftinv(dataset_name, shifts):
    samples, labels = load(dataset_name)
    samples = torch.cat([samples.roll(shift, dims=(-2, -1)) for shift in shifts])
    labels = torch.cat([labels] * len(shifts))
    dataset = samples, labels
    return dataset


def main():
    """Generate data sets and save them to disk."""

    print(">> Generating MNIST data set ... ", end="", flush=True)
    samples, labels = load_mnist()
    dataset = normalize_radius(samples), labels
    save(dataset, "mnist")
    print("Done.")

    print(">> Generating CIFAR data set ... ", end="", flush=True)
    samples, labels = load_cifar10()
    dataset = normalize_radius(samples), labels
    save(dataset, "cifar")
    print("Done.")

    print(">> Generating XOR data set ... ", end="", flush=True)
    dataset = generate_xor(n_samples=500, noise=0.35, seed=2) # Seed from 2d example.
    save(dataset, "xor")
    print("Done.")

    print(">> Generating rotation invariant XOR data set ... ", end="", flush=True)
    samples, labels = dataset
    dataset = torch.cat((samples, -samples)), torch.cat((labels, labels))
    save(dataset, "xor-rotinv")
    print("Done.")

    print(">> Generating shift invariant MNIST data set ... ", end="", flush=True)
    dataset = make_shiftinv("mnist", [(0, i) for i in (0, 7, 14, 21)])
    save(dataset, "mnist-shiftinv")
    print("Done.")

    print(">> Generating shift invariant CIFAR data set ... ", end="", flush=True)
    dataset = make_shiftinv("cifar", [(0, i) for i in (0, 8, 16, 24)])
    save(dataset, "cifar-shiftinv")
    print("Done.")


if __name__ == "__main__":
    main()
