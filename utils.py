import pathlib
import random
import functools
import operator as op
import pickle
import hashlib

import numpy as np
import torch


RESULTS_PATH = pathlib.Path("results")


def prod(values):
    """Compute a product of values from an iterable."""

    return functools.reduce(op.mul, values, 1)


def save(results, filename):
    """Save results. Overwrites existing files."""

    path = RESULTS_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return torch.save(results, path)


def exists(filename):
    """Check if results exist."""

    path = RESULTS_PATH / filename
    return path.exists()


def load(filename):
    """Load results."""

    path = RESULTS_PATH / filename
    return torch.load(path)


def manual_seed(seed):
    """Set a manual random seed in Python, NumPy and PyTorch all at once."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_hash(obj):
    """Return a stable (does not differ between runs) hash of an object.
    
    The built-in *hash()* function changes between runs."""

    serialized = pickle.dumps(obj, protocol=4)
    hex_hash = hashlib.sha256(serialized).hexdigest()
    int_hash = int(hex_hash, 16)
    bounded_hash = abs(int_hash) % (2**32)
    return bounded_hash
