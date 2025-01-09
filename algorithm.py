import random
import operator as op

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Disabling the cuDNN backend yielded much better performance when evaluating the kernel_*
# functions for our setup.
torch.backends.cudnn.enabled = False


def threshold(inputs, inplace=True):
    """Return `1` if `inputs > 0`, otherwise return `0`."""

    out = inputs if inplace else None
    return torch.heaviside(inputs, torch.tensor(0, dtype=inputs.dtype), out=out)


def random_tessellation(in_features, out_features, maximal_bias=0.0, dtype=None, device=None):
    """Construct a function that performs a random hyperplane tessellation.
    
    This is used to implement the first layer of the construction for custom transformations."""

    A = torch.empty(out_features, in_features, dtype=dtype, device=device).normal_(0.0, 1.0)
    b = torch.empty(out_features, dtype=dtype, device=device).uniform_(-maximal_bias, maximal_bias)

    def fn(inputs):
        out = inputs @ A.T
        out += b
        return threshold(out)
    return fn


def random_conv2d_circular(in_channels, out_channels, input_size, stride=1, maximal_bias=0.0,
                           dtype=None, device=None):
    """Construct a function that performs a random 2d convolution.
    
    This is used to implement the first layer of the construction for translations."""

    assert input_size % stride == 0, "`input_size` must be multiple of `stride`, otherwise this is not a group"

    weights = torch.empty(out_channels, in_channels, input_size, input_size, dtype=dtype, device=device).normal_(0.0, 1.0)
    bias = torch.empty(out_channels, dtype=dtype, device=device).uniform_(-maximal_bias, maximal_bias)

    def fn(inputs):
        assert inputs.dim() >= 4, "Does the input have the correct shape, including color channel?"

        padding = (0, inputs.shape[-1] - 1, 0, inputs.shape[-2] - 1)
        inputs = F.pad(inputs, padding, mode="circular")

        out = F.conv2d(inputs, weights, bias, stride=stride)
        return threshold(out)
    return fn


def kernel_linear(A, B):
    return A @ B.mT


def kernel_conv1d_circular(A, B, batch_size=5_000):
    padding = (0, B.shape[-1] - 1)
    B = F.pad(B, padding, mode="circular")
    batches = [slice(i, i + batch_size) for i in range(0, A.shape[0], batch_size)]
    out = torch.cat([F.conv1d(B, A[batch]).transpose_(0, 1).amin(-1) for batch in batches])

    return out


def kernel_conv2d_circular(A, B, batch_size=5_000):
    padding = (0, B.shape[-1] - 1, 0, B.shape[-2] - 1)
    B = F.pad(B, padding, mode="circular")
    
    out = torch.empty((A.shape[0], B.shape[0]), dtype=B.dtype, device=B.device)
    for i in range(0, A.shape[0], batch_size):
        batch = slice(i, i + batch_size)
        out[batch] = F.conv2d(B, A[batch]).transpose_(0, 1).amin((-1, -2))
        
    return out


def random_covering(samples, mask_positive, kernel_fn, device=None, verbose=True):
    INFINITY = torch.finfo(samples.dtype).max

    n_samples = samples.shape[0]
    n_positive = mask_positive.sum().item()
    indices_samples = torch.where(mask_positive)[0]

    assert n_positive < n_samples, "There must be at least one negative sample!"

    # Because *kernel_fn* expects 2-dim tensors and returns a 2-dim tensor, but
    # the algorithm only ever uses a 1-dim tensor in the second argument we
    # define the following for convenience.
    kernel_fn_vec = lambda A, B: kernel_fn(A, B[None, :])[:, 0]

    covering = []
    index_accepted = []

    if verbose:
        pbar = tqdm(desc="cover", total=n_positive, leave=False, maxinterval=0.5)

    mask_candidates = torch.full((n_positive,), True, device=device)
    while (n_candidates := mask_candidates.sum().item()) > 0:
        if verbose:
            pbar.update((n_positive - n_candidates) - pbar.n)

        i = random.choice(torch.where(mask_candidates)[0]).item()
        mask_candidates[i] = False

        u = (samples[indices_samples[i]] <= 0).to(samples.dtype)
        kernel_values = kernel_fn_vec(samples, u)
        m = torch.where(mask_positive, INFINITY, kernel_values).min()

        if m <= 0: # Skip empty component in constant time.
            continue

        mask_covered = kernel_values[mask_positive] < m
        mask_candidates &= ~mask_covered

        covering.append(mask_covered.cpu()) # keep in CPU memory
        index_accepted.append(i)

    if verbose:
        pbar.update(pbar.total - pbar.n)
        pbar.close()

    # Run a version of the pruning algorithm that retains full interpolation
    # and only prunes components that do not contribute to the covering.
    mask_not_pruned = torch.full((len(covering),), True)
    multiplicity = sum(covering)
    for i, component in enumerate(covering):
        if (multiplicity[component] >= 2).all():
            mask_not_pruned[i] = False
            multiplicity -= component.to(multiplicity.dtype)

    index_accepted = [i for m, i in zip(mask_not_pruned, index_accepted) if m]
    index_accepted = torch.tensor(index_accepted)

    covering = [c for m, c in zip(mask_not_pruned, covering) if m]
    covering = torch.stack(covering).T if covering else torch.empty((n_positive, 0), dtype=bool)

    return index_accepted, covering


def random_covering_multiclass(samples, labels, kernel_fn, verbose=True, device=None):
    classes = labels.unique()
    if verbose:
        classes = tqdm(classes, desc="class", leave=False)

    results = {}
    ns_covered = []
    for class_ in classes:
        mask_positive = labels == class_
        indices, covering = random_covering(samples, mask_positive, kernel_fn, verbose=verbose, device=device)
        ns_covered.append(prune(covering, device=device))

        results[f"class_{class_}"] = {"indices": indices}

    results["ns_covered"] = prune_multiclass(ns_covered, device=device).cpu()

    return results


def prune(covering, verbose=True, device=None):
    INFINITY = torch.iinfo(torch.int64).max
    covering = covering.to(device)

    multiplicity = covering.sum(1)
    mask_pruned = torch.full((covering.shape[1],), False, device=device)

    if verbose:
        pbar = tqdm(desc="prune", total=len(mask_pruned), leave=False, maxinterval=0.5)

    n_covered = (multiplicity > 0).sum().item()
    ns_covered = [n_covered]
    while n_covered > 0:
        contribution = covering[multiplicity == 1].sum(0)
        contribution[mask_pruned] = INFINITY
        i = contribution.argmin().item()

        mask_pruned[i] = True
        multiplicity -= covering[:, i].to(multiplicity.dtype)

        n_covered = (multiplicity > 0).sum().item()
        ns_covered.append(n_covered)

        if verbose:
            pbar.update()

    return torch.tensor(ns_covered, device=device)


def prune_multiclass(ns_covered, verbose=True, device=None):
    INFINITY = torch.iinfo(torch.int32).max

    n_classes = len(ns_covered)
    n_decrements = sum(map(len, ns_covered)) - n_classes
    ns_covered_multi = torch.empty((n_classes, n_decrements + 1), dtype=torch.int32, device=device)
    ns_covered_multi[:, 0] = torch.tensor([ns[0] for ns in ns_covered])

    decrements = [map(op.sub, ns[:-1], ns[1:]) for ns in ns_covered]
    ds = torch.tensor([next(d, INFINITY) for d in decrements])

    if verbose:
        pbar = tqdm(desc="prune_m", total=n_decrements, leave=False, maxinterval=0.5)

    for j in range(1, n_decrements + 1):
        i = ds.argmin().item()

        ns_covered_multi[:, j] = ns_covered_multi[:, j-1]
        ns_covered_multi[i, j] -= ds[i]

        ds[i] = next(decrements[i], INFINITY)

        if verbose:
            pbar.update()

    return ns_covered_multi
