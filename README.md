# Memorization Capacity of Invariant Convolutional Neural Nets

This repository contains code to reproduce the results of

> **Memorization Capacity of Invariant Convolutional Neural Nets**\
> Sjoerd Dirksen, Patrick Finke and Paul Geuchen, preprint [arXiv:2610.38938](https://arxiv.org/abs/2610.38938) \[stat.ML\], 2026


## Repository structure

These are the core files of the repository:
- `data.py` generates datasets and writes them to `data/`
- `ex_*.py` run the experiments, write results to `results/` and save plots to `plots/`
- `algorithm.py` (and `experiment.py`) contain the implementation of the algorithm

The rest are auxiliary files:
- `plot.py` contains functions to generate the plots
- `utils.py` contains some utility functions


## Reproducing results

The code requires Python 3.14+ and the packages listed in `requirements.txt`. To reproduce the results from the paper, follow these steps:

1. Clone the repository, set up a virtual environment and install the dependencies.

```bash
git clone https://github.com/patrickfinke/invariant-memo
cd invariant-memo

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Generate the datasets. This is also required to generate the plots from our precomputed results.

```bash
python data.py
```

3. Run the experiments, which also generates the plots. To use the precomputed results, unzip them first.

```bash
unzip results.zip         # (optional)

                          # Figure(s)
python ex_2d.py           # 2
python ex_mnist.py        # 3, 7a
python ex_mnist_small.py  # 4
python ex_cifar.py        # 5, 7b
python ex_cifar_small.py  # 6
python ex_reduction.py    # 1
```


## Citation

```bibtex
@misc{dirksen2026memorization,
  title={Memorization Capacity of Invariant Convolutional Neural Nets}, 
  author={Sjoerd Dirksen and Patrick Finke and Paul Geuchen},
  year={2026},
  eprint={2610.38938},
  archivePrefix={arXiv},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2610.38938}, 
}
```
