# MaxCal Network

MaxCal Network contains the code and data used for network inference with maximum caliber methods. It simulates neural activity, converts spike trains into Markov states, and infers effective network connections.

## Paper

An earlier version of this work is available on [arXiv](https://arxiv.org/abs/2405.15206). The revised manuscript has been accepted for publication in *PLOS Computational Biology*.

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yaml
conda activate maxcal-network
```

Run all commands from the repository root.

## Repository structure

```text
data/                   Data used by analysis scripts
notebooks/              Interactive demonstration
scripts/benchmarks/     GLM and Granger-causality comparisons
scripts/foundations/    Core MaxCal studies
scripts/inference/      Motif, learning, and coarse-graining studies
scripts/retina/         Retinal-data analysis
scripts/scans/          Parameter scans
scripts/simulations/    LIF and large-network simulations
src/maxcal_network/     Reusable package functions
tests/                  Unit tests
```

## Reproduce manuscript figures

| Figure | Command |
| --- | --- |
| Fig 2 | `python scripts/inference/motifs/MaxCal_err.py` |
| Fig 3 | `python scripts/inference/motifs/MaxCal_motif.py` |
| Fig 4 | `python scripts/benchmarks/GC_plus.py`<br>`python scripts/benchmarks/GC_linear.py`<br>`python scripts/benchmarks/glm_test.py` |
| Fig 5 | `python scripts/inference/coarse_graining/MaxCal_C5_3.py` |
| Fig 6 | `python scripts/simulations/large_net.py` |
| Fig 7 | `python scripts/retina/retina_sample.py` |
| Fig 8 | `python scripts/scans/scan_stim.py` |

Supporting-information scripts:

- `python scripts/scans/scan_dof.py`
- `python scripts/scans/scan_net.py`
- `python scripts/inference/learning/MaxCal_block.py`
- `python scripts/inference/exploratory/MaxCal_delay.py`
- `python scripts/inference/learning/MaxCal_scale_t.py`
- `python scripts/inference/motifs/MaxCal_spk.py`

## Demo

Open the demonstration notebook:

```bash
jupyter lab notebooks/maxcal_net_demo.ipynb
```

## Tests

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same tests for pushes and pull requests.
