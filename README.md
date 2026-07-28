# MaxCal Network

MaxCal Network contains research code for network inference with maximum caliber methods. The code simulates neural activity, converts spike trains into Markov states, and infers effective network connections.

## Main features

- Simulate leaky integrate-and-fire networks.
- Convert spike trains into continuous-time Markov chain states.
- Infer transition rates and effective connections.
- Compare MaxCal inference with GLM and Granger-causality methods.
- Study motifs, coarse-graining, finite-data effects, and retinal recordings.

## Installation

Clone the repository and create the Conda environment:

```bash
conda env create -f environment.yaml
conda activate maxcal-network
```

The environment installs the local `maxcal_network` package. Run scripts from the repository root:

```bash
python scripts/benchmarks/GC_plus.py
```

To install the package in an existing environment:

```bash
python -m pip install -e .
```

## Repository structure

```text
data/                   Required-data lists and figure-data mapping
scripts/
  benchmarks/           GLM and Granger-causality comparisons
  foundations/          Core MaxCal model studies
  inference/
    coarse_graining/    Hidden-neuron and reduced-network studies
    exploratory/        Experimental temporal and ISI studies
    learning/           Constraint and finite-data learning studies
    motifs/             Three-neuron motif studies
  retina/               Retinal-data studies
  scans/                Parameter scans
  simulations/          LIF and large-network simulations
notebooks/              Interactive demonstrations
src/maxcal_network/     Reusable package functions
tests/                  Automated tests
```

## Demo notebook

Open `notebooks/maxcal_net_demo.ipynb` in JupyterLab. The notebook simulates a three-neuron LIF network, converts spikes into CTMC observations, and runs MaxCal inference.

```bash
jupyter lab notebooks/maxcal_net_demo.ipynb
```

## Reproduce manuscript figures

Run commands from the repository root.

| Figure | Scripts |
| --- | --- |
| Figure 2 | `python scripts/inference/motifs/MaxCal_err.py` |
| Figure 3 | `python scripts/inference/motifs/MaxCal_motif.py` |
| Figure 4 | `python scripts/benchmarks/GC_plus.py`<br>`python scripts/benchmarks/GC_linear.py`<br>`python scripts/benchmarks/glm_test.py` |
| Figure 5 | `python scripts/inference/coarse_graining/MaxCal_C5_3.py` |
| Figure 6 | `python scripts/simulations/large_net.py` |
| Figure 7 | `python scripts/retina/retina_sample.py` |
| Figure 8 | `python scripts/scans/scan_stim.py` |

Supporting-information scripts:

- `python scripts/scans/scan_dof.py`
- `python scripts/scans/scan_net.py`
- `python scripts/inference/learning/MaxCal_block.py`
- `python scripts/inference/exploratory/MaxCal_delay.py`
- `python scripts/inference/learning/MaxCal_scale_t.py`
- `python scripts/inference/motifs/MaxCal_spk.py`

## Data

Use `data/` as the target location for required `.pkl` and `.mat` files. Benchmark scripts already read from this folder.

Some scripts still contain commented pickle-save examples. These examples do not write files unless a user removes the comment markers.

## Package modules

- `dynamics.py`: state conversion and CTMC operations
- `optimization.py`: MaxCal objectives and constraints
- `metrics.py`: inference-quality metrics
- `simulation.py`: CTMC and LIF simulation helpers

Public functions are available from the package:

```python
from maxcal_network import compute_tauC, param2M, spk2statetime
```

## Tests

Run the automated tests:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same tests for pushes and pull requests.

## Project status

This repository contains active research code. The scripts preserve exploratory analyses and manuscript workflows. Review simulation length, input paths, and commented configuration options before a long run.
