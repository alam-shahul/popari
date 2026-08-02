(installation)=

# Installation

`popari` requires Python version >=3.10.

Install `popari` with pip.

```
pip install popari
```

To use the Jupyter Lab-based GUI for designing simulated multisample spatially resolved transcriptomics, install the optional dependencies:

```
pip install popari[simulation]
```

Hydra-based local training is included in the core package. To track runs and model artifacts with Weights & Biases, install the optional tracking dependency:

```
pip install popari[wandb]
```

````{note}
Since Popari lies at the cutting edge of the mSRT analysis frontier, it is possible that the most
updated version of Popari depends on some package versions which are not yet available on PyPI.
Thus, it may be necessary to install as follows:

```bash
git clone https://github.com/alam-shahul/popari.git
cd popari
pip install .[wandb,simulation]
```
````
