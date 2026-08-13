(installation)=

# Installation

Popari requires Python 3.12 or newer. We recommend using
[uv](https://docs.astral.sh/uv/) to install Python, manage the environment, and
run Popari commands.

## Install with uv

Install `uv` on Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, use PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create a project environment and install Popari from PyPI:

```bash
mkdir popari-analysis
cd popari-analysis
uv init --python 3.12
uv add popari
```

Run Python inside the managed environment without activating it:

```bash
uv run python
```

For example, verify the installed version:

```bash
uv run python -c "import popari; print(popari.__version__)"
```

To work in JupyterLab, install Popari's Jupyter dependencies and launch the
server through `uv`:

```bash
uv add "popari[jupyter]"
uv run jupyter lab
```

Weights & Biases support and the simulation interface are also available as
optional dependencies:

```bash
uv add "popari[wandb,simulation]"
```

## Install from source

To use the latest development version, clone the repository and synchronize
its environment:

```bash
git clone https://github.com/alam-shahul/popari.git
cd popari
uv sync --extra wandb --extra simulation
uv run python
```

All repository scripts and tests should likewise be launched through the
managed environment, for example:

```bash
uv run python scripts/train.py --help
uv run pytest
```

## Install with pip

Popari can alternatively be installed into an existing Python 3.12
environment with `pip`:

```bash
pip install popari
```

Optional dependencies use the same extras:

```bash
pip install "popari[jupyter,wandb,simulation]"
```
