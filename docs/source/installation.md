(installation)=

# Installation

`popari` requires Python version >=3.8.

Install `popari` with pip.

```
pip install popari
```

To use the optional experiment tracking/grid search functionality, install Popari with MLflow:

```
pip install popari[mlflow]
```

````{note}
Since Popari lies at the cutting edge of the mSRT analysis frontier, it is possible that the most
updated version of Popari depends on some package versions which are not yet available on PyPI.
Thus, it may be necessary to install as follows:

```bash
git clone https://github.com/alam-shahul/popari.git
cd popari
pip install .[mlflow,simulation]
````

```
```
