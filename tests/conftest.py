import anndata as ad
import matplotlib
import numpy as np
import pytest
import torch
from scipy.sparse import csr_array

from popari import tl
from popari.model import Popari

matplotlib.use("Agg")


def _grid_coordinates(num_cells: int) -> np.ndarray:
    width = int(np.ceil(np.sqrt(num_cells)))
    xs, ys = np.meshgrid(np.arange(width), np.arange(width))
    coordinates = np.column_stack([xs.ravel(), ys.ravel()])[:num_cells].astype(float)
    return coordinates


def _make_dataset(
    rng: np.random.Generator,
    replicate_name: str,
    num_cells: int,
    num_genes: int,
    latent_dim: int,
    offset: float = 0.0,
) -> ad.AnnData:
    metagenes = rng.gamma(shape=2.0, scale=1.0, size=(latent_dim, num_genes))
    assignments = np.arange(num_cells) % latent_dim
    weights = np.full((num_cells, latent_dim), 0.05)
    weights[np.arange(num_cells), assignments] = 1.0 + offset
    expression = weights @ metagenes + 0.05 * rng.random((num_cells, num_genes))

    dataset = ad.AnnData(X=csr_array(expression))
    dataset.obs["batch"] = replicate_name
    dataset.obs["cell_type"] = np.array([f"type_{index}" for index in assignments], dtype=object)
    dataset.obs["domain"] = np.where(assignments % 2 == 0, "left", "right")
    dataset.obsm["spatial"] = _grid_coordinates(num_cells)
    dataset.var_names = [f"gene_{index}" for index in range(num_genes)]
    dataset.popari.compute_spatial_neighbors()
    return dataset


@pytest.fixture(scope="session")
def context():
    return {"device": "cpu", "dtype": torch.float64}


@pytest.fixture(scope="session")
def float32_context():
    return {"device": "cpu", "dtype": torch.float32}


@pytest.fixture(scope="session")
def dataset_factory():
    def factory(
        *,
        num_replicates: int = 2,
        num_cells: int = 24,
        num_genes: int = 12,
        latent_dim: int = 3,
        seed: int = 0,
        replicate_names: list[str] | None = None,
    ) -> list[ad.AnnData]:
        rng = np.random.default_rng(seed)
        if replicate_names is None:
            replicate_names = [str(index) for index in range(num_replicates)]

        return [
            _make_dataset(
                rng,
                replicate_name=name,
                num_cells=num_cells,
                num_genes=num_genes,
                latent_dim=latent_dim,
                offset=0.1 * index,
            )
            for index, name in enumerate(replicate_names)
        ]

    return factory


@pytest.fixture
def mock_datasets(dataset_factory):
    return dataset_factory()


@pytest.fixture(scope="session")
def shared_model_factory(context, dataset_factory):
    def factory(**overrides):
        datasets = overrides.pop("datasets", dataset_factory())
        replicate_names = overrides.pop("replicate_names", [dataset.obs["batch"].iloc[0] for dataset in datasets])
        return Popari(
            K=overrides.pop("K", 3),
            datasets=datasets,
            replicate_names=replicate_names,
            lambda_Sigma_x_inv=overrides.pop("lambda_Sigma_x_inv", 1e-3),
            metagene_mode=overrides.pop("metagene_mode", "shared"),
            spatial_affinity_mode=overrides.pop("spatial_affinity_mode", "shared lookup"),
            initialization_method=overrides.pop("initialization_method", "svd"),
            torch_context=overrides.pop("torch_context", context),
            initial_context=overrides.pop("initial_context", context),
            random_state=overrides.pop("random_state", 0),
            verbose=overrides.pop("verbose", 0),
            **overrides,
        )

    return factory


@pytest.fixture(scope="session")
def differential_model_factory(context, dataset_factory):
    def factory(**overrides):
        datasets = overrides.pop("datasets", dataset_factory())
        replicate_names = overrides.pop("replicate_names", [dataset.obs["batch"].iloc[0] for dataset in datasets])
        return Popari(
            K=overrides.pop("K", 3),
            datasets=datasets,
            replicate_names=replicate_names,
            lambda_Sigma_x_inv=overrides.pop("lambda_Sigma_x_inv", 1e-3),
            metagene_mode="differential",
            lambda_M=overrides.pop("lambda_M", 0.5),
            spatial_affinity_mode=overrides.pop("spatial_affinity_mode", "differential lookup"),
            lambda_Sigma_bar=overrides.pop("lambda_Sigma_bar", 1e-3),
            initialization_method=overrides.pop("initialization_method", "svd"),
            torch_context=overrides.pop("torch_context", context),
            initial_context=overrides.pop("initial_context", context),
            random_state=overrides.pop("random_state", 0),
            verbose=overrides.pop("verbose", 0),
            **overrides,
        )

    return factory


@pytest.fixture(scope="session")
def hierarchical_model_factory(context, dataset_factory):
    def factory(**overrides):
        datasets = overrides.pop("datasets", dataset_factory(num_cells=36))
        replicate_names = overrides.pop("replicate_names", [dataset.obs["batch"].iloc[0] for dataset in datasets])
        return Popari(
            K=overrides.pop("K", 3),
            datasets=datasets,
            replicate_names=replicate_names,
            lambda_Sigma_x_inv=overrides.pop("lambda_Sigma_x_inv", 1e-3),
            initialization_method=overrides.pop("initialization_method", "svd"),
            spatial_affinity_mode=overrides.pop("spatial_affinity_mode", "differential lookup"),
            torch_context=overrides.pop("torch_context", context),
            initial_context=overrides.pop("initial_context", context),
            hierarchical_levels=overrides.pop("hierarchical_levels", 2),
            binning_downsample_rate=overrides.pop("binning_downsample_rate", 0.5),
            superresolution_lr=overrides.pop("superresolution_lr", 1e-2),
            random_state=overrides.pop("random_state", 0),
            verbose=overrides.pop("verbose", 0),
            **overrides,
        )

    return factory


@pytest.fixture
def shared_mock_model(shared_model_factory):
    return shared_model_factory()


@pytest.fixture
def tmp_h5_path(tmp_path):
    return tmp_path / "model.h5ad"


@pytest.fixture(scope="session")
def trained_shared_model(shared_model_factory, dataset_factory):
    model = shared_model_factory(datasets=dataset_factory(num_cells=48))
    for _ in range(2):
        model.estimate_parameters()
        model.estimate_weights()
    return model


@pytest.fixture(scope="session")
def analyzed_shared_model(trained_shared_model):
    model = trained_shared_model
    tl.preprocess_embeddings(model)
    tl.pca(model, joint=False, n_comps=3)
    tl.pca(model, joint=True, n_comps=3)
    tl.compute_columnwise_autocorrelation(model, uns="M")
    tl.compute_empirical_correlations(model, output="empirical_correlation")
    tl.compute_spatial_gene_correlation(model)
    tl.cluster_domains(model, target_domains=2)
    return model


@pytest.fixture(scope="session")
def clustered_shared_model(analyzed_shared_model):
    tl.leiden(analyzed_shared_model, joint=True, target_clusters=3)
    tl.compute_ari_scores(analyzed_shared_model, labels="cell_type", predictions="leiden")
    tl.compute_silhouette_scores(analyzed_shared_model, labels="cell_type", embeddings="normalized_X")
    tl.evaluate_classification_task(analyzed_shared_model, labels="cell_type", embeddings="normalized_X", joint=False)
    tl.evaluate_classification_task(analyzed_shared_model, labels="cell_type", embeddings="normalized_X", joint=True)
    return analyzed_shared_model


@pytest.fixture(scope="session")
def shared_reference_metrics(clustered_shared_model):
    return {
        "nll": -657.06558928,
        "sigma_yx": [0.14147329, 0.13269758],
        "metagene_sum": 3.0,
        "embedding_sum_0": 140.5735742798036,
        "spatial_affinity_sum_0": 25.173517287732157,
        "pca_norms": [30.21869468688965, 32.63429260253906],
        "ari": [1.0, 0.5607476635514018],
        "silhouette": [0.9893147404134058, 0.45012760617995057],
        "microprecision_validation": [0.6388888888888888, 0.6388888888888888],
        "macroprecision_validation": [0.6551051051051051, 0.6551051051051051],
    }
