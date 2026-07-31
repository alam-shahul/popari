import anndata as ad
import matplotlib
import numpy as np
import pytest
import torch
from scipy.sparse import csr_array

from popari import pp, tl
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
    pp.compute_spatial_neighbors(dataset)
    return dataset


@pytest.fixture(scope="session")
def context():
    return {"device": "cpu", "dtype": torch.float64}


@pytest.fixture(scope="session")
def float32_context():
    return {"device": "cpu", "dtype": torch.float32}


@pytest.fixture(scope="session")
def gpu_context():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU training tests.")
    return {"device": "cuda", "dtype": torch.float64}


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


def _model_adata(datasets, sample_key="batch"):
    datasets = [dataset.copy() for dataset in datasets]
    sample_names = []
    for dataset in datasets:
        try:
            sample_names.append(dataset.popari.name)
        except ValueError:
            names = dataset.obs["batch"].astype(str).unique()
            if len(names) != 1:
                raise ValueError("Test datasets must contain exactly one sample.")
            sample_names.append(str(names[0]))
    for dataset, name in zip(datasets, sample_names, strict=True):
        dataset.obs[sample_key] = name
        if sample_key != "batch":
            dataset.obs.drop(columns=["batch"], errors="ignore", inplace=True)
    adata = ad.concat(datasets, index_unique="-", pairwise=True)
    adata.obs[sample_key] = adata.obs[sample_key].astype("category")
    adata.obs[sample_key] = adata.obs[sample_key].cat.reorder_categories(sample_names, ordered=True)
    return adata


@pytest.fixture(scope="session")
def adata_factory(dataset_factory):
    def factory(*, sample_key="batch", **dataset_kwargs):
        return _model_adata(dataset_factory(**dataset_kwargs), sample_key=sample_key)

    return factory


@pytest.fixture(scope="session")
def shared_model_factory(context, adata_factory):
    def factory(**overrides):
        sample_key = overrides.pop("sample_key", None)
        adata = overrides.pop("adata", None)
        if adata is None:
            adata = adata_factory(sample_key=sample_key or "batch")
        return Popari(
            K=overrides.pop("K", 3),
            adata=adata,
            sample_key=sample_key,
            lambda_Sigma_x_inv=overrides.pop("lambda_Sigma_x_inv", 1e-3),
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
def differential_model_factory(context, adata_factory):
    def factory(**overrides):
        sample_key = overrides.pop("sample_key", None)
        adata = overrides.pop("adata", None)
        if adata is None:
            adata = adata_factory(sample_key=sample_key or "batch")
        return Popari(
            K=overrides.pop("K", 3),
            adata=adata,
            sample_key=sample_key,
            lambda_Sigma_x_inv=overrides.pop("lambda_Sigma_x_inv", 1e-3),
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
def hierarchical_model_factory(context, adata_factory):
    def factory(**overrides):
        sample_key = overrides.pop("sample_key", None)
        adata = overrides.pop("adata", None)
        if adata is None:
            adata = adata_factory(num_cells=36, sample_key=sample_key or "batch")
        return Popari(
            K=overrides.pop("K", 3),
            adata=adata,
            sample_key=sample_key,
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
def trained_shared_model(shared_model_factory, adata_factory, context):
    model = shared_model_factory(
        adata=adata_factory(num_cells=48),
        torch_context=context,
        initial_context=context,
    )
    for _ in range(2):
        model.estimate_parameters(spatial_affinity_epochs=50)
        model.estimate_weights()
    return model


@pytest.fixture(scope="session")
def initialized_shared_model(shared_model_factory, adata_factory):
    return shared_model_factory(adata=adata_factory(num_cells=48))


@pytest.fixture(scope="session")
def preprocessed_shared_model(initialized_shared_model):
    model = initialized_shared_model
    tl.postprocess_embeddings(model.adata)
    pp.pca(model.adata, n_comps=3)
    return model


@pytest.fixture(scope="session")
def analyzed_shared_model(preprocessed_shared_model):
    model = preprocessed_shared_model
    tl.compute_columnwise_autocorrelation(model.adata, uns="M")
    tl.compute_empirical_correlations(model.adata, output="empirical_correlation")
    tl.compute_spatial_gene_correlation(model.adata)
    tl.cluster_domains(model.adata, target_domains=2)
    return model


@pytest.fixture(scope="session")
def clustered_shared_model(preprocessed_shared_model):
    model = preprocessed_shared_model
    tl.leiden(model.adata, target_clusters=3)
    tl.compute_ari_scores(model.adata, labels="cell_type", predictions="leiden")
    tl.compute_silhouette_scores(model.adata, labels="cell_type", embeddings="normalized_X")
    tl.evaluate_classification_task(model.adata, labels="cell_type", embeddings="normalized_X")
    return model


@pytest.fixture(scope="session")
def shared_model_expected_metrics():
    return {
        "nll": -704.4400195133358,
        "sigma_yx": [0.15010631381825246, 0.14708547226059387],
        "metagene_sum": 3.0,
        "embedding_sum_0": 139.86554004948852,
        "spatial_affinity_sum_0": 2.866319315960922,
    }
