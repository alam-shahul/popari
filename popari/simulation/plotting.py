"""Plotting helpers for simulation evaluations."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from popari.plotting import all_embeddings, multireplicate_heatmap, spatial_affinity_heatmap
from popari.plotting.utils import _highlight_cell


def _sample_result(adata, key: str, sample: str):
    """Return a sample-keyed simulation annotation."""

    value = adata.uns[key]
    if isinstance(value, dict):
        return value[sample]
    if len(adata.popari.sample_names) == 1:
        return value
    raise ValueError(f"`uns[{key!r}]` must be keyed by sample for multisample results.")


def plot_pairwise_comparison(
    primary_evaluation,
    other_evaluations,
    metrics,
):
    """Compare one evaluation against several alternatives for each metric.

    Each subplot compares the primary evaluation on the x-axis with one other
    evaluation on the y-axis. Every point represents one result ID after
    averaging across its datasets and selecting its best random-state score:
    the maximum when larger values are better and the minimum when smaller
    values are better. The diagonal denotes equal performance, and the green
    region denotes better performance by the primary evaluation.

    A subplot is omitted when the corresponding alternative evaluation does
    not contain that metric.

    Args:
        primary_evaluation: Evaluation displayed on the x-axis of every
            subplot.
        other_evaluations: Evaluations displayed in separate columns on the
            y-axis.
        metrics: Ordered mapping from metric names to ``"larger"`` or
            ``"smaller"``, indicating which direction represents better
            performance.

    Returns:
        A figure with one row per metric and one column per alternative
        evaluation.

    """

    overall_fig = plt.figure(constrained_layout=True, dpi=600)
    subfigs = np.atleast_1d(overall_fig.subfigures(nrows=len(metrics), ncols=1))

    for metric_index, (metric, direction) in enumerate(metrics.items()):
        ax_row = subfigs[metric_index].subplots(nrows=1, ncols=len(other_evaluations), squeeze=False)
        subfigs[metric_index].suptitle(f"{metric}")
        subfigs[metric_index].supxlabel(primary_evaluation.model_name)

        for other_evaluation, ax in zip(other_evaluations, ax_row.flat):
            if not other_evaluation.has_metric(metric):
                ax.set_axis_off()
                continue

            primary_scores = primary_evaluation.metric_scores(metric)
            other_scores = other_evaluation.metric_scores(metric)

            score_reduction = "max" if direction == "larger" else "min"
            primary_scores = getattr(primary_scores, score_reduction)(axis="columns")
            other_scores = getattr(other_scores, score_reduction)(axis="columns").reindex(primary_scores.index)
            if other_scores.isna().any():
                raise ValueError("Evaluations must contain the same result IDs.")

            all_scores = [*primary_scores, *other_scores]
            score_ptp = np.ptp(all_scores)
            max_score = np.max(all_scores) + 0.05 * score_ptp
            min_score = np.min(all_scores) - 0.05 * score_ptp

            domain = np.linspace(min_score, max_score, 50)
            boundary = np.full(len(domain), min_score) if direction == "larger" else np.full(len(domain), max_score)
            ax.fill_between(domain, domain, boundary, color="green", alpha=0.25, linewidth=0)

            ax.scatter(
                primary_scores.to_numpy(),
                other_scores.to_numpy(),
                edgecolors="none",
                alpha=0.75,
                s=10,
                c="black",
            )
            ax.set_ylabel(other_evaluation.model_name)
            ax.plot(domain, domain, linestyle="--", c="black", linewidth=1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim([min_score, max_score])
            ax.set_ylim([min_score, max_score])

    return overall_fig


def plot_best_in_situ_result(
    primary_evaluation,
    other_evaluations,
    results_by_model,
    metagene_indices,
    dataset_index: int = 0,
    size=None,
):
    """Plot in situ embeddings from preloaded single-level results."""

    overall_fig = plt.figure(constrained_layout=True, figsize=(14, 2 * (2 + len(other_evaluations))), dpi=600)
    subfigs = overall_fig.subfigures(nrows=len(other_evaluations) + 2, ncols=1)

    primary_dataset = results_by_model[primary_evaluation.model_name]
    primary_sample = primary_dataset.popari.sample_names[dataset_index]

    _, num_features = primary_dataset.obsm["truncated_ground_truth_X"].shape
    first_row = np.atleast_1d(subfigs[0].subplots(nrows=1, ncols=num_features, squeeze=True))
    all_embeddings(
        primary_dataset,
        samples=primary_sample,
        fig=subfigs[0],
        embedding_key="truncated_ground_truth_X",
        ax=first_row,
        colorbar=False,
        connectivity_key=None,
        edgecolors="none",
        cmap="Reds",
        size=size,
    )

    second_row = np.atleast_1d(subfigs[1].subplots(nrows=1, ncols=num_features, squeeze=True))
    all_embeddings(
        primary_dataset,
        samples=primary_sample,
        fig=subfigs[1],
        embedding_key="truncated_matched_X",
        ax=second_row,
        colorbar=False,
        connectivity_key=None,
        edgecolors="none",
        cmap="Reds",
        size=size,
    )

    for value, ax in zip(
        _sample_result(
            primary_dataset,
            "embedding_spatial_wasserstein",
            primary_sample,
        )[
            metagene_indices[dataset_index]
        ].min(axis=1),
        second_row,
    ):
        ax.set_title(f"{value:.2f}")

    for other_evaluation, subfig in zip(other_evaluations, subfigs[2:]):
        dataset = results_by_model[other_evaluation.model_name]
        sample = dataset.popari.sample_names[dataset_index]

        _, num_features = dataset.obsm["X"].shape
        row = np.atleast_1d(subfig.subplots(nrows=1, ncols=num_features, squeeze=True))
        all_embeddings(
            dataset,
            samples=sample,
            fig=subfig,
            embedding_key="truncated_matched_X",
            ax=row,
            colorbar=False,
            connectivity_key=None,
            edgecolors="none",
            cmap="Reds",
            size=size,
        )

        for value, ax in zip(
            _sample_result(
                dataset,
                "embedding_spatial_wasserstein",
                sample,
            )[
                metagene_indices[dataset_index]
            ].min(axis=1),
            row,
        ):
            ax.set_title(f"{value:.2f}")

    for subfig in subfigs:
        for ax in subfig.axes:
            ax.set_ylabel("")
            ax.set_xlabel("")

    for ax in first_row:
        ax.set_title("")

    subfigs[0].axes[0].set_ylabel("GT")
    subfigs[1].axes[0].set_ylabel(primary_evaluation.model_name)
    for other_evaluation, subfig in zip(other_evaluations, subfigs[2:]):
        subfig.axes[0].set_ylabel(other_evaluation.model_name)

    return overall_fig


def plot_best_affinity_correlation_result(
    primary_evaluation,
    other_evaluations,
    results_by_model,
    correlation_truth_key: str = "ground_truth_correlation",
    spatial_affinity_key: str = "permuted_Sigma_x_inv",
    use_residuals: bool = False,
):
    """Plot affinity matrices from preloaded single-level results."""

    from scipy.stats import rankdata

    overall_fig = plt.figure(constrained_layout=True, dpi=600)
    subfigs = np.atleast_1d(overall_fig.subfigures(nrows=len(other_evaluations) + 2, ncols=1))

    primary_dataset = results_by_model[primary_evaluation.model_name]
    primary_samples = primary_dataset.popari.sample_names

    def store_ranked_affinities(dataset, affinity_key: str, output_key: str):
        dataset.uns[output_key] = {}
        for sample in dataset.popari.sample_names:
            affinities = dataset.uns[affinity_key][sample]
            dataset.uns[output_key][sample] = rankdata(affinities).reshape(affinities.shape)

    first_affinity = primary_dataset.uns[spatial_affinity_key][primary_samples[0]]
    num_metagenes = first_affinity.shape[0]
    mask = np.ones((num_metagenes, num_metagenes), dtype=bool)
    mask[np.triu_indices_from(mask)] = 0

    first_row = np.atleast_1d(subfigs[0].subplots(nrows=1, ncols=len(primary_samples), squeeze=True))
    if use_residuals:
        truth_rank_key = f"{correlation_truth_key}_rank"
        store_ranked_affinities(primary_dataset, affinity_key=correlation_truth_key, output_key=truth_rank_key)
        multireplicate_heatmap(
            primary_dataset,
            uns=truth_rank_key,
            label_font_size=1.5,
            mask=mask,
            axes=first_row,
            cmap="ocean",
            vmin=0,
        )
    else:
        spatial_affinity_heatmap(
            primary_dataset,
            spatial_affinity_key=correlation_truth_key,
            label_values=False,
            label_font_size=1.5,
            mask=mask,
            axes=first_row,
        )

    second_row = np.atleast_1d(subfigs[1].subplots(nrows=1, ncols=len(primary_samples), squeeze=True))
    if use_residuals:
        affinity_rank_key = f"{spatial_affinity_key}_rank"
        store_ranked_affinities(
            primary_dataset,
            affinity_key=spatial_affinity_key,
            output_key=affinity_rank_key,
        )
        multireplicate_heatmap(
            primary_dataset,
            uns=affinity_rank_key,
            label_font_size=1.5,
            mask=mask,
            axes=second_row,
            cmap="ocean",
            vmin=0,
        )
    else:
        spatial_affinity_heatmap(
            primary_dataset,
            spatial_affinity_key=spatial_affinity_key,
            label_values=False,
            label_font_size=1.5,
            mask=mask,
            axes=second_row,
        )

    for sample, ax in zip(primary_samples, second_row):
        ax.set_title(f"{_sample_result(primary_dataset, 'affinity_correlation', sample):.2f}")

    for other_evaluation, subfig in zip(other_evaluations, subfigs[2:]):
        other_dataset = results_by_model[other_evaluation.model_name]
        other_samples = other_dataset.popari.sample_names

        row = np.atleast_1d(subfig.subplots(nrows=1, ncols=len(other_samples), squeeze=True))
        if use_residuals:
            affinity_rank_key = f"{spatial_affinity_key}_rank"
            store_ranked_affinities(
                other_dataset,
                affinity_key=spatial_affinity_key,
                output_key=affinity_rank_key,
            )
            multireplicate_heatmap(
                other_dataset,
                uns=affinity_rank_key,
                label_font_size=1.5,
                mask=mask,
                axes=row,
                cmap="ocean",
                vmin=0,
            )
        else:
            spatial_affinity_heatmap(
                other_dataset,
                spatial_affinity_key=spatial_affinity_key,
                label_values=False,
                label_font_size=1.5,
                mask=mask,
                axes=row,
            )

        for sample, ax in zip(other_samples, row):
            ax.set_title(f"{_sample_result(other_dataset, 'affinity_correlation', sample):.2f}")

    for subfig in subfigs:
        for ax in subfig.axes:
            for i in range(num_metagenes):
                for j in range(i, num_metagenes):
                    _highlight_cell(j, i, color="gray", ax=ax, linewidth=0.5)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    return overall_fig


__all__ = [
    plot_pairwise_comparison.__name__,
    plot_best_in_situ_result.__name__,
    plot_best_affinity_correlation_result.__name__,
]
