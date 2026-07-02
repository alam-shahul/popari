"""Plotting helpers for simulation evaluations."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from popari.simulation.evaluation import (
    best_metric_difference_index,
    find_run_for_score_index,
    metric_score_array,
    run_id,
)


def highlight_cell(x, y, ax=None, **kwargs):
    """Draw a rectangle around a heatmap cell."""

    rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def store_ranked_affinity_matrices(
    datasets,
    affinity_key: str = "Sigma_x_inv",
    output_key: str = "Sigma_x_inv_rank",
):
    """Store rank-transformed affinity matrices for plotting."""

    from scipy.stats import rankdata

    for dataset in datasets:
        affinities = dataset.uns[affinity_key][dataset.name]
        dataset.uns[output_key] = {
            dataset.name: rankdata(affinities).reshape(affinities.shape),
        }


def affinity_matrix_dataframe(model, labels: Sequence[str] | None = None):
    """Return the first dataset's learned spatial affinity matrix as a
    DataFrame."""

    dataset = model.datasets[0]
    matrix = dataset.uns["Sigma_x_inv"][dataset.name]
    if labels is None:
        labels = [f"m{index}" for index in range(matrix.shape[0])]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def display_spatial_affinity_results(models_by_label, labels: Sequence[str] | None = None):
    """Display learned spatial affinity matrices and return them as
    DataFrames."""

    from IPython.display import display

    affinity_matrices = {}
    for label, model in models_by_label.items():
        print(label)
        matrix = affinity_matrix_dataframe(model, labels=labels)
        affinity_matrices[label] = matrix
        plot_spatial_affinity_matrix_panel({label: model}, labels=labels, title=str(label), figsize=(2.6, 2.6))
        display(matrix)
    return affinity_matrices


def lambda_sweep_affinity_range_summary(models_by_lambda, labels: Sequence[str] | None = None):
    """Summarize minimum and maximum learned affinity for one lambda sweep."""

    rows = []
    for lambda_value, model in models_by_lambda.items():
        matrix = affinity_matrix_dataframe(model, labels=labels).to_numpy()
        rows.append(
            {
                "lambda_Sigma_x_inv": lambda_value,
                "minimum": matrix.min(),
                "maximum": matrix.max(),
            },
        )
    return pd.DataFrame(rows).sort_values("lambda_Sigma_x_inv")


def plot_lambda_sweep_affinity_range(
    models_by_lambda,
    *,
    labels: Sequence[str] | None = None,
    title=None,
    figsize=None,
    dpi=300,
):
    """Plot affinity minimum/maximum values across lambda sweeps."""

    values = list(models_by_lambda.values())
    is_single_sweep = all(hasattr(value, "datasets") for value in values)

    if is_single_sweep:
        summaries_by_title = {
            title or "lambda sweep": lambda_sweep_affinity_range_summary(models_by_lambda, labels=labels),
        }
    else:
        summaries_by_title = {
            dataset_name: lambda_sweep_affinity_range_summary(sweep_models, labels=labels)
            for dataset_name, sweep_models in models_by_lambda.items()
        }

    num_panels = len(summaries_by_title)
    if figsize is None:
        figsize = (4, 3) if is_single_sweep else (2.4 * num_panels, 2.4)
    fig, axes = plt.subplots(1, num_panels, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes.flat

    positive_lambdas = np.concatenate(
        [summary["lambda_Sigma_x_inv"].to_numpy(dtype=float) for summary in summaries_by_title.values()],
    )
    positive_lambdas = np.sort(np.unique(positive_lambdas[positive_lambdas > 0]))
    zero_position = positive_lambdas.min() / 10 if len(positive_lambdas) else 1e-6
    ticks = np.concatenate([[zero_position], positive_lambdas])
    tick_labels = ["0", *[f"{value:.0e}" for value in positive_lambdas]]

    for ax, (panel_title, summary) in zip(axes, summaries_by_title.items()):
        lambdas = summary["lambda_Sigma_x_inv"].to_numpy(dtype=float)
        plotted_lambdas = lambdas.copy()
        plotted_lambdas[lambdas == 0] = zero_position

        ax.plot(plotted_lambdas, summary["minimum"], marker="o", label="Minimum")
        ax.plot(plotted_lambdas, summary["maximum"], marker="o", label="Maximum")
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_title(panel_title)
        ax.grid(True, axis="x", which="both", alpha=0.25)

    axes[0].set_ylabel("Spatial affinity value")
    for ax in axes:
        ax.set_xlabel(r"$\lambda_{\Sigma_x^{-1}}$")
    axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout()

    if is_single_sweep:
        return next(iter(summaries_by_title.values())), fig, axes[0]
    return summaries_by_title, fig, axes


def plot_spatial_affinity_matrix_panel(
    models_by_label,
    *,
    labels: Sequence[str] | None = None,
    title=None,
    figsize=(8, 2.5),
    dpi=300,
    cmap="bwr",
    vlim=None,
):
    """Plot one learned spatial affinity matrix per model."""

    matrices = {label: affinity_matrix_dataframe(model, labels=labels) for label, model in models_by_label.items()}

    fig, axes = plt.subplots(1, len(matrices), figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes.flat
    for ax, (label, matrix) in zip(axes, matrices.items()):
        matrix_values = matrix.to_numpy()
        matrix_vlim = np.abs(matrix_values).max() if vlim is None else vlim
        image = ax.imshow(matrix_values, cmap=cmap, vmin=-matrix_vlim, vmax=matrix_vlim, interpolation="nearest")
        ax.set_title(label)
        ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
        ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    return matrices, fig, axes


def plot_pairwise_comparison(
    primary_evaluation,
    other_evaluations,
    general_metrics,
    spatial_metrics,
    reduction: str = "average",
    level: int = 0,
):
    """Plot pairwise metric comparisons between a primary model and
    baselines."""

    general_metric_names, general_metric_directions = zip(*general_metrics)
    spatial_metric_names, spatial_metric_directions = zip(*spatial_metrics)
    metrics = [*general_metric_names, *spatial_metric_names]
    directions = [*general_metric_directions, *spatial_metric_directions]

    overall_fig = plt.figure(constrained_layout=True, dpi=600)
    subfigs = np.atleast_1d(overall_fig.subfigures(nrows=len(metrics), ncols=1))

    for metric_index, (metric, direction) in enumerate(zip(metrics, directions)):
        ax_row = subfigs[metric_index].subplots(nrows=1, ncols=len(other_evaluations), squeeze=False)
        subfigs[metric_index].suptitle(f"{metric}")
        subfigs[metric_index].supxlabel(primary_evaluation.model_name)

        for other_evaluation, ax in zip(other_evaluations, ax_row.flat):
            if metric_index >= len(general_metrics) and not other_evaluation.is_spatial:
                ax.set_axis_off()
                continue

            primary_scores = metric_score_array(primary_evaluation, metric, level=level)
            other_scores = metric_score_array(other_evaluation, metric, level=level)

            score_reduction = np.max if direction == "larger" else np.min
            primary_scores = score_reduction(primary_scores, axis=-1)
            other_scores = score_reduction(other_scores, axis=-1)

            all_scores = [*primary_scores, *other_scores]
            score_ptp = np.ptp(all_scores)
            max_score = np.max(all_scores) + 0.05 * score_ptp
            min_score = np.min(all_scores) - 0.05 * score_ptp

            domain = np.linspace(min_score, max_score, 50)
            boundary = np.full(len(domain), min_score) if direction == "larger" else np.full(len(domain), max_score)
            ax.fill_between(domain, domain, boundary, color="green", alpha=0.25, linewidth=0)

            ax.scatter(
                primary_scores.flatten(),
                other_scores.flatten(),
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
    metagene_indices,
    metric: str,
    dataset_index: int = 0,
    best_index=None,
    best_models=None,
    size=None,
    level: int = 0,
    verbose: bool = False,
):
    """Plot in situ embeddings for the run with the largest metric
    disagreement."""

    from popari._dataset_utils import _plot_all_embeddings

    if best_models is None:
        best_models = {}

    if best_index is None:
        best_index = best_metric_difference_index(primary_evaluation, other_evaluations, metric, level=level)
    if verbose:
        print(best_index)

    overall_fig = plt.figure(constrained_layout=True, figsize=(14, 2 * (2 + len(other_evaluations))), dpi=600)
    subfigs = overall_fig.subfigures(nrows=len(other_evaluations) + 2, ncols=1)

    best_primary_run = find_run_for_score_index(primary_evaluation, best_index)
    if "primary" not in best_models:
        if verbose:
            print(f"Loading {primary_evaluation.model_name} run {run_id(best_primary_run)}")
        best_models["primary"] = primary_evaluation.load_and_evaluate(best_primary_run, metagene_indices)
    elif verbose:
        print(f"Using cached {primary_evaluation.model_name} result")
    best_primary_result = best_models["primary"]
    primary_dataset = best_primary_result[level][dataset_index]

    _, num_features = primary_dataset.obsm["truncated_ground_truth_X"].shape
    first_row = np.atleast_1d(subfigs[0].subplots(nrows=1, ncols=num_features, squeeze=True))
    _plot_all_embeddings.__wrapped__(
        primary_dataset,
        fig=subfigs[0],
        embedding_key="truncated_ground_truth_X",
        ax=first_row,
        colorbar=False,
        edgecolors="none",
        cmap="Reds",
        size=size,
    )

    second_row = np.atleast_1d(subfigs[1].subplots(nrows=1, ncols=num_features, squeeze=True))
    _plot_all_embeddings.__wrapped__(
        primary_dataset,
        fig=subfigs[1],
        embedding_key="truncated_matched_X",
        ax=second_row,
        colorbar=False,
        edgecolors="none",
        cmap="Reds",
        size=size,
    )

    for value, ax in zip(
        primary_dataset.uns["embedding_spatial_wasserstein"][metagene_indices[dataset_index]].min(axis=1),
        second_row,
    ):
        ax.set_title(f"{value:.2f}")

    for other_evaluation, subfig in zip(other_evaluations, subfigs[2:]):
        best_other_run = find_run_for_score_index(other_evaluation, best_index)
        if other_evaluation.model_name not in best_models:
            if verbose:
                print(f"Loading {other_evaluation.model_name} run {run_id(best_other_run)}")
            best_models[other_evaluation.model_name] = other_evaluation.load_and_evaluate(
                best_other_run,
                metagene_indices,
            )
        elif verbose:
            print(f"Using cached {other_evaluation.model_name} result")
        best_other_result = best_models[other_evaluation.model_name]
        dataset = best_other_result[level][dataset_index]

        _, num_features = dataset.obsm["X"].shape
        row = np.atleast_1d(subfig.subplots(nrows=1, ncols=num_features, squeeze=True))
        _plot_all_embeddings.__wrapped__(
            dataset,
            fig=subfig,
            embedding_key="truncated_matched_X",
            ax=row,
            colorbar=False,
            edgecolors="none",
            cmap="Reds",
            size=size,
        )

        for value, ax in zip(
            dataset.uns["embedding_spatial_wasserstein"][metagene_indices[dataset_index]].min(axis=1),
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

    return overall_fig, best_models


def plot_best_affinity_correlation_result(
    primary_evaluation,
    other_evaluations,
    metagene_indices,
    metric: str,
    correlation_truth_key: str = "ground_truth_correlation",
    spatial_affinity_key: str = "permuted_Sigma_x_inv",
    best_index=None,
    best_models=None,
    use_residuals: bool = False,
    level: int = 0,
    verbose: bool = False,
):
    """Plot ground-truth and learned affinity matrices for the best-disagreement
    run."""

    from popari._dataset_utils import _multireplicate_heatmap, _spatial_affinity_heatmap

    if best_models is None:
        best_models = {}

    if best_index is None:
        best_index = best_metric_difference_index(primary_evaluation, other_evaluations, metric, level=level)
    if verbose:
        print(best_index)

    overall_fig = plt.figure(constrained_layout=True, dpi=600)
    subfigs = np.atleast_1d(overall_fig.subfigures(nrows=len(other_evaluations) + 2, ncols=1))

    best_primary_run = find_run_for_score_index(primary_evaluation, best_index)
    if "primary" not in best_models:
        if verbose:
            print(f"Loading {primary_evaluation.model_name} run {run_id(best_primary_run)}")
        best_models["primary"] = primary_evaluation.load_and_evaluate(best_primary_run, metagene_indices)
    elif verbose:
        print(f"Using cached {primary_evaluation.model_name} result")
    primary_datasets = best_models["primary"][level]

    first_affinity = primary_datasets[0].uns[spatial_affinity_key][primary_datasets[0].name]
    num_metagenes = first_affinity.shape[0]
    mask = np.ones((num_metagenes, num_metagenes), dtype=bool)
    mask[np.triu_indices_from(mask)] = 0

    first_row = np.atleast_1d(subfigs[0].subplots(nrows=1, ncols=len(primary_datasets), squeeze=True))
    if use_residuals:
        truth_rank_key = f"{correlation_truth_key}_rank"
        store_ranked_affinity_matrices(primary_datasets, affinity_key=correlation_truth_key, output_key=truth_rank_key)
        _multireplicate_heatmap(
            primary_datasets,
            uns=truth_rank_key,
            label_font_size=1.5,
            mask=mask,
            axes=first_row,
            cmap="ocean",
            vmin=0,
        )
    else:
        _spatial_affinity_heatmap(
            primary_datasets,
            spatial_affinity_key=correlation_truth_key,
            label_values=False,
            label_font_size=1.5,
            mask=mask,
            axes=first_row,
        )

    second_row = np.atleast_1d(subfigs[1].subplots(nrows=1, ncols=len(primary_datasets), squeeze=True))
    if use_residuals:
        affinity_rank_key = f"{spatial_affinity_key}_rank"
        store_ranked_affinity_matrices(
            primary_datasets,
            affinity_key=spatial_affinity_key,
            output_key=affinity_rank_key,
        )
        _multireplicate_heatmap(
            primary_datasets,
            uns=affinity_rank_key,
            label_font_size=1.5,
            mask=mask,
            axes=second_row,
            cmap="ocean",
            vmin=0,
        )
    else:
        _spatial_affinity_heatmap(
            primary_datasets,
            spatial_affinity_key=spatial_affinity_key,
            label_values=False,
            label_font_size=1.5,
            mask=mask,
            axes=second_row,
        )

    for dataset, ax in zip(primary_datasets, second_row):
        ax.set_title(f"{dataset.uns['affinity_correlation']:.2f}")

    for other_evaluation, subfig in zip(other_evaluations, subfigs[2:]):
        best_other_run = find_run_for_score_index(other_evaluation, best_index)
        if other_evaluation.model_name not in best_models:
            if verbose:
                print(f"Loading {other_evaluation.model_name} run {run_id(best_other_run)}")
            best_models[other_evaluation.model_name] = other_evaluation.load_and_evaluate(
                best_other_run,
                metagene_indices,
            )
        elif verbose:
            print(f"Using cached {other_evaluation.model_name} result")
        other_datasets = best_models[other_evaluation.model_name][level]

        row = np.atleast_1d(subfig.subplots(nrows=1, ncols=len(other_datasets), squeeze=True))
        if use_residuals:
            affinity_rank_key = f"{spatial_affinity_key}_rank"
            store_ranked_affinity_matrices(
                other_datasets,
                affinity_key=spatial_affinity_key,
                output_key=affinity_rank_key,
            )
            _multireplicate_heatmap(
                other_datasets,
                uns=affinity_rank_key,
                label_font_size=1.5,
                mask=mask,
                axes=row,
                cmap="ocean",
                vmin=0,
            )
        else:
            _spatial_affinity_heatmap(
                other_datasets,
                spatial_affinity_key=spatial_affinity_key,
                label_values=False,
                label_font_size=1.5,
                mask=mask,
                axes=row,
            )

        for dataset, ax in zip(other_datasets, row):
            ax.set_title(f"{dataset.uns['affinity_correlation']:.2f}")

    for subfig in subfigs:
        for ax in subfig.axes:
            for i in range(num_metagenes):
                for j in range(i, num_metagenes):
                    highlight_cell(j, i, color="gray", ax=ax, linewidth=0.5)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    return overall_fig, best_models


__all__ = [
    highlight_cell.__name__,
    affinity_matrix_dataframe.__name__,
    display_spatial_affinity_results.__name__,
    lambda_sweep_affinity_range_summary.__name__,
    plot_lambda_sweep_affinity_range.__name__,
    plot_spatial_affinity_matrix_panel.__name__,
    plot_pairwise_comparison.__name__,
    plot_best_in_situ_result.__name__,
    plot_best_affinity_correlation_result.__name__,
]
