"""Gene-set analysis helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact

from popari.util import get_metagene_signature


def _resolve_gene_set_libraries(gene_sets, *, organism: str, gseapy) -> list[dict[str, set[str]]]:
    """Resolve named and custom gene-set libraries to term memberships."""

    specifications = [gene_sets] if isinstance(gene_sets, (str, Mapping)) else list(gene_sets)
    libraries = []
    for specification in specifications:
        if isinstance(specification, str):
            definitions = gseapy.get_library(name=specification, organism=organism)
        elif isinstance(specification, Mapping):
            definitions = specification
        else:
            raise TypeError("gene_sets must contain library names or term-to-gene mappings.")
        libraries.append({term: set(genes) for term, genes in definitions.items()})
    return libraries


def _normalize_enrichment_term(term: str) -> str:
    """Normalize presentation-only differences in enrichment term labels."""

    normalized = " ".join(term.split()).casefold()
    return re.sub(r"\s*\([a-z]+:\d+\)\s*$", "", normalized)


def _enrichment_term_identifier(term: str) -> str | None:
    """Extract an ontology accession such as ``GO:0006357``."""

    match = re.search(r"\b[a-z]+:\d+\b", term, flags=re.IGNORECASE)
    return None if match is None else match.group(0).casefold()


def run_enrichr(
    gene_list: Iterable[str],
    *,
    gene_sets,
    organism: str,
    background: Iterable[str],
) -> pd.DataFrame:
    """Run Enrichr over-representation analysis for one gene list.

    Args:
        gene_list: Genes to test for enrichment.
        gene_sets: Gene-set libraries or custom gene-set definitions accepted
            by :func:`gseapy.enrichr`.
        organism: Organism name accepted by Enrichr.
        background: Measured genes defining the enrichment universe.

    Returns:
        Enrichr results sorted by adjusted p-value, with explicit
        ``Overlap Count``, ``Gene Set Size``, and ``Overlap Ratio`` columns.

    Raises:
        ValueError: If ``gene_list`` or ``background`` is empty.

    """

    genes = list(dict.fromkeys(gene_list))
    measured_genes = list(dict.fromkeys(background))
    if not genes:
        raise ValueError("gene_list must contain at least one gene.")
    if not measured_genes:
        raise ValueError("background must contain at least one gene.")

    import gseapy as gp

    enrichment_result = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_sets,
        organism=organism,
        background=measured_genes,
        outdir=None,
        no_plot=True,
    )
    results = enrichment_result.results.copy()
    libraries = _resolve_gene_set_libraries(gene_sets, organism=organism, gseapy=gp)

    memberships = {}
    memberships_by_identifier = {}
    for library in libraries:
        for term, members in library.items():
            memberships.setdefault(_normalize_enrichment_term(term), members)
            identifier = _enrichment_term_identifier(term)
            if identifier is not None:
                memberships_by_identifier.setdefault(identifier, members)

    normalized_terms = results["Term"].map(_normalize_enrichment_term)
    term_identifiers = results["Term"].map(_enrichment_term_identifier)
    term_memberships = pd.Series(
        [
            memberships.get(normalized_term) or memberships_by_identifier.get(identifier)
            for normalized_term, identifier in zip(normalized_terms, term_identifiers)
        ],
        index=results.index,
        dtype=object,
    )
    missing_terms = results.loc[term_memberships.isna(), "Term"].sort_values().tolist()
    if missing_terms:
        raise ValueError(
            "Could not find returned Enrichr terms in the resolved gene-set libraries: " f"{missing_terms[:5]}",
        )

    query_genes = set(genes)
    results["Overlap Count"] = term_memberships.map(lambda members: len(query_genes & members))
    results["Gene Set Size"] = term_memberships.map(len)
    results["Overlap Ratio"] = results["Overlap Count"] / results["Gene Set Size"]
    return results.sort_values("Adjusted P-value").reset_index(drop=True)


def compute_metagene_enrichment(
    dataset: ad.AnnData,
    *,
    gene_sets,
    organism: str,
    metagene_indices: Iterable[int] | None = None,
    sensitivity: float = 0.5,
    signature_type: str = "upregulated",
    background: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run Enrichr on signatures derived from learned Popari metagenes.

    Args:
        dataset: Dataset containing learned metagenes.
        gene_sets: Gene-set libraries or custom definitions accepted by
            :func:`run_enrichr`.
        organism: Organism name accepted by Enrichr.
        metagene_indices: Metagenes to analyze. By default, analyze all
            metagenes.
        sensitivity: Knee-detection sensitivity used to define signatures.
        signature_type: Select the upregulated or downregulated signature.
        background: Enrichment universe. By default, use measured genes.

    Returns:
        Combined Enrichr results with a leading ``metagene`` column.

    """

    if "M" not in dataset.uns:
        raise KeyError('dataset.uns["M"] is missing.')
    metagenes = np.asarray(dataset.uns["M"])
    indices = range(metagenes.shape[1]) if metagene_indices is None else metagene_indices
    measured_genes = list(dataset.var_names if background is None else background)
    results = []

    for index in indices:
        signature = get_metagene_signature(
            metagenes[:, index],
            dataset.var_names,
            sensitivity=sensitivity,
            type=signature_type,
            show_plot=False,
        )
        if not signature:
            continue

        enrichment = run_enrichr(
            signature,
            gene_sets=gene_sets,
            organism=organism,
            background=measured_genes,
        )
        if enrichment.empty:
            continue

        enrichment.insert(0, "metagene", f"m{index}")
        results.append(enrichment)

    if not results:
        return pd.DataFrame(columns=["metagene"])
    return pd.concat(results, ignore_index=True)


def order_columns_by_best_row(scores):
    """Order matrix columns by strongest row association, strongest columns
    first."""

    return np.argsort(np.argmax(scores, axis=0) - np.max(scores, axis=0) / (np.max(scores) + 1))


def compute_gene_set_auroc(
    dataset,
    gene_sets,
    metagene_key: str = "M",
):
    """Compute metagene correspondence to gene-set columns."""

    from scipy.stats import mannwhitneyu
    from sklearn.metrics import auc, roc_curve

    if metagene_key not in dataset.uns:
        raise KeyError(f"dataset.uns[{metagene_key!r}] is missing.")
    metagenes = np.asarray(dataset.uns[metagene_key])
    _, num_metagenes = metagenes.shape
    num_gene_sets = len(gene_sets.columns)

    aurocs = np.zeros((num_gene_sets, num_metagenes))
    pvalues = np.zeros(aurocs.shape)
    for gene_set_index, gene_set_name in enumerate(gene_sets):
        gene_set = gene_sets[gene_set_name]
        gene_set_membership = dataset.var_names.isin(gene_set)

        for metagene_index, metagene in enumerate(metagenes.T):
            metagene_subset = metagene[gene_set_membership]
            metagene_subset_complement = metagene[~gene_set_membership]

            fpr, tpr, _ = roc_curve(gene_set_membership, metagene)
            aurocs[gene_set_index, metagene_index] = auc(fpr, tpr)
            pvalues[gene_set_index, metagene_index] = mannwhitneyu(
                metagene_subset,
                metagene_subset_complement,
            ).pvalue

    return aurocs, pvalues


def compute_gene_set_enrichment(
    query_sets: Mapping[str, Iterable[str]],
    reference_sets: Mapping[str, Iterable[str]],
    *,
    background: Iterable[str],
) -> pd.DataFrame:
    """Test pairwise enrichment between query and reference gene sets.

    Args:
        query_sets: Named gene sets whose enrichment will be evaluated.
        reference_sets: Named gene sets used as enrichment targets.
        background: Measured genes defining the contingency-table universe.

    Returns:
        One row per query-reference pair, including set sizes, overlapping
        genes, Fisher's exact-test results, and Benjamini-Hochberg-adjusted
        p-values.

    Raises:
        ValueError: If the background, query sets, or reference sets are empty.

    """

    universe = set(background)
    if not universe:
        raise ValueError("background must contain at least one gene.")
    if not query_sets:
        raise ValueError("query_sets must contain at least one named gene set.")
    if not reference_sets:
        raise ValueError("reference_sets must contain at least one named gene set.")

    records = []
    for query_name, query_genes in query_sets.items():
        query = set(query_genes) & universe
        for reference_name, reference_genes in reference_sets.items():
            reference = set(reference_genes) & universe
            overlap = query & reference
            result = fisher_exact(
                [
                    [len(overlap), len(query - reference)],
                    [len(reference - query), len(universe - (query | reference))],
                ],
                alternative="greater",
            )
            records.append(
                {
                    "query": query_name,
                    "reference": reference_name,
                    "query_size": len(query),
                    "reference_size": len(reference),
                    "overlap_size": len(overlap),
                    "overlap_genes": tuple(sorted(overlap)),
                    "odds_ratio": result.statistic,
                    "pvalue": result.pvalue,
                },
            )

    results = pd.DataFrame.from_records(records)
    results["adjusted_pvalue"] = false_discovery_control(results["pvalue"].to_numpy(), method="bh")
    return results


__all__ = [
    run_enrichr.__name__,
    compute_metagene_enrichment.__name__,
    order_columns_by_best_row.__name__,
    compute_gene_set_auroc.__name__,
    compute_gene_set_enrichment.__name__,
    get_metagene_signature.__name__,
]
