"""Regenerate canonical analysis results from a historical Popari W&B run."""

import argparse
import shutil
import uuid
from pathlib import Path

import torch

from popari.io import load_anndata_hierarchy, save_anndata_hierarchy
from popari.model import load_pretrained
from popari.schema import SCHEMA_VERSION
from popari.train import Trainer
from popari.wandb_util import load_popari_anndata_from_wandb, log_popari_results

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT.parent / "mlflow_experiments" / "regenerated_popari_results"


def replace_directory(source: Path, destination: Path) -> None:
    """Replace a result directory while preserving the old copy on failure."""

    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.replace(backup)
    try:
        source.replace(destination)
    except Exception:
        if backup.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def regenerate_results(
    run_id: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cuda:0",
    entity: str = "popari",
    project: str = "revisions",
    upload: bool = True,
) -> Path:
    """Reconstruct, optionally superresolve, save, and upload one historical
    run."""

    import wandb

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} was requested, but CUDA is unavailable.")

    api_run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    source_artifacts = [
        artifact.name
        for artifact in api_run.logged_artifacts()
        if artifact.type == "popari-model" and "latest" in artifact.aliases
    ]
    if not source_artifacts:
        raise ValueError(f"Run {run_id} has no latest historical popari-model artifacts.")

    source_hierarchy = load_popari_anndata_from_wandb(
        run_id,
        entity=entity,
        project=project,
        artifact_stem=None,
        legacy=True,
    )
    model = load_pretrained(
        source_hierarchy[0],
        reloaded_hierarchy=source_hierarchy,
        hierarchical_levels=len(source_hierarchy),
        context={"device": device, "dtype": torch.float64},
    )
    if model.hierarchical_levels > 1:
        trainer = Trainer(model, iterations=0, verbose=1)
        trainer.superresolve()

    destination = output_root / run_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_anndata_hierarchy(temporary, model.materialize_results())
        hierarchy = load_anndata_hierarchy(temporary)
        replace_directory(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    if upload:
        metadata = {
            "source_run_id": run_id,
            "source_artifacts": source_artifacts,
            "schema_version": SCHEMA_VERSION,
            "hierarchical_levels": len(hierarchy),
            "superresolution": model.hierarchical_levels > 1,
            "superresolution_schedule": (
                [
                    {"learning_rate": 1e-1, "epochs": 10_000, "update_spatial_affinities": False},
                    {"learning_rate": 1e-2, "epochs": 10_000, "update_spatial_affinities": True},
                ]
                if model.hierarchical_levels > 1
                else []
            ),
        }
        with wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume="allow",
            job_type="popari-result-regeneration",
        ) as run:
            log_popari_results(run, destination, metadata=metadata)

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Historical W&B run ID.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--entity", default="popari")
    parser.add_argument("--project", default="revisions")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    output = regenerate_results(
        args.run_id,
        output_root=args.output_root,
        device=args.device,
        entity=args.entity,
        project=args.project,
        upload=not args.no_upload,
    )
    print(output)


if __name__ == "__main__":
    main()
