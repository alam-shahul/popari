import numpy as np
import torch


@torch.no_grad()
def project_M(M, M_constraint):
    result = M.clone()
    if M_constraint == "simplex":
        result = project2simplex_(result, dim=0, minimum_value=1e-5)
    elif M_constraint == "unit sphere":
        result = M.div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    elif M_constraint == "nonneg unit sphere":
        result = M.clip(1e-10).div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return result


def project_M_(M, M_constraint):
    result = M.clone()
    if M_constraint == "simplex":
        result = project2simplex_(result, dim=0, minimum_value=1e-5)
    elif M_constraint == "unit sphere":
        result = M.div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    elif M_constraint == "nonneg unit sphere":
        result = M.clip(1e-10).div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return result


def project2simplex(y, dim: int = 0, minimum_value: float = 1e-10) -> torch.Tensor:
    """Return the Euclidean projection of ``y`` onto a unit simplex."""

    return project2simplex_(y.clone(), dim=dim, minimum_value=minimum_value)


def project2simplex_(y, dim: int = 0, minimum_value: float = 1e-10) -> torch.Tensor:
    """Project ``y`` onto a unit simplex in place using an active-set sort."""

    if not np.isfinite(minimum_value) or minimum_value < 0:
        raise ValueError("minimum_value must be finite and nonnegative.")

    num_components = y.shape[dim]
    if num_components == 0:
        raise ValueError("Cannot project an empty dimension onto the simplex.")

    simplex_mass = 1.0 - num_components * minimum_value
    if simplex_mass < 0:
        raise ValueError(
            f"minimum_value={minimum_value} is infeasible for a simplex with " f"{num_components} components.",
        )

    values = y.movedim(dim, -1)
    if simplex_mass == 0:
        values.fill_(minimum_value)
        return y

    # Projection is invariant to a common offset. Centering prevents loss of
    # precision when every component has a large shared magnitude.
    shifted = values - values.amax(dim=-1, keepdim=True)
    ordered = shifted.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(dim=-1).sub_(simplex_mass)
    ranks = torch.arange(
        1,
        num_components + 1,
        device=y.device,
        dtype=y.dtype,
    )
    active = ordered - cumulative / ranks > 0
    active_count = active.sum(dim=-1, keepdim=True)
    threshold = cumulative.gather(dim=-1, index=active_count - 1) / active_count.to(y.dtype)
    values.copy_(shifted.sub_(threshold).clamp_min_(0).add_(minimum_value))
    return y
