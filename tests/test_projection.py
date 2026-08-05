import numpy as np
import pytest
import torch

from popari.optim.projection import project2simplex, project2simplex_


def _simplex_reference(values, dim, minimum_value):
    moved = np.moveaxis(values.detach().cpu().numpy().astype(np.longdouble), dim, -1)
    flattened = moved.reshape(-1, moved.shape[-1])
    projected = np.empty_like(flattened)
    simplex_mass = np.longdouble(1) - flattened.shape[-1] * np.longdouble(minimum_value)
    ranks = np.arange(1, flattened.shape[-1] + 1, dtype=np.longdouble)

    for index, vector in enumerate(flattened):
        shifted = vector - vector.max()
        ordered = np.sort(shifted)[::-1]
        cumulative = np.cumsum(ordered, dtype=np.longdouble) - simplex_mass
        active = ordered - cumulative / ranks > 0
        active_count = np.flatnonzero(active)[-1] + 1
        threshold = cumulative[active_count - 1] / active_count
        projected[index] = np.maximum(shifted - threshold, 0) + minimum_value

    projected = projected.reshape(moved.shape)
    return np.moveaxis(projected, -1, dim).astype(values.detach().cpu().numpy().dtype)


@pytest.mark.baseline
def test_project2simplex_projects_columns_to_simplex():
    projection_input = torch.tensor(
        [[0.2, -1.0, 3.0], [1.2, 0.5, -2.0], [0.8, 2.5, 1.0]],
        dtype=torch.float32,
    )

    projection_output = project2simplex(projection_input, dim=0)

    assert torch.all(projection_output >= 0)
    assert torch.allclose(projection_output.sum(dim=0), torch.ones(3), atol=1e-4)


@pytest.mark.baseline
def test_project2simplex_inplace_matches_functional():
    projection_input = torch.tensor(
        [[0.2, -1.0, 3.0], [1.2, 0.5, -2.0], [0.8, 2.5, 1.0]],
        dtype=torch.float32,
    )

    original = projection_input.clone()
    projected = project2simplex(projection_input, dim=1)
    assert torch.equal(projection_input, original)
    projected_inplace = project2simplex_(projection_input, dim=1)

    assert torch.allclose(projected, projected_inplace, atol=1e-5)
    assert projected_inplace is projection_input
    assert not torch.equal(projection_input, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("step_size", [1e-3, 5e-2, 5e-1])
def test_project2simplex_matches_reference_for_random_updates(dtype, dim, step_size):
    generator = torch.Generator().manual_seed(7)
    shape = (257, 10)
    initial = torch.rand(shape, generator=generator, dtype=dtype)
    initial /= initial.sum(dim=dim, keepdim=True)
    values = initial + step_size * torch.randn(shape, generator=generator, dtype=dtype)

    projected = project2simplex(values, dim=dim)
    reference = torch.from_numpy(_simplex_reference(values, dim, 1e-10))
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12

    torch.testing.assert_close(projected.cpu(), reference, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        projected.sum(dim=dim),
        torch.ones(projected.shape[1 - dim], dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
    assert torch.all(projected >= 1e-10)
    torch.testing.assert_close(project2simplex(projected, dim=dim), projected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype,offset", [(torch.float32, 1e6), (torch.float64, 1e12)])
def test_project2simplex_is_stable_with_large_common_offset(dtype, offset):
    generator = torch.Generator().manual_seed(11)
    values = offset + torch.randn((128, 10), generator=generator, dtype=dtype)
    projected = project2simplex(values, dim=1)
    reference = torch.from_numpy(_simplex_reference(values, 1, 1e-10))
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12

    torch.testing.assert_close(projected.cpu(), reference, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(projected.sum(dim=1), torch.ones(128, dtype=dtype), atol=tolerance, rtol=tolerance)


def test_project2simplex_respects_minimum_value_and_kkt_conditions():
    values = torch.tensor(
        [[2.0, -3.0, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]],
        dtype=torch.float64,
    )
    minimum_value = 0.1
    projected = project2simplex(values, dim=1, minimum_value=minimum_value)
    active = projected > minimum_value + 1e-12
    residual = values - projected

    for vector_residual, vector_active in zip(residual, active):
        threshold = vector_residual[vector_active][0]
        torch.testing.assert_close(vector_residual[vector_active], threshold.expand(vector_active.sum()))
        assert torch.all(vector_residual[~vector_active] <= threshold + 1e-12)

    assert torch.all(projected >= minimum_value)
    torch.testing.assert_close(projected.sum(dim=1), torch.ones(2, dtype=torch.float64))


@pytest.mark.parametrize("minimum_value", [-1e-3, 0.26, float("nan")])
def test_project2simplex_rejects_invalid_minimum_value(minimum_value):
    with pytest.raises(ValueError, match="minimum_value"):
        project2simplex(torch.ones((2, 4)), dim=1, minimum_value=minimum_value)


def test_project2simplex_handles_fully_reserved_simplex():
    values = torch.randn((3, 4), dtype=torch.float64)
    projected = project2simplex(values, dim=1, minimum_value=0.25)

    assert torch.equal(projected, torch.full_like(values, 0.25))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_project2simplex_runs_on_cuda(dtype, gpu_context):
    device = gpu_context["device"]
    generator = torch.Generator(device=device).manual_seed(19)
    values = torch.randn((1024, 20), generator=generator, device=device, dtype=dtype)

    projected = project2simplex(values, dim=1)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12

    assert torch.isfinite(projected).all()
    assert torch.all(projected >= 1e-10)
    torch.testing.assert_close(
        projected.sum(dim=1),
        torch.ones(1024, device=device, dtype=dtype),
        rtol=tolerance,
        atol=tolerance,
    )
