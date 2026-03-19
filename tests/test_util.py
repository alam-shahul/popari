import pytest
import torch

from popari.util import project2simplex, project2simplex_


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

    projected = project2simplex(projection_input, dim=1)
    projected_inplace = project2simplex_(projection_input, dim=1)

    assert torch.allclose(projected, projected_inplace, atol=1e-5)
