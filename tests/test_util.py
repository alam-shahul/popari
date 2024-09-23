import numpy as np
import pytest
import torch

from popari.util import project2simplex, project2simplex_


def test_project2simplex():
    data = np.load("tests/test_data/util/projection_input.npy").astype(np.float32)
    projection_input = torch.from_numpy(data)

    projection_output = project2simplex(projection_input)

    expected_output = np.load("tests/test_data/util/projection_output.npy")
    assert np.allclose(expected_output, projection_output)


def test_project2simplex_():
    data = np.load("tests/test_data/util/projection_input.npy").astype(np.float32)
    projection_input = torch.from_numpy(data)

    projection_output = project2simplex_(projection_input)

    expected_output = np.load("tests/test_data/util/projection_output.npy")
    assert np.allclose(expected_output, projection_output)
