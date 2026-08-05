import pytest
import torch

from popari.optim.simplex_integral import integrate_of_exponential_over_simplex

pytestmark = [pytest.mark.baseline, pytest.mark.cheap]


def _legacy_integral(eta, eps=1e-30):
    _, num_factors = eta.shape
    log_abs = torch.empty_like(eta)
    signs = torch.empty_like(eta)
    for factor in range(num_factors):
        difference = eta - eta[:, [factor]]
        difference[:, factor] = 1
        signs[:, factor] = difference.sign().prod(dim=1)
        difference = difference.abs().add(eps).log()
        difference[:, factor] = eta[:, factor]
        log_abs[:, factor] = difference.sum(dim=1)

    log_abs.neg_()
    maxima = log_abs.max(dim=1, keepdim=True).values
    signed_sum = log_abs.sub(maxima).exp().mul(signs).sum(dim=-1).clamp_min(eps)
    return signed_sum.log().add(maxima.squeeze(dim=-1))


def _well_separated_eta(*, dtype, device):
    generator = torch.Generator(device=device).manual_seed(0)
    offsets = torch.arange(6, dtype=dtype, device=device).mul(0.75)
    return offsets + 0.05 * torch.randn((8, 6), dtype=dtype, device=device, generator=generator)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 2e-4, 2e-5),
        (torch.float64, 1e-10, 1e-11),
    ],
)
def test_simplex_log_integral_matches_legacy_values_and_gradients(dtype, rtol, atol):
    eta = _well_separated_eta(dtype=dtype, device="cpu")
    legacy_eta = eta.clone().requires_grad_()
    vectorized_eta = eta.clone().requires_grad_()

    legacy = _legacy_integral(legacy_eta)
    vectorized = integrate_of_exponential_over_simplex(vectorized_eta)
    legacy_gradient = torch.autograd.grad(legacy.sum(), legacy_eta)[0]
    vectorized_gradient = torch.autograd.grad(vectorized.sum(), vectorized_eta)[0]

    torch.testing.assert_close(vectorized, legacy, rtol=rtol, atol=atol)
    torch.testing.assert_close(vectorized_gradient, legacy_gradient, rtol=rtol, atol=atol)


def test_simplex_log_integral_matches_analytic_low_dimensional_cases():
    one_factor = torch.tensor([[0.5], [-1.25]], dtype=torch.float64)
    torch.testing.assert_close(
        integrate_of_exponential_over_simplex(one_factor),
        -one_factor.squeeze(dim=1),
    )

    two_factors = torch.tensor([[0.25, 1.5], [-0.5, 0.75]], dtype=torch.float64)
    first, second = two_factors.unbind(dim=1)
    expected = ((torch.exp(-first) - torch.exp(-second)) / (second - first)).log()
    torch.testing.assert_close(integrate_of_exponential_over_simplex(two_factors), expected)


@pytest.mark.gpu
def test_simplex_log_integral_matches_legacy_on_cuda(gpu_context):
    eta = _well_separated_eta(dtype=gpu_context["dtype"], device=gpu_context["device"])
    legacy_eta = eta.clone().requires_grad_()
    vectorized_eta = eta.clone().requires_grad_()

    legacy = _legacy_integral(legacy_eta)
    vectorized = integrate_of_exponential_over_simplex(vectorized_eta)
    legacy_gradient = torch.autograd.grad(legacy.sum(), legacy_eta)[0]
    vectorized_gradient = torch.autograd.grad(vectorized.sum(), vectorized_eta)[0]

    torch.testing.assert_close(vectorized, legacy, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(vectorized_gradient, legacy_gradient, rtol=1e-10, atol=1e-11)
