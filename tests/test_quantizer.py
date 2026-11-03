import numpy as np
import pytest
import torch
from src.common import seed_everything
from src.rqvae import RQVAE


def test_nearest_vectors_use_successive_residuals():
    model = RQVAE(dimensions=(2, 2), levels=3, codebook_size=2)
    with torch.no_grad():
        model.codebooks[0].weight.copy_(torch.tensor([[0., 0.], [2., 0.]]))
        model.codebooks[1].weight.copy_(torch.tensor([[0., 0.], [0., 1.]]))
        model.codebooks[2].weight.copy_(torch.tensor([[0., 0.], [.5, 0.]]))
    quantized, codes, _, _ = model.quantize(torch.tensor([[2.4, .9]]))
    assert codes.tolist() == [[1, 1, 1]]
    assert quantized.tolist() == [[2.5, 1.0]]


def test_straight_through_and_auxiliary_gradient_routing():
    seed_everything(42)
    model = RQVAE(dimensions=(4, 8, 2), codebook_size=4)
    x = torch.randn(12, 4)
    output = model(x)
    output["reconstruction"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.codebooks.parameters())
    model.zero_grad(set_to_none=True)
    model(x)["loss"].backward()
    for group in (model.encoder, model.decoder, model.codebooks):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters())
        assert sum(p.grad.abs().sum().item() for p in group.parameters()) > 0


def test_tiny_batch_reconstruction_improves_and_initialization():
    seed_everything(42)
    model = RQVAE(dimensions=(4, 16, 2), codebook_size=4)
    x = torch.randn(16, 4)
    model.initialize_codebooks(x, 42)
    before = model(x)["reconstruction"].item()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(200):
        optimizer.zero_grad()
        model(x)["loss"].backward()
        optimizer.step()
    assert model(x)["reconstruction"].item() < before * .8
    with pytest.raises(ValueError, match="at least"):
        model.initialize_codebooks(x[:2], 42)
