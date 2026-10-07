"""A small check of attention weights, head projection, and residual."""

import torch

import config
from attention import attention
from model import model


def test_attention_uniform_head_adds_mean_value_to_residual():
    network = model()
    with torch.no_grad():
        for weight in (network.W_Q, network.W_K, network.W_V, network.W_O):
            weight.zero_()
        network.W_V[0, 0, 0] = 1
        network.W_O[0, 0, 0] = 1

    x0 = torch.zeros(1, config.d_context, config.d_model)
    x0[0, :, 0] = torch.tensor([1, 2, 3])
    x1 = attention(network, x0)

    expected = x0.clone()
    expected[0, :, 0] += 2  # Zero Q and K give equal weight to all three values.
    assert x1.shape == x0.shape
    assert torch.allclose(x1, expected)
