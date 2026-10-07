"""A small check of the MLP's activation and bias placement."""

import torch

import config
from mlp import mlp
from model import model


def test_mlp_adds_hidden_bias_before_relu():
    network = model()
    with torch.no_grad():
        for weight in (network.W_i, network.b_i, network.W_o, network.b_o):
            weight.zero_()
        network.W_i[0, 0] = 1
        network.b_i[0] = 0.5
        network.W_o[0, 0] = 2
        network.b_o[0] = 1

    x1 = torch.zeros(1, config.d_context, config.d_model)
    x1[0, :, 0] = torch.tensor([-2, 1, 0])
    x2 = mlp(network, x1)

    expected = torch.zeros_like(x1)
    expected[0, :, 0] = torch.tensor([1, 4, 2])
    assert torch.allclose(x2, expected)
