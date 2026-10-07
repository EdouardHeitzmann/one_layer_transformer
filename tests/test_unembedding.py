"""A small check of the supplied unembedding helper."""

import torch

import config
from model import model
from unembedding import unembedding


def test_unembedding_maps_model_width_to_token_logits():
    network = model()
    with torch.no_grad():
        network.W_U.zero_()
        network.W_U[0, 0] = 2

    x2 = torch.zeros(1, config.d_context, config.d_model)
    x2[0, 0, 0] = 3
    logits = unembedding(network, x2)

    expected = torch.zeros(1, config.d_context, config.d_vocab)
    expected[0, 0, 0] = 6
    assert torch.equal(logits, expected)
