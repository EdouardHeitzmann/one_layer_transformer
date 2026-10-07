"""A small check of token and position embeddings."""

import torch

import config
from embedding import embedding
from model import model


def test_embedding_combines_token_and_position_weights():
    network = model()
    with torch.no_grad():
        network.W_E.zero_()
        network.W_p.zero_()
        network.W_E[0, 0] = 2
        network.W_p[:, 0] = torch.tensor([1, 2, 3])

    x = torch.zeros(1, config.d_context, config.d_vocab)
    x[0, 0, 0] = 1
    x[0, 1, 0] = 1
    x0 = embedding(network, x)

    expected = torch.zeros(1, config.d_context, config.d_model)
    expected[0, :, 0] = torch.tensor([3, 4, 3])
    assert torch.equal(x0, expected)
