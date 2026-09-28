"""A small contract check for the assembled model's forward pass."""

import torch

import config
from model import model
from tokenizer import tokenize_values


def test_forward_one_equation():
    network = model()
    inputs = tokenize_values(0, config.modulus - 1, config.modulus).unsqueeze(0)
    logits = network(inputs)
    assert logits.shape == (1, config.d_context, config.d_vocab)
    assert torch.isfinite(logits).all()
    logits[0, -1, :].sum().backward()
    assert network.W_E.grad is not None
