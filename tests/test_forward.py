"""A small contract check for the assembled model's forward pass."""

import torch

from config import d_vocab
from model import model
from tokenizer import tokenize_values


def test_forward_output_shape():
    in_test = torch.zeros(1, 3, d_vocab)
    m = model()
    output = m.forward(in_test)
    assert output.shape == in_test.shape
