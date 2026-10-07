"""A small contract check for the assembled model's forward pass."""

import torch

from config import *
from model import *
from encoding import *


def test_forward_output_shape():
    c = config(modulus=23)
    in_test = torch.zeros(1, 3, c.d_vocab)
    m = model(c)
    output = m.forward(in_test)
    assert output.shape == in_test.shape
