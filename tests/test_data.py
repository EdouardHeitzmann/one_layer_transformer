"""Reference checks for the integer and one-hot data generators."""

import torch

from config import *
from data import generate_data, generate_data_tensor


def test_generate_data_output_length():
    c = config(modulus=10)
    n = c.modulus**2
    inputs, targets = generate_data(c)
    assert inputs.shape[0] == n
    assert targets.shape[0] == n
