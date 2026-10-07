"""Reference checks for the integer and one-hot data generators."""

import torch

import config
from data import generate_data, generate_data_tensor


<<<<<<< Updated upstream
=======
def test_generate_data_output_length():
    n = min(10, config.modulus**2)
    inputs, targets = generate_data(n)
    assert inputs.shape[0] == n
    assert targets.shape[0] == n
>>>>>>> Stashed changes
