"""Reference checks for the integer and one-hot data generators."""

import torch

import config
from data import generate_data, generate_data_tensor


def test_generate_data_output_length():
    n = min(10, config.modulus**2)
    inputs, targets = generate_data(n)
    assert inputs.shape[0] == n
    assert targets.shape[0] == n

def test_generate_data_all_pairs_once():
    inputs, targets = generate_data(config.modulus**2)
    assert inputs.shape == targets.shape == (config.modulus**2, 3)
    assert len({tuple(row) for row in targets[:, :2].tolist()}) == config.modulus**2
    assert torch.equal(inputs[:, :2], targets[:, :2])
    assert torch.all(inputs[:, -1] == config.modulus)
    assert torch.equal(targets[:, -1], (targets[:, 0] + targets[:, 1]) % config.modulus)


def test_generate_data_tensor_one_example():
    inputs, targets = generate_data_tensor(1)
    assert inputs.shape == targets.shape == (1, 3, config.d_vocab)
    assert inputs.dtype == targets.dtype == torch.float32
    assert torch.all(inputs.sum(dim=-1) == 1)
    assert torch.all(targets.sum(dim=-1) == 1)
    a, b, c = targets.argmax(dim=-1)[0].tolist()
    assert inputs.argmax(dim=-1)[0].tolist() == [a, b, config.modulus]
    assert c == (a + b) % config.modulus
