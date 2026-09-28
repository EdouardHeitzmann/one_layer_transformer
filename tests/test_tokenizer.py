"""Small reference checks; groups should add edge cases of their own."""

import torch

import config
from tokenizer import tokenize_string, tokenize_tensor, tokenize_values


def test_tokenize_tensor_one_integer():
    token_id = min(1, config.modulus - 1)
    encoded = tokenize_tensor(torch.tensor(token_id))
    assert encoded.shape == (config.d_vocab,)
    assert encoded.dtype == torch.float32
    assert torch.equal(encoded, torch.nn.functional.one_hot(torch.tensor(token_id), config.d_vocab).float())


def test_tokenize_values_keeps_order():
    first, second = 0, config.modulus - 1
    encoded = tokenize_values(first, second)
    assert encoded.shape == (2, config.d_vocab)
    assert encoded.argmax(dim=-1).tolist() == [first, second]


def test_tokenize_string_adds_unknown_answer():
    first, second = 0, config.modulus - 1
    encoded = tokenize_string(f"{first} {second}")
    assert encoded.shape == (3, config.d_vocab)
    assert encoded.argmax(dim=-1).tolist() == [first, second, config.modulus]
