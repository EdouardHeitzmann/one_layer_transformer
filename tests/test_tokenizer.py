"""Small reference checks; groups should add edge cases of their own."""

import torch

from config import *
from tokenizer import tokenize_string, tokenize_tensor, tokenize_values


def test_tokenize_string_shape():
    a = 2 % modulus
    b = 3 % modulus
    t = tokenize_string(f'{a} {b}')
    assert type(t) == torch.Tensor
    assert t.shape == (3, d_vocab)


