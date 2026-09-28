"""The loss must score the answer position, not the known input tokens."""

import math

import torch

import config
from loss import loss_fn


def test_loss_uses_last_token_only():
    target = torch.zeros(1, config.d_context, config.d_vocab)
    target[0, :, 2] = 1
    logits = torch.zeros_like(target)
    logits[0, 0, 2] = -100  # A bad prediction at a known input position.
    actual = loss_fn(target, logits)
    assert actual.ndim == 0
    assert torch.isclose(actual, torch.tensor(math.log(config.d_vocab)), atol=1e-6)
