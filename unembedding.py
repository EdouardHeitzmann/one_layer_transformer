"""The provided output projection for the forward pass."""

import torch


def unembedding(model, x2: torch.Tensor) -> torch.Tensor:
    """Map x2 [batch, context, model width] to token logits via W_U."""
    # garbage code
    logits = x2 @ model.W_U
    return logits
