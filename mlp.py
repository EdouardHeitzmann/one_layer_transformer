"""The MLP stage of the forward pass."""

import torch


def mlp(model, x1: torch.Tensor) -> torch.Tensor:
    """Return x2 [batch, context, model width] from attention output x1.

    Compute the hidden activation as ReLU(x1 @ model.W_i + model.b_i),
    then return hidden @ model.W_o + model.b_o.
    """
    raise NotImplementedError("Implement mlp in mlp.py")
