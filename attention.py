"""The four-head attention stage of the forward pass."""

import torch


def attention(model, x0: torch.Tensor) -> torch.Tensor:
    """Return x1 [batch, context, model width] from embedded input x0.

    For each of the four heads h, compute Q=x0@W_Q[h], K=x0@W_K[h],
    V=x0@W_V[h], score matrix S=Q@K.T, and attention weights A=softmax(S)
    over keys (the last dimension). Project A@V through W_O[h], sum the
    projected head outputs, and add the residual x0. The head weights are
    slices of model.W_Q, W_K, W_V, and W_O. This model uses no causal mask
    or score scaling.
    """
    raise NotImplementedError("Implement attention in attention.py")
