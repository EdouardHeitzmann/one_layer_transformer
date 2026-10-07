"""The four-head attention stage of the forward pass."""

import torch


def attention(self, x0: torch.Tensor) -> torch.Tensor:
    """Return x1 of shape [batch, context, model width] from embedded input x0.

    For each of the four heads h, compute Q=x0*W_Q[h], K=x0*W_K[h],
    V=x0*W_V[h], score matrix S=Q*K^t, and attention weights A=softmax(S). 
    Output of each head is A*V*W_O[h], sum the
    projected head outputs, and add the residual x0.     
    """
    S = x0.unsqueeze(-3) @ self.W_Q.unsqueeze(-4) @  \
        self.W_K.unsqueeze(-4).mT @ x0.unsqueeze(-3).mT
    S -= S.max(dim=-1,keepdim=True).values
    A = S.softmax(dim=-1)
    x1 = (A @ x0.unsqueeze(-3) @ self.W_V @ self.W_O).sum(dim=-3) + x0
    return x1
