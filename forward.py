
import torch

from embedding import embedding
from attention import attention
from mlp import mlp
from unembedding import unembedding


def forward_pass(model, x: torch.Tensor) -> torch.Tensor:
    """Apply the model stages to one-hot input x [batch, context, vocab].

    Return unnormalized token logits [batch, context, vocab]. Tokenization
    happens before this function, and the loss is computed afterward.
    """
    x0 = embedding(model, x)
    x1 = attention(model, x0)
    x2 = mlp(model, x1)
    return unembedding(model, x2)
