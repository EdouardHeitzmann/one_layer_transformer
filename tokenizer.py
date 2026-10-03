

import torch
from torch import nn

import config

def tokenize_tensor( tokens ) :
    """One-hot encode integer token IDs of any shape; append a d_vocab axis.

    Return float32 values. With N = config.modulus, IDs 0 through N-1
    represent numbers, and ID N represents the unknown answer token.
    """
    return nn.functional.one_hot( tokens, config.d_vocab ).to( torch.float32 )

def tokenize_values( *tokens ) :
    """One-hot encode a sequence of integer token IDs into (count, d_vocab)."""
    return tokenize_tensor( torch.tensor(tokens) )

def tokenize_string( s ) :
    """Parse two space-separated integers and encode (a, b, unknown answer).

    Return a float32 tensor of shape (3, d_vocab).
    """
    l = s.split(' ')
    v = [ int(l[0]), int(l[1]), config.d_vocab-1 ]
    return tokenize_values(*v)
