

import torch
from torch import nn

from config import *

def encode_tensor( c : config, tokens ) :
    """One-hot encode integer token IDs of any shape; append a d_vocab axis.

    Return float32 values. IDs 0 through modulus-1 represent numbers, and
    the final ID (modulus) represents the unknown answer token.
    """
    if c == None : c = config.default_config
    output = nn.functional.one_hot( tokens, c.d_vocab ).to( torch.float32 )
    return output

def encode_values( c : config, *tokens ) :
    """One-hot encode a sequence of integer token IDs into (count, d_vocab)."""
    return encode_tensor( c, torch.tensor(tokens) )

def encode_string( c : config, s ) :
    """Parse two space-separated integers and encode (a, b, unknown answer).

    Return a float32 tensor of shape (3, d_vocab).
    """
    l = s.split(' ')
    v = [ int(l[0]), int(l[1]), c.d_vocab-1 ]
    return encode_values(c, *v)
