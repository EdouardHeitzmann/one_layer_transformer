import torch
from torch import nn

LR = .001
WEIGHT_DECAY = 1.

# N in the class handout: numbers are added modulo this value.
modulus = 23
d_vocab = modulus + 1
d_model = 128
d_head = 32
num_heads = 4
d_mlp = 512
d_context = 3
