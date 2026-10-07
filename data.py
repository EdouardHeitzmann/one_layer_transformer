
import torch

from config import *
from encoding import encode_tensor

def generate_data( conf : config, n : int = None ) :
    """
    Choose n distinct pairs (a, b) from 0..modulus-1, in random order.
    Return integer tensors of shape (n, 3): inputs (a, b, modulus), where
    modulus is the unknown-answer token, and targets (a, b, (a+b) modulo modulus).
    Use 0 <= n <= modulus^2.
    """
    return in_data, out_data

def generate_data_tensor( conf : config, n : int = None ) :
    """
    Generate the same input/target pairs as generate_data(n), then one-hot
    encode every token. Return two float32 tensors of shape (n, 3, d_vocab).
    """
    in_data, out_data = generate_data(conf, n)
    return encode_tensor(conf, in_data), encode_tensor(conf, out_data) 
