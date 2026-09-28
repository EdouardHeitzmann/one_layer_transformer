
import torch

from config import *
from tokenizer import tokenize_tensor

def generate_data( n : int ) :
    """
    Choose n distinct pairs (a, b) from 0..modulus-1, in random order.
    Return integer tensors of shape (n, 3): inputs (a, b, modulus), where
    modulus is the unknown-answer token, and targets (a, b, (a+b) % modulus).
    Use 0 <= n <= modulus**2.
    """
    return in_data, out_data

def generate_data_tensor( n : int ) :
    """
    Generate the same input/target pairs as generate_data(n), then one-hot
    encode every token. Return two float32 tensors of shape (n, 3, d_vocab).
    """
    in_data, out_data = generate_data(n)
    return tokenize_tensor(in_data), tokenize_tensor(out_data) 
