

def tokenize_tensor( tokens : torch.tensor ) :
    """
    This function takes a torch tensor with shape (3) as input.
    It returns a torch tensor with shape (3, d_vocab).  Each row
    of the output is the one-hot vector in the position indicated by 
    the corresponding entry of the input.
    """
    return nn.functional.one_hot( tokens, d_vocab ).to( torch.float32 )

def tokenize_values( *tokens ) :
    return tokenize_tensor( torch.tensor(tokens) )

def tokenize_string( s : str ) :
    """
    Takes as input a string 'a b' where a and b are integers between 
    0 and modulus-1, inclusive.  Returns a torch tensor with shape
    (3, d_vocab) whose first row is one-hot in position a, second row
    is 1-hot in position b, and third row is one-hot in position modulus.
    """
    l = s.split(' ')
    v = [ int(l[0]), int(l[1]), d_vocab-1 ]
    return tokenize_values(*v)
