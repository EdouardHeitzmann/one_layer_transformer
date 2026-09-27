import torch
from torch import nn

LR = .001
WEIGHT_DECAY = 1.

modulus = 7
d_vocab = modulus + 1
d_model = 128
d_head = 32
num_heads = 4
d_mlp = 512
d_context = 3

def tokenize_tensor( tokens ) :
    return nn.functional.one_hot( tokens, d_vocab ).to( torch.float32 )

def tokenize_values( *tokens ) :
    return tokenize_tensor( torch.tensor(tokens) )

def tokenize_string( s ) :
    l = s.split(' ')
    v = [ int(l[0]), int(l[1]), d_vocab-1 ]
    return tokenize_values(*v)

def loss_fn( y, z ) :
    """
    y is training data, shape (..., d_context, d_vocab), in probabilities
    z is model output, shape (..., d_contact, d_vocab), in logits
    computes the cross entropy loss at the last token only
    """
    return - ( y * z.log_softmax(dim=-1) ).sum(dim=-1)[:,...].mean()


def generate_data( n : int ) :
    a,b = torch.meshgrid( torch.arange(modulus), torch.arange(modulus) )
    c = (a + b) % modulus
    abc = torch.stack((a, b, c), dim=-1)
    out_data = abc.reshape(modulus**2,3)[torch.randperm(modulus**2)[:n]]
    in_data = out_data.clone()
    in_data[:,-1] = modulus
    return in_data, out_data

def generate_data_tensor( n : int ) :
    in_data, out_data = generate_data(n)
    return tokenize_tensor(in_data), tokenize_tensor(out_data) 

    



class model( nn.Module ) :
    def __init__( self ) :
        super().__init__()

        # initialize model parameters
        self.W_E = nn.Parameter( torch.randn(d_vocab, d_model, dtype=torch.float32) 
                                    / d_vocab**.5 )
        self.W_p = nn.Parameter( torch.randn(d_context, d_model, dtype=torch.float32) 
                                    / d_context**.5 )

        self.W_Q = nn.Parameter( torch.randn(num_heads, d_model, d_head, 
                                             dtype=torch.float32) / d_model**.5 )
        self.W_K = nn.Parameter( torch.randn(num_heads, d_model, d_head, 
                                             dtype=torch.float32) / d_model**.5 )
        self.W_V = nn.Parameter( torch.randn(num_heads, d_model, d_head, 
                                             dtype=torch.float32) / d_model**.5 )

        self.W_O = nn.Parameter( torch.randn(num_heads, d_head, d_model, 
                                             dtype=torch.float32) / d_head**.5 )

        self.W_i = nn.Parameter( torch.randn(d_model, d_mlp, 
                                             dtype=torch.float32) / d_model**.5 )
        self.b_i = nn.Parameter( torch.randn(d_mlp, dtype=torch.float32 ) )

        self.W_o = nn.Parameter( torch.randn(d_mlp, d_model, 
                                             dtype=torch.float32) / d_mlp**.5 )
        self.b_o = nn.Parameter( torch.randn(d_model, dtype=torch.float32 ) )

        self.W_U = nn.Parameter( torch.randn(d_model, d_vocab, 
                                             dtype=torch.float32) / d_model**.5 )

        # setup optimizer
        self.optimizer = torch.optim.AdamW( self.parameters(), lr=LR, 
                                            weight_decay=WEIGHT_DECAY )


    def forward( self, x : torch.Tensor ) :
        x0 = x @ self.W_E + self.W_p
        S = x0.unsqueeze(-3) @ self.W_Q.unsqueeze(-4) @  \
            self.W_K.unsqueeze(-4).mT @ x0.unsqueeze(-3).mT
        S -= S.max(dim=-1,keepdim=True).values
        A = S.softmax(dim=-1)
        x1 = (A @ x0.unsqueeze(-3) @ self.W_V @ self.W_O).sum(dim=-3) + x0
        mlp = torch.relu(x1 @ self.W_i) + self.b_i
        x2 = mlp @ self.W_o + self.b_o
        logits = x2 @ self.W_U 
        return logits.squeeze().expand_as(x)


    def train( self, in_tensor, out_tensor, dry_run=False ) :
        self.optimizer.zero_grad()
        logits = self.forward(in_tensor)
        loss = loss_fn( out_tensor, logits )
        loss.backward()
        if not dry_run : 
            self.optimizer.step()
        return loss





        
