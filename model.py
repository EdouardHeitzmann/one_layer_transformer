from config import *
from encoding import encode_tensor, encode_string, encode_values
from loss import loss_fn
from data import generate_data_tensor


from embedding import embedding
from attention import attention
from mlp import mlp
from unembedding import unembedding
    
import torch
from torch import nn



class model( nn.Module ) :
    def __init__( self, c : config ) :
        """Create the model's learnable weights and AdamW optimizer."""
        super().__init__()

        # initialize model parameters
        self.W_E = nn.Parameter( torch.randn( c.d_vocab, c.d_model, dtype=torch.float32) 
                                    / c.d_vocab**.5 )
        self.W_p = nn.Parameter( torch.randn( c.d_context, c.d_model, dtype=torch.float32) 
                                    / c.d_context**.5 )

        self.W_Q = nn.Parameter( torch.randn(c.num_heads, c.d_model, c.d_head, 
                                             dtype=torch.float32) / c.d_model**.5 )
        self.W_K = nn.Parameter( torch.randn(c.num_heads, c.d_model, c.d_head, 
                                             dtype=torch.float32) / c.d_model**.5 )
        self.W_V = nn.Parameter( torch.randn(c.num_heads, c.d_model, c.d_head, 
                                             dtype=torch.float32) / c.d_model**.5 )

        self.W_O = nn.Parameter( torch.randn(c.num_heads, c.d_head, c.d_model, 
                                             dtype=torch.float32) / c.d_head**.5 )

        self.W_i = nn.Parameter( torch.randn(c.d_model, c.d_mlp, 
                                             dtype=torch.float32) / c.d_model**.5 )
        self.b_i = nn.Parameter( torch.randn(c.d_mlp, dtype=torch.float32 ) )

        self.W_o = nn.Parameter( torch.randn(c.d_mlp, c.d_model, 
                                             dtype=torch.float32) / c.d_mlp**.5 )
        self.b_o = nn.Parameter( torch.randn(c.d_model, dtype=torch.float32 ) )

        self.W_U = nn.Parameter( torch.randn(c.d_model, c.d_vocab, 
                                             dtype=torch.float32) / c.d_model**.5 )

        # setup optimizer
        self.optimizer = torch.optim.AdamW( self.parameters(), lr=c.LR, 
                                            weight_decay=c.WEIGHT_DECAY )


    def forward( self, x : torch.Tensor ) :
        """Return logits for a batch of token sequences."""
        x0 = embedding(self, x)
        x1 = attention(self, x0)
        x2 = mlp(self, x1)
        return unembedding(self, x2)


    def train( self, in_tensor, out_tensor, dry_run=False ) :
        """Run one training step and return its scalar loss.

        Clear gradients, compute logits and loss, and backpropagate. Update
        weights unless dry_run is True; dry_run still computes gradients.
        """
        self.optimizer.zero_grad()
        logits = self.forward(in_tensor)
        loss = loss_fn( out_tensor, logits )
        loss.backward()
        if not dry_run : 
            self.optimizer.step()
        return loss





        
