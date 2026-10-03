from config import *
from tokenizer import tokenize_tensor, tokenize_string, tokenize_values
from loss import loss_fn
from data import generate_data_tensor
from forward import forward_pass
from training import *

    



class model( nn.Module ) :
    def __init__( self ) :
        """Create the model's learnable weights and AdamW optimizer."""
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
        """Return logits for a batch of token sequences via forward_pass."""
        return forward_pass( self, x )


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





        
