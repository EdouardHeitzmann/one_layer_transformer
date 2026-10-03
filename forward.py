
import torch

def forward_pass( self, x : torch.Tensor ) :
    """
    Run a batch of one-hot token sequences through the model's weights.
    x has shape (batch, d_context, d_vocab). Apply token and position
    embeddings, attention, and the MLP, then return unnormalized token
    logits with the same shape as x.
    """
    x0 = x @ self.W_E + self.W_p
    S = x0.unsqueeze(-3) @ self.W_Q.unsqueeze(-4) @  \
        self.W_K.unsqueeze(-4).mT @ x0.unsqueeze(-3).mT
    S -= S.max(dim=-1,keepdim=True).values
    A = S.softmax(dim=-1)
    x1 = (A @ x0.unsqueeze(-3) @ self.W_V @ self.W_O).sum(dim=-3) + x0
    mlp = torch.relu(x1 @ self.W_i + self.b_i)
    x2 = mlp @ self.W_o + self.b_o
    logits = x2 @ self.W_U 
    return logits.squeeze().expand_as(x)
