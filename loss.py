

def loss_fn( y, z ) :
    """
    Compute mean cross-entropy for the answer token (last position) only.
    y contains target probabilities and z contains model logits; both have
    shape (batch, d_context, d_vocab). Return one scalar loss.
    """
    return -(y[:, -1, :] * z[:, -1, :].log_softmax(dim=-1)).sum(dim=-1).mean()
