
import torch
from model import *

def train( m : model, N : int, n : int, 
           in_training : torch.tensor, 
           out_training : torch.tensor,
           in_testing : torch.tensor | None = None, 
           out_testing : torch.tensor | None = None ) :
    """
    Trains on N rounds of batches of size n from training data.  Returns a record of
    loss change on the training data and testing data (if supplied).
    """

    loss_history = []

    for i in range(N) :
        p = torch.randperm(in_training.shape[0])[:n]
        testing_loss = m.train( in_testing, out_testing, dry_run=True )
        training_loss = m.train( in_training[p], out_training[p] )
        loss_history.append((training_loss, testing_loss))
        print(f'round {i} : training loss {training_loss}, testing loss {testing_loss}')

    return loss_history



