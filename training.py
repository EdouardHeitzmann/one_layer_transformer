
import torch
from model import *

def train( m : model, N : int, n : int, 
           in_training : torch.tensor, 
           out_training : torch.tensor,
           in_testing : torch.tensor  = None, 
           out_testing : torch.tensor  = None ) :
    """
    Trains on N rounds of batches of size n from training data.  Returns a record of
    loss change on the training data and testing data (if supplied).
    """

    loss_history = []

    for i in range(N) :
        p = torch.randperm(in_training.shape[0])[:n]
        if in_testing != None :
            testing_loss = m.train( in_testing, out_testing, dry_run=True )
        training_loss = m.train( in_training[p], out_training[p] )
        loss_history.append((training_loss, testing_loss))
        if in_testing != None :
            print(f'round {i} : training loss {training_loss}, testing loss {testing_loss}')
        else :
            print(f'round {i} : training loss {training_loss}')


    return loss_history



class trainer :
    def __init__( self, m : model ):
        self.m = m
        self.epoch = 0

    def train( self, in_training, out_training, n = None ) :
        N = len(in_training)
        if n == None : n = N
        p = torch.randperm(in_training.shape[0])
        for i in range(0,N,n) :
            training_loss = self.m.train( in_training[p[i:i+n]], out_training[p[i:i+n]] )

        self.m.writer.add_scalar( 'train/loss', training_loss, self.epoch )
        self.epoch += 1


    def test( self, in_testing, out_testing ) :
        testing_loss = self.m.train( in_testing, out_testing, dry_run=True )
        self.m.writer.add_scalar( 'test/loss', testing_loss, self.epoch )


