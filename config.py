
from dataclasses import dataclass

@dataclass
class config:
    modulus : int = 7

    LR : float = .001
    WEIGHT_DECAY : float = 1.

    d_model : int = 128
    d_head : int = 32
    num_heads : int = 4
    d_mlp : int = 512
    d_context : int = 3

    logdir : str = 'runs'

    @property
    def d_vocab( self ) -> int :
        return self.modulus + 1





    

default_config = config()
