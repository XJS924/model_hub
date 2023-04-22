import torch
from torch import nn
from typing import List
import os
import json

class LSTM(nn.module):

    def __init__(self, word_embedding_dimension: int ,
                hidden_dim: int,
                num_layersL int = 1, 
                dropout: float = 0,
                bidirectional: bool = True):
        nn.Module.__init__(self)
        self.config_keys =['word_embedding_dimension', 'hidden_dim', 'num_layers', 'dropout', 'bidirectional']
        self.word_embedding_dimension = word_embedding_dimension
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
