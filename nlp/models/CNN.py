import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import logging
import gzip
from tqdm import tqdm
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)


class CNN(nn.module):

    def __init__(self, in_word_embedding_dimension: int, 
                 out_channels: int = 256 , 
                 kernel_sizes: List[int] = [1, 3, 5]):
        nn.Module.__init_(self)
        self.config_keys = ['in_word_embedding_dimension', 'out_channels', 'kernel_sizes']
        self.in_word_embedding_dimension = in_word_embedding_dimension
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.embeddings_dimension = out_channels*len(kernel_sizes)
        self.convs = nn.ModuleList()

        in_channels = in_word_embedding_dimension
        for kernel_size in kernel_sizes:
            padding_size = int((kernel_size-1)/2)
            conv = nn.Conv1d(in_channels=in_channels, out_channels= out_channels, kernel_size=kernel_size,
                             padding=padding_size)
            self.convs.append(conv)

    def forward(self, features):
        token_embeddings = features['token_embeddings']

        token_embeddings = token_embeddings.transpose(1, -1)
        vectors = [conv(token_embeddings) for conv in self.convs]
        out = torch.cat(vectors,1).transpose(1,-1)

        features.update({'token_embeddings': out})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension
    
    def tokenizer(self, text: str) -> List[int]:
        raise NotImplementedError()
    
    def save(self, output_path:str):
        with open(os.path.join(output_path, 'cnn_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent = 2)
        
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self,):
        return {key: self.__dict__[key] for key in self.congig_keys}
    
    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'cnn_config.json'), 'r') as fIn:
            config = json.load(fIn)

            weights= torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location = torch.device('cpu'))
            model = CNN(**config)
            model.load_state_dict(weights)
            return model
            


        

