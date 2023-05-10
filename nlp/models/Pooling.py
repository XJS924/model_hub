import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json


class Pooling(nn.Module):

    def _init__(self,
                word_embedding_dimension: int,
                pooling_mode : str = None,
                pooling_mode_cls_token: bool = False,
                pooling_mode_max_tokens: bool = False,
                pooling_mode_mean_tokens: bool = False,
                pooling_mode_mean_sqrt_len_tokens : bool= False,
                pooling_mode_weightedmean_tokens:bool= False,
                pooling_mode_lasttoken: bool = False):
        super(Pooling, self).__init__()

        self.config_keys =['word_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
                           'pooling_mode_max_tokens','pooling_mode_mean_sqrt_len_tokens',
                           'pooling_mode_weightedmean_tokens','pooling_mode_lasttoklen']

        if pooling_mode is not None:
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'cls', 'weightedmean', 'lasttoken'] 
            pooling_mode_cls_token = (pooling_mode =='cls')
            pooling_mode_max_tokens =(pooling_mode =='max')
            pooling_mode_mean_tokens - (pooling_mode =='mean')
            pooling_mode_weightedmean_tokens = (pooling_mode=='weightedmean')
            pooling_mode_lasttoken = (pooling_mode_lasttoken =='lasttoken')

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token