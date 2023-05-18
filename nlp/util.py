import requests
from torch import Tensor, device
from typing import List, Callable
from tqdm.autonotebook import tqdm
import sys
import importlib
import os
import torch
import numpy as np
import queue
import logging
from typing import Dict, Optional, Union
from pathlib import Path

import huggingface_hub
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub import HfApi, hf_hub_url, cached_download, HfFolder
import fnmatch
from packaging import version
import heapq

logger = logging.getLogger(__name__)

def pytorch_cos_sim(a:Tensor, b:Tensor):
    return cos_sim(a, b )

def cos_sim(a: Tensor, b : Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.Tensor(a)
    
    if not isinstance(b, torch.Tensor):
        b = torch.Tensor(b)

    if len(a.shape) ==1:
        a = a.unsqueeze(0)
    
    if len(b.shape) ==1:
        b = b.unsequeeze(0)

    a_norm = torch.nn.functional.normalize(a, p= 2, dim =1)
    b_norm = torch.nn.functional.normalize(b, p= 2, dim =1)
    return torch.mm(a_norm, b_norm.transpose(0,1))

def dot_score(a:Tensor ,b :Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.Tensor(a)
    
    if not isinstance(b, torch.Tensor):
        b = torch.Tensor(b)

    if len(a.shape) ==1:
        a = a.unsqueeze(0)
    
    if len(b.shape) ==1:
        b = b.unsequeeze(0)

    return torch.mm(a, b.transpose(0, 1))

def pairwise_dot_score(a:Tensor ,b :Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.Tensor(a)
    
    if not isinstance(b, torch.Tensor):
        b = torch.Tensor(b)

    return (a *b ).sum(dim=-1)

def pairwise_cos_sim(a:Tensor ,b :Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.Tensor(a)
    
    if not isinstance(b, torch.Tensor):
        b = torch.Tensor(b)
    
    return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))

def normalize_embeddings(embeddings:Tensor):
    return torch.nn.functional.normalize(embeddings, p = 2, dim=1)

def paraphrash_mining(model, sentences: List[str],
                      show_progress_bar: bool=False,
                      batch_size:int=32,
                      *args,
                      **kwargs):
    embeddings = model.encode(sentences, show_progress_bar=show_progress_bar, batch_size = batch_size, convert_to_tensor = True)
    return paraphrash_mining_embeddings(embeddings, *args, **kwargs)

def paraphrash_mining_embeddings(embeddings: Tensor, 
                                 query_chunk_size: int =5000,
                                 corpus_chunk_size: int =100000,
                                 max_pairs: int = 500000,
                                 top_k:int =100,
                                 score_function: Callable[[Tensor, Tensor],Tensor]= cos_sim):
    top_k +=1

    pairs = queue.PriorityQueue()
    min_score= -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings),corpus_chunk_size):
        pass
