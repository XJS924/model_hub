import logging 
from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor , device
import transformers 
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosin_similarity
from sklearn.preprocess import normalize
from typing import List, Dict , Tuple, Type, union

logging.basicConfig(format='%(asctime)s- %(levelname)s - %(name)s  - %(messsage)s', datefmt= '%m/%d/%Y %H:%M:%S',level = logging.INFO)

logger = logging.getLogger(__name__)

class SimCSE(object):

    def __init__(self, model_name_or_path:str,
                device: str = None, 
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_naem_or_path)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.index= None
        self.is_faiss_index = False
        self.num_cells =num_cells
        self.num_cells_in_search = num_cells_in_search

        if pooler is not None:
            pass

