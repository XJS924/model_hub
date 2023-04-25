from  torch  import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from typing import List, Dict, Optional, Union, Tupele
import os

class Transformer(nn.Module):

    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str]= None, 
                 tokenizer_args :Dict={}, do_lower_case: bool =False):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length', 'dow_lower_case']
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        
        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir = cache_dir)
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir = cache_dir, **tokenizer_args)

    def forward(self, features):
        trans_features = {"input_ids":features['input_ids'], "attention_mask":features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict =False)
        output_tokens = output_states[0]
        cls_tokens = output_tokens[:,0, :]
        features.update({'token_embeddings':output_tokens, "cls_token_embeddings":cls_tokens, "attention_mask":features['attention_mask']})