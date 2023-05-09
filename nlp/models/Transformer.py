from  torch  import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
import json
from typing import List, Dict, Optional, Union, Tupele
import os

class Transformer(nn.Module):

    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str]= None, 
                 tokenizer_args :Dict={}, do_lower_case: bool =False,
                 tokenizer_name_or_path: str = None):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length', 'dow_lower_case']
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        
        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir = cache_dir)

        self._load_model(model_name_or_path, config, cache_dir, **model_args)
        # self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir = cache_dir, **tokenizer_args)

        if max_seq_length is None:
            if hasattr(self.auto_model, 'config') and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, cache_dir, **model_args)

    
    def _load_t5_modle(self, model_name_or_path, config, cache_dir, **model_args):
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, config = config, cache_dir= cache_dir, **model_args)

    def _load_mt5_mdoel(self, model_name_or_path, config, cache_dir, **model_args):
        from transformers import MT5EncoderModel
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ['decoder.*']
        self.auto_model = MT5EncoderModel.from_pretrained(model_name_or_path, config =config,  cache_dir =cache_dir, **model_args)

    def __repr__(self):
        return f"Transformer({self.get_config_dict()} with Transformer model:{self.auto_model.__class__.__name__})"

    def forward(self, features):
        trans_features = {"input_ids":features['input_ids'], "attention_mask":features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict =False)
        output_tokens = output_states[0]
        cls_tokens = output_tokens[:,0, :]
        features.update({'token_embeddings':output_tokens, "cls_token_embeddings":cls_tokens, "attention_mask":features['attention_mask']})