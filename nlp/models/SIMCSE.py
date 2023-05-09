import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaForMaskedLM
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertForMaskedLM
from transformers.activations import gelu_fast
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings
)

from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.model):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, ** kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class Similarity(nn.Module):
    
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim = -1)
    
    def forward(self, x, y ):
        return self.cos(x,y)/ self.temp
    
class Pooler(nn.Module):
    
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], f"unrecognized pooling type {self.pooler_type}"

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_bofore_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1)/ attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type =='avg_first_last':
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_output = ((first_hidden+ last_hidden)/2.0 * attention_mask.unsqueeze(-1)).sum(1)/ attention_mask.sum(-1).unsqueeze(-1)
            return pooled_output
        elif self.pooler_type =='avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_output = ((second_last_hidden+ last_hidden)/2.0*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(-1).unsqueeze(-1)
            return pooled_output
        else:
            return NotImplementedError
        

    def cls_init(cls, config):
        cls_pooler_type = cls.model_args.pooler_type
        cls.pooler = Pooler(cls.model_args.pooler_type)

        if cls.model_args.pooler_type =='cls':
            cls.mlp = MLPLayer(config)
        cls.sim = Similarity(tmp = cls.model_args.temp)
        cls.init_weight()

    def cls_forward(cls,
                    encoder,
                    input_ids =None,
                    attention_mask = None, 
                    token_type_ids = None, 
                    position_ids = None,
                    head_mask = None, 
                    inputs_embeds = None,
                    labels = None,
                    output_attentions = None,
                    output_hidden_states = None, 
                    return_dict = None,
                    mlp_input_ids = None, 
                    mlm_labels = None):
        


