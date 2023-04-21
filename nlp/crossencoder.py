from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import numpy as np
import transformers
import torch
import os
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm.autonotebook import tqdm, trange
import logging
from typing import Dict, Type, Callable, List

logger = logging.getLogger(__name__)


class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int=None , max_length:int=None, device:str=None , tokenizer_args:Dict={}):
        self.config = AutoConfig.from_pretrained(model_name)
        if num_labels == None:
            num_labels = self.config.num_labels

        self.max_length = max_length
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,config = self.config)
        self.tokenizer= AutoTokenizer.from_pretrained(model_name)
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else "cpu"
            logger.info(f"Use Pytorch device {device} ")
        self._target_device = torch.device(device)
    
    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())
            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors = 'pt',max_length= self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels ==1 else torch.long).to(self._target_device)
        for name in tokenized:
            tokenized[name]=tokenized[name].to(self._target_device)

        return tokenized,labels
    
    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        
        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())
        
        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors = 'pt',max_length= self.max_length)
        
        for name in tokenized:
            tokenized[name]=tokenized[name].to(self._target_device)

        return tokenized
    
    def _get_scheduler(self, optimizer, scheduler: str, 
                     warmup_steps: int,
                     t_total: int):
        scheduler = scheduler.lower()
        if scheduler=='constantlr':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler=='warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler=='warmcosine':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError(f'unknown schedule {scheduler}')

    def fit(self,train_dataloader: DataLoader,
            evaluateor,
            epochs: int=1,
            activation_fct=nn.Identity(),
            scheduler: str="WarmupLinear",
            warmup_steps : int = 10000,
            optimizer_class: Type[Optimizer]=transformers.Adamw,
            optimizer_params: Dict[str,object]={'lr':1e-5},
            weight_decay: float=0.01,
            evaluation_steps: int=0,
            output_path: str=None,
            save_best_model: bool = True,
            max_grad_norm : float = 1,
            use_amp: bool=False,
            callback: Callable[[float, int, int], None]= None,
            ):
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.amp.GradScaler()
        
        self.model.to(self._target_device)
        
        if output_path is not None:
            os.mkdir(output_path, exist_ok=True)
        
        self.best_score= -9999999
        num_train_steps = int(len(train_dataloader)* epochs)
        
        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters)
        no_decay = ['bias', 'LayerNorm.bias','LayerNorm.weight']

        optimizer_grounded_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n ,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer  = optimizer_class(optimizer_grounded_parameters, **optimizer_params)
        if isinstance(schedule, str):
            schedule = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels==1 else nn.CrossEntropyLoss()

        









