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
            evaluator,
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
            show_progress_bar: bool = True
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

        skip_scheduler =  False
        for epoch in trange(epochs, desc = 'Epoch', disbale= not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc= ' Iteration', smoothing = 0.05):
                if use_amp:
                    with autocast():
                        model_predictions=  self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.conifg.nium_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scaler()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scaler() != scale_before_step

                else:
                    model_predictions = self.model(**features, return_dict = True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels ==1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.paramters(), max_grad_norm)
                    optimizer.step()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)
                    self.model.zero_grad()
                    self.model.train()
            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        if evaluator is not None:
            score = evaluator(self, output_path= output_path, epoch = epoch ,steps = steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def predict(self, sentences: List[List[str]],
                batch_size : int = 32,
                show_process_bar: bool = None, 
                num_workers : int = 0,
                actiavtion_fct =None, 
                apply_softmax = False, 
                convert_to_numpy: bool = True,
                convert_to_tensor: bool = False):
        input_was_string = False
        if isinstance(sentences[0], str):
            sentences = [sentences]
            input_was_string =True

        input_dataloader  = DataLoader(sentences, batch_size, collate_fn = self.smart_batching_collate_text_only,
                                       num_workers = num_workers, shuffle = False)
        if show_process_bar is None:
            show_process_bar = (logger.getEffectiveLevel() ==logger.INFO or logger.getEffectiveLevel() == logger.DEBUG)

        iterator = input_dataloader
        if show_process_bar:
            iterator = tqdm(input_dataloader, desc= 'Batches')
        
        if actiavtion_fct is None:
            actiavtion_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)

        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict = True)
                logits = actiavtion_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim = 1)
                pred_scores.extent(logits)
        
        if self.config.num_labels ==1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
        
        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores
        
        
    def save(self, path):
        if path is None:
            return 
        logger.info(f"Save model to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        return self.save(path)
            






        









