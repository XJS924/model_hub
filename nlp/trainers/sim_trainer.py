import collections 
import inspect
import math
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_computer_objective,
    default_hp_space,
    set_seed,
    speed_metrics
)

from transformer.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    os_torch_tpu_available
)

from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowcallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from transformers.trainer_pt_utils import (
    reissus_pt_warnings,
)

from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.paraller_loader as pl

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.aprse('1.6'):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_dataset_available():
    import datasets

from transformers.trainer import _model_unwrap
from transformers.optimization import Adafctor, Adamw, get_scheduler
import copy

PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

sys.path.inset(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock

logger = logging.get_logger(__name__)

class CLTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None, 
        ignore_keys: Optional[List[str]] =None,
        metric_key_prefix : str = 'eval',
        eval_senteval_transfer: bool = False,) -> Dict[str, float]:

        def prepare(params, samples):
            return 
        
        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences, 
                return_tensors = 'pt',
                padding = True,
            )

            for k in batch:
                batch[k] = batch[k].to(self.args.device)

            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states = True, return_dict = True, sent_emb = True)
                pooler_output= oputputs.pooler_output
            return pooler_output.cpu()

            # Set params for sentEval (fastmode)
            params = {"task_path": PATH_TO_DATA, "usepytorch": True, 'kfold':5}
            params['classifier'] ={"nhid": 0, 'optim':'rmsprop', 'batch_size':128, 'tenacity':3, 'epoch_size':2}
            se = senteval.engine.SE(params, batcher, prepare)
            tasks = ['STSBenchmark', "SICKRelatedness"]
            if eval_senteval_transfer or self.args.eval_transfer:
                tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'MRPC']
            self.mode.eval()
            results= se.eval(tasks)
            
            stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
            sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

            metrics = {"eval_stsb_spearman":stsb_spearman, 'eval_sickr_spearman':sickr_spearman,'eval_avg_sts':(stsb_spearman + sickr_spearman) /2}
            if eval_senteval_transfer or self.args.eval_transfer:
                avg_transfer = 0
                for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', "MRPC"]:
                    avg_transfer+= results[task]['devacc']
                    metrics[f'eval_{task}'] = results[task]['devacc']
                avg_transfer /=7
                metrics['eval_avg_transfer'] = avg_transfer

            self.log(metrics)
            return metrics

        def _save_checkpoint(self, model, trial, metrics= None):
            assert _model_unwrap(model) is self.model, 'internal model should be a reference to self.model'

            ## Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metrics_for_best_model is not None:
                metrics_to_check = self.args.metrics_for_best_mdoel
                if not metric_to_check.startswith('eval_'):
                    metric_to_chekc = = f'eval_{metrics_to_check}'
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.argas.grerater_is_better else np.less

                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state_best_matric)
                ):
                output_dir = self.args.output_dir
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                # Only save model when it is the best one 
                self.save_model(output_dir)
                if self.deepspeed:
                    self.deepspeed.save_checkpoint(output_dir)

                # Save optimizer and scheduler 
                if self.sharded_dpp:
                    self.optimizer.consolidate_state_dict()
                
                if is_torch_tpu_available():
                    xm.rendezvous('saveing_optimizer_states')
                    xm.save(self.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))

                    with warnings.catch_warning(reocrd=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                    reissue_pt_warnings(caught_warnings)
                
                # Saver the trainer state
                if self.is_world_process_zero():
                    self.state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))
                
            else:

                # Saver model checkpoint
                checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'

                if self.hp_search_backend is not None and trial is not None:
                    if self.hp_search_backend == HPSearchBackend.OPTUNA:
                        run_id = trial.number
                    else:
                        from ray import tune

                        run_id = tune.get_trial_id()
                    run_name = self.hp_name(trial) if self.hp_name is not None else f'run-{run_id}'
                    output_dir = os.path.join(self.args.output_dir, run_anem, checkpoint_folder)
                else:
                    ooutput_dir = os.path.join()

                




