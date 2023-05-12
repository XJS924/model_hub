import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random

from datasets import load_dataset

import transformers

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer, 
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerbase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.fuile_utils import cached_property , torch_required, is_torch_available, is_torch_tpu_available
from models.SIMCSE import RobertaForCL, BertForCL
from trainers.simcse_trainer import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:

    model_name_or_path : Optional[str] =field(
        default=None,
        metadata={
            "help":"The model checkpoint for weights initialization\
            Don't set if you want to train a model from scratch"
        }
    )

    model_type : Optional[str]=field(
        default=None,
        metadata={'help': 'if training from scratch., pass a model type from the list: '+ ','.join(MODEL_TYPES)}
    )

    config_name: Optional[str] = field(
        default=None,
        metadata={"help":"Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name:Optional[str]=field(
        default= None,
        metadata={"help":"Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir:Optional[str] = field(
        default=None,
        metadata={"help":"where yuou want to store the pretrained models download from huggingface.co"}
    )

    use_fast_tokenizer: bool = field(
        default=None,
        metadata={"help":'whether to use one of the fast tokenizer (backed by the tokenizers library ) or not '}
    )

    model_revision: str= field(
        default=None,
        metadata={"help": 'The specfic model version to use (can be a branch name tag name or commit id).'}
    )

    use_auth_token: bool = field(
        default= False,
         metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # simcse arguments
    temp: float= field(
        default= 0.05,
        metafault={
            'help':'Temperature for softmax'
        }
    )

    pooler_type: str = field(
        default="cls",
        metadata={"help": 'what kind of pooler to use (cls, csl_before_pooler, avg, avg_top2, avg_first_last)'}
    )

    hard_negative_weight: float = field(
        default=0,
        metadata={
            'help':'the  logit of weight for head negatives (only effective if hard negatives are used)'
        }
    )

    do_mlm: bool = field(
        default = False,
        metadata={
            'help':'whether to use mlm auxiliary objective '
        }
    )

    mlm_weight: float=field(
        default=0.1,
        metadata=(
            'help':'weight for mlm auxiliary objective only effective if --do_mlm'
        )
    )

    mlp_only_train : bool = (
        default=False,
        metadata={
            'help':'Use MLP only druing training'
        }
    )

@dataclass
class DataTrainingArguments:

    dataset_name:Optional[str] = field(
        default = None, metadata = {'help':"the name of the dataset to use (via the datasets library)"}
    )

    dataset_config_name: Optional[str]=filed(
        default=None, metadata={'help':"The configuratin name of the dataset"}
    )

    overwrite_cache: bool = field(
        default=None, metadata={'help':'Overwrite the cache training and evaluation sets'}
    )

    validation_split_percentage: Optional[int]=field(
        default=5, metadata={"help":'The precentage of the train set used as validation set in case'}
    )

    preprocessing_num_workers : Optional[int]= field(
        default=None , metadata = {'help':"The number of processes to use for the preprocessing"}
    )


    # SimCSE's argument
    train_file: Optional[str]=field(
        default=None, metadata={"help":"the training file "}
    )

    max_seq_length: Optional[int]=field(
        default=32, metadata={'help':'the maximum total input sequence lenght after tokenization'}
    )

    pad_to_max_length: bool=field(
        default=False,metadata={'help':"whether to pad all samples to max_seq_length. if False, will pad the samples\
            dynamicall when batching to the maximum length in the batch"}
    )

    mlm_probability: float = field(
        default=0.15,
        metadata={'help':"Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post__init__(self):
        if self.dataset_naem is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training / validation file")
        else:
            if self.train_file is not None:
                extension = self.train_file.split('.')[-1]
                assert extension in ['csv' ,'json', 'txt'] , "`trian_file` should be a csv, a json or a txt file."

@dataclass
class OurTrainingAugments(TrainingAugments):

    eval_transfer:  bool =field(
        default=None, metadata={'help':"Evaluation transfer taks dev sets (in validation)"}
    )


    @cache_property
    @torch_required
    def _setup_devices(self) -> 'torch.device':
        logger.info("Pytorch: setting up devices")

        if self.no_cuda:
            device = torch.device('cpu')
            self._n_gpu  = 0

        elif is_torch_tpu_available():
            import torch_xla.core_xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            self._n_gpu = torch.cuda.device_count()

        else:
            # deepspeed
            device = torch.device("cuda", self.lock_rank)
            self._n_gpu =1 
        
        if device.type =='cuda':
            torch.cuda.set_device(device)
        return device

    
def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv)==2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args= parser.parse_json_file(json_file = os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args= parser.parse_args_into_dataclasses()
    
    if (
        oa.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_trian
        and not training_args.overwrite_output_dir
    ):
        raise ValueError( f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # log on each process the small summary
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.loggings.enable_explicit_format()
    logger.info(f'Training/evaluation parameters {training_args}')

    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files['train'] = data_argas.train_file
    extension = data_args.train_file.split('.')[-1]
    if extension == 'txt':
        extension = 'text'
    if extension =='csv':
        datasets = load_dataset(extension, data_files =data_files, cache_dir = './data/',
                            delimiter='\t' if 'tsv' in data_args.train_file else ',')
    else:
        datasets = load_dataset(extension, data_files =data_files, cache_dir = './data/')
    

    config_kwargs = {
        "cache_dir":model_args.cache_dir,
        "revision": model_args.model_revision,
        'use_auth_token': True if model_args.use_auth_token else None
    }
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance fro mscratch")

    tokenizer_kwargs = {
        "cache_dir":model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        'use_auth_token': True if model_args.use_auth_token else None
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer= AutoTokenizer.from_pretrained(model_args.model_name_or_path,**tokenizer_kwargs)
    else:
        raise ValueError( "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name.")
    
    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf = bool('.ckpt' in model_args.model_name_or_path),
                config= config,
                cache_dir = cache_dir,
                revision = model_args.model_revision,
                use_auth_token = True if model_args.use_auth_token else None,
                model_args = model_args
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf = bool('.ckpt' in model_args.model_name_or_path),
                config= config,
                cache_dir = cache_dir,
                revision = model_args.model_revision,
                use_auth_token = True if model_args.use_auth_token else None,
                model_args = model_args
            )

            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError
        logger.info('Training new model from scratch')
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    column_names = datasets['train'].columns_names
    sent2_cname = None

    if len(column_names) ==2:
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names)==3:
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        total= len(examples[sent0_cname])

        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = ' '
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = ' '
        sentences = examples[sent0_cname] + examples[sent1_cname]

        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = ' '
            sentences += examples[sent2_cname]
        
        sent_features = tokenizer(
            sentences,
            max_length = data_args.max_seq_length,
            truncation = True,
            padding = 'max_length' if data_args.pad_to_max_length else False
        )

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
        
        return features

    if training_args.do_train:
        train_dataset = datasets['train'].map(
            prepare_features,
            batch = True,
            num_proc  = data_args.preprocessing_num_workers,
            remove_columns = columns_name,
            load_from_cache_file = not data_args.overwrite_cache,
        )

    
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of : Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]])-> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return

            flat_features = []

            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: features[k][i] for k in special_keys else feature[k] for k in feature})

            
            batch = self.tokenizer.pad(
                flat_features,
                paddding = slef.padding,
                max_length = self.max_length,
                pad_to_multiple_of = self.pad_to_multiple_of,
                return_tensors = "pt",
            )

            if model_args.do_mlm:
                batch['mlm_input_ids'], batch['mlm_labels'] = self.mask_tokens(batch['input_ids'])

            batch = {k:batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs ,num_sent, -1)[:, 0] for k in batch}

            if 'label' in batch:
                batch['labels'] = batch['label']
                del batch['label']
            if 'label_ids' in batch:
                batch['labels'] = batch['label_ids']
                del batch['label_ids']
            
            return batch
        
        def mask_tokens(self, inputs: torch.Tensor, special_token_mask:Optional[torch.Tensor] =None) -> Tupel[torch.Tensor, torch.Tensor]:
            inputs= inputs.clone()
            labels = labels.clone()
            
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_token_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens = True) for val in labels in tolist()
                ]

                special_tokens_mask = torch.tensor(special_tokens_mask, dtype = torch.bool)

            else:
                special_tokens_mask = special_tokens_mask.bool()
            
            probability_matrix.masked_fill_(special_tokens_mask).bool()
            labels[~masked_indices] = -100

            indices_replaced = torch.bernoulli(otrch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype = torch.long)
            inputs[indices_random] = random_words[indices_rondom]

            return inputs, labels
        
    data_collator = default_data_collator  if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    trainer  = CLTrainer(
        model =model, 
        args  = training_args,
        train_dataset = train_dataset if training_args.do_train else None,
        tokenizer = tokenizer,
        data_collator = data_collator,
    )

    trainer.model_args = model_args

    #Training 
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_naem_or_path))
            else None
        )
        train_result = trainer.train(model_path = model_path)
        trainer.save_model()

        output_train_file = os.path.join(trianing_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, 'w') as writer:
                logger.info("******* Train results ****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f'{key} = {value}\n')

            trainer.state.save_to_json(os.path.join(trianing_args.output_dir, "trainer_state.json"))

    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_trainsfer=True)

        ouput_eval_file = os.path.join(trianing_args.output_dir, "eval_results.txt")

        if trainer.is_world_process_zero():
            with open(output_eval_file, 'w') as writer:
                logger.info("*** Eval results ***")    
                for key ,value in sorted(results.items()):
                    logger.info(f" {key} = {value}")
                    writer.write(f"{key} = {value}\n")
    return results

def _mp_fn(index):
    main()


if __name__ == '__main__':
    main()


        


