import random
from transformers import AutoConfig, BertModel, BertConfig, BertTokenizer, AutoTokenizer,AutoModelForSequenceClassification, Trainer, TrainingArguments,set_seed
from torch.utils.data import Dataset
from transformers.trainer_utils import IntervalStrategy
import torch
from torch import nn

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import time

set_seed(42)
model_path="bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)

class Model(nn.Module):
    def __init__(self, model_path):
        super(Model,self).__init__()
        self.config = BertConfig.from_pretrained(model_path)
        # self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path, config = self.config)
        
    def forward(self, input_ids, attention_mask, pooler_type = 'cls'):
        output = self.model(input_ids, attention_mask, output_hidden_states=True)
        final_encoding = None
        if pooler_type == 'first-last-avg':
            first = output.hidden_states[1]
            last = output.hidden_states[-1]
            seq_length  = first.size(1)
            first_avg = torch.avg_pool1d(first.transpose(1,2),kernel_size=seq_length).sequeeze(-1)
            last_avg = torch.avg_pool1d(last.transpose(1,2),kernel_size=seq_length).sequeeze(-1)
            final_encoding = torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1),last_avg.unsqueeze(1)],dim=1).transpose(1,2),kernal_size=2).squeeze(-1)
        elif pooler_type == 'last-avg':
            sequence_output = output.last_hidden_state
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1,2),kernal_size=seq_length).sequeeze(-1)
        elif pooler_type == 'cls':
            sequence_output = output.last_hidden_state
            final_encoding = sequence_output[:, 0 ]
        elif pooler_type == 'pooler':
            final_encoding = output.pooler_output
        return final_encoding
    
def load_data(path):
    with open(path, 'r', encoding='UTF-8') as f:
        data=[]
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            data.append((content,int(label)))
    random.shuffle(data)
    return data

class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.texts = [line[0] for line in data]
        self.labels = [line[1] for line in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        return [text, label]

def collate_fn(features):
    text=[line[0] for line in features]
    label=[line[1] for line in features]
    tokenized = tokenizer(text,padding=True,return_tensors='pt')
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    labels = torch.tensor(label)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_data=load_data("./data/train.txt")
dev_data=load_data("./data/dev.txt")
test_data=load_data("./data/test.txt")

train = MyDataset(train_data)
valid = MyDataset(dev_data)
test = MyDataset(test_data)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds.flatten(), average='macro')
    return {'accuracy': round((labels == preds).mean(), 6),
            'f1': round(f1, 6),
            'precision': round(precision, 6),
            'recall': round(recall, 6),
            "our":1}

training_args=TrainingArguments(output_dir='./weights',
                                    do_train=True,
                                    do_eval=True,
                                    do_predict=True,
                                    per_device_train_batch_size=64,
                                    per_device_eval_batch_size=64,
                                    eval_steps=400,
                                    logging_steps=100,
                                    report_to=["wandb"],
                                    save_steps=400,
                                    metric_for_best_model='f1',
                                    logging_strategy=IntervalStrategy.STEPS,
                                    evaluation_strategy=IntervalStrategy.STEPS,
                                    save_strategy=IntervalStrategy.STEPS,
                                    save_total_limit=1
                               )

model =  Model(model_path)
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train,
                  eval_dataset=valid,
                  data_collator=collate_fn,
                  compute_metrics=compute_metrics
                  )

start = time.time()
trainer.train()
model.save_pretrained("./best/")
end = time.time()
print("训练耗时：{}分钟".format((end - start) / 60))

res=trainer.predict(test)
print(res.metrics)
print("finish")