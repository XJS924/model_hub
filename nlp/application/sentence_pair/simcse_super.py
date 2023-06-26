import json
import numpy as np
import  torch 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import scipy.stats 
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using {device} device ')

model_path = ''
save_path = ""

tokenizer = BertTokenizer.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path)

output_way = 'pooler'
batch_size =128
learning_rate = 2e-5
max_len = 64

data_path = ""
dev_file = ''
test_file = ''

def load_data(path):
    data = []
    with open(path,'r' ,encoding='utf8') as f:
        for i in f:
            d = i.split('||')
            sent1 = d[1]
            sent2 = d[2]
            score = d[3]
            data.append([sent1, sent2, score])
        return data

dev_data = load_data(os.path.join(data_path, dev_file))    
test_data = load_data(os.path.join(data_path, test_file))

class TrainDataset(Dataset):
    def __init__(self, data ,tokenizer, max_len , transform = None, target_transform = None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        origin = source['origin']
        entailment = source['entailment']
        contradiction =source['contradiction']
        sample =self.tokenizer([origin,entailment,contradiction], max_length = self.max_len, truncation=True, padding='max_length',return_tensor='pt')
        return sample
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])
    
class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
    
    def text_to_id(self, source):
        sample =self.tokenizer(source, max_length = self.max_len, truncation=True, padding='max_length',return_tensor='pt')
        return sample
    
    def get_data(self):
        return self.target_idxs,self.source_idxs, self.label_list

class NerualNetwork(nn.Module):
    def __init__(self, model_path, output_way):
        super(NerualNetwork,self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.output_way = output_way
        assert output_way in ['cls', 'pooler']
    
    def forawrd(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask= attention_mask, token_type_ids=token_type_ids)
        if self.output_way =='cls':
            output = x1.last_hidden_state[:,0]
        elif self.output_way=='pooler':
            output = x1.pooler_output
        return  output

model = NerualNetwork(model_path, output_way).to(device)
optimizer= torch.optim.AdamW(model.parameters(),lr = learning_rate)

training_data = TrainDataset(s, tokenizer, max_len)
train_dataloader = DataLoader(training_data, batch_size=batch_size)

testing_data = TestDataset(test_data,tokenizer,max_len)
deving_data = TestDataset(dev_data,tokenizer,max_len)

def compute_corrcoef(x, y ):
    return scipy.stats.spearmanr(x,y).correlation

def compute_loss(y_pred, lamda = 0.05):
    row = torch.arange(0, y_pred.shape[0], 3, device='cuda')
    col = torch.arange(y_pred.shape[0],device='cuda')
    col = torch.where(col % 3 !=0)[0].cuda()
    y_true = torch.arange(0,len(col),2, device='cuda')
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0),dim=2)
    similarities = torch.index_select(similarities, 0 ,row)
    similarities = torch.index_select(similarities,1, col)
    loss = F.cross_entropy(similarities,y_true)
    return torch.mean(loss)

def test(test_data,model):
    target_idxs, source_idxs, label_list = test_data.get_data()
    with torch.no_grad():
        target_pred = model(**target_idxs)

        source_pred= model(**source_idxs)

        similarities_list = F.cosine_similarity(target_pred, source_pred)
        similarities_list = similarities_list.cpu().numpy()
        label_list = np.array(label_list)
        corrcoef = compute_corrcoef(label_list,similarities_list)
    return corrcoef

def train(dataloader, testdata, model, optimizer):
    model.train()
    size = len(dataloader.dataset)
    max_corrcoef = 0
    not_up_batch = 0
    for batch, data in enumerate(dataloader):
        input_ids = data['input_ids'].view(len(data['input_ids'])*3,-1).to(device)
        attention_mask = data['attention_mask'].view(len(data['attention_mask'])*3, -1).to(device)
        token_type_ids = data['token_type_ids'].view(len(data['token_type_ids'])*3,-1).to(device)
        pred = model(input_ids, attention_mask, token_type_ids )
        loss= compute_loss(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%10 ==0:
            loss ,current = loss.item(), batch*int(len(input_ids)/3)
            print(f"loss:{loss:>7f}, [{current:>5d}/ {size:>5d}]")
            model.eval()
            corrcoef = test(test_data, model)
            model.train()
            print(f"corrcoef:{corrcoef:.4f}")

            if corrcoef> max_corrcoef:
                max_corrcoef = corrcoef
                not_up_batch = 0
                torch.save(model.state_dict(),save_path)

                print(f"higher corrcoef :{(max_corrcoef):.4f} saved PyTorch Model stat to model.pth")
            else:
                not_up_batch += 1
                if not_up_batch > 10:
                    print(f"Corrcoef didn't up for %s batch, early stop!" % not_up_batch)
                    break

if __name__=='__main__':
    epochs= 1
    for t in range(epochs):
        print(f"Epoch {t+1 }\n ------------------")
        train(train_dataloader, test_data, model, optimizer)
    print("Train Done")
    model.load_state_dict(torch.load(save_path))
    corrcoef = test(dev_data, model)
    print(f"dev corrcoef {corrcoef:.4f}")


