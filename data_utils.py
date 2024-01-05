import pickle as pickle
import torch
from tqdm.auto import tqdm 

from torch.utils.data import Dataset 

class ReDataset(Dataset):
    def __init__(self, args, X, y, types='train'):
        self.types = types 
        self.tokenizer = args.tokenizer 
        self.datasets = X
        self.labels = y 

        self.max_length = args.max_length
        self.datasets = self.tokenizing(self.datasets)

    def __getitem__(self, idx):
        if self.types in ['train', 'dev']:
            item = {key: val[idx].clone().detach() for key, val in self.datasets.items()}
            item['label'] = torch.tensor(self.labels[idx])
            return item
        else:
            item = {key: val[idx].clone().detach() for key, val in self.datasets.items()}
            return item, torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)

    def df2list(self, data):
        return data.values.tolist()
    
    def tokenizing(self, datasets):
        concat_entity = []
        for idx, rows in tqdm(datasets.iterrows(), total=datasets.shape[0], desc=f'{self.types.title()}set Tokenizing...'):
            temp = rows['subject_entity'] + '[SEP]' + rows['object_entity'] + '[SEP]' + rows['sentence']
            concat_entity.append(temp)
            
        tokenized_sentences = self.tokenizer(
            concat_entity, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            add_special_tokens=True
        )
        return tokenized_sentences
