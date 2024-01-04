import pickle as pickle
import os
import pandas as pd
import torch
from ast import literal_eval

from torch.utils.data import Dataset 
from transformers import AutoTokenizer 
from utils import label_to_num

class ReDataset(Dataset):
    def __init__(self, args, types='train'):
        if types=='train':
            self.datasets = self.preprocessing(self.load_data(args.train_path))
            self.labels = label_to_num(self.load_data(args.train_path)['label'])
        elif types=='dev':
            self.datasets = self.preprocessing(self.load_data(args.dev_path))
            self.labels = label_to_num(self.load_data(args.dev_path)['label'])
        else:
            self.datasets = self.preprocessing(self.load_data(args.test_path))
            self.labels = label_to_num(self.load_data(args.test_path)['label'])
            
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.max_length = args.max_length

        self.datasets = self.tokenizing(self.datasets)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.datasets.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def df2list(self, data):
        return data.values.tolist()

    def preprocessing(self, datasets):
        target_col = ['sentence', 'subject_entity', 'object_entity', 'label']
        for column in ['subject_entity', 'object_entity']:
            datasets.loc[:, column] = datasets.loc[:, column].apply(lambda x: literal_eval(x)['word'])
        return datasets.loc[:, target_col].copy()
    
    def load_data(self, path):
        return pd.read_csv(path)

    def tokenizing(self, datasets):
        concat_entity = []
        for idx, rows in datasets.iterrows():
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
