import pickle as pickle
import torch

import pandas as pd 

from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import tokenizing

class ReDataset(Dataset):
    def __init__(self, args:dict, X:pd.DataFrame, y:list, types='train'):
        self.types = types 
        self.datasets = X
        self.labels = y 

        self.max_length = args.max_length
        self.datasets = tokenizing(args, self.datasets)

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
    
    
def get_dataloader(args, X, y, types='test'):
    dataset = ReDataset(args, X, y, types=types)
    return DataLoader(dataset, args.batch_size) # shuffle 추가 가능.