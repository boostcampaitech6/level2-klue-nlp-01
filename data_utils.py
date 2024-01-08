import pickle as pickle
import torch
from tqdm.auto import tqdm 

from torch.utils.data import Dataset 
from utils import tokenizing

class ReDataset(Dataset):
    def __init__(self, args, X, y, types='train'):
        self.types = types 
        self.tokenizer = args.tokenizer 
        self.datasets = X
        self.labels = y 

        self.max_length = args.max_length
        self.datasets = tokenizing(self.datasets)

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