'''
Author: DongEon, Kim
'''
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch, argparse, hanja 
import types, pickle, yaml

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import seaborn as sns 

from settings import * 

def load_data(path):
    return pd.read_csv(path)

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

def save_pkl(file, fname):
    with open(f'{fname}.pkl', mode='wb') as f:
        pickle.dump(file, f)
    print(f'Success Saving File! PATH: {fname}')

def load_pkl(path):
    with open(path, mode='rb') as f:
        file = pickle.load(f)
    return file 

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def version_check(names=None):
    if not names:
        raise TypeError('please input package name! for example, numpy, pandas, matplotlib, etc.')

    if isinstance(names, types.ModuleType):
        exec('print(f"{names.__name__} version is {names.__version__}")')
        
    elif isinstance(names, list) and all(isinstance(name, types.ModuleType ) for name in names):
        for name in names:
            exec('print(f"{name.__name__} version is {name.__version__}")')
    else:
        raise ValueError

def hanja_to_hangul(sentence):
    return hanja.translate(sentence, 'substitution')


def get_unk_tokens(entity, tokenizer, unk_token_ids=3, verbose=True):
    unk_list = []
    token_ids = tokenizer.encode(entity)
    decode_entity = tokenizer.decode(token_ids)[5:-5]
    if unk_token_ids in token_ids:
        for src, dst in zip(entity.split(), decode_entity.split()):
            if verbose:
                if src != dst and dst == '[UNK]':
                    print(f'Entity: {entity}')
                    print(f'[UNK] Tokens: {decode_entity}')
                    print(f'{dst} = {src}\n')
                    unk_list.append(src)
    return unk_list 

def build_unk_tokens(dataset, tokenizer, unk_token_ids=3, verbose=True):
    unk_list = []
    for _, rows in dataset.iterrows():
        sbj, obj = rows['subject_entity']['word'], rows['object_entity']['word']
        sbj_unk = get_unk_tokens(sbj, tokenizer, unk_token_ids, verbose)
        obj_unk = get_unk_tokens(obj, tokenizer, unk_token_ids, verbose)
        
        if sbj_unk != []:
            unk_list.extend(sbj_unk)
        
        if obj_unk != []:
            unk_list.extend(obj_unk)
    return unk_list 


def label_to_num(label):
    num_label = []
    path = os.path.join(DATA_DIR, 'dict_label_to_num.pkl')
    dict_label_to_num = load_pkl(path)
    for v in label:
        num_label.append(dict_label_to_num[v])
    return num_label


def num_to_label(label):
  origin_label = []
  path = os.path.join(DATA_DIR,'dict_num_to_label.pkl')
  dict_num_to_label = load_pkl(path)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def train_valid_split(dataset, test_size=0.2, random_state=0, version='v.0.0.2'):
    '''
    train valid split 
    Input:
        dataset: pd.DataFrame
        test_size: float 
        random_state: int
        version: e.g.) v.0.0.2
    Return:
        train: pd.DataFrame 
        valid: pd.DataFrame
    '''
    train, valid = train_test_split(dataset, test_size=test_size, random_state=random_state, stratify=dataset['label'])
    
    print(f'dataset length is {dataset.__len__():,}')
    print(f'train dataset length is {train.__len__():,}')
    print(f'valid dataset length is {valid.__len__():,}')
    
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    
    train.to_csv(os.path.join(DATA_DIR, f'train-{version}.csv'), index=False)
    valid.to_csv(os.path.join(DATA_DIR, f'dev-{version}.csv'), index=False)
    print(f'\nSucess Save Dataset [train, valid]')
    
    return train, valid 


def plot_hist(li: list, bins=50, title=None, xlabel=None, ylabel=None, f_name='figure.png', save=False) -> None:
    '''
    plot histogram with kde 
    Input: 
        li: list
        bins: int 
        title: str 
        xlabel: str 
        ylabel: str 
        f_name: str 
        save: bool
    '''
    sns.histplot(li, bins=bins, kde=True)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    if save:
        plt.savefig(os.path.join(FIG_DIR, f'{f_name}'), dpi=200)
    plt.show()

def tapt_apply(pretrained_dict, model):
    model_dict = model.state_dict() # 현재 신경망 상태 로드
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    """
    #HEAD만(나머지 freeze)
    for name, param in model.named_parameters():
        if name.split('.')[0] == 'classifier':
            pass
        else :
            param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    """
    return model



# data split
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', default=0, type=int
    )
    parser.add_argument(
        '--test_size', default=0.2, type=float
    )
    parser.add_argument(
        '--version', '-V', default='v.0.0.2',type=str 
    )
    parser.add_argument(
        '--train_path', default='train/train.csv', type=str
    )
    
    args = parser.parse_args()
    
    dataset = pd.read_csv(os.path.join(DATA_DIR, args.train_path))
    train, valid = train_valid_split(dataset, test_size=args.test_size, random_state=args.seed, version=args.version)