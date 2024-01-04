import os, types, pickle
import pandas as pd 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import seaborn as sns 

from settings import * 

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

def label_to_num(label):
    num_label = []
    with open(os.path.join(DATA_DIR, 'dict_label_to_num.pkl'), 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
  
    return num_label

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
    
    train.to_csv(os.path.join(TRAIN_DIR, f'train-{version}.csv'), index=False)
    valid.to_csv(os.path.join(DEV_DIR, f'dev-{version}.csv'), index=False)
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
