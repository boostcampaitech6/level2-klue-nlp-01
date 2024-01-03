import os 
import pandas as pd 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import seaborn as sns 

from settings import * 

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
    valid.to_csv(os.path.join(DEV_DIR, f'dev-{version}.csv'))
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
