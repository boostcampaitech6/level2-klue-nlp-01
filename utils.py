import os, types, pickle 
import pandas as pd 

import argparse
from ast import literal_eval

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import seaborn as sns 

from settings import * 
from preprocess.preprocess_marker import *


def load_data(path):
    return pd.read_csv(path)

def preprocessing(datasets):
    target_col = ['sentence', 'subject_entity', 'object_entity', 'label']
    for column in ['subject_entity', 'object_entity']:
        datasets.loc[:, column] = datasets.loc[:, column].apply(lambda x: literal_eval(x)['word'])
    return datasets.loc[:, target_col].copy()

def preprocessing_MARKER(df: pd.DataFrame, method:str='none') -> pd.DataFrame:
    """Generate dataframe after Marker preprocessing

        Args:
        df (pd.DataFrame): raw dataset
        method (str): em, tem, temp     
            em: Entity Mask
            ex) Bill was born in Seattle-> [SUBJ-PERSON] was born in [OBJ-CITY]

            tem: Typed Entity Marker
            ex) Bill was born in Seattle -> 
            <S:PERSON>Bill</S:PERSON> was born in <O:CITY>Seattle</O:CITY>

            temp: Typed entity marker (punct)
            ex) Bill was born in Seattle -> 
            @*person*Bill@ was born in #^Seattle^#

    Returns:
        pd.DataFrame: dataframe with MARKER
    """
    df_mark = df.copy()

    for idx in range(len(df)):
        # Data Sentence, Subject_entity, Object_entity 추출
        row = df.iloc[idx,:]
        sen, sbj, obj = row['sentence'], literal_eval(row['subject_entity']), literal_eval(row['object_entity'])

        sbj_word, obj_word = get_marker_tag(sen, sbj, obj, method)

        sen = replace_string_index(sen, sbj, obj, sbj_word, obj_word)
        df_mark.iloc[idx,1] = sen

    return df_mark

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
        try:
            num_label.append(dict_label_to_num[v])
        except:
            num_label.append(10)
  
    return num_label

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open(os.path.join(DATA_DIR,'dict_num_to_label.pkl'), 'rb') as f:
    dict_num_to_label = pickle.load(f)
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
    
    args = parser.parse_args()
    
    dataset = pd.read_csv(os.path.join(TRAIN_DIR, 'train.csv'))
    train, valid = train_valid_split(dataset, test_size=0.2, random_state=args.seed, version='v.0.0.2')
