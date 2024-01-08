from tqdm.auto import tqdm 
from ast import literal_eval
from utils.utils import load_data, label_to_num

############################### Preprocessing ###############################
def preprocessing(path):
    datasets = load_data(path)
    X_col, y_col = ['sentence', 'subject_entity', 'object_entity'], ['label']
    
    X = datasets.loc[:, X_col].copy()
    y = label_to_num(datasets.loc[:, y_col])
    
    for column in ['subject_entity', 'object_entity']:
        X.loc[:, column] = X.loc[:, column].apply(lambda x: literal_eval(x)['word']) # default.
        
    return X, y

def tokenizing(datasets, tokenizer, max_length):
    concat_entity = []
    for idx, rows in tqdm(datasets.iterrows(), total=datasets.shape[0], desc='Dataset Tokenizing...'):
        temp = rows['subject_entity'] + '[SEP]' + rows['object_entity'] + '[SEP]' + rows['sentence']
        concat_entity.append(temp)
        
    tokenized_sentences = tokenizer(
        concat_entity, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        add_special_tokens=True
    )
    return tokenized_sentences
############################### Preprocessing ###############################