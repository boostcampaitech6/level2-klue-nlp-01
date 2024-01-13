from tqdm.auto import tqdm 
from ast import literal_eval
from utils.utils import load_data, label_to_num, hanja_to_hangul
import re 
"""
작성자: 김인수, 이재형
"""

############################### Preprocessing ###############################
def preprocess(path, marker_type = 'temp', translate = True):
    """_summary_

    Args:
        path (str): data path
        marker_type (str, optional): marker type sellection. Defaults to 'temp'.
        translate (bool, optional): translate hanja to hangul. Defaults to True.

    Returns:
        X, y: 'sentence', 'subject_entity', 'object_entity' column Data, label column data
    """    
    datasets = load_data(path)
    X_col, y_col = ['sentence', 'subject_entity', 'object_entity'], 'label'
    for idx, row in tqdm(datasets.iterrows(), total = datasets.shape[0], desc = "Dataset Preprocessing..."):
        sen, sbj, obj = row['sentence'], literal_eval(row['subject_entity']), literal_eval(row['object_entity'])
        if translate:
            sen = hanja_to_hangul(sen)
        sen = re.sub('\*','', sen)

        sbj_word, obj_word = get_marker_tag(sbj, obj, marker_type, translate)
        sen = replace_string_index(sen, sbj, obj, sbj_word, obj_word)

        # marker_type에 따라 변형된 Sentence
        datasets.at[idx, 'sentence'] = sen
        if translate:
            sbj['word'] = hanja_to_hangul(sbj['word'])
            obj['word'] = hanja_to_hangul(obj['word'])
            datasets.at[idx, 'subject_entity'] = sbj 
            datasets.at[idx, 'object_entity'] = obj

    X = datasets.loc[:, X_col].copy()
    try:
        y = label_to_num(datasets.loc[:, y_col].values)
    except:
        y = [100] * X.shape[0]
    return X, y

def tokenizing(args:dict, datasets, marker_type = 'temp'):
    """_summary_

    Args:
        args (dict): max_length
        datasets (pd.Dataframe): 'sentence', 'subject_entity', 'object_entity' columns Dataframe
        marker_type (str, optional): Defaults to 'temp'.

    Returns:
        tokenized_sentences: tokenized_sentences
    """    
    
    concat_entity = []
    for idx, row in tqdm(datasets.iterrows(), total = datasets.shape[0], desc = "Dataset Tokenizing..."):
        sbj, obj = row['subject_entity'], row['object_entity']
        sbj_word, sbj_type = sbj['word'], sbj['type']
        obj_word, obj_type = obj['word'], obj['type']

        if marker_type == 'temp':
            relation = temp = f"@*{sbj_type}*{sbj_word}@와 #^{obj_type}^{obj_word}#의 관계"
        else:
            relation = sbj_word + '[SEP]' + obj_word
        temp = relation + '[SEP]' + row['sentence']
        concat_entity.append(temp)
        
    tokenized_sentences = args.tokenizer(
        concat_entity, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length= args.max_length, 
        add_special_tokens=True
    )
    return tokenized_sentences
############################### Preprocessing ###############################

def get_marker_tag(sbj:dict, obj:dict, method:str = 'temp', translate:bool = True) -> (str, str):
    """_summary_

    Args:
        sen (str): Original sentence
        sbj (dict): dictionary of subject entity
        obj (dict): dictionary of object entity
        method (str): take one method in em, tem, temp, default

    Returns:
        str, str: return marked sbj_word, obj_word
    """
    # Subject_entity, Object_entity Word, Start_index, End_index, Type 추출
    sbj_word, sbj_type, obj_word, obj_type = sbj['word'], sbj['type'], obj['word'], obj['type']

    if method=="em":
        sbj_word=f"[주체-{get_entity_kor(sbj_type)}]"
        obj_word=f"[대상-{get_entity_kor(obj_type)}]"
    elif method=='tem':
        sbj_word=f"<주:{get_entity_kor(sbj_type)}> {sbj_word} </주:{get_entity_kor(sbj_type)}>"
        obj_word=f"<대:{get_entity_kor(obj_type)}> {obj_word} </대:{get_entity_kor(obj_type)}>"
    elif method=='temp':
        sbj_word=f"#^{get_entity_kor(sbj_type)}^" + sbj_word + f"#"
        obj_word=f"@*{get_entity_kor(obj_type)}*" + obj_word + f"@"
    elif method == 'default':
        sbj_word = sbj_word
        obj_word = obj_word

    return sbj_word, obj_word

def get_entity_kor(ent:str)->str:
    """Get KOR entity using entity dict.

    Args:
        ent (str): Entity

    Returns:
        str: return KOR translated entity
    """
    # Entity Type Dictionary를 통해 영 -> 한 변경
    type_dic = {"PER": "사람", "ORG": "단체", "DAT": "날짜", "LOC": "위치", "POH": "기타", "NOH": "수량"}
    return type_dic[ent]

def replace_string_index(sen:str, sbj:dict, obj:dict, sbj_word:str, obj_word:str)->str:
    """Replace original word into markered one.

    Args:
        sen (str): Original sentence
        sbj (dict): dictionary of subject entity
        obj (dict): dictionary of object entity
        sbj_word (str): subject word marked
        obj_word (str): object word marked

    Returns:
        str: marked sentence
    """
    sbj_start_idx, sbj_end_idx=sbj['start_idx'], sbj['end_idx']
    obj_start_idx, obj_end_idx=obj['start_idx'], obj['end_idx']
    if sbj_start_idx<obj_start_idx:
        return sen[:sbj_start_idx] + sbj_word + sen[sbj_end_idx+1:obj_start_idx] + obj_word + sen[obj_end_idx+1:]
    else:
        return sen[:obj_start_idx] + obj_word + sen[obj_end_idx+1:sbj_start_idx] + sbj_word + sen[sbj_end_idx+1:]