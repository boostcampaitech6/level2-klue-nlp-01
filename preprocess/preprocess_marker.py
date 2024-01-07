"""
작성자: 김인수, 이재형

"""


def get_marker_tag(sen: str, sbj:dict, obj:dict, method:str) -> (str, str):
    """_summary_

    Args:
        sen (str): Original sentence
        sbj (dict): dictionary of subject entity
        obj (dict): dictionary of object entity
        method (str): take one method in em, tem, temp     

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

