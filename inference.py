from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import os 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm.auto import tqdm 

from utils.utils import num_to_label, load_pkl
from utils.preprocessing import preprocess
from data_utils.data_utils import get_dataloader
from settings import * 

def inference(args, model, dataloader):
    model.eval()
    output_pred, output_prob = [], []
    for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        with torch.no_grad():
            X = {k:v.to(args.device) for k, v in batch[0].items()}
            pred_y = model(**X)
        logits = pred_y[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        
        output_pred.append(result)
        output_prob.append(prob)
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model dir
<<<<<<< HEAD
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--save_path', default='/data/ephemeral/parameters/roberta-large-64-5e-05_tapt.pt', type=str)
    parser.add_argument('--inference', required=True, type=str) # test or dev
    parser.add_argument('--batch_size', '-b', default=128)
    parser.add_argument('--num_labels', default=30, type=int)
    parser.add_argument('--f_name', default='submission')
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--test_path', default='test_data.csv', type=str)
    parser.add_argument('--dev_path', default='dev-v.0.0.2.csv', type=str)
>>>>>>> e51595cdf5095e9b377cbc806a92fd9568e19adb
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Preprocessing...
    if args.inference == 'test':
      x_test, y_test = preprocess(os.path.join(DATA_DIR, args.test_path))
    elif args.inference == 'dev':
      x_test, y_test = preprocess(os.path.join(DATA_DIR, args.dev_path))
    
    # Get DataLoader 
    test_loader = get_dataloader(args, x_test, y_test, types='test')    
    
    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    # Add unk tokens 
    #unk_list = load_pkl(os.path.join(DATA_DIR, 'unk_tokens.pkl'))
    #args.tokenizer.add_tokens(unk_list)
    model.resize_token_embeddings(len(args.tokenizer))
    
    # Load Model
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, args.save_path)))
    
    # Inference  
    model.to(args.device)
    pred_answer, output_prob = inference(args, model, test_loader)
    pred_answer = num_to_label(pred_answer)
    
    if args.inference == 'dev':
        label_list = ["no_relation", "org:dissolved", "org:founded", "org:place_of_headquarters", 
                "org:alternate_names", "org:member_of", "org:members","org:political/religious_affiliation", 
                "org:product", "org:founded_by","org:top_members/employees", "org:number_of_employees/members", 
                "per:date_of_birth", "per:date_of_death", "per:place_of_birth", "per:place_of_death", 
                "per:place_of_residence", "per:origin","per:employee_of", "per:schools_attended", "per:alternate_names", 
                "per:parents", "per:children", "per:siblings", "per:spouse", "per:other_family", "per:colleagues", 
                "per:product", "per:religion", "per:title"]
  
        y_test = num_to_label(y_test)
        cm = confusion_matrix(y_test, pred_answer)
        
  
        plt.figure(figsize=(18, 18))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_list, yticklabels=label_list,
                    square=False) 
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.jpg', dpi=300, pad_inches=0.5)
        
    elif args.inference == 'test':
        # Submission
        output = pd.DataFrame({'id':list(range(len(pred_answer))),'pred_label':pred_answer,'probs':output_prob,})
        output.to_csv(os.path.join(OUT_DIR, f'{args.f_name}.csv'))
        
    print('Success!')
    
    
    #### python inference.py --model_name  --save_path  --inference dev (test시에는 test로 작성하시면 됩니다.)