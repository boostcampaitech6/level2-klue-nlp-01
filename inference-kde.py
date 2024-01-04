from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader 
import pandas as pd
import torch
import os 

import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm.auto import tqdm 

from utils import num_to_label
from data_utils import ReDataset

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
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--save_path', required=True, type=str)
    parser.add_argument('--batch_size', '-b', default=32)
    parser.add_argument('--num_labels', default=30, type=int)
    parser.add_argument('--f_name', default='submission')
    parser.add_argument('--max_length', default=256)
    args = parser.parse_args()
    args.dev_path = os.path.join(DEV_DIR, 'dev-v.0.0.2.csv')
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', num_labels=30)
    
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'parameters/roberta-large/checkpoint-12000/pytorch_model.bin')))

    test_dataset = ReDataset(args, types='dev')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.to(args.device)
    pred_answer, output_prob = inference(args, model, test_loader)
    pred_answer = num_to_label(pred_answer)
    
    output = pd.DataFrame({'id':list(range(len(pred_answer))),'pred_label':pred_answer,'probs':output_prob,})
    if not os.path.exists(os.path.join(BASE_DIR, 'prediction')):
      os.mkdir(os.path.join(BASE_DIR, 'prediction'))
      
    output.to_csv(f'./prediction/{args.f_name}.csv')
