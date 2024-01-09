from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import os 

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
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--save_path', default='/data/ephemeral/parameters/roberta-large-64-5e-05.pt', type=str)
    parser.add_argument('--batch_size', '-b', default=64)
    parser.add_argument('--num_labels', default=30, type=int)
    parser.add_argument('--f_name', default='submission')
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--test_path', default='test_data.csv', type=str)
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.test_path = os.path.join(DATA_DIR, args.test_path)

    # Preprocessing...
    x_test, y_test = preprocess(args.test_path)
    
    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    # Add unk tokens 
    #unk_list = load_pkl(os.path.join(DATA_DIR, 'unk_tokens.pkl'))
    #args.tokenizer.add_tokens(unk_list)
    model.resize_token_embeddings(len(args.tokenizer))
    
    # Load Model
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, args.save_path)))
    
    # Get DataLoader 
    test_loader = get_dataloader(args, x_test, y_test, types='test')

    # Inference  
    model.to(args.device)
    pred_answer, output_prob = inference(args, model, test_loader)
    pred_answer = num_to_label(pred_answer)
    
    # Submission
    output = pd.DataFrame({'id':list(range(len(pred_answer))),'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv(os.path.join(OUT_DIR, f'{args.f_name}.csv'))
    print('Success!')