import argparse 
import os
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from settings import * 
from metrics import compute_metrics
from data_utils import ReDataset 
import wandb
import re
from utils import *

import yaml
from box import Box


conf_url = 'config.yaml'
with open(conf_url, 'r') as f:
	config_yaml = yaml.load(f, Loader=yaml.FullLoader)
config = Box(config_yaml)

def train(args):
    x_train = preprocessing(load_data(args.train_path))
    y_train = label_to_num(load_data(args.train_path)['label'])
    
    x_valid = preprocessing(load_data(args.dev_path))
    y_valid = label_to_num(load_data(args.dev_path)['label'])
    
    for data in [x_train, x_valid]:
        subject = data.loc[:, 'subject_entity'].values.tolist()
        subject = [re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ ]', '', sub).strip() for sub in subject]
        subject = list(set([sub for sub in subject if sub and ' ' not in sub]))
    
        objects = data.loc[:, 'object_entity'].values.tolist()
        objects = [re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ ]', '', obj).strip() for obj in objects]
        objects = list(set([obj for obj in objects if obj and ' ' not in obj]))
        args.tokenizer.add_tokens(subject)
        args.tokenizer.add_tokens(objects)
    
        print(f'Add Token size is {len(objects) + len(subject)}')
    print(f'Total Tokenizer length: {len(args.tokenizer)}')
    
    train_set = ReDataset(args,x_train, y_train, types='train')
    dev_set = ReDataset(args, x_valid, y_valid, types='dev')
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    model.to(args.device)
    model.resize_token_embeddings(len(args.tokenizer))
    
    train_args = TrainingArguments(
        output_dir = f'{args.model_name}-{args.batch_size}-{args.learning_rate}', 
        save_total_limit=10, 
        save_steps=1000, 
        num_train_epochs=20, 
        learning_rate=args.learning_rate, 
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size, 
        warmup_steps=500, 
        weight_decay=args.weight_decay, 
        logging_dir = './logs', 
        logging_steps=100, 
        evaluation_strategy='steps', 
        eval_steps=500, 
        load_best_model_at_end=True 
    )

    trainer = Trainer(
        model=model, 
        args=train_args, 
        train_dataset=train_set, 
        eval_dataset=dev_set, 
        compute_metrics=compute_metrics
    )
    trainer.train()
    torch.save(model.state_dict(), os.path.join(f'{train_args.output_dir}/{args.model_name.split("/")[-1]}-{args.batch_size}-{args.learning_rate}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', default=config.model_name, type=str 
    )
    parser.add_argument(
        '--max_length', '-len', default=config.max_length, type=int
    )
    parser.add_argument(
        '--num_labels', '-l', default=config.num_labels, type=int
    )
    parser.add_argument(
        '--batch_size', '-b', default=config.batch_size, type=int
    )
    parser.add_argument(
        '--weight_decay', '-wd', default=config.weight_decay, type=float
    )
    parser.add_argument(
        '--learning_rate', '-lr', default=config.learning_rate, type=float
    )
    parser.add_argument(
         '--max_epoch', '-epoch', default=config.max_epoch, type=int
    )
    
    args = parser.parse_args()
    args.train_path = os.path.join(TRAIN_DIR, config.train_path)
    args.dev_path = os.path.join(DEV_DIR, config.dev_path)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    wandb.login()
    wandb.init(
        entity='boostcamp-ai-tech-01',
        project= 'Level02', 
        name=f'{args.model_name}-{args.batch_size}-{args.learning_rate}',
        config= {
            'learning_rate': args.learning_rate, 
            'batch_size': args.batch_size, 
            'model_name': args.model_name
        }
    )
    train(args)
