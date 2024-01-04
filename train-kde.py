import argparse 
import os
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from settings import * 
from metrics import compute_metrics
from data_utils import ReDataset 

import yaml
from box import Box
conf_url = 'config.yaml'
with open(conf_url, 'r') as f:
	config_yaml = yaml.load(f, Loader=yaml.FullLoader)
config = Box(config_yaml)

def train(args):
    train_set = ReDataset(args, types='train')
    dev_set = ReDataset(args, types='dev')
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    model.to(args.device)
    
    train_args = TrainingArguments(
        output_dir = f'{PARAM_DIR}/{args.model_name.split("/")[-1]}', 
        save_total_limit=10, 
        save_steps=500, 
        num_train_epochs=args.max_epoch, 
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
    torch.save(model, os.path.join(f'{args.model_name.split("/")[-1]}-{args.batch_size}-{args.learning_rate}.pt'))

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
    
    train(args)
