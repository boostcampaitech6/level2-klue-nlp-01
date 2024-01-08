import argparse 
import os
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer 
from settings import * 

from utils import load_data, preprocessing, label_to_num
from metrics.metrics import compute_metrics, FocalLoss
from data_utils.data_utils import ReDataset 
import wandb
from transformers import Trainer 


def train(args):
    # x_train = preprocessing(load_data(args.train_path))
    # y_train = label_to_num(load_data(args.train_path)['label'])
    
    # x_valid = preprocessing(load_data(args.dev_path))
    # y_valid = label_to_num(load_data(args.dev_path)['label'])
    x_train, y_train = preprocessing(args.train_path)
    x_valid, y_valid = preprocessing(args.dev_path)
    

    train_set = ReDataset(args,x_train, y_train, types='train')
    dev_set = ReDataset(args, x_valid, y_valid, types='dev')
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    model.to(args.device)
    model.resize_token_embeddings(len(args.tokenizer))
    print(f'Tokenizer Size is {len(args.tokenizer)}')
    
    train_args = TrainingArguments(
        output_dir = f'{args.model_name.split("/")[-1]}-{args.batch_size}-{args.learning_rate}', 
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
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get('labels')
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss_fct = FocalLoss(gamma=2)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss 

    trainer = CustomTrainer(
        model=model, 
        args=train_args, 
        train_dataset=train_set, 
        eval_dataset=dev_set, 
        compute_metrics=compute_metrics
    )
    trainer.train()
    torch.save(model.state_dict(), os.path.join(PARAM_DIR, f'{args.model_name.split("/")[-1]}-{args.batch_size}-{args.learning_rate}.pt'))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', required=True, type=str 
    )
    parser.add_argument(
        '--max_length', '-len', default=256, type=int
    )
    parser.add_argument(
        '--num_labels', '-l', default=30, type=int
    )
    parser.add_argument(
        '--batch_size', '-b', default=32, type=int
    )
    parser.add_argument(
        '--weight_decay', '-wd', default=0.01, type=float
    )
    parser.add_argument(
        '--learning_rate', '-lr', default=5e-5, type=float
    )
    parser.add_argument(
        '--gamma', '-g', default=2., type=float
    )
    parser.add_argument(
        '--train_path', default='train-v.0.0.2.csv', type=str
    )
    parser.add_argument(
        '--dev_path', default='dev-v.0.0.2.csv', type=str
    )
    parser.add_argument(
        '--test_path', default='test_data.csv', type=str
    )
    parser.add_argument(
        '--device', default='cuda:0', type=str
    )
    parser.add_argument(
        '--wandb', default=False, action='store_true'
    )
    parser.add_argument(
        '--entity', '-e', default='boostcamp-ai-tech-01', type=str
    )
    parser.add_argument(
        '--project', default='Level02', type=str
    )
    
    
    args = parser.parse_args()
    args.train_path = os.path.join(DATA_DIR, args.train_path)
    args.dev_path = os.path.join(DATA_DIR, args.dev_path)
    args.test_path = os.path.join(DATA_DIR, args.test_path)
    
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.wandb:
        wandb.login()
        wandb.init(
            entity=args.entity,
            project= args.project, 
            name=f'{args.model_name}-{args.batch_size}-{args.learning_rate}',
            config= {
                'learning_rate': args.learning_rate, 
                'batch_size': args.batch_size, 
                'model_name': args.model_name
            }
        )
    train(args)


