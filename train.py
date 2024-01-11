import argparse 
import os
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoTokenizer 
from settings import * 

from utils.preprocessing import preprocess
from utils.utils import load_pkl, build_unk_tokens, save_pkl, set_seed, tapt_apply
from metrics.metrics import compute_metrics
from data_utils.data_utils import ReDataset 
import wandb
from trainer import CustomTrainer



def train(args):
    x_train, y_train = preprocess(args.train_path)
    x_valid, y_valid = preprocess(args.dev_path)
    x_test, _ = preprocess(args.test_path)
    if args.unk_token:
        if not os.path.exists(os.path.join(DATA_DIR, 'unk_tokens.pkl')):
            unk_list = []
            for data in [x_train, x_valid, x_test]:
                unk = build_unk_tokens(data, args.tokenizer, verbose=args.verbose)
                unk_list.extend(unk)
            save_pkl(unk_list, os.path.join(DATA_DIR, 'unk_tokens.pkl'))
        else:
            unk_list = load_pkl(os.path.join(DATA_DIR, 'unk_tokens.pkl'))
            
        args.tokenizer.add_tokens(unk_list)

    train_set = ReDataset(args, x_train, y_train, types='train')
    dev_set = ReDataset(args, x_valid, y_valid, types='dev')
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    # Tapt
    if args.tapt:
        model = tapt_apply(torch.load(args.tapt_pretrained_path), model)
    model.to(args.device)

    
    model.resize_token_embeddings(len(args.tokenizer))
    print(f'Tokenizer Size is {len(args.tokenizer)}')
    
    train_args = TrainingArguments(
        output_dir = f'{args.model_name.split("/")[-1]}-{args.batch_size}-{args.learning_rate}', 
        save_total_limit=5, 
        save_steps=500, 
        num_train_epochs=30,
        seed=42, 
        learning_rate=args.learning_rate, 
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size, 
        warmup_steps=500, 
        weight_decay=args.weight_decay, 
        logging_dir = './logs', 
        logging_steps=100, 
        evaluation_strategy='steps', 
        eval_steps=500, 
        load_best_model_at_end=True, 
        metric_for_best_model='micro f1 score', 
        greater_is_better=True
    )
    
    train_args.loss_type = args.loss_type
    train_args.gamma = args.gamma 
    
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
    
    # model name 
    parser.add_argument(
        '--model_name', required=True, type=str 
    )
    
    # hyper-parameters 
    parser.add_argument(
        '--max_length', '-len', default=128, type=int
    )
    parser.add_argument(
        '--num_labels', '-l', default=30, type=int
    )
    parser.add_argument(
        '--batch_size', '-b', default=64, type=int
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
        '--device', default='cuda:0', type=str
    )
    
    # path 
    parser.add_argument(
        '--train_path', default='train-v.0.0.2.csv', type=str
    )
    parser.add_argument(
        '--dev_path', default='dev-v.0.0.2.csv', type=str
    )
    parser.add_argument(
        '--test_path', default='test.csv', type=str
    )

    # wandb
    parser.add_argument(
        '--wandb', default=True, action='store_true'
    )
    parser.add_argument(
        '--entity', '-e', default='boostcamp-ai-tech-01', type=str
    )
    parser.add_argument(
        '--project', default='Level02', type=str
    )
    
    # loss 
    # select between focal, ldam, labsm (else) cross entropy
    parser.add_argument(
        '--loss_type', '-lt', required=True, type=str
    )
    
    # Add unk token
    parser.add_argument(
        '--unk_token', action='store_true'
    )
    parser.add_argument(
        '--verbose', action='store_true'
    )
    
    # Apply Tapt
    parser.add_argument(
        '--tapt', default=False, action='store_true'
    )
    parser.add_argument(
        '--tapt_pretrained_path', '-pre_path', default='Tapt-roberta-large-pretrained.pt', action='store_true'
    )


    
    args = parser.parse_args()
    args.train_path = os.path.join(DATA_DIR, args.train_path)
    args.dev_path = os.path.join(DATA_DIR, args.dev_path)
    args.test_path = os.path.join(DATA_DIR, args.test_path)

    args.tapt_pretrained_path = os.path.join(PARAM_DIR,args.tapt_pretrained_path)

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.wandb:
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


