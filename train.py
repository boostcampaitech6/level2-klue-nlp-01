import argparse 
import os
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoTokenizer 
from settings import * 

from utils.preprocessing import preprocess
from utils.utils import load_pkl, build_unk_tokens, save_pkl
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

    train_set = ReDataset(args,x_train, y_train, types='train')
    dev_set = ReDataset(args, x_valid, y_valid, types='dev')
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    # TAPT
    pretrained_dict = torch.load('/data/ephemeral/roberta-large-pretrained/roberta-large-64-5e-05.pt') # pretrained 상태 로드
    model_dict = model.state_dict() # 현재 신경망 상태 로드
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    """
    #HEAD만
    for name, param in model.named_parameters():
        if name.split('.')[0] == 'classifier':
            pass
        else :
            param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    """

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
        load_best_model_at_end=True, 
        metric_for_best_model='micro f1 score', 
        greater_is_better=True,
        seed=42
    )
    
    train_args.focal_loss = args.focal_loss
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
        '--model_name', default='klue/roberta-large', type=str 
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
        '--test_path', default='test_data.csv', type=str
    )

    # wandb
    parser.add_argument(
        '--wandb', default=True, action='store_true'
    )
    parser.add_argument(
        '--entity', '-e', default='boostcamp-ai-tech-01', type=str
    )
    parser.add_argument(
        '--project', default='kunha98', type=str
    )
    
    # focal loss 
    parser.add_argument(
        '--focal_loss', action='store_true'
    )
    
    # Add unk token
    parser.add_argument(
        '--unk_token', action='store_true'
    )
    parser.add_argument(
        '--verbose', action='store_true'
    )
    
    
    args = parser.parse_args()
    args.train_path = os.path.join(DATA_DIR, args.train_path)
    args.dev_path = os.path.join(DATA_DIR, args.dev_path)
    args.test_path = os.path.join(DATA_DIR, args.test_path)
    
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


