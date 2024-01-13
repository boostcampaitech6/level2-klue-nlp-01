import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, get_linear_schedule_with_warmup, LineByLineTextDataset, Trainer, TrainingArguments, EarlyStoppingCallback 
#import wandb
import argparse
from settings import * 

def train_tapt(args):   
    # fetch pretrained model for MaskedLM training 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device(args.device)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    # config 지정 불러오기 필요## 
    model.to(device)

    # Read txt file which is consisted of sentences from train.csv
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=os.path.join(DATA_DIR, 'merged-kde.txt'),
        block_size=128 # block size needs to be modified to max_position_embeddings
    )

    data_collator = DataCollatorForLanguageModeling( 
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2 
    )

    # need to change arguments 
    training_args = TrainingArguments(
        output_dir=f'{args.model_name}-{args.batch_size}-{args.learning_rate}',
        learning_rate=args.learning_rate,
        num_train_epochs=100, 
        per_device_train_batch_size=args.batch_size,
        save_steps=100,
        save_total_limit=2,
        seed=0,
        save_strategy='epoch',
        gradient_accumulation_steps=8,
        logging_steps=100,
        evaluation_strategy='epoch',
        resume_from_checkpoint=True,
        fp16=False,
        fp16_opt_level='O1',
        load_best_model_at_end=True,

        adam_epsilon=1e-6
    ) 

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, eps=training_args.adam_epsilon, weight_decay=0.01, betas=(0.9,0.98))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(trainer.train_dataset) * training_args.num_train_epochs)

    trainer.optimizer = optimizer
    trainer.scheduler = scheduler

    trainer.train()
    torch.save(model.state_dict(), os.path.join(f'{PARAM_DIR}/Tapt-{args.model_name.split("/")[-1]}-pretrained.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model name 
    parser.add_argument(
        '--model_name', default='kykim/electra-kor-base', type=str 
    )

    parser.add_argument(
        '--device', default='cuda:0', type=str
    )
    parser.add_argument(
        '--batch_size', default=64, type=int
    )
    parser.add_argument(
        '--learning_rate', default=5e-5, type=float
    )

    # # wandb
    # parser.add_argument(
    #     '--wandb', default=True, action='store_true'
    # )
    # parser.add_argument(
    #     '--entity', '-e', default='boostcamp-ai-tech-01', type=str
    # )
    # parser.add_argument(
    #     '--project', default='sujong', type=str
    # )

    args = parser.parse_args()

    # if args.wandb:
    #     wandb.init(
    #         entity=args.entity,
    #         project= args.project, 
    #         name=f'{args.model_name}-tapt-pretraining',
    #     )
    
    train_tapt(args)

