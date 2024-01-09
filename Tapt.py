from transformers import AutoTokenizer, RobertaForMaskedLM, ElectraForMaskedLM, BertForMaskedLM, AutoConfig, DataCollatorWithPadding, DataCollatorForLanguageModeling, AutoModelForMaskedLM
import torch
from transformers import get_linear_schedule_with_warmup
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import wandb
import os

# fetch pretrained model for MaskedLM training 
MODELNAME = 'klue/roberta-small'
tokenizer = AutoTokenizer.from_pretrained(MODELNAME)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelForMaskedLM.from_pretrained(MODELNAME)
# config 지정 불러오기 필요## 
model.to(device)

# Read txt file which is consisted of sentences from train.csv
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='/data/ephemeral/dataset/merged.txt',
    block_size=128 # block size needs to be modified to max_position_embeddings
)

data_collator = DataCollatorForLanguageModeling( 
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2 
)

# need to change arguments 
training_args = TrainingArguments(
    output_dir=f"./{MODELNAME.split('/')[-1]}-pretrained",
    overwrite_output_dir=True,
    learning_rate=5e-05,
    num_train_epochs=10, 
    per_device_train_batch_size=64,
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
wandb.init(
    entity='boostcamp-ai-tech-01',
    project= 'kunha98', 
    name=f'{MODELNAME.split("/")[-1]}-pretrained-{training_args.per_device_train_batch_size}-{training_args.learning_rate}',
    config= {
        'learning_rate': training_args.learning_rate, 
        'batch_size': training_args.per_device_train_batch_size, 
        'model_name': MODELNAME.split("/")[-1]
    }
)

trainer.train()
torch.save(model.state_dict(), os.path.join(f'{training_args.output_dir}/{MODELNAME.split("/")[-1]}-{training_args.per_device_train_batch_size}-{training_args.learning_rate}.pt'))
#trainer.save_model("./klue-roberta-large-pretrained")


#89에서 early stopping