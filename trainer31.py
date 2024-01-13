from transformers import Trainer 
from metrics.metrics import FocalLoss, LDAMLoss, LabelSmoothingLoss
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader 
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        cls_num_list = torch.bincount(labels)
        cls_num_list = torch.cat((cls_num_list,torch.zeros(31-cls_num_list.shape[0], device='cuda')),axis=0)
        
        if self.args.loss_type=='focal':
            loss_fct = FocalLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        elif self.args.loss_type=='ldam':
            loss_fct = LDAMLoss(cls_num_list)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        elif self.args.loss_type=='labsm':
            loss_fct = LabelSmoothingLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = F.cross_entropy(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
        
class ImbalancedSamplerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        model.cuda()
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        if self.args.loss_type=='focal':
            loss_fct = FocalLoss(gamma=2)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        elif self.args.loss_type=='ldam':
            loss_fct = LDAMLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        elif self.args.loss_type=='labsm':
            loss_fct = LabelSmoothingLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = F.cross_entropy(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        def get_label(dataset):
            return dataset.labels

        train_sampler = ImbalancedDatasetSampler(
            train_dataset, callback_get_label=get_label
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers            
        )