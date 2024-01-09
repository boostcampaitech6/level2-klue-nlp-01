from transformers import Trainer 
from metrics.metrics import FocalLoss
import torch.nn.functional as F

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.args.focal_loss:
            loss_fct = FocalLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        else:
            loss = F.cross_entropy(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss