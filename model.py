import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class Model(nn.Module):
    def __init__(self, MODEL_NAME):
        super().__init__()

        self.config= AutoConfig.from_pretrained(MODEL_NAME)
        self.config.num_labels= 30
        self.model= AutoModel.from_pretrained(MODEL_NAME, config= self.config)
        self.hidden_dim= self.config.hidden_size # roberta hidden dim = 1024

        self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 2, dropout= 0.2,
                            batch_first= True, bidirectional= True)
        self.fc= nn.Linear(self.hidden_dim*2, self.config.num_labels)


    def forward(self, input_ids, attention_mask, token_type_ids, labels = None):
        # BERT output= (16, 244, 1024) (batch, seq_len, hidden_dim)
        output= self.model(input_ids= input_ids, attention_mask= attention_mask, token_type_ids = token_type_ids)[0]

        # LSTM last hidden, cell state shape : (2, 244, 1024) (num_layer, seq_len, hidden_size)
        hidden, (last_hidden, last_cell)= self.lstm(output)

        # (16, 1024) (batch, hidden_dim)
        cat_hidden= torch.cat((last_hidden[-1], last_hidden[-2]), dim= 1)
        logits= self.fc(cat_hidden)
        
        return {'logits': logits}
        # return logits