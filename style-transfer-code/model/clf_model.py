import torch
from torch import nn

import pytorch_lightning as pl
from transformers import AutoModel


class TexClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_size, num_labels, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # takes [CLS] token representation as input
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x


class ClfModel(nn.Module):
    def __init__(self, model_key='google/muril-base-cased'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_key, add_pooling_layer=False)
        self.task_head = TexClassificationHead(self.model.config.hidden_size, 2, 0.1)
    
    def forward(self, input_ids, attention_mask):
        # loading to cuda devices
        # input_seq = input_seq.to(self.transformer.device)
        # attention_mask = attention_mask.to(self.transformer.device)
        # calculating the output logits
        doc_rep = self.model(input_ids, attention_mask=attention_mask)[0]
        output_logits = nn.functional.softmax(self.task_head(doc_rep), dim=-1)
        
        return output_logits


def load_clf_model(model_key, model_path):
    clf_model = ClfModel(model_key)
    ckpt = torch.load(model_path)
    del ckpt['state_dict']['model.embeddings.position_ids']
    clf_model.load_state_dict(ckpt['state_dict'])
    clf_pred = clf_model.eval()

    return clf_pred
