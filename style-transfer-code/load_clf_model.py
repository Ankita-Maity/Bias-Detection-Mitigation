import os
import torch
import pytorch_lightning as pl
from torch import nn
from transformers import AutoModel, AutoTokenizer


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
        output_logits = nn.functional.softmax(self.task_head(doc_rep))
        
        return output_logits


model_key = 'google/muril-base-cased'
model_path = '/scratch/user/bn-en-gu-hi-kn-mr-ta-te-google-muril-base-cased-15-1e-06-60-wnc-0.1-0.001-epoch=14.ckpt'

tokenizer = AutoTokenizer.from_pretrained(model_key)
clf_model = ClfModel()
ckpt = torch.load(model_path)
clf_model.load_state_dict(ckpt['state_dict'])

sentence = 'bla bla'
sent_tok = tokenizer(sentence, return_tensors='pt', return_token_type_ids=False)
clf_pred = clf_model(**sent_tok)

print('Sentence:', sentence)
print('Prediction:', clf_pred)
