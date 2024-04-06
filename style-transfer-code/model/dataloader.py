import json

from icecream import ic
from torch.utils.data import Dataset
from langdetect import detect, DetectorFactory
import pandas as pd

class ModelDataset(Dataset):
    def __init__(self, path, tokenizer, max_source_length, max_target_length, is_mt5, isTest, multilingual, language):
        
        self.df = pd.read_csv(path)
        self.df = self.df.dropna()
        self.df.columns = ["input_text", "target_text"]
        self.df.reset_index(inplace=True, drop=True)
        
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_mt5 = is_mt5
        self.isTest = isTest
        self.multilingual = multilingual
        self.language = language
        self.lang_map = {
        'bn' : 'bn_IN',
        'en' : 'en_XX',
        'gu' : 'gu_IN',
        'hi' : 'hi_IN',
        'kn' : 'kn_IN',
        'mr' : 'mr_IN',
        'ta' : 'ta_IN',
        'te' : 'te_IN'
        }

        DetectorFactory.seed = 0

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        
        input_text = self.df['input_text'][index]
        target_text = self.df['target_text'][index]
        
        if self.is_mt5:
            if self.isTest and self.multilingual=='True':
                sencoding = self.tokenizer(f'Generate in {self.language}: {input_text} </s>', return_tensors='pt', max_length=self.max_source_length, padding='max_length', truncation=True)
                tencoding = self.tokenizer(f'{target_text} </s>', return_tensors='pt', max_length=self.max_target_length, padding='max_length', truncation=True)
            else:
                sencoding = self.tokenizer(f'{input_text} </s>', return_tensors='pt', max_length=self.max_source_length, padding='max_length', truncation=True)
                tencoding = self.tokenizer(f'{target_text} </s>', return_tensors='pt', max_length=self.max_target_length, padding='max_length', truncation=True)
                
                
        else: #IndicBART
            if self.isTest == 0: #getting the language code for IndicBART training using GenModel
                self.language = target_text[:5]
            sencoding = self.tokenizer(f'{input_text}', return_tensors='pt', max_length=self.max_source_length, padding='max_length', truncation=True, add_special_tokens=False)
            tencoding = self.tokenizer(f'{target_text}', return_tensors='pt', max_length=self.max_target_length, padding='max_length', truncation=True, add_special_tokens=False)
            
        input_ids, attention_mask = sencoding['input_ids'], sencoding['attention_mask']
        labels, tgt_mask = tencoding['input_ids'], tencoding['attention_mask']

        if self.is_mt5:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze(), 'language': self.language, 'tgt_mask': tgt_mask}
