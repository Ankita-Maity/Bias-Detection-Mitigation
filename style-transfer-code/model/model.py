import os
import numpy as np
import torch
from icecream import ic
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    MT5ForConditionalGeneration,
    MBartForConditionalGeneration,
    AutoConfig,
)

class GenModel(torch.nn.Module):
    def __init__(self,
                learning_rate,
                weight_decay,
                model_name_or_path,
                config,
                is_mt5,
                eval_beams,
                tgt_max_seq_len,
                tokenizer,
                model_gpus,
                isTest,
                language,
                multilingual,
                final_checkpoint='',
                checkpoint=''
                 ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_name_or_path = model_name_or_path
        self.is_mt5 = is_mt5
        self.eval_beams = eval_beams
        self.tgt_max_seq_len = tgt_max_seq_len
        self.tokenizer = tokenizer
        self.model_gpus = model_gpus
        self.checkpoint = checkpoint
        self.isTest = isTest
        self.final_checkpoint = final_checkpoint
        self.language = language
        self.multilingual = multilingual

        print("Loading Main Model")
        self.config = AutoConfig.from_pretrained(config)
        if self.is_mt5:
            self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
            self.model.resize_token_embeddings(len(self.tokenizer))
            #ckpt_state_dict = torch.load("mT5.ckpt")["state_dict"]
            #self.model.load_state_dict(ckpt_state_dict)
        else:
            self.config.dropout = 0.1
            self.model = MBartForConditionalGeneration.from_pretrained(self.model_name_or_path, config=self.config)
            self.bos_id = self.tokenizer._convert_token_to_id_with_added_voc("<s>")
            self.eos_id = self.tokenizer._convert_token_to_id_with_added_voc("</s>")
            self.pad_id = self.tokenizer._convert_token_to_id_with_added_voc("<pad>")
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.isTest:
            ic(self.checkpoint)
            a = torch.load(self.checkpoint)
            b = OrderedDict([(k[13:], v) for (k, v) in a.items()])
            self.model.load_state_dict(b)
        else:
            if self.final_checkpoint is not None: #there's a final trained checkpoint to resume training from
                #a = torch.load(self.final_checkpoint, map_location='cuda:0')
                a = torch.load(self.final_checkpoint)
                b = OrderedDict([(k[13:], v) for (k, v) in a.items()])
                self.model.load_state_dict(b)

        print("Main Model successfully loaded")

    def forward(self, batch):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        return outputs

    def test(self, batch):
        
        if self.is_mt5:
            generated_ids = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=True, num_beams=self.eval_beams, max_length=self.tgt_max_seq_len)
                
        else:
            generated_ids = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=True, num_beams=self.eval_beams, max_length=self.tgt_max_seq_len, pad_token_id=self.pad_id, bos_token_id=self.bos_id, eos_token_id=self.eos_id, decoder_start_token_id=self.tokenizer._convert_token_to_id_with_added_voc(self.language))
              

        input_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        pred_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if self.is_mt5:
            batch['labels'][batch['labels'] == -100] = self.tokenizer.pad_token_id
        gold_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        return {'input_text':input_text, 'pred_text': pred_text, 'gold_text': gold_text}

    def middle(self, batch):
        if self.is_mt5:
            generated_ids = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=True, num_beams=self.eval_beams, max_length=self.tgt_max_seq_len)
                
        else:
            generated_ids = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=True, num_beams=self.eval_beams, max_length=self.tgt_max_seq_len, pad_token_id=self.pad_id, bos_token_id=self.bos_id, eos_token_id=self.eos_id, decoder_start_token_id=self.tokenizer._convert_token_to_id_with_added_voc(batch['language'][0]))                
        
        input_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        pred_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if self.is_mt5:
            batch['labels'][batch['labels'] == -100] = self.tokenizer.pad_token_id

        gold_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        outputs = self(batch)
        loss, logits = outputs['loss'], outputs['logits']
        return {'main_loss': loss, 'logits': logits, 'input_text': input_text, 'pred_text': pred_text, 'gold_text': gold_text}


    # def genOutput(self, batch):
    #     generated_ids = self.model.generate(
    #         input_ids=batch['input_ids'],
    #         attention_mask=batch['attention_mask'],
    #         use_cache=True,
    #         num_beams=self.eval_beams,
    #         max_length=self.tgt_max_seq_len
    #     )

    #     input_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    #     if self.is_mt5:
    #         batch['labels'][batch['labels'] == -100] = self.tokenizer.pad_token_id
    #     ref_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
    #     pred_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #     return pred_text, ref_text, input_text
