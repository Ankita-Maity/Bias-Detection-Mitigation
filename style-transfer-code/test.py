import os
import json
import wandb
import torch
import argparse

from tqdm import tqdm
from icecream import ic
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model.model import GenModel
from model.dataloader import ModelDataset

def main(args):

    test_path = args.test_path

    tokenizer_name_or_path = args.tokenizer
    model_name_or_path = args.model
    is_mt5 = args.is_mt5

    if args.config is not None:
        config = args.config
    else:
        config = model_name_or_path

    checkpoint = args.checkpoint

    EXP_NAME = args.exp_name
    test_batch_size = args.test_batch_size
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    save_dir = args.save_dir
    multilingual = args.multilingual
    language = args.language



    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f'{save_dir}/preds.csv'

    model_device = args.model_device
    model_gpus = [int(c) for c in args.model_gpus.split(',')]

    isTrial = args.isTrial
    isTest = args.isTest

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    test_dataset = ModelDataset(
        test_path,
        tokenizer,
        max_source_length,
        max_target_length,
        is_mt5,
        isTest,
        multilingual,
        language
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=6,
        shuffle=False
    )

    
    model = GenModel(
        learning_rate=1e-3,
        weight_decay=0.01,
        model_name_or_path=model_name_or_path,
        config = config,
        is_mt5=is_mt5,
        eval_beams=10,
        tgt_max_seq_len=max_target_length,
        tokenizer=tokenizer,
        model_gpus=model_gpus,
        isTest=isTest,
        checkpoint=checkpoint, #test checkpoint
        language = language,
        multilingual = multilingual
        #final checkpoint is the final trained checkpoint (to resume training)
    )
    wandb.init(
        project='RL-baselines-new',
        config={
            'batch_size': test_batch_size
        }
    )
    

    wandb.run.name = EXP_NAME
    wandb.run.save()
    model_device = torch.device(f"{model_device}" if torch.cuda.is_available() else "cpu")
    model.to(model_device)
    ic(next(model.parameters()).is_cuda)

    test_output = []

    for batch in tqdm(test_loader):
        batch['input_ids'] = batch['input_ids'].to(model_device)
        batch['attention_mask'] = batch['attention_mask'].to(model_device)
        batch['labels'] = batch['labels'].to(model_device)
        with torch.no_grad():
            outputs = model.test(batch)

        # ic(outputs)
        # ic(batch['clang'], batch['tlang'], batch['arttitle'], batch['domain'])

        #for i in range(test_batch_size):
        for i in range(len(outputs['input_text'])):
            test_output.append({
                'input_text': outputs['input_text'][i],
                'gold_text': outputs['gold_text'][i],
                'pred_text': outputs['pred_text'][i],
            })

        if len(test_output) >= 20 and isTrial:
            break

    df = pd.DataFrame(test_output)
    df.to_csv(save_path, index = False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input parameters for extractive stage')
    parser.add_argument('--test_path', help='path to output json file for a given domain in given language')
    parser.add_argument('--config', default=None, help='which config file to use')
    parser.add_argument('--checkpoint', default=None, help='which checkpoint file to use')
    parser.add_argument('--tokenizer', default='google/muril-base-cased', help='which tokenizer to use')
    parser.add_argument('--model', default='google/muril-base-cased', help='which model to use')
    parser.add_argument('--is_mt5', type=int, help='is the model mt5')
    parser.add_argument('--exp_name', help='experiment name')
    parser.add_argument('--save_dir', default='predictions/', help='where to save the test output')
    parser.add_argument('--test_batch_size', default=4, type=int, help='test batch size')
    parser.add_argument('--max_source_length', default=250, type=int, help='max source length')
    parser.add_argument('--max_target_length', default=250, type=int, help='max target length')
    parser.add_argument('--model_gpus', default='0', type=str, help='multiple gpus on which main model will be loaded')
    parser.add_argument('--model_device', default='cuda:0', type=str, help='device to run the main generation model on')
    parser.add_argument('--isTest', default=1, type=int, help='test run')
    parser.add_argument('--isTrial', default=0, type=int, help='toy run')
    parser.add_argument('--multilingual', help="True/False")
    parser.add_argument('--language', help='language code needed only during inference e.g. <2bn> for IndicBART or Bengali for mT5')

    args = parser.parse_args()

    main(args)
