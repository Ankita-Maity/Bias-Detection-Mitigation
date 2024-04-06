import os
import wandb
import torch
import argparse

import sys
import random
import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed import ReduceOp
from earlystopping import EarlyStopping

from tqdm import tqdm
from icecream import ic
import datetime
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.model import GenModel
from model.dataloader import ModelDataset
from model.rewards import embedding_reward, unchanged_reward, kl_reward
from model.clf_model import load_clf_model

from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    Adafactor,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)

SEED = 42
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def cleanup():
    dist.destroy_process_group()

def calcTestReward(
        batch,
        logits,
        input_text,
        pred_text,
        ref_text,
        mask,
        emb_device,
        embmodel,
        embtok,
        clf_device,
        clf_model,
        clf_tok,
        model_device

):

    emb_reward = embedding_reward(ref_text, pred_text, emb_device, embmodel, embtok)
    unchange_reward = unchanged_reward(input_text, ref_text, pred_text, embtok)
    kl_div_reward = kl_reward(ref_text, pred_text, clf_device, clf_model, clf_tok)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    argmax = torch.amax(probs, dim=2)
    bestaction = torch.log(argmax)
    emb_reward = emb_reward.unsqueeze(1)
    unchange_reward = unchange_reward.unsqueeze(1)
    kl_div_reward = kl_div_reward.unsqueeze(1)

    embloss = -bestaction * emb_reward.to(model_device) * mask
    embloss = (embloss.sum(-1)/mask.sum(-1)).mean()

    unchanged_loss = -bestaction * unchange_reward.to(model_device) * mask
    unchanged_loss = (unchanged_loss.sum(-1)/mask.sum(-1)).mean()
    
    klloss = -bestaction * kl_div_reward.to(model_device) * mask
    klloss = (klloss.sum(-1)/mask.sum(-1)).mean()
    
    return {'embloss': embloss, 'unchanged_loss': unchanged_loss, 'klloss': klloss}

@record
def main(args):
    
    local_rank = int(os.environ['LOCAL_RANK'])
    ic(local_rank)
    args.is_master = local_rank == 0

    train_path = args.train_path
    val_path = args.val_path

    tokenizer_name_or_path = args.tokenizer
    model_name_or_path = args.model
    is_mt5 = args.is_mt5

    if args.config is not None:
        config = args.config
    else:
        config = model_name_or_path

    EXP_NAME = args.exp_name
    num_epochs = args.num_epochs
    lr = args.lr
    wd = args.wd
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    save_dir = args.save_dir
    multilingual = args.multilingual
    language = args.language
    wandb_log = args.wandb
    
    emb_coeff = args.emb_coeff
    unchanged_coeff = args.unchanged_coeff
    clf_coeff = args.clf_coeff

    torch.autograd.set_detect_anomaly(True)
    es = EarlyStopping(patience=3)
    
    rank = local_rank

    # model_device = args.model_device
    #model_gpus = [int(c) for c in args.model_gpus.split(',')]
    model_gpus = [local_rank]
    device = torch.cuda.device(local_rank)
    ic(f"starting setup {local_rank}")
    #setup(rank, args.world_size)
    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed_all(SEED)

#     ner_device = args.ner_device
#     ner_model_path = args.ner_model_path
#     ner_tok = args.ner_tok

#     ner_f_device = args.ner_f_device
#     ner_f_model_path = args.ner_f_model_path
#     ner_f_tok = args.ner_f_tok

    emb_device = args.emb_device
    emb_model_path = args.emb_model_path
    emb_tok = args.emb_tok
    clf_device = args.clf_device
    clf_model_path = args.clf_model_path
    clf_tok = args.clf_tok

    isTrial = args.isTrial
    isTest = args.isTest

    #foreign = {'en'}

    ic(f"getting tokenizer {local_rank}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    

    ic(f"getting datasets {local_rank}")
    train_dataset = ModelDataset(
        train_path,
        tokenizer,
        max_source_length,
        max_target_length,
        is_mt5,
        isTest,
        multilingual,
        language
    )

    val_dataset = ModelDataset(
        val_path,
        tokenizer,
        max_source_length,
        max_target_length,
        is_mt5,
        isTest,
        multilingual,
        language
    )
    
    if is_mt5:
        train_sampler = DistributedSampler(train_dataset, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, drop_last=False)
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=False, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, num_workers=6)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size, num_workers=6)

    ic(f"got datasets {local_rank}")
    
    start_epoch = 0
    final_checkpoint = None 

    if os.path.exists(save_dir):
        filenames = [f for f in os.listdir(save_dir) if "half" not in f]
        if len(filenames) > 0:
            filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
            final_checkpoint = filenames[-1]
            start_epoch = int(final_checkpoint.split('.')[0]) + 1
            final_checkpoint = f'{save_dir}/{final_checkpoint}'
            
    ic(f"getting model {local_rank}")
    model_device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    model = GenModel(
        learning_rate=lr,
        weight_decay=wd,
        model_name_or_path=model_name_or_path,
        config = config,
        is_mt5=is_mt5,
        eval_beams=4,
        tgt_max_seq_len=max_target_length,
        tokenizer=tokenizer,
        model_gpus=local_rank,
        isTest=isTest,
        final_checkpoint=final_checkpoint,
        language = language,
        multilingual = multilingual
    )
    
    #ic(f"got model {local_rank} {next(model.parameters()).device}")


#     print("Loading Embedding Model")
#     embeddingModel =  AutoModel.from_pretrained(emb_model_path, output_hidden_states=True).to(emb_device)
#     embeddingTok =  AutoTokenizer.from_pretrained(emb_tok, padding='max_length', truncation='max_length', max_length=512)
#     #ic(torch.cuda.current_device(), torch.cuda.device_count())
#     embeddingModel.eval()
#     #ic(next(embeddingModel.parameters()).device)
#     #torch.cuda.empty_cache()
#     print("Loaded Embedding Model")

#     print('Loading classification model')
#     #clfModel =  AutoModel.from_pretrained(clf_tok, output_hidden_states=True).to(clf_device)
#     #ic(torch.cuda.current_device(), torch.cuda.device_count())
#     #clfModel.eval()
#     clfModel = load_clf_model(clf_tok, clf_model_path).to(clf_device)
#     clfTok =  AutoTokenizer.from_pretrained(clf_tok, padding='max_length', truncation='max_length', max_length=512)
#     #ic(next(clfModel.parameters()).device)
#     print('Loaded classification model')
    
#     torch.cuda.empty_cache()

#     print("Loading Section-Title Model")
#     titlemodel = AutoModelForSequenceClassification.from_pretrained(
#         sectitle_model_path,
#         num_labels=2
#     ).to(sectitle_device)
#     titletok = AutoTokenizer.from_pretrained(sectitle_tok)
#     titlemodel.eval()
#     print("Loaded Section-Title Model")

#     print("Loading IndicNER Model")
#     nermodel = AutoModelForTokenClassification.from_pretrained(
#         ner_model_path
#     ).to(ner_device)
#     nertok = AutoTokenizer.from_pretrained(ner_tok)
#     nermodel.eval()
#     print("Loaded IndicNER Model")

#     print("Loading Foreign NER Model")
#     nerfmodel = AutoModelForTokenClassification.from_pretrained(
#         ner_f_model_path
#     )
#     nerftok = AutoTokenizer.from_pretrained(ner_f_tok)
#     nerfmodel.eval()
#     nerf_pipeline = pipeline("ner", model=nerfmodel, tokenizer=nerftok, device=ner_f_device)
#     print("Loaded Foreign NER Model")

    if(local_rank == 0):
        if wandb_log == 0:
            wandb.init(
                mode='disabled'
            )
            print("Wandb logging disabled")
        else:
            wandb.init(
            project='RL-baselines-new',
            config={
                'learning_rate': lr,
                'epochs': num_epochs,
                'batch_size': train_batch_size
            },
        )
            print("Wandb logging enabled")
            wandb.run.name = EXP_NAME
            wandb.run.save()

    model_device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    #ic(model_device)
    if is_mt5:
        optimizer = Adafactor(model.parameters(), lr=lr, weight_decay=wd, relative_step=False)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay = wd)

    #ic(model_gpus)
    model.to(model_device)
    model = DDP(model, device_ids=[local_rank])
    #ic(f" on device {local_rank} model device {next(model.parameters()).device} but {torch.cuda.is_available()} and model_device_var {model_device}")



    ic(start_epoch, final_checkpoint)
    lowest, vallosses, count = [float('inf'), float('inf')], {}, 0
    
    pbar = tqdm(range(start_epoch, num_epochs))
    for epoch in pbar:
        train_loader.sampler.set_epoch(epoch)
        pbar.set_postfix(loss=local_rank)
        avg_train_loss = []
        avg_val_loss = []

        dist.barrier()  
        pbar2 = tqdm(train_loader)
        for batch in pbar2:
            pbar2.set_postfix(model_number=local_rank)
            batch['input_ids'] = batch['input_ids'].to(model_device)
            batch['attention_mask'] = batch['attention_mask'].to(model_device)
            batch['labels'] = batch['labels'].to(model_device)
            batch['tgt_mask'] = batch['tgt_mask'].to(model_device)
            
            #outputs = model(batch)
            middle_output = model.module.middle(batch)
            main_loss, logits, input_text, pred_text, gold_text = middle_output['main_loss'], middle_output['logits'], middle_output['input_text'], middle_output['pred_text'], middle_output['gold_text']

#             reward_loss = calcTestReward(
#                 batch=batch,
#                 logits=logits,
#                 input_text=input_text,
#                 pred_text=pred_text,
#                 ref_text=gold_text,
#                 mask = batch['tgt_mask'],
#                 emb_device = emb_device,
#                 embmodel = embeddingModel,
#                 embtok = embeddingTok,
#                 clf_device = clf_device,
#                 clf_model = clfModel,
#                 clf_tok = clfTok,
#                 model_device=model_device
#             )

        

#             reward_loss = calcReward(
#                 batch=batch,
#                 logits=logits,
#                 foreign=foreign,
#                 input_text=input_text,
#                 pred_text=pred_text,
#                 titlemodel=titlemodel,
#                 titletok=titletok,
#                 titledevice=sectitle_device,
#                 nertok=nertok,
#                 nermodel=nermodel,
#                 nerdevice=ner_device,
#                 nerf_pipeline=nerf_pipeline
#             )

#             embloss, unchanged_loss, klloss = reward_loss['embloss'], reward_loss['unchanged_loss'], reward_loss['klloss']
#             main_coeff = 1 - (emb_coeff + unchanged_coeff + clf_coeff)
#             total_loss = (main_coeff*main_loss) + (emb_coeff*embloss) + (unchanged_coeff*unchanged_loss) + (clf_coeff*klloss)
            total_loss = 1.0*main_loss
       

            avg_train_loss.append(total_loss.item())
            total_loss.backward()

            if len(avg_train_loss) == 10 and isTrial:
                break

#             if len(avg_train_loss) == len(train_loader)//2:
#                 if not os.path.exists(save_dir):
#                     os.makedirs(save_dir)
#                 save_path = f'{save_dir}/{epoch}_half.pt'
#                 torch.save(model.state_dict(), save_path)


            optimizer.step()
            optimizer.zero_grad()
            
#         if(local_rank == 0):
#             ic(f"Model number {local_rank} saving")
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             save_path = f'{save_dir}/{epoch}.pt'
#             torch.save(model.state_dict(), save_path)
            
        dist.barrier()

        for batch in tqdm(val_loader):
            batch['input_ids'] = batch['input_ids'].to(model_device)
            batch['attention_mask'] = batch['attention_mask'].to(model_device)
            batch['labels'] = batch['labels'].to(model_device)
            with torch.no_grad():
                #outputs = model(batch)
                middle_output = model.module.middle(batch)
            loss = middle_output['main_loss']
            avg_val_loss.append(loss.item())

            if len(avg_val_loss) == 10 and isTrial:
                break

        valloss = sum(avg_val_loss)/len(avg_val_loss)
        trainloss = sum(avg_train_loss)/len(avg_train_loss)
        vallosses[valloss] = epoch

        def save():
            if(local_rank == 0):
                ic(f"Model number {local_rank} saving")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = f'{save_dir}/{epoch}.pt'
                torch.save(model.state_dict(), save_path)

        if(local_rank == 0):
            if(lowest[0] > valloss) or (lowest[1] > valloss):
                if(lowest[1] > lowest[0]):
                    if count > 1:
                        del_ckpt = vallosses[lowest[1]]
                    lowest[1] = valloss
                else:
                    if count > 1:
                        del_ckpt = vallosses[lowest[0]]
                    lowest[0] = valloss
                save()
                if count > 1:
                    ic(del_ckpt)
                    if os.path.exists(f'{save_dir}/{del_ckpt}.pt'):
                        os.remove(f'{save_dir}/{del_ckpt}.pt')

            count += 1
        
        if(local_rank == 0) and wandb_log == 1:
            wandb.log({
                'val_loss': valloss,
                'train_loss': trainloss,
                'epoch': epoch
            })
            
            
        flag_tensor = torch.zeros(1).to(model_device)
        if local_rank == 0:
            if es.step(valloss):
                flag_tensor += 1
        dist.all_reduce(flag_tensor,op=ReduceOp.SUM)
        if flag_tensor == 1:
            break # early stop criterion is met, we can stop now
       
    cleanup()

if __name__ == '__main__':
    
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=10000))
    #dist.init_process_group("nccl")
    parser = argparse.ArgumentParser(description='Input parameters for extractive stage')
    parser.add_argument('--train_path', help='path to input json file for a given domain in given language')
    parser.add_argument('--val_path', help='path to intermediate output json file for a given domain in given language')
    parser.add_argument('--test_path', help='path to output json file for a given domain in given language')
    parser.add_argument('--config', default=None, help='which config file to use')
    parser.add_argument('--tokenizer', default='facebook/mbart-large-50', help='which tokenizer to use')
    parser.add_argument('--model', default='facebook/mbart-large-50', help='which model to use')
    parser.add_argument('--is_mt5', type=int, help='is the model mt5')
    parser.add_argument('--exp_name', help='experiment name')
    parser.add_argument('--save_dir', default='checkpoints/', help='where to save the logs and checkpoints')
    parser.add_argument('--lr', default=1e-3, help='learning rate for main model') #1e-3 for non RL and lesser for RL
    parser.add_argument('--wd', default=0.01, help='weight decay for main model') #0.01 for non RL
    parser.add_argument('--num_epochs', default=5, type=int, help='number of epochs')
    parser.add_argument('--train_batch_size', default=4, type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=4, type=int, help='val batch size')
    parser.add_argument('--test_batch_size', default=4, type=int, help='test batch size')
    parser.add_argument('--max_source_length', default=250, type=int, help='max source length')
    parser.add_argument('--max_target_length', default=250, type=int, help='max target length')
    # parser.add_argument('--model_device', default='cuda:0', type=str, help='device to run the main generation model on')
    parser.add_argument('--model_gpus', default='0', type=str, help='multiple gpus on which main model will be loaded')
    parser.add_argument('--ner_device', default='cuda:1', type=str, help='device to load IndicNER on')
    parser.add_argument('--ner_model_path', default='ai4bharat/IndicNER', type=str, help='path to the NER model checkpoint')
    parser.add_argument('--ner_tok', default='ai4bharat/IndicNER', type=str, help='tokenizer for NER model')
    parser.add_argument('--ner_f_device', default=3, type=int, help='device to load Foreign NER on')
    parser.add_argument('--ner_f_model_path', default='Babelscape/wikineural-multilingual-ner', type=str, help='path to the NER model checkpoint')
    parser.add_argument('--ner_f_tok', default='Babelscape/wikineural-multilingual-ner', type=str, help='tokenizer for NER model')
    parser.add_argument('--emb_device', default='cuda:3', type=str, help='device to load embedding model on')
    parser.add_argument('--emb_model_path', default='sentence-transformers/distiluse-base-multilingual-cased-v2 ', type=str, help='path to the embedding model')
    parser.add_argument('--emb_tok', default='sentence-transformers/distiluse-base-multilingual-cased-v2 ', type=str, help='tokenizer for embedding model')
    parser.add_argument('--clf_device', default='cuda:3', type=str, help='device to load classification model on')
    parser.add_argument('--clf_model_path', default='google/muril-base-cased', type=str, help='path to the classificaton model checkpoint')
    parser.add_argument('--clf_tok', default='google/muril-base-cased', type=str, help='tokenizer for classification model')
    parser.add_argument('--isTest', default=0, type=int, help='test run')
    parser.add_argument('--isTrial', default=0, type=int, help='toy run')
    parser.add_argument('--world_size', default=4, type=int, help="world size")
    parser.add_argument('--multilingual', help="True/False")
    parser.add_argument('--language', default=' ', help='language code needed only during inference e.g. <2bn> for IndicBART or Bengali for mT5')
    parser.add_argument('--wandb', default=1, type=int, help="1 or 0 based on wandb on or off")
    parser.add_argument('--emb_coeff', default=0.1, type=float, help="coefficient for the embedding reward")
    parser.add_argument('--unchanged_coeff', default=0.1, type=float, help="coefficient for the unchanged percentage reward")
    parser.add_argument('--clf_coeff', default=0.1, type=float, help="coefficient for the classifier reward")
    args = parser.parse_args()

    main(args)
