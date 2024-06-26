import os,sys
import random
import json
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score,precision_score,recall_score,matthews_corrcoef,accuracy_score

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
os.environ['WANDB_API_KEY'] = '599b2c499431b83b20ee24cdf7feb9dd42df2e92'
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

from logger import MyLogger, LOG_LEVELS
from dataloader import get_dataset_loaders
from utils import handle_multiple_languages, merge_dataset_across_languages, dataset_exists

base_dir = os.path.dirname(os.path.realpath(__file__))


# allow deterministic psuedo-random-initialization
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

class ModelWrapper(pl.LightningModule):
    def __init__(self, args):
        super(ModelWrapper, self).__init__()
        self.config_args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/tmp/huggingface_new")
        if args.use_pretrained:
            # using pretrained transformers
            self.model = AutoModel.from_pretrained(args.model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False, cache_dir="/tmp/hugginface_new")
        else:
            # training transformer from scratch
            self.model = AutoModel.from_config(AutoModel.from_pretrained(
                args.model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False))
        self.task_head = TexClassificationHead(self.model.config.hidden_size, 2, args.dropout_rate)
        #metrics
        self.train_metric = pl.metrics.Accuracy()
        self.val_metric = pl.metrics.Accuracy()
        self.test_metric = [pl.metrics.Accuracy()] #] pl.metrics.Accuracy()
        self.predict_labels=[]
        self.labels=[]
        self.output_file_name =args.output_file_path

    def forward(self, input_ids, attention_mask):
        # loading to cuda devices
        # input_seq = input_seq.to(self.transformer.device)
        # attention_mask = attention_mask.to(self.transformer.device)
        # calculating the output logits
        doc_rep = self.model(input_ids, attention_mask=attention_mask)[0]
        output_logits = self.task_head(doc_rep)
        return output_logits
        
    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config_args.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config_args.learning_rate)
        # optimizer = Adafactor(optimizer_grouped_parameters, lr=self.config_args.learning_rate, 
        #                                                   scale_parameter=False, relative_step=False, warmup_init=False)
        
        # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.config_args.learning_rate, eps=1e-6)

        if self.config_args.enable_scheduler:
            total_dataset_count = self.config_args.train_dataset_count
            total_steps = int(np.ceil((self.config_args.epochs * total_dataset_count) /
                              (self.config_args.batch_size*self.config_args.gpus)))

            scheduler = {
                # 'scheduler': get_constant_schedule_with_warmup(optimizer, self.config_args.warmup_steps*total_steps)
                'scheduler': get_linear_schedule_with_warmup(optimizer, self.config_args.warmup_steps*total_steps, total_steps),
                'interval': 'step',
            }
            return [optimizer], [scheduler]

        return optimizer

    def _step(self, batch, step_type):
        if step_type == 'train':
            step_metric = self.train_metric
        elif step_type == 'val':
            step_metric = self.val_metric
        else:
            step_metric = self.test_metric
        
        input_ids, attention_mask = batch[:-1]
        model_output = self(input_ids, attention_mask)
        
        if step_type!='test':
            label_ids = batch[-1]

        return_map = {}
        online_logger_data = {}
        pbar = {}
        if step_type!='test':
            task_loss = F.cross_entropy(model_output, label_ids.long())
            acc = step_metric(model_output.softmax(dim=-1), label_ids.long())
            if step_type == 'val':
                return_map['val_loss'] = task_loss
                return_map['val_acc'] = acc 
            else:
                return_map['loss'] = task_loss
                pbar['acc'] = acc
        if step_type=='test':

            batch_input_sentences = self.tokenizer.batch_decode(batch[0],skip_special_tokens=True)
            label_ids = batch[-1]
            task_loss = F.cross_entropy(model_output, label_ids.long())
            acc = step_metric[0](model_output.softmax(dim=-1), label_ids.long())
            return_map['test_loss'] = task_loss
            return_map['test_acc'] = acc 
            return_map['predicted_logits']=model_output.softmax(dim=-1)
            logit_0=model_output.softmax(dim=-1)[:,0]
            logit_1=model_output.softmax(dim=-1)[:,1]
            self.predict_labels.append(model_output.softmax(dim=-1).argmax(dim=-1).numpy())
            self.labels.append(label_ids.numpy())
            with open(self.output_file_name,'a',encoding='utf-8') as file_handler:
                for idx in range(len(batch_input_sentences)):
                    file_handler.write("sentence - %s,predict_label - %s, true_label - %s, logit_0 - %s, logit_1 - %s \n"%(
                        batch_input_sentences[idx],self.predict_labels[-1][idx],self.labels[-1][idx],logit_0[idx].item()
                        ,logit_1[idx].item()))
            # updating the online logger
            # online_logger_data.update(pbar)
            # online_logger_data.update(return_map)
            # self.logger.log_metrics(online_logger_data)

            if len(pbar):
                return_map['progress_bar'] = pbar
        
        return return_map

    def _epoch_end(self, step_outputs, end_type):
        if end_type == 'train':
            end_metric = self.train_metric
        elif end_type == 'test':
            end_metric = self.test_metric
        else:
            end_metric = self.val_metric
        
        loss_label = 'loss'
        if end_type == 'val':
            loss_label = 'val_loss'
        elif end_type == 'test':
            loss_label = 'test_loss'

        if end_type!='test':
            avg_loss = torch.stack([x[loss_label] for x in step_outputs]).mean()
            overall_acc = end_metric.compute()
            self.config_args.logger.info('epoch : %d - average_%s_loss : %f, overall_%s_acc : %f' % (self.current_epoch, end_type, avg_loss.item(),
                                                                                                            end_type, overall_acc.item()))
            # logging to weight and bias if online mode is enabled
            self.logger.log_metrics(
                {'avg_%s_loss' % end_type: avg_loss, 'overall_%s_acc' % end_type: overall_acc})
            self.log('avg_%s_loss' % end_type, avg_loss, prog_bar=True)
            self.log('overall_%s_acc' % end_type, overall_acc, prog_bar=True)
        if end_type == 'test':
            # print("In test epoch:"*30 , type(step_outputs[0]),"  ",(step_outputs[0]))
            # print("IN test epoch"*30 , len(self.labels[0]),len(self.predict_labels[0]))
            y = np.concatenate(self.labels)
            y_hat = np.concatenate(self.predict_labels)
            tp=0
            fp=0
            tn=0
            fn=0
            for i in range(y.shape[0]):
                if y[i]==1 and y_hat[i]==1:
                    tp+=1
                elif y[i]==0 and y_hat[i]==0:
                    tn+=1
                elif y[i]==1 and y_hat[i]==0:
                    fn+=1
                else:
                    fp+=1
            print("LETS DOTHIS: %d %d %d %d "% (tp,tn,fp,fn))
                
            print(y.shape, y_hat.shape)
            macro_f1 = f1_score(y, y_hat, average='macro')
            micro_f1 = f1_score(y, y_hat, average='micro')
            macro_precision = precision_score(y, y_hat, average='macro')
            micro_precision = precision_score(y, y_hat, average='micro')
            macro_recall = recall_score(y, y_hat, average='macro')
            micro_recall = recall_score(y, y_hat, average='micro')
            mcc_score = matthews_corrcoef(y,y_hat)
            accuracy_score_sklearn = accuracy_score(y,y_hat)
            print("accuracy: %f" %accuracy_score_sklearn)
            print()
            print("macro_precision: %f"%macro_precision)
            print()
            print("macro_recall: %f"%macro_recall)
            print()
            print("macro_f1: %f"%macro_f1)
            print()
            print("matthews correlation: %f"%mcc_score)
            print()
            # print("macro_f1: %f , micro_f1: %f , macro_recall: %f , micro_recall: %f , macro_precision: %f , micro_precision: %f , mcc_score: %f , accuracy_sklearn_score: %f"%(macro_f1,micro_f1,macro_recall,micro_recall,macro_precision,micro_precision,mcc_score,accuracy_score_sklearn))

            avg_loss = torch.stack([x[loss_label] for x in step_outputs]).mean()
            overall_acc = end_metric[0].compute()
            self.config_args.logger.info('epoch : %d - average_precalculated_%s_loss : %f, overall_%s_acc : %f ,f1_score_macro_sk %f , f1_score_micro_sk %f ,recall_score_macro_sk %f , recall_score_micro_sk %f , precision_score_macro_sk %f , precision_score_micro_sk %f , mcc_score %f, 0/1 accuracy %f'% (self.current_epoch, 
                                                                                                                                                                                                                                                                                                                            end_type, avg_loss.item(),
                                                                                                            end_type, overall_acc.item(),macro_f1,micro_f1,macro_recall,micro_recall,macro_precision,micro_precision,mcc_score,accuracy_score_sklearn))    
            # logging to weight and bias if online mode is enabled
            self.logger.log_metrics(
                {'avg_%s_loss' % end_type: avg_loss, 'overall_micro_%s_acc' % end_type: accuracy_score_sklearn, 'overall_macro_%s_f1' % end_type: macro_f1, 'overall_micro_%s_f1' % end_type: micro_f1})
            self.log('overall_%s_acc' % end_type, accuracy_score_sklearn, prog_bar=True)
            self.log('avg_%s_loss' % end_type, avg_loss, prog_bar=True)
            self.log('overall_macro_%s_f1' % end_type, macro_f1, prog_bar=True)
            self.log('overall_macro_%s_precision' % end_type, macro_precision, prog_bar=True)
            self.log('overall_micro_%s_recall' % end_type, micro_recall, prog_bar=True)
            self.log('overall_macro_%s_recall' % end_type, macro_recall, prog_bar=True)
            self.log('overall_micro_%s_f1' % end_type, micro_f1, prog_bar=True)
            self.log('overall_micro_%s_precision' % end_type, micro_precision, prog_bar=True)


    def training_step(self, batch, batch_idx):

        return self._step(batch, 'train')

    def training_epoch_end(self, train_step_outputs):
        self._epoch_end(train_step_outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def validation_epoch_end(self, val_step_outputs):
        self._epoch_end(val_step_outputs, 'val')
    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')
    def test_epoch_end(self, test_step_outputs):
        # print(test_step_outputs)
        self._epoch_end(test_step_outputs, 'test')


class TextDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.args = args

    def _intiate_dataset_merging(self, dataset_type, dataset_dir, languages, logger):
        logger.critical('%s: merging the %d %s different languages dataset' % (dataset_type, len(languages), languages))
        final_dir_path = os.path.join(os.path.abspath(dataset_dir), '-'.join(languages))
        os.makedirs(final_dir_path, exist_ok=True)
        # check if dataset already exists
        if dataset_exists(final_dir_path, required_files=['%s.jsonl' % dataset_type]):
            logger.info("%s dataset is already present." % dataset_type)
        else:
            merge_dataset_across_languages(dataset_dir, languages, dataset_type, os.path.join(final_dir_path, "%s.jsonl" % dataset_type))
        return final_dir_path

    def val_dataloader(self):
        if len(self.args.lang)==1:
            dev_file_path = os.path.join(os.path.abspath(self.args.dataset_path), self.args.lang[0], 'val.jsonl')
        else:
            merged_directory = self._intiate_dataset_merging('val', self.args.dataset_path, self.args.lang, 
                                                                            self.args.logger)
            dev_file_path = os.path.join(os.path.abspath(merged_directory), 'val.jsonl')
        
        val_dataset = get_dataset_loaders(self.tokenizer, dev_file_path, self.args.logger, dataset_count=self.args.val_dataset_count, 
                                          batch_size=self.args.batch_size, max_seq_len=self.args.max_seq_len)
        return val_dataset

    def test_dataloader(self):
        if len(self.args.lang)==1:
            test_file_path = os.path.join(os.path.abspath(self.args.dataset_path), self.args.lang[0], 'test.jsonl')
        else:
            merged_directory = self._intiate_dataset_merging('test', self.args.dataset_path, self.args.lang, 
                                                                            self.args.logger)
            test_file_path = os.path.join(os.path.abspath(merged_directory), 'test.jsonl')
        
        test_dataset = get_dataset_loaders(self.tokenizer, test_file_path, self.args.logger, dataset_count=self.args.test_dataset_count, 
                                          batch_size=self.args.batch_size, max_seq_len=self.args.max_seq_len)
        return test_dataset

    def train_dataloader(self):

        if len(self.args.lang)==1:
            train_file_path = os.path.join(os.path.abspath(self.args.dataset_path), self.args.lang[0], 'train.jsonl')
        else:
            merged_directory = self._intiate_dataset_merging('train', self.args.dataset_path, self.args.lang, 
                                                                            self.args.logger)
            train_file_path = os.path.join(os.path.abspath(merged_directory), 'train.jsonl')

        train_dataset = get_dataset_loaders(self.tokenizer, train_file_path, self.args.logger, dataset_count=self.args.train_dataset_count,
                                            batch_size=self.args.batch_size,  max_seq_len=self.args.max_seq_len)
        return train_dataset
    

def get_checkpoint_file(checkpoint_path, logger):
    file_list = []
    for file_name in os.listdir(checkpoint_path):
        if not file_name.endswith('ckpt'):
            continue
        last_modified_time = os.path.getmtime(
            os.path.join(checkpoint_path, file_name))
        file_list.append([file_name, last_modified_time])

    logger.info(
        'total number of files within checkpoint directory [%s]: %d' % (checkpoint_path, len(file_list)))
    if len(file_list) == 0:
        return False, ""
    # if multiple files exists then choose the last modified checkpoint path
    file_list = sorted(file_list, key=lambda x: x[1], reverse=True)
    return True, os.path.join(checkpoint_path, file_list[0][0])

def start_training(args):
    model_name = args.logger_exp_name

    args.logger.debug('initiating training process...')

    final_checkpoint_path = os.path.join(args.checkpoint_path, model_name)
    os.makedirs(final_checkpoint_path, exist_ok=True)

    # Load datasets
    dm = TextDataModule(args)

    call_back_parameters = {
        'filepath': final_checkpoint_path,
        'save_top_k': 1,
        'verbose': True,
        'monitor': 'overall_val_acc',
        'mode': 'max',
    }

    # checkpoint callback to used by the Trainer
    checkpoint_callback = ModelCheckpoint(**call_back_parameters)

    early_stop_callback = EarlyStopping(
        monitor='overall_val_acc',
        patience=args.patience,
        verbose=True,
        mode='max',
    )
    model = ModelWrapper(args)

    args.logger.debug(model)
    args.logger.info('Model has %d trainable parameters' %
                     count_parameters(model))

    callback_list = [checkpoint_callback, early_stop_callback]

    precision_val = 16 if args.fp16 > 0 else 32

    global_callback_params = {
        "callbacks": callback_list,
        "max_epochs": args.epochs,
        "min_epochs": 1,
        "gradient_clip_val": args.clip_grad_norm,
        "gpus": args.gpus,
        "distributed_backend": "ddp",
        "logger": args.online_logger,
        "precision": precision_val,
        "plugins": "deepspeed_stage_2",
    }

    #checking whether checkpoint already exists or not
    checkpoint_exists, checkpoint_file = get_checkpoint_file(final_checkpoint_path, args.logger)
    if checkpoint_exists:
        global_callback_params.update({'resume_from_checkpoint': checkpoint_file})
        args.logger.info('resuming training from checkpoint : %s' % checkpoint_file)

    trainer = pl.Trainer(**global_callback_params)
    # finally train the model
    args.logger.debug('about to start training loop...')
    trainer.fit(model, dm)
    if args.online_mode > 0:
        _, checkpoint_file = get_checkpoint_file(final_checkpoint_path, args.logger)
        # args.online_logger.experiment.save(checkpoint_file, base_path=final_checkpoint_path)
    args.logger.debug('training done.')


if __name__ == "__main__":
    parser = ArgumentParser()

    default_checkpoint_path = os.path.join(base_dir, 'lightning_checkpoints')
    
    # Global model configuration
    parser.add_argument('--checkpoint_path', default=default_checkpoint_path, type=str,
                        help='directory where checkpoints are stored')
    parser.add_argument('--dataset_path', required=True, type=str,
                        help='directory where dataset exits')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='specify the learning rate')
    parser.add_argument('--clip_grad_norm', default=0.0, type=float,
                        help='clip gradients with norm above specified value, 0 value will disable it.')
    parser.add_argument('--weight_decay', default=0.001, type=float,
                        help='specify the weight decay.')
    parser.add_argument('--dropout_rate', default=0.1, type=float,
                        help='specify the dropout rate for all layer, also applies to all transformer layers.')
    parser.add_argument('--patience', default=0, type=int,
                        help='specify patience for early stop algorithm. if its 0 then disable this feature.')
    # parser.add_argument('--seed', default=42, type=int,
    # help='seed value for random initialization.')
    parser.add_argument("--enable_scheduler", action='store_true',
                        help='activates the linear decay scheduler.')
    parser.add_argument("--warmup_steps", default=0.01, type=float,
                        help="percentage of total step used as linear warmup while training the model.")
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help="specify the maximum sequence length for processed dataset.")
    # below three arguments are for debugging purpose
    parser.add_argument("--train_dataset_count", type=int, default=0,
                        help="specify number of training data to use. (for debugging purpose). If zero then takes all the available dataset.")
    parser.add_argument("--val_dataset_count", type=int, default=0,
                        help="specify number of validation data to use. (for debugging purpose). If zero then takes all the available dataset.")
    parser.add_argument("--test_dataset_count", type=int, default=0,
                        help="specify number of test data to use. (for debugging purpose). If zero then takes all the available dataset.")
    parser.add_argument("--output_file_path", type=str, default="model_outputs.txt",
                        help="specify File path where you want to store your outputs in case of test (for debugging purpose).")
    # logger configs
    
    parser.add_argument('--online_mode', default=0, type=int,
                        help='disables weight and bias syncronization if 0 is passed')
    # if "combined" architecture is active, we can disable mtl (on textual entailment data) associated with it
    parser.add_argument('--model_name', type=str, default='microsoft/infoxlm-base',
                        help='specify pretrained transformer model to use.')
    parser.add_argument('--use_pretrained', type=int, default=1,
                        help='loads pretrained transformer model.')
    # GPU memory utilization optimizations
    parser.add_argument('--fp16', type=int, default=1,
                        help='enable the automatic mixed precision training')
    parser.add_argument('--lang', type=str, required=True, 
                        help='specify the target language iso code. Mutliple languages supported if their iso codes are separated using ",".')
    
    args = parser.parse_args()

    args.lang = handle_multiple_languages(args.lang)
    if(len(args.lang)==0):
        print('Invalid language(s) specified !!!')
        sys.exit(0)
    datasetname=args.dataset_path.split("_")[5]
    args.logger_exp_name = "%s-%s-%s-%s-%s-%s-%s-%s" % ('-'.join(args.lang), args.model_name, args.epochs, args.learning_rate,args.batch_size,datasetname,args.dropout_rate,args.weight_decay)
    args.logger_exp_name = args.logger_exp_name.replace('/', '-')

    # offline logger
    args.logger = MyLogger('', os.path.join(base_dir, "%s.log" % args.logger_exp_name),
                           use_stdout=True, log_level=LOG_LEVELS.DEBUG, overwrite=True)

    # get the arguments passed to this program
    params = {}
    for arg in vars(args):
        if arg in ["online_logger", "logger"]:
            continue
        params[arg] = getattr(args, arg)

    logger_args = {
        'project': args.lang[0]+'-WNC-INFOXLM',    # first create a project on weight & bias with local account
        'name': args.logger_exp_name,
        'config': params,
        'tags': ['pytorch-lightning'],
        'log_model': False
    }

    # turn off the online sync
    if args.online_mode == 0:
        logger_args.update({'offline': True}),

    # configure and add logger to arguments
    args.online_logger = WandbLogger(**logger_args)

    # get the arguments passed to this program
    args.logger.info('\ncommand line argument captured ..')
    args.logger.info('--'*30)

    for key, value in params.items():
        args.logger.info('%s - %s' % (key, value))
    args.logger.info('--'*30)
    start_training(args)
    # model = ModelWrapper(args)
    # # # ckpt_path = get_checkpoint_file(args.logger_exp_name,args.logger)[1]
    # ckpt_path="/scratch/user/bn-en-gu-hi-kn-mr-ta-te-google-muril-base-cased-6-1e-06/epoch=5.ckpt"
    # # print(type(ckpt_path), ckpt_path)
    # checkpoint = torch.load(ckpt_path)
    # model.load_state_dict(checkpoint["state_dict"])
    # global_callback_params = {

    #     "gradient_clip_val": args.clip_grad_norm,
    #     "gpus": args.gpus,  
    #     "distributed_backend": "ddp",
    #     "logger": args.online_logger,
    # }
    # trainer = pl.Trainer()
    # dm = TextDataModule(args)
    # results = trainer.test(model=model, datamodule=dm, verbose=True)


    # Command to run --> 
