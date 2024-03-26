import logging
import argparse
import math
import os
import sys
import random
import numpy
import csv
from sklearn import metrics
from time import strftime, localtime

from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, \
                         AlbertTokenizer, AlbertConfig, AlbertModel,AdamW, \
                         AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN
from models.bert_spc import BERT_SPC
from models.bert_rn import BERT_RN
from transformers import logging as tlog
 
tlog.set_verbosity_warning()
tlog.set_verbosity_error()

BASEPATH = ''

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        if '-' in self.opt.dataset and self.opt.dataset[-1] != '-':
            log_path = f'../log/{self.opt.model_name}/cross/'
        elif self.opt.dataset[-1] == '-' or 'vast' in self.opt.dataset:
            log_path = f'../log/{self.opt.model_name}/zero/'
        else:
            log_path = f'../log/{self.opt.model_name}/in-domain/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger = get_logger(f'{log_path}{self.opt.dataset}_{self.opt.input_type}_{self.opt.add_num}_{self.opt.label_ratio}%_{self.opt.FAD_ratio}%_{self.opt.RAD_ratio}%.log',name='normal')
        self.logger = logger
        best_logger = get_logger(f'{log_path}best_{self.opt.dataset}_{self.opt.input_type}_{self.opt.add_num}_{self.opt.label_ratio}%_{self.opt.FAD_ratio}%_{self.opt.RAD_ratio}%.log',name='best')
        self.best_logger = best_logger
        
        if 'bert' in opt.model_name:
            bert = AutoModel.from_pretrained(opt.pretrained_name)
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_name)
            if opt.model_name == 'bert_rn':
                bert2 = AutoModel.from_pretrained(opt.pretrained_name) 
                self.model = opt.model_class(bert,bert2, opt).to(opt.device)
            else:
                self.model = opt.model_class(bert, opt).to(opt.device)
        elif 'bart' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_name)
            self.model = opt.model_class(opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test'], opt.dataset_file['front_add'], opt.dataset_file['reverse_add']],
                max_seq_len=opt.max_seq_len,
                dat_fname='../dat/{0}_{1}_tokenizer.dat'.format(opt.dataset,opt.add_num),
                add_num = opt.add_num)
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='../dat/{0}_{1}_{2}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset,opt.add_num))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer,self.opt)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer,self.opt,tt_model='test')
        if 'vast' in opt.dataset:
            self.valset = ABSADataset(opt.dataset_file['val'], tokenizer,self.opt,tt_model='test')
        else:
            assert 0 <= opt.valset_ratio < 1
            if opt.valset_ratio > 0:
                valset_len = int(len(self.trainset) * opt.valset_ratio)
                self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
            else:
                self.valset = self.testset
        self.frontaddset = ABSADataset(opt.dataset_file['front_add'], tokenizer,self.opt,tt_model='add',ratio=self.opt.FAD_ratio)
        self.reverseaddset = ABSADataset(opt.dataset_file['reverse_add'], tokenizer,self.opt,tt_model='add',ratio=self.opt.RAD_ratio)
        self.trainset = ConcatDataset([self.trainset,self.frontaddset,self.reverseaddset])
        # self.trainset = ConcatDataset([self.trainset,self.frontaddset])
        print(len(self.trainset),len(self.testset),len(self.valset))


        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in vars(self.opt):
            self.logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        if 'bart' not in self.opt.model_name:
            for child in self.model.children():
                if type(child) != BertModel and type(child) != RobertaModel and type(child) != AutoModel:  # skip bert params !!
                    for p in child.parameters():
                        if p.requires_grad:
                            if len(p.shape) > 1:
                                self.opt.initializer(p)
                            else:
                                stdv = 1. / math.sqrt(p.shape[0])
                                torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        else:
            for n, p in self.model.named_parameters():
                if 'bart' not in n:
                    if p.requires_grad:
                            if len(p.shape) > 1:
                                self.opt.initializer(p)
                            else:
                                stdv = 1. / math.sqrt(p.shape[0])
                                torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader,test_data_loader,bert_optimizer = None):
        best_val = []
        best_test = []
        max_maf1a = 0
        max_val_epoch = 0
        # global_step = 0
        path = None
        if self.opt.log_step == -1:
            self.opt.log_step = len(train_data_loader) // 3
        ii_batch = 0
        for i_epoch in range(self.opt.num_epoch):
            self.logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            for i_batch, batch in enumerate(train_data_loader):
                self.model.train()
                # global_step += 1
                optimizer.zero_grad()
                if bert_optimizer != None:
                    bert_optimizer.zero_grad()
                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if bert_optimizer != None:
                    bert_optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                ii_batch += 1
                if ii_batch % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    self.logger.info(f'{ii_batch}/{len(train_data_loader)}\ttrain_loss: {train_loss}\ttrain_acc: {round(train_acc * 100,2)}')
                    val_acc, val_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a = self._evaluate_acc_f1(val_data_loader)
                    self.logger.info(f'Val: ma_f1: {round(val_f1*100,2)}\tacc: {round(val_acc*100,2)}\tavg_f1: {round(avg_f1*100,2)}\tma_all_f1: {round(maf1a*100,2)}\tmi_all_f1: {round(mif1a*100,2)}\tfavor_f1: {round(f_f1*100,2)}\tagainst_f1: {round(a_f1*100,2)}\tnone_f1: {round(n_f1*100,2)}')
                    if 'vast' in self.opt.dataset:
                        sign_fa = maf1a
                    else:
                        sign_fa = val_f1
                    if sign_fa > max_maf1a:
                        best_val = [val_acc,val_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a,i_epoch]
                        max_maf1a = sign_fa
                        max_val_epoch = i_epoch
                        test_acc, test_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a = self._evaluate_acc_f1(test_data_loader)
                        best_test = [test_acc, test_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a,i_epoch]
                        self.logger.info(f'Test: ma_f1: {round(test_f1*100,2)}\tacc: {round(test_acc*100,2)}\tavg_f1: {round(avg_f1*100,2)}\tma_all_f1: {round(maf1a*100,2)}\tmi_all_f1: {round(mif1a*100,2)}\tfavor_f1: {round(f_f1*100,2)}\tagainst_f1: {round(a_f1*100,2)}\tnone_f1: {round(n_f1*100,2)}')
                        if self.opt.save_model:
                            path = f'../state_dict/edda_{self.opt.model_name}_{self.opt.dataset}_{self.opt.input_type}_{self.opt.label_ratio}_{self.opt.FAD_ratio}_{self.opt.RAD_ratio}'
                            torch.save(self.model.state_dict(), path)
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break
        self.logger.info(f'Best_test: epoch:{best_test[-1]}\tma_f1: {round(best_test[1]*100,2)}\tacc: {round(best_test[0]*100,2)}\tavg_f1: {round(best_test[2]*100,2)}\tma_all_f1: {round(best_test[6]*100,2)}\tmi_all_f1: {round(best_test[7]*100,2)}\tfavor_f1: {round(best_test[3]*100,2)}\tagainst_f1: {round(best_test[4]*100,2)}\tnone_f1: {round(best_test[5]*100,2)}')
        self.best_logger.info(f'Best_test: epoch:{best_test[-1]}\tma_f1: {round(best_test[1]*100,2)}\tacc: {round(best_test[0]*100,2)}\tavg_f1: {round(best_test[2]*100,2)}\tma_all_f1: {round(best_test[6]*100,2)}\tmi_all_f1: {round(best_test[7]*100,2)}\tfavor_f1: {round(best_test[3]*100,2)}\tagainst_f1: {round(best_test[4]*100,2)}\tnone_f1: {round(best_test[5]*100,2)}')
        self.logger.info(f'Best_val: epoch:{best_val[-1]}\tma_f1: {round(best_val[1]*100,2)}\tacc: {round(best_val[0]*100,2)}\tavg_f1: {round(best_val[2]*100,2)}\tma_all_f1: {round(best_val[6]*100,2)}\tmi_all_f1: {round(best_val[7]*100,2)}\tfavor_f1: {round(best_val[3]*100,2)}\tagainst_f1: {round(best_val[4]*100,2)}\tnone_f1: {round(best_val[5]*100,2)}')
# =====  
        if self.opt.save_model:
            torch.save(self.model.state_dict(), f'../state_dict/edda_last_{self.opt.model_name}_{self.opt.dataset}_{self.opt.input_type}_{self.opt.label_ratio}_{self.opt.FAD_ratio}_{self.opt.RAD_ratio}')
        return path

    def _evaluate_acc_f1(self, data_loader):
        t_targets_all, t_outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_outputs = self.model(t_inputs)
                t_targets = t_batch['polarity'].to(self.opt.device)
                if t_targets_all is None:
                    t_targets_all = t_targets.cpu()
                    t_outputs_all = t_outputs.cpu()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.cpu()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.cpu()), dim=0)
        acc = metrics.accuracy_score(t_targets_all, torch.argmax(t_outputs_all, -1))
        maf1a = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro')
        mif1a = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='micro')
        maf1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro',labels=[0,1])
        mif1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='micro',labels=[0,1])
        f_f1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro',labels=[1])
        a_f1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro',labels=[0])
        n_f1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), average='macro',labels=[2])
        avg_f1 = (mif1 + maf1)/2
        return acc, maf1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a

    def run(self):

        criterion = nn.CrossEntropyLoss()
        # print('='*30)
        # for child in self.model.children():
        #     for n,p in child.named_parameters():
        #         print(n)
        #         print(n.startswith('bart.encoder.layer'))

        #     print(type(child) == "<class 'models.bart.Encoder'>")
        #     print(type(child) == 'models.bart.Encoder')
        _params = [
            # {'params': [p for p in child.parameters()]} for child in self.model.children() if type(child) != BertModel and type(child) != RobertaModel and type(child) != AutoModel
            {'params': [p for n,p in child.named_parameters()],'lr':self.opt.lr} for child in self.model.children() if type(child) != BertModel and type(child) != RobertaModel and type(child) != AutoModel
        ]
        # optimizer = self.opt.optimizer(_params, lr=self.opt.lr,weight_decay=0.0001)

        no_decay = ['bias', 'LayerNorm.weight']
        if 'bert' in self.opt.model_name:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5,'lr':self.opt.bert_lr},
                {'params': [p for n, p in self.model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':self.opt.bert_lr},
            ]
            if self.opt.model_name == 'bert_rn':
                new_optimizer_parameters = [
                    {'params': [p for n, p in self.model.bert2.named_parameters() if not any(nd in n for nd in no_decay)],'lr': self.opt.bert_lr, 'weight_decay': 1e-5},
                    {'params': [p for n, p in self.model.bert2.named_parameters() if any(nd in n for nd in no_decay)], 'lr': self.opt.bert_lr, 'weight_decay': 0.0},
                ]
                optimizer_grouped_parameters += new_optimizer_parameters
            optimizer_grouped_parameters += _params
        elif 'bart' in self.opt.model_name:
            for n, p in self.model.named_parameters():
                if "bart.shared.weight" in n or "bart.encoder.embed" in n:
                    p.requires_grad = False
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if n.startswith('bart.encoder.layer')] , 'lr': self.opt.bert_lr},
                {'params': [p for n, p in self.model.named_parameters() if not n.startswith('bart.encoder.layer')] , 'lr': self.opt.lr},
                ]
        else:
            optimizer_grouped_parameters = _params

        optimizer = self.opt.optimizer(optimizer_grouped_parameters)


        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        print(len(self.testset))
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader,test_data_loader)
        if self.opt.save_model:
            self.model.load_state_dict(torch.load(best_model_path))
            test_acc, test_f1,avg_f1,f_f1,a_f1,n_f1,maf1a,mif1a = self._evaluate_acc_f1(test_data_loader)
            self.logger.info(f'Best_test: ma_f1: {round(test_f1*100,2)}\tacc: {round(test_acc*100,2)}\tavg_f1: {round(avg_f1*100,2)}\tma_all_f1: {round(maf1a*100,2)}\tmi_all_f1: {round(mif1a*100,2)}\tfavor_f1: {round(f_f1*100,2)}\tagainst_f1: {round(a_f1*100,2)}\tnone_f1: {round(n_f1*100,2)}')



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lstm', type=str)
    parser.add_argument('--input_type', default='tt', type=str)
    parser.add_argument('--dataset', default='-dt', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--bert_lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--num_epoch', default=50, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=-1, type=int)
    # parser.add_argument('--pretrained_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--patience', default=10, type=int)                  
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    parser.add_argument('--save_model', default=False, type=bool)
    parser.add_argument('--seed', default=2023, type=int, help='set seed for reproducibility')
    parser.add_argument('--add_num', default=0, type=int, help='front_add and reverse_add')
    parser.add_argument('--valset_ratio', default=0.15, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--FAD_ratio', default=1, type=float, help='set ratio between 0 and 1 for front data augmention')
    parser.add_argument('--RAD_ratio', default=1, type=float, help='set ratio between 0 and 1 for reverse data augmention')
    parser.add_argument('--label_ratio', default=1, type=float, help='set ratio between 0 and 1 for label')
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--lambadd', default=0.2, type=float)
    opt = parser.parse_args()
    Model2Path = {
        'bert': 'bert-base-uncased',
        'bert_rn': 'bert-base-uncased',
        'roberta': 'roberta-base',
        'bart': 'facebook/bart-large-mnli',
        'bart_rn': 'facebook/bart-large-mnli',
    }
    if opt.model_name in Model2Path:
        opt.pretrained_name = Model2Path[opt.model_name]
    if 'bert' in opt.model_name:
        opt.patience = 4
        # opt.num_epoch = 10
    elif 'bart' in opt.model_name:
        opt.num_epoch = 4
    else:
        opt.patience = 5
        opt.num_epoch = 25
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'bert': BERT_SPC,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'bert_rn': BERT_RN,
        'roberta':BERT_SPC,
    }
    def dataset_files(data_target):
        if '-' not in data_target:
            # return {
            #     'test': f'{BASEPATH}/sem16/IF-Then/{data_target}/test.csv',
            #     'train': f'{BASEPATH}/sem16/IF-Then/{data_target}/train.csv',
            #     'val': f'{BASEPATH}/sem16/IF-Then/{data_target}/val.csv',
            #     'front_add': f'{BASEPATH}/front_add/{data_target}.csv',
            #     'reverse_add': f'{BASEPATH}/reverse_add/{data_target}.csv',
            # }

            return {
                'train': f'{BASEPATH}/llam-datasets/sem16/IF-Then/{data_target}/train_10.csv',
                'test': f'{BASEPATH}/llam-datasets/sem16/IF-Then/{data_target}/test.csv',
                'val': f'{BASEPATH}/llam-datasets/sem16/IF-Then/{data_target}/val.csv',
                # 'front_add': f'/home/dingdaijun/data_list/dingdaijun/code/OpenStance/data/VAST/vast_mask_sentence.csv',
                # 'reverse_add': f'/home/dingdaijun/data_list/dingdaijun/code/OpenStance/data/VAST/vast_mask_topic.csv',
                'front_add': f'{BASEPATH}/llam-datasets/front_add/vast2/{data_target}.csv',
                'reverse_add': f'{BASEPATH}/llam-datasets/reverse_add/vast2/{data_target}.csv',
            }
        else:
            source_target, destin_target = data_target.split('-')
            if source_target != '' and destin_target != '':
                return {
                    'train': f'{BASEPATH}/sem16/IF-Then/{source_target}/{source_target}.csv',
                    'test': f'{BASEPATH}/sem16/IF-Then/{destin_target}/test.csv',
                    'front_add': f'{BASEPATH}/front_add/{source_target}.csv',
                    'reverse_add': f'{BASEPATH}/reverse_add/{source_target}.csv',
                }
            else:
                all_target = ['dt','la','fm','hc']
                destin_target = next(filter(lambda x: x != '', [source_target,destin_target]))
                all_target.remove(destin_target)
                assert len(all_target) == 3
                return {
                    'train': [
                        f'{BASEPATH}/llam-datasets/sem16/IF-Then/{all_target[1]}/{all_target[1]}.csv',
                        f'{BASEPATH}/llam-datasets/sem16/IF-Then/{all_target[0]}/{all_target[0]}.csv',
                        f'{BASEPATH}/llam-datasets/sem16/IF-Then/{all_target[2]}/{all_target[2]}.csv',
                        ],
                    'test': f'{BASEPATH}/sem16/IF-Then/{destin_target}/test.csv',
                    'front_add': [
                        f'{BASEPATH}/front_add/{all_target[0]}.csv',
                        f'{BASEPATH}/front_add/{all_target[1]}.csv',
                        f'{BASEPATH}/front_add/{all_target[2]}.csv',
                        ],
                    'reverse_add': [
                        f'{BASEPATH}/reverse_add/{all_target[0]}.csv',
                        f'{BASEPATH}/reverse_add/{all_target[1]}.csv',
                        f'{BASEPATH}/reverse_add/{all_target[2]}.csv',
                        ],
                }

    input_colses = {
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['bert_text_target_indices'],
        'bert_rn_tt': ['bert_text_target_inputs','bert_text_target_type','bert_text_target_mask','bert_reason_inputs','bert_reason_type','bert_reason_mask','polarity'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw':AdamW,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files(opt.dataset)
    opt.inputs_cols = input_colses[f'{opt.model_name}_{opt.input_type}']
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
