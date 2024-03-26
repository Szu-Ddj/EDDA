# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, \
                         AlbertTokenizer, AlbertConfig, AlbertModel, \
                         AutoTokenizer
import csv
import random

def build_tokenizer(fnames, max_seq_len, dat_fname, add_num):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            if type(fname) == list:
                for sign_fname in fname:
                    if 'add' not in sign_fname:
                        _add_num = 1000000
                    else:
                        _add_num = add_num
                    with open(sign_fname,'r') as f:
                        l1s = csv.DictReader(f)
                        for l1,_ in zip(l1s,range(_add_num)):
                            text += l1['Tweet'] + ' ' + l1['Reason'] + ' ' + l1['Target']
            else:
                if 'add' not in fname:
                    _add_num = 1000000
                else:
                    _add_num = add_num
                with open(fname,'r') as f:
                    l1s = csv.DictReader(f)
                    for l1,_ in zip(l1s,range(_add_num)):
                        text += l1['Tweet'] + ' ' + l1['Reason'] + ' ' + l1['Target']

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else '/home/dingdaijun/data_list/dingdaijun/glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        words_num = {}
        for word in words:
            if words_num.get(word) == None:
                words_num[word] = 0
            words_num[word] += 1
        for word in words_num:
            if word not in self.word2idx and words_num[word] > 1:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
        print('tokenizer is over:', len(self.word2idx))

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_name):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.max_seq_len = max_seq_len
        self.tokenizer.add_tokens([])

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):

        if len(text) == 1:
            sequence = self.tokenizer.encode_plus(text[0],max_length=self.max_seq_len + add,padding='max_length',truncation=True,return_tensors='pt').values()
        else:
            sequence = self.tokenizer.encode_plus(text[0],text_pair=text[1],max_length=self.max_seq_len + add,padding='max_length',truncation=True,return_tensors='pt').values()
        sequence = [item.squeeze(0) for item in sequence]

        if len(sequence[0]) > self.max_seq_len + add:
            # print("===========it's Prompt==========")
            for index in range(len(sequence)):
                sequence[index] = sequence[index][len(sequence[index]) - self.max_seq_len - add:]
        return sequence

import csv
import random
class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer,opt,tt_model='train',ratio=1):
        all_data = []
        match = {'AGAINST':0,'FAVOR':1,'NONE':2,'0':0,'1':1,'2':2,'SUPPORT':1,'NEUTRAL':2,'OPPOSED':0}

        def deal_data(sign_fname):
            len_path_data = sum(1 for _ in open(sign_fname)) - 1
            rand_pp = [0] * len_path_data
            ran_po = random.sample(range(len_path_data),int(opt.label_ratio * len_path_data))
            for i in ran_po:
                rand_pp[i] = 1
            sall_data = []
            if tt_model == 'add':
                max_data_len = opt.add_num
            else:
                max_data_len = len_path_data + 10000
            with open(sign_fname,'r',encoding='utf-8') as f:
                lines = csv.DictReader(f)
                for index,(line,_) in enumerate(zip(lines,range(max_data_len))):
                    text = line['Tweet']
                    reason = line['Reason']
                    if len(reason) <= 1:
                        reason = 'empty'
                    target = line['Target']

                    if tt_model == 'test' or tt_model == 'train':
                        polarity_s = match[line['Stance'].upper()]
                        polarity = polarity_s
                    elif tt_model == 'add':
                        polarity_a = match[line['Attitude'].upper()]
                        polarity = polarity_a

                    if  'bart' in opt.model_name or 'bert' in opt.model_name:
                        bert_text = tokenizer.text_to_sequence([text])
                        bert_text_target = tokenizer.text_to_sequence([text,target])
                        bert_text_reason = tokenizer.text_to_sequence([text, reason])
                        bert_reason = tokenizer.text_to_sequence([reason])
                        bert_reason_target = tokenizer.text_to_sequence([reason,target])
                        # bert_text_reason_target = tokenizer.text_to_sequence([text + '[SEP]' + reason,target])
                        
                        if 'bert' in opt.model_name:
                            data = {
                                'bert_text_inputs': bert_text[0],
                                'bert_text_type': bert_text[1],
                                'bert_text_mask': bert_text[2],

                                'bert_text_target_inputs': bert_text_target[0],
                                'bert_text_target_type': bert_text_target[1],
                                'bert_text_target_mask': bert_text_target[2],

                                'bert_text_reason_inputs': bert_text_reason[0],
                                'bert_text_reason_type': bert_text_reason[1],
                                'bert_text_reason_mask': bert_text_reason[2],

                                'bert_reason_inputs': bert_reason[0],                            
                                'bert_reason_type': bert_reason[1],                            
                                'bert_reason_mask': bert_reason[2],     

                                'bert_reason_target_inputs': bert_reason_target[0],
                                'bert_reason_target_type': bert_reason_target[1],
                                'bert_reason_target_mask': bert_reason_target[2],

                                # 'bert_text_reason_target_inputs': bert_text_reason_target[0],
                                # 'bert_text_reason_target_type': bert_text_reason_target[1],
                                # 'bert_text_reason_target_mask': bert_text_reason_target[2],

                                'polarity': polarity,
                            }
                        else:
                            data = {
                                'bert_text_inputs': bert_text[0],
                                'bert_text_mask': bert_text[1],

                                'bert_text_target_inputs': bert_text_target[0],
                                'bert_text_target_mask': bert_text_target[1],

                                'bert_text_reason_inputs': bert_text_reason[0],
                                'bert_text_reason_mask': bert_text_reason[1],

                                'bert_reason_inputs': bert_reason[0],                            
                                'bert_reason_mask': bert_reason[1],     

                                'bert_reason_target_inputs': bert_reason_target[0],
                                'bert_reason_target_mask': bert_reason_target[1],

                                'bert_text_reason_target_inputs': bert_text_reason_target[0],
                                'bert_text_reason_target_mask': bert_text_reason_target[1],

                                'polarity': polarity,
                            }
                    else:
                        text_target_indices = tokenizer.text_to_sequence(text + ' ' + target)
                        text_reason_indices = tokenizer.text_to_sequence(text + ' ' + reason)
                        text_indices = tokenizer.text_to_sequence(text)
                        reason_indices = tokenizer.text_to_sequence(reason)
                        if reason_indices[0] == 0:
                            print(reason,len(reason))
                            assert False
                        reason_target_indices = tokenizer.text_to_sequence(reason + ' ' + target)
                        text_reason_target_indices = tokenizer.text_to_sequence(text + ' ' + reason + ' ' + target)
                        data = {
                            'text_target_indices': text_target_indices,
                            'text_reason_indices': text_reason_indices,
                            'text_reason_target_indices': text_reason_target_indices,
                            'text_indices': text_indices,
                            'reason_indices': reason_indices,
                            'reason_target_indices': reason_target_indices,

                            'polarity': polarity,
                        }
                    sall_data.append(data)
            return sall_data


        if type(fname) == list:
            for sign_fname in fname:
                sall_data = deal_data(sign_fname)
                # all_data.extend(sall_data)
                all_data.extend(random.sample(sall_data,int(ratio * len(sall_data))))
        else:
            sall_data = deal_data(fname)
            # print('ratio * len(sall_data)',ratio * len(sall_data),ratio,len(sall_data))
            # all_data.extend(sall_data)
            all_data.extend(random.sample(sall_data,int(ratio * len(sall_data))))

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
