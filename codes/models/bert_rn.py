# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiAttRule(nn.Module):
    def __init__(self,hid_dim,heads=4,batch_first=True):
        super(MultiAttRule, self).__init__()
        self.att = nn.MultiheadAttention(hid_dim,heads,batch_first=batch_first)
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        
    def forward(self,q,k,v):
        return self.att(self.w_q(q),self.w_k(k),self.w_v(v))


class BERT_RN(nn.Module):
    def __init__(self, bert,bert2, opt):
        super(BERT_RN, self).__init__()
        self.bert = bert
        self.bert2 = bert2
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.att = MultiAttRule(self.bert.config.hidden_size)
        self.layernorm = nn.LayerNorm(self.bert.config.hidden_size, eps=1e-12)
        self.lambadd = opt.lambadd

    def forward(self, inputs):
        inputs_id, token_type_ids, attention_mask,r_inputs_id, r_token_type_ids, r_attention_mask,polarity = inputs

        last_hidden,pooled = self.bert(inputs_id, token_type_ids=token_type_ids,attention_mask=attention_mask,return_dict=False)
        rlast_hidden,rpooled = self.bert2(r_inputs_id, token_type_ids=r_token_type_ids,attention_mask=r_attention_mask,return_dict=False)
        # rx_l = r_attention_mask.sum(1).to('cuda')
        out,_score = self.att(pooled.unsqueeze(1),rlast_hidden,rlast_hidden)
        out = nn.ReLU()(out.squeeze(1))
        return F.softmax(self.dense(self.lambadd * out + pooled), -1)

