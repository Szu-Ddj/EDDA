# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)

    def forward(self, inputs):
        # print(len(inputs))
        # if len(inputs) == 3:
        inputs_id, token_type_ids, attention_mask,polarity = inputs
        pooled_output = self.bert(inputs_id, token_type_ids=token_type_ids,attention_mask=attention_mask)[1]
        # elif len(inputs) == 2:
        #     inputs_id, attention_mask = inputs
        #     # print(inputs_id.size(),attention_mask.size())
        #     pooled_output = self.bert(inputs_id,attention_mask=attention_mask)[1]
        # with open('datt/add.txt','a+') as f:
        #     f.write(str((pooled_output).tolist()))
        #     f.write('\n')
        # text_bert_indices = inputs[0]
        # pooled_output = self.bert(text_bert_indices, attention_mask=bert_segments_ids)[1]
        # print(text_bert_indices.size())
        # pooled_output = self.bert(text_bert_indices)[1]
        # pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)[1]
        # with open('/home/dingdaijun/data_list/dingdaijun/DataAugmention/Classification/datt/open_badd.txt','a+') as f:
        #     for i,j in zip(pooled_output,polarity):
        #         f.write(str((i).tolist()))
        #         f.write('\n')
        #         f.write(str((j.item())))
        #         f.write('\n')
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        # with open('/home/dingdaijun/data_list/dingdaijun/DataAugmention/Classification/datt/open_test.txt','a+') as f:
        #     for i,j in zip(logits,polarity):
        #         f.write(str((i).tolist()))
        #         f.write('\n')
        #         f.write(str((j.item())))
        #         f.write('\n')
        # print(logits.size(), logits.device)
        return logits
        # return pooled_output
