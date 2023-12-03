# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
# from layers.squeeze_embedding import SqueezeEmbedding
from transformers import AutoModel


class BERT_SPC(nn.Module):
    def __init__(self, opt):
        super(BERT_SPC, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        # bert_dim = 768,
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, attention_mask = inputs
        # text_bert_len = torch.sum(text_bert_indices != 0, dim=-1)
        # text_bert_indices = self.squeeze_embedding(text_bert_indices, text_bert_len)
        # bert_segments_ids = self.squeeze_embedding(bert_segments_ids, text_bert_len)
        pooled_output = self.bert(text_bert_indices, attention_mask)[-1]
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
