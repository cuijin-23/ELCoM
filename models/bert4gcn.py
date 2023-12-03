# -*- coding: utf-8 -*-
import os
from typing import Any, List, Tuple
# from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.gcn import GCNWithPosition
# from models.bilstm_embedding import BiLSTMEmbedding
from models.relativa_position import RelativePosition


class BERT4GCN(nn.Module):
    def __init__(self,
                 model_name_or_path: str = 'bert-base-uncased',
                 bert_layers: Tuple = (1, 5, 9, 12),
                 bert_dim: int = 768,
                 emb_dim: int = 300,
                 upper: float = 0.25,
                 lower: float = 0.01,
                 hidden_dim: int = 300,
                 window: int = 3,
                 gnn_drop: float = 0.8,
                 guidance_drop: float = 0.8,
                 freeze_emb: bool = True):

        super(BERT4GCN, self).__init__()
        # self.save_hyperparameters(logger=False)
        # embedding_matrix = self._build_embedding_matrix()
        self.bert_layers = bert_layers
        self.upper = upper
        self.lower = lower
        # self.lstm_emb = BiLSTMEmbedding(embedding_matrix, emb_dim, hidden_dim, freeze=freeze_emb)
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(hidden_dim * 2, 3)

        self.position_emb = RelativePosition(hidden_dim * 2, window)

        self.gnn = nn.ModuleList()
        self.guidance_trans = nn.ModuleList()
        for _ in range(len(bert_layers)):
            self.guidance_trans.append(nn.Linear(bert_dim, hidden_dim * 2))
            self.gnn.append(GCNWithPosition(hidden_dim * 2, hidden_dim * 2))


        self.gc1 = GraphConvolution(bert_dim, bert_dim)
        self.gc2 = GraphConvolution(bert_dim, bert_dim)
        self.gc3 = GraphConvolution(bert_dim, bert_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.gnn_drop = nn.Dropout(gnn_drop)
        self.guidance_drop = nn.Dropout(guidance_drop)
        # self.reset_parameter()
    #
    # def forward(self, x):
    #
    #     input_ids, attention_mask, token_type_ids, adj, token_starts, token_starts_mask, aspect_in_text, aspect_in_text_mask = x
    #
    #     batch_size, max_len = input_ids.shape[0], input_ids.shape[1]
    #
    #     # encode text
    #     # feature = self.lstm_emb(text_raw_indices)
    #     outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True, output_attentions=True)
    #     # select hidden_states of word start tokens
    #     stack_hidden_states = outputs.last_hidden_state # torch.stack(outputs.hidden_states) #
    #     hidden_states_list = []
    #     for i in range(batch_size):
    #         start_tokens_hidden_states = torch.index_select(stack_hidden_states[i, :], dim=0, index=token_starts[i])
    #         hidden_states_list.append(start_tokens_hidden_states)
    #     guidance_states = torch.stack(hidden_states_list, dim=0)
    #     # print(guidance_states)
    #     feature = guidance_states
    #     # # select attention weights of word start tokens
    #     stack_attentions = torch.stack(outputs.attentions).clone().detach().mean(dim=2)
    #     attentions_list = []
    #
    #     x = F.relu(self.gc1(feature, adj))
    #     x = F.relu(self.gc2(x, adj))
    #     feature = F.relu(self.gc3(x, adj))
    #
    #
    #     # # (bs,seq,dim) * (bs,seq,1)
    #     # aspects = (feature * aspect_in_text_mask.unsqueeze(2)).sum(dim=1) / aspect_in_text_mask.sum(dim=1, keepdim=True)
    #     # aspects = feature.sum(dim=1)
    #     feature = feature.sum(dim=1)
    #     logits = self.classifier(feature)
    #
    #     return logits

    def forward(self, x):
        input_ids, attention_mask, token_type_ids, adj, token_starts, token_starts_mask, aspect_in_text, aspect_in_text_mask = x
        #     input_ids, attention_mask, token_type_ids, adj, token_starts, token_starts_mask, aspect_in_text, aspect_in_text_mask = x

        batch_size, max_len = input_ids.shape[0], input_ids.shape[1]

        # encode text
        # feature = self.lstm_emb(text_raw_indices)
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True, output_attentions=True)

        # select hidden_states of word start tokens
        stack_hidden_states = torch.stack(outputs.hidden_states)
        hidden_states_list = []
        for i in range(batch_size):
            start_tokens_hidden_states = torch.index_select(stack_hidden_states[:, i], dim=1, index=token_starts[i])
            hidden_states_list.append(start_tokens_hidden_states)
        guidance_states = torch.stack(hidden_states_list, dim=1)

        # select attention weights of word start tokens
        stack_attentions = torch.stack(outputs.attentions).clone().detach().mean(dim=2)
        attentions_list = []
        for i in range(batch_size):
            sample_attentions = stack_attentions[:, i]  # (n,max_len,max_len)
            sample_attentions = sample_attentions * token_starts_mask[i].view(max_len, 1) * token_starts_mask[i].view(1, max_len)
            start_tokens_attentions_row2col = torch.index_select(sample_attentions, dim=1, index=token_starts[i])
            start_tokens_attentions_col2row = torch.index_select(start_tokens_attentions_row2col, dim=2, index=token_starts[i])
            attentions_list.append(start_tokens_attentions_col2row)
        guidance_attentions = torch.stack(attentions_list, dim=1)

        pos = self.position_emb(max_len, max_len).unsqueeze(0).expand(batch_size, -1, -1, -1)

        for index, layer in enumerate(self.bert_layers):
            layer_hidden_states = guidance_states[layer]
            guidance = F.relu(self.guidance_trans[index](layer_hidden_states))
            node_embeddings = self.guidance_drop(guidance) #+ feature
            feature = self.layer_norm(node_embeddings)

            if index < len(self.bert_layers) - 1:
                layer_attentions = guidance_attentions[layer - 1]  # (batch, seq, seq)
                upper_att_adj = torch.gt(layer_attentions, self.upper)
                enhanced_adj = torch.logical_or(adj, upper_att_adj)
                lower_att_adj = torch.le(layer_attentions, self.lower)
                enhanced_adj = torch.logical_and(enhanced_adj, ~lower_att_adj)
                gnn_out = F.relu(self.gnn[index](feature, enhanced_adj.float(), pos))
                feature = self.gnn_drop(gnn_out)

        # (bs,seq,dim) * (bs,seq,1)
        aspects = (feature * aspect_in_text_mask.unsqueeze(2)).sum(dim=1) / aspect_in_text_mask.sum(dim=1, keepdim=True)
        logits = self.classifier(aspects)
        return logits


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
