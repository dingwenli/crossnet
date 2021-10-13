#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 10 2021

@author: dingwenli

Multi-modal fusion and CrossNet models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
from models.model_utils import FeatureRegression, TemporalDecay, StaticFeatureRegression

class MultiModal(nn.Module):
    def __init__(self, static_size, rnn_hid_size, impute_weight):
        super(MultiModal, self).__init__()
        
        self.static_size = static_size
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        
        self.build()
        
    def build(self):
        # initialize dense layer for static input representation learning
        self.static_feat_reg = StaticFeatureRegression(self.static_size)
        self.dense1 = nn.Linear(self.static_size, self.rnn_hid_size)
        self.dropout1 = nn.Dropout(p = 0.25)
        self.dense2 = nn.Linear(self.rnn_hid_size, self.rnn_hid_size)
        self.dropout2 = nn.Dropout(p = 0.25)
        self.dense3 = nn.Linear(self.rnn_hid_size, self.rnn_hid_size)
        self.dropout3 = nn.Dropout(p = 0.25)
        
    def forward(self, data, masks):
        statics, static_masks = data, masks
        # static values go through several fully connected layers
        s_h = self.static_feat_reg(statics)
        s_c = static_masks * statics + (1-static_masks) * s_h
        s_tc = s_c
        
        s_loss = torch.sum(torch.abs(statics - s_c) * static_masks) / (torch.sum(static_masks) + 1e-5)
        # regularization terms
        lambda2 = 0.1
        static_feat_reg_params = self.static_feat_reg.W.data
        s_loss += self.impute_weight * lambda2 * torch.sum(torch.norm(torch.diagonal(static_feat_reg_params, 0),p=1))
        
        s_c = self.dense1(s_c)
        s_c = nn.ReLU()(s_c)
        s_c = self.dropout1(s_c)
        s_c = self.dense2(s_c)
        s_c = nn.ReLU()(s_c)
        s_c = self.dropout2(s_c)
        s_c = self.dense3(s_c)
        s_c = nn.ReLU()(s_c)
        s_c = self.dropout3(s_c)
        
        return s_c, s_tc, s_loss
    
class CrossNet(nn.Module):
    def __init__(self, static_size, ts_size, rnn_hid_size, seq_len, impute_weight):
        super(CrossNet, self).__init__()

        self.static_size = static_size
        self.ts_size = ts_size
        self.rnn_hid_size = rnn_hid_size
        self.seq_len = seq_len
        self.impute_weight = impute_weight

        self.build()

    def build(self):
        self.multi_modal = MultiModal(self.static_size, self.rnn_hid_size, self.impute_weight)
        self.rnn_cell = nn.LSTMCell(self.ts_size * 2, self.rnn_hid_size)
        self.temp_decay_h = TemporalDecay(input_size = self.ts_size, output_size = self.rnn_hid_size, diag = False)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.ts_size)
        self.st_reg = nn.Linear(self.static_size, self.ts_size)
        self.real_reg = nn.Linear(self.ts_size, self.ts_size)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)

    def forward(self, values, masks, deltas, statics, static_masks, labels):
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        # static values go through several fully connected layers
        s_c, s_tc, s_loss = self.multi_modal(statics, static_masks)
        
        h = s_c

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0
        r_loss = 0.0
        
        imputations = []

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)

            h = h * gamma_h

            x_h = self.hist_reg(h) + self.st_reg(s_tc)

            x_r = self.real_reg(x)
            x_c = x_r + x_h
            x_loss += torch.sum(torch.abs(x - x_c) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * x_c

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))
            
            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        
        y_h = self.out(h)
        y_loss = F.binary_cross_entropy_with_logits(y_h[:,0], labels)
        
        # regularization terms
        lambda1 = 0.01
        lambda2 = 0.1
        real_reg_params = self.real_reg.weight.data
        hist_reg_params = self.hist_reg.weight.data
        st_reg_params = self.st_reg.weight.data
        r_loss += lambda1 * torch.sum(torch.norm(real_reg_params,p=1))
        r_loss += lambda1 * torch.sum(torch.norm(hist_reg_params,p=1))
        r_loss += lambda1 * torch.sum(torch.norm(st_reg_params,p=1))
        r_loss += lambda2 * torch.sum(torch.norm(torch.diagonal(real_reg_params, 0),p=1))
        
        y_loss += r_loss + self.impute_weight*x_loss + s_loss
            
        y_h = torch.sigmoid(y_h)

        return {'loss': y_loss, 'predictions': y_h, 'imputations': imputations, 'labels': labels}

    # call this method to train the model by batches
    def run_on_batch(self, values, masks, deltas, statics, static_masks, labels, optimizer, epoch = None):
        ret = self(values, masks, deltas, statics, static_masks, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
