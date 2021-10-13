#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 10 2021

@author: dingwenli

Bi-directional version of CrossNet following BRITS's implementation
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.crossnet import CrossNet


class BiCrossNet(nn.Module):
    def __init__(self, static_size, ts_size, rnn_hid_size, seq_len, impute_weight):
        super(BiCrossNet, self).__init__()

        self.static_size = static_size
        self.ts_size = ts_size
        self.rnn_hid_size = rnn_hid_size
        self.seq_len = seq_len
        self.impute_weight = impute_weight
        
        self.build()

    def build(self):
        self.crossnet_f = CrossNet(self.static_size, self.ts_size, self.rnn_hid_size, self.seq_len, self.impute_weight)
        self.crossnet_b = CrossNet(self.static_size, self.ts_size, self.rnn_hid_size, self.seq_len, self.impute_weight)

    def forward(self, f_values, f_masks, f_deltas, b_values, b_masks, b_deltas, statics, static_masks, labels):
        ret_f = self.crossnet_f(f_values, f_masks, f_deltas, statics, static_masks, labels)
        ret_b = self.reverse(self.crossnet_b(b_values, b_masks, b_deltas, statics, static_masks, labels))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    # call this method to train the model by batches
    def run_on_batch(self, f_values, f_masks, f_deltas, b_values, b_masks, b_deltas, statics, static_masks, labels, optimizer, epoch = None):
        ret = self(f_values, f_masks, f_deltas, b_values, b_masks, b_deltas, statics, static_masks, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

