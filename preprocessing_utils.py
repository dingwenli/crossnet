#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 10 2021

@author: dingwenli

Preprocessing methods
"""

import numpy as np

def parse_delta(masks):
    deltas = np.zeros(masks.shape)

    N, T, D = masks.shape
    
    for i in range(N):
        for h in range(T):
            if h == 0:
                deltas[i,h,:] = 0
            else:
                deltas[i,h,:] = np.ones(masks[i,h,:].shape) + np.multiply(np.ones(masks[i,h,:].shape) - masks[i,h,:], deltas[i,h-1,:])

    return np.array(deltas)