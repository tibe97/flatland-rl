# Code from https://github.com/Kaixhin/Rainbow
# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F, Linear, BatchNorm1d, Dropout
from VRSPConv import VRSPConv



# TODO Check hidsizes and pass them as params
class DQN(nn.Module):
    '''
    DQN with Rainbow but without dueling networks (we don't have actions, but just state values)
    '''
    def __init__(self, feature_size, hidsizes=[10,5], out_size=5):
        super(DQN, self).__init__()
        #self.atoms = args.atoms
        self.conv1 = VRSPConv(2*feature_size, hidsizes[0])
        self.bn1 = BatchNorm1d(num_features=3*hidsizes[0])
        self.drop1 = Dropout(p=0.3)
        self.conv2 = VRSPConv(6*hidsizes[0], hidsizes[1])
        self.bn2 = BatchNorm1d(num_features=3*hidsizes[1])
        self.drop2 = Dropout(p=0.25)
        self.conv3 = VRSPConv(6*hidsizes[1], out_size)
        self.bn3 = BatchNorm1d(num_features=3*out_size)
        self.drop3 = Dropout(p=0.2)
        self.linear1 = Linear(3*out_size, out_size)
        self.bn4 = BatchNorm1d(num_features=out_size)
        self.drop4 = Dropout(p=0.2)
        self.linear2 = Linear(out_size, 1)

    def forward(self, x, edge_index, agents_messages=None, log=False):
        '''
        If in training mode, we need to pass the agent handle and its position, so we can store its message
        at the different levels.
        When doing forward pass we need to gather the messages of the other agents present on the track we encounter.
        '''
        x = F.relu(self.bn1(self.conv1(x, edge_index, agents_messages)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, agents_messages)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index, agents_messages)))
        x = self.drop3(x)
        x = F.relu(self.bn4(self.linear1(x)))
        x = self.drop4(x)
        x = self.linear2(x)
        return x

    def reset_noise(self):
        for name, module in self.named_children():
            if 'noisy' in name:
                module.reset_noise()
