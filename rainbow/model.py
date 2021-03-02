# Code from https://github.com/Kaixhin/Rainbow
# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F, Linear, BatchNorm1d, Dropout
from VRSPConv import VRSPConv



# TODO Check hidsizes and pass them as params
class DQN_value(nn.Module):
    '''
    DQN with Rainbow but without dueling networks (we don't have actions, but just state values)
    '''
    def __init__(self, feature_size, hidsizes=[12,9,7,30,10], out_size=1):
        super(DQN_value, self).__init__()
        #self.atoms = args.atoms
        self.conv1 = VRSPConv(2*feature_size, hidsizes[0])
        self.bn1 = BatchNorm1d(num_features=3*hidsizes[0])
        #self.drop1 = Dropout(p=0.3)
        self.conv2 = VRSPConv(6*hidsizes[0], hidsizes[1])
        self.bn2 = BatchNorm1d(num_features=3*hidsizes[1])
        #self.drop2 = Dropout(p=0.25)
        self.conv3 = VRSPConv(6*hidsizes[1], hidsizes[2])
        self.bn3 = BatchNorm1d(num_features=3*hidsizes[2])
        #self.drop3 = Dropout(p=0.2)
        
        self.linear1 = Linear(3*hidsizes[2], hidsizes[3])
        self.linear2 = Linear(hidsizes[3], hidsizes[4])
        self.bn4 = BatchNorm1d(num_features=hidsizes[4])
        #self.drop4 = Dropout(p=0.2)
        self.out = Linear(hidsizes[4], out_size)

        
        '''
        self.conv1 = VRSPConv(2*feature_size, hidsizes[0])
        self.drop1 = Dropout(p=0.3)
        self.conv2 = VRSPConv(6*hidsizes[0], hidsizes[1])
        self.drop2 = Dropout(p=0.25)
        self.conv3 = VRSPConv(6*hidsizes[1], out_size)
        self.drop3 = Dropout(p=0.2)
        self.linear1 = Linear(3*out_size, out_size)
        self.drop4 = Dropout(p=0.2)
        self.linear2 = Linear(out_size, 1)
        '''
    def forward(self, x, edge_index, agents_messages=None, log=False):
        '''
        If in training mode, we need to pass the agent handle and its position, so we can store its message
        at the different levels.
        When doing forward pass we need to gather the messages of the other agents present on the track we encounter.
        '''
        
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index, agents_messages)))
        #x = self.drop1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index, agents_messages)))
        #x = self.drop2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index, agents_messages)))
        #x = self.drop3(x)
        x_value = F.leaky_relu(self.linear1(x))
        x_value = F.leaky_relu(self.bn4(self.linear2(x_value)))
        #x = self.drop4(x)
        out = self.out(x_value)
        
        '''
        x = F.relu(self.conv1(x, edge_index, agents_messages))
        x = self.drop1(x)
        x = F.relu(self.conv2(x, edge_index, agents_messages))
        x = self.drop2(x)
        x = F.relu(self.conv3(x, edge_index, agents_messages))
        x = self.drop3(x)
        x = F.relu(self.linear1(x))
        x = self.drop4(x)
        x = self.linear2(x)
        '''
        return out


class DQN_action(nn.Module):
    '''
    DQN with Rainbow but without dueling networks (we don't have actions, but just state values)
    '''
    def __init__(self, feature_size, hidsizes=[12,9,7,30,10], out_size=2):
        super(DQN_action, self).__init__()
        #self.atoms = args.atoms
        self.conv1 = VRSPConv(2*feature_size, hidsizes[0])
        self.bn1 = BatchNorm1d(num_features=3*hidsizes[0])
        #self.drop1 = Dropout(p=0.3)
        self.conv2 = VRSPConv(6*hidsizes[0], hidsizes[1])
        self.bn2 = BatchNorm1d(num_features=3*hidsizes[1])
        #self.drop2 = Dropout(p=0.25)
        self.conv3 = VRSPConv(6*hidsizes[1], hidsizes[2])
        self.bn3 = BatchNorm1d(num_features=3*hidsizes[2])
        #self.drop3 = Dropout(p=0.2)
        

        self.linear1 = Linear(3*hidsizes[2], hidsizes[3])
        self.linear2 = Linear(hidsizes[3], hidsizes[4])
        self.bn4 = BatchNorm1d(num_features=hidsizes[4])
        #self.drop4 = Dropout(p=0.2)
        self.out = Linear(hidsizes[4], out_size)
        
        '''
        self.conv1 = VRSPConv(2*feature_size, hidsizes[0])
        self.drop1 = Dropout(p=0.3)
        self.conv2 = VRSPConv(6*hidsizes[0], hidsizes[1])
        self.drop2 = Dropout(p=0.25)
        self.conv3 = VRSPConv(6*hidsizes[1], out_size)
        self.drop3 = Dropout(p=0.2)
        self.linear1 = Linear(3*out_size, out_size)
        self.drop4 = Dropout(p=0.2)
        self.linear2 = Linear(out_size, 1)
        '''
    def forward(self, x, edge_index, agents_messages=None, log=False):
        '''
        If in training mode, we need to pass the agent handle and its position, so we can store its message
        at the different levels.
        When doing forward pass we need to gather the messages of the other agents present on the track we encounter.
        '''
        
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index, agents_messages)))
        #x = self.drop1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index, agents_messages)))
        #x = self.drop2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index, agents_messages)))
        #x = self.drop3(x)
        
        x_action = F.leaky_relu(self.linear1(x))
        x_action = F.leaky_relu(self.bn4(self.linear2(x_action)))
        #x = self.drop4(x)
        out = self.out(x_action)
        '''
        x = F.relu(self.conv1(x, edge_index, agents_messages))
        x = self.drop1(x)
        x = F.relu(self.conv2(x, edge_index, agents_messages))
        x = self.drop2(x)
        x = F.relu(self.conv3(x, edge_index, agents_messages))
        x = self.drop3(x)
        x = F.relu(self.linear1(x))
        x = self.drop4(x)
        x = self.linear2(x)
        '''
        return F.softmax(out, dim=1)

