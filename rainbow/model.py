# Code from https://github.com/Kaixhin/Rainbow
# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F, Linear, BatchNorm1d, Dropout
from VRSPConv import VRSPConv
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn.conv import GATConv



# TODO Check hidsizes and pass them as params
"""
class DQN_value(nn.Module):
    '''
    DQN with Rainbow but without dueling networks (we don't have actions, but just state values)
    '''
    def __init__(self, feature_size, hidsizes=[12,9,7,30,10], out_size=1):
        super(DQN_value, self).__init__()
        #self.atoms = args.atoms
        self.conv1 = VRSPConv(2*feature_size, hidsizes[0])
        #self.bn1 = BatchNorm1d(num_features=3*hidsizes[0])
        #self.drop1 = Dropout(p=0.3)
        self.conv2 = VRSPConv(6*hidsizes[0], hidsizes[1])
        #self.bn2 = BatchNorm1d(num_features=3*hidsizes[1])
        #self.drop2 = Dropout(p=0.25)
        self.conv3 = VRSPConv(6*hidsizes[1], hidsizes[2])
        #self.bn3 = BatchNorm1d(num_features=3*hidsizes[2])
        #self.drop3 = Dropout(p=0.2)
        
        self.linear1 = Linear(3*hidsizes[2], hidsizes[3])
        self.linear2 = Linear(hidsizes[3], hidsizes[4])
        #self.bn4 = BatchNorm1d(num_features=hidsizes[4])
        #self.drop4 = Dropout(p=0.2)
        self.out = Linear(hidsizes[4], out_size)

        
       
    def forward(self, x, edge_index, agents_messages=None, log=False):
        '''
        If in training mode, we need to pass the agent handle and its position, so we can store its message
        at the different levels.
        When doing forward pass we need to gather the messages of the other agents present on the track we encounter.
        '''
        
        x = F.leaky_relu(self.conv1(x, edge_index, agents_messages))
        #x = self.drop1(x)
        x = F.leaky_relu(self.conv2(x, edge_index, agents_messages))
        #x = self.drop2(x)
        x = F.leaky_relu(self.conv3(x, edge_index, agents_messages))
        #x = self.drop3(x)
        x_value = F.leaky_relu(self.linear1(x))
        x_value = F.leaky_relu(self.linear2(x_value))
        #x = self.drop4(x)
        out = self.out(x_value)
        
        return out
"""
class DQN_value(nn.Module):
    '''
    DQN with Rainbow but without dueling networks (we don't have actions, but just state values)
    '''
    def __init__(self, feature_size, hidsizes=[12,9,7], out_size=1):
        super(DQN_value, self).__init__()
        self.conv1 = VRSPConv(2*feature_size, hidsizes[0])
        self.conv2 = VRSPConv(6*hidsizes[0], hidsizes[1])
        self.gat = GATConv(3*hidsizes[1], hidsizes[2], negative_slope=0.2, heads=8, concat=True, flow='target_to_source')

        self.out = Linear(hidsizes[2] * 8, out_size)

        
       
    def forward(self, x, edge_index, agents_messages=None, log=False):
        '''
        If in training mode, we need to pass the agent handle and its position, so we can store its message
        at the different levels.
        When doing forward pass we need to gather the messages of the other agents present on the track we encounter.
        '''
        x = F.elu(self.conv1(x, edge_index, agents_messages))
        x = F.elu(self.conv2(x, edge_index, agents_messages))
        x = F.elu(self.gat(x, edge_index))
        return self.out(x)


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
       
        return F.softmax(out, dim=1)




class GAT_value(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, alpha, nheads, flow, use_bn):
        """Dense version of GAT."""
        super(GAT_value, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.attentions = []
        self.use_bn = use_bn
        self.batch_norms = []
        
        # for now all layers have same input and output size, except first attention layer 
    
        for l in range(nlayers):
            input_size = nfeat if l==0 else nhid * nheads
            self.attentions.append(GATConv(input_size, nhid, heads=nheads, dropout=dropout, negative_slope=alpha, concat=True, flow=flow))
            if self.use_bn:
                self.batch_norms.append(BatchNorm1d(num_features=nhid*nheads))

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            if self.use_bn:
                self.add_module('bn_{}'.format(i), self.batch_norms[i])
        
        '''
        for l in range(nlayers):
            input_size = nfeat if l==0 else nhid * nheads
            self.attentions.append([GATConv(input_size, nhid, dropout=dropout, negative_slope=alpha, concat=True) for _ in range(nheads)])
            self.batch_norms.append(BatchNorm1d(num_features=nhid*nheads))
        for i, attention in enumerate(self.attentions):
            for j, att_head in enumerate(attention):
                self.add_module('attention_{}_head_{}'.format(i, j), att_head)
        '''
        self.out_att = GATConv(nhid * nheads, nclass, negative_slope=alpha, concat=False, flow=flow)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        for l in range(self.nlayers):
            if l > 0: 
                residual = x
            x = self.attentions[l](x, adj)
            #x = torch.cat([att(x, adj) for att in self.attentions[l]], dim=1)
            if self.use_bn:
                x = self.batch_norms[l](x)
            if l > 0:
                x += residual # residual connection
            x = F.elu(x) 
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x


class GAT_action(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, alpha, nheads, flow, use_bn):
        """Dense version of GAT."""
        super(GAT_action , self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.attentions = []
        self.use_bn = use_bn
        self.batch_norms = []

        # for now all layers have same input and output size, except first attention layer 
        
        for l in range(self.nlayers):
            input_size = nfeat if l==0 else nhid * nheads
            self.attentions.append(GATConv(input_size, nhid, heads=nheads, dropout=dropout, negative_slope=alpha, concat=True, flow=flow))
            if self.use_bn:
                self.batch_norms.append(BatchNorm1d(num_features=nhid*nheads))

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            if self.use_bn:
                self.add_module('bn_{}'.format(i), self.batch_norms[i])
        
        '''
        for l in range(nlayers):
            input_size = nfeat if l==0 else nhid * nheads
            self.attentions.append([GATConv(input_size, nhid, dropout=dropout, negative_slope=alpha, concat=True) for _ in range(nheads)])
            self.batch_norms.append(BatchNorm1d(num_features=nhid*nheads))
        for i, attention in enumerate(self.attentions):
            for j, att_head in enumerate(attention):
                self.add_module('attention_{}_head_{}'.format(i, j), att_head)
        '''
        self.out_att = GATConv(nhid * nheads, nclass, negative_slope=alpha, concat=False, flow=flow)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        for l in range(self.nlayers):
            if l > 0: 
                residual = x
            x = self.attentions[l](x, adj)
            #x = torch.cat([att(x, adj) for att in self.attentions[l]], dim=1)
            if self.use_bn:
                x = self.batch_norms[l](x)
            if l > 0:
                x += residual # residual connection
            x = F.elu(x) 
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)
        
class FC_action(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(FC_action, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        
        self.layer1 = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(True))
        self.layer3 = nn.Linear(n_hidden, n_out)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = nn.functional.softmax(self.layer3(x))
        
        return x
        
