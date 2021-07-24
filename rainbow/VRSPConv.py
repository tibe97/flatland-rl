import torch
import math
from torch import Tensor, nn
from typing import Optional
from torch.nn import Sequential as Seq, Linear, ReLU, functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import segment_csr, scatter
from collections import defaultdict


class VRSPConv(MessagePassing):
    '''
        Graph convolutional layer for Vehicle ReScheduling Problem.
        We don't need to classify every node but only the one where the agent is present.
        Each agent has a different graph representation, because node features are different for every agent.
        Communication between different agents is achieved by considering hidden representations of
        more agents, which are passed to the MLP in case other agents are on the path of the agent taken
        into consideration.
    '''

    def __init__(self, in_channels, out_channels):
        super(
            VRSPConv,
            self).__init__(
            aggr=None,
            flow="target_to_source")  # Use custom aggregator
        # self.lin = NoisyLinear(3 * in_channels, out_channels)
        #self.lin = Linear(in_channels, out_channels)
        self.mlp = Seq(
            #Linear(in_channels, out_channels)
            Linear(in_channels, in_channels * 2),
            ReLU(),
            Linear(in_channels * 2, out_channels)
        )

    def forward(self, x, edge_index, agents_messages=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index):
        '''
        BE CAREFUL: when passing the adjacency matrix, add symmetric edges for layers greater than 1 because we
        also consider joining paths, not only choice points.
        '''
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # x_j_message = self.agents_messages[edge_index] # retrieve other
        # agent's message if present on node j

        tmp = torch.cat([x_i, x_j], dim=1)
        # tmp = torch.cat([x_i, x_j, x_j_message], dim=1)  # tmp has shape [E,
        # 2 * in_channels]
        return self.mlp(tmp)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        dim = self.node_dim
        out_mean = scatter(inputs, index, dim=dim, dim_size=dim_size,
                           reduce='mean')
        out_max = scatter(inputs, index, dim=dim, dim_size=dim_size,
                          reduce='max')
        out_min = scatter(inputs, index, dim=dim, dim_size=dim_size,
                          reduce='min')
        return torch.cat([out_mean, out_max, out_min], dim=-1)
