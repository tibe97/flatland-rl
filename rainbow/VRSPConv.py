import torch
import math
from torch import Tensor, nn
from typing import Optional
from torch.nn import Sequential as Seq, Linear, ReLU, functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import segment_csr, scatter
from collections import defaultdict


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
          return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
          return F.linear(x, self.weight_mu, self.bias_mu)


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
        super(VRSPConv, self).__init__(aggr=None, flow="target_to_source") # Use custom aggregator
        # self.lin = NoisyLinear(3 * in_channels, out_channels)
        #self.lin = Linear(in_channels, out_channels)
        self.mlp = Seq(
            Linear(in_channels, in_channels//2),
            ReLU(),
            Linear(in_channels//2, out_channels)
        )
        self.agents_messages = {}
        

    def forward(self, x, edge_index, agents_messages=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        if agents_messages is not None:
            self.agents_messages = agents_messages
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index):
        '''
        BE CAREFUL: when passing the adjacency matrix, add symmetric edges for layers greater than 1 because we 
        also consider joining paths, not only choice points.
        '''
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # x_j_message = self.agents_messages[edge_index] # retrieve other agent's message if present on node j
        
        tmp = torch.cat([x_i, x_j], dim=1)
        # tmp = torch.cat([x_i, x_j, x_j_message], dim=1)  # tmp has shape [E, 2 * in_channels]
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