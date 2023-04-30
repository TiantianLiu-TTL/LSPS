import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from utils.utils import get_activation
from torch_geometric.utils import add_self_loops, degree
# from utils.utils import get_activation
class EGraphSage(MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels,
                 edge_channels, activation, edge_mode,
                 normalize_emb,
                 aggr):
        super(EGraphSage, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.edge_mode = edge_mode

        if edge_mode == 0:
            self.message_lin = nn.Linear(in_channels, out_channels)
            self.attention_lin = nn.Linear(2*in_channels+edge_channels, 1)
        elif edge_mode == 1:
            self.message_lin = nn.Linear(in_channels+edge_channels, out_channels)
        elif edge_mode == 2:
            self.message_lin = nn.Linear(2*in_channels+edge_channels, out_channels)
        elif edge_mode == 3:
            self.message_lin = nn.Sequential(
                    nn.Linear(2*in_channels+edge_channels, out_channels),
                    get_activation(activation),
                    nn.Linear(out_channels, out_channels),
                    )
        elif edge_mode == 4:
            self.message_lin = nn.Linear(in_channels, out_channels*edge_channels)
        elif edge_mode == 5:
            self.message_lin = nn.Linear(2*in_channels, out_channels*edge_channels)

        self.agg_lin = nn.Linear(in_channels+out_channels, out_channels)

        self.message_activation = get_activation(activation)
        self.update_activation = get_activation(activation)
        self.normalize_emb = normalize_emb

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]
        if self.edge_mode == 0:
            attention = self.attention_lin(torch.cat((x_i,x_j, edge_attr),dim=-1))
            m_j = attention * self.message_activation(self.message_lin(x_j))
        elif self.edge_mode == 1:
            m_j = torch.cat((x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 2 or self.edge_mode == 3:
            m_j = torch.cat((x_i,x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 4:
            E = x_j.shape[0]
            w = self.message_lin(x_j)
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        elif self.edge_mode == 5:
            E = x_j.shape[0]
            w = self.message_lin(torch.cat((x_i,x_j),dim=-1))
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        return m_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x),dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class EdgeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]

        tmp = torch.cat([x_i, x_j-x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

class IncreGnn(MessagePassing):
    def __init__(self):
        super(IncreGnn, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        pass

    def message(self, x_i, x_j):
        pass

    def update(self, aggrout):
        return aggrout

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError


class MLPNet(torch.nn.Module):
    def __init__(self,
         		input_dims, output_dim,
         		hidden_layer_sizes=(64,),
         		hidden_activation='relu',
         		output_activation=None,
                dropout=0.):
        super(MLPNet, self).__init__()

        layers = nn.ModuleList()
        input_dim = np.sum(input_dims)
        for layer_size in hidden_layer_sizes:
        	hidden_dim = layer_size
        	layer = nn.Sequential(
        				nn.Linear(input_dim, hidden_dim),
        				get_activation(hidden_activation),
        				nn.Dropout(dropout),
        				)
        	layers.append(layer)
        	input_dim = hidden_dim

        layer = nn.Sequential(
        				nn.Linear(input_dim, output_dim),
        				get_activation(output_activation),
        				)
       	layers.append(layer)
       	self.layers = layers

    def forward(self, inputs):
    	if torch.is_tensor(inputs):
    		inputs = [inputs]
    	input_var = torch.cat(inputs,-1)
    	for layer in self.layers:
    		input_var = layer(input_var)
    	return input_var




