import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from torch_scatter import scatter_sum, scatter_mean

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from torch_scatter import scatter_sum

class Readout(nn.Module):
    def __init__(self,
                 node_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 global_input_dim: int = 0,
                 dropout: float = 0.):
        super(Readout, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Dropout(p=dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + global_input_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self,
                x: FloatTensor,
                node2graph: LongTensor = None,
                global_x: FloatTensor = None):
        x = self.fc1(x) * self.fc2(x)
        if node2graph is not None:
            Z1 = scatter_sum(x, node2graph, dim=0)
            Z2 = scatter_mean(x, node2graph, dim=0)
        else:
            Z1 = x.sum(dim=0, keepdim=True)
            Z2 = x.mean(dim=0, keepdim=True)
        if global_x is not None:
            Z = torch.cat([Z1, Z2, global_x], dim=-1)
        else:
            Z = torch.cat([Z1, Z2], dim=-1)
        return self.fc3(Z)