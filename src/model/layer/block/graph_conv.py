import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, LayerNorm
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

class ResidualBlock(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 dropout: float = 0.
                 ):
        super(ResidualBlock, self).__init__()
        self.conv1 = GINEConv(nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        ), eps=0, train_eps=True)
        
        self.norm1 = LayerNorm(node_dim, affine=True)

        self.conv2 = GINEConv(nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        ), eps=0, train_eps=True)
        
        self.norm2 = LayerNorm(node_dim, affine=True)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr: Tensor,
                node2graph: OptTensor = None
                ):
        out = self.conv1(x, edge_index, edge_attr)
        out = self.norm1(out, node2graph)
        out = self.relu(out)

        out = self.conv2(out, edge_index, edge_attr)
        out = self.norm2(out, node2graph)

        out = (out + x) / 2
        return self.relu(out)