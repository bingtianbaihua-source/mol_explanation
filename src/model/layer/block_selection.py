import torch
import torch.nn as nn
from torch import FloatTensor

class BlockSelectionModel(nn.Module):
    def __init__(self,
                 core_graph_vector_dim: int,
                 block_graph_vector_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.):
        super(BlockSelectionModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(core_graph_vector_dim + block_graph_vector_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self,
                Z_core: FloatTensor,
                Z_block: FloatTensor):
        Z_concat = torch.cat([Z_core, Z_block], dim=-1)
        return self.mlp(Z_concat).squeeze(-1)