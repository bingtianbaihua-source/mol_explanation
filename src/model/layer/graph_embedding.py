import torch
import torch.nn as nn
from torch import LongTensor
from torch_geometric.typing import Adj
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_scatter import scatter_sum, scatter_mean
from model.layer.block.graph_conv import ResidualBlock
from model.layer.block.readout import Readout
from utils.typing import NodeVector, EdgeVector, GlobalVector

class GraphEmbeddingModel(nn.Module):
    def __init__(self, 
                 node_input_dim: int,
                 edge_input_dim: int,
                 global_input_dim: int = 0,
                 hidden_dim: int = 128,
                 graph_vector_dim: int = 0,
                 n_block: int = 2,
                 dropout: float = 0.):
        super(GraphEmbeddingModel, self).__init__()
        
        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_dim + global_input_dim, hidden_dim),
            nn.SiLU()
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(node_dim=hidden_dim, edge_dim=hidden_dim, dropout=dropout)
            for _ in range(n_block)
        ])

        self.final_node_embedding = nn.Sequential(
            nn.Linear(hidden_dim + node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )

        if graph_vector_dim > 0:
            self.readout = Readout(node_dim=hidden_dim, 
                                   hidden_dim=graph_vector_dim,
                                   output_dim=graph_vector_dim, 
                                   global_input_dim=global_input_dim,
                                   dropout=dropout)
        else:
            self.readout = None

    def forward(self,
                x_inp: NodeVector,
                edge_index: Adj,
                edge_attr: EdgeVector,
                global_x: GlobalVector = None,
                node2graph: LongTensor = None):
        x = self.concat_features(x_inp, global_x, node2graph)
        x_emb = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for block in self.blocks:
            x_emb = block(x_emb, edge_index, edge_attr, node2graph)

        x_emb = torch.cat([x_emb, x_inp], dim=-1)
        x_emb = self.final_node_embedding(x_emb)

        if self.readout is not None:
            Z = self.readout(x_emb, node2graph, global_x)
        else:
            Z = None

        return x_emb, Z
    
    def forward_batch(self,
                      batch: PyGBatch | PyGData):
        node2graph = batch.batch if isinstance(batch, PyGBatch) else None
        global_x = batch.get('global_x', None)
        return self.forward(batch.x, batch.edge_index, batch.edge_attr, global_x, node2graph)

    def concat_features(self,
                        x: NodeVector,
                        global_x: GlobalVector,
                        node2graph: LongTensor):
        if global_x is not None:
            global_x = global_x[node2graph] if node2graph is not None else global_x.repeat(x.size(0), 1)
            x = torch.cat([x, global_x], dim=-1)
        return x