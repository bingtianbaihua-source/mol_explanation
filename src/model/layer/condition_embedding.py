import torch
from torch import FloatTensor, LongTensor
import torch.nn as nn

class ConditionEmbeddingModel(nn.Module):
    def __init__(self,
                 core_node_vector_dim: int,
                 core_graph_vector_dim: int,
                 condition_dim: dict[str, int],
                 dropout: float = 0.):
        super(ConditionEmbeddingModel, self).__init__()

        self.property_keys = condition_dim
        self.embeddings = nn.ModuleDict({
            key: nn.Embedding(num_embeddings, embedding_dim)
            for key, (num_embeddings, embedding_dim) in condition_dim.items()
        })

        self.node_mlp = nn.Sequential(
            nn.Linear(core_node_vector_dim + condition_dim, core_node_vector_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(core_node_vector_dim, core_node_vector_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )
        self.graph_mlp = nn.Sequential(
            nn.Linear(core_graph_vector_dim + condition_dim, core_graph_vector_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(core_graph_vector_dim, core_graph_vector_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self,
                h_core: FloatTensor,
                Z_core: FloatTensor,
                condition: dict[str, int],
                node2graph: LongTensor = None):
        if node2graph is not None:
            h_condition = condition[node2graph]
        else:
            h_condition = condition.repeat(h_core.size(0), 1)
        
        Z_condition = self.embedding_property(condition)
        h_core = torch.cat([h_core, h_condition], dim=-1)
        Z_core = torch.cat([Z_core, Z_condition], dim=-1)
        return self.node_mlp(h_core), self.graph_mlp(Z_core)
    
    def embedding_property(self, property: dict[str, int]):
        assert sorted(property.keys()) == sorted(self.property_keys), f'Invalid keys, Input: {set(property.keys())} Required {set(self.property_keys)}'
        
        embedded_properties = [self.embeddings[key](torch.tensor([value])) for key, value in property.items()]
        return torch.cat(embedded_properties, dim=-1)