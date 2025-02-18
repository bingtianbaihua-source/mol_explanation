import torch
from torch import FloatTensor
import torch.nn as nn

class MultiHeadPropertyClassificationModel(nn.Module):
    def __init__(self, 
                 core_graph_vector_dim: int,
                 num_classes_list: list,
                 hidden_dim: int = 128,
                 dropout: float = 0.):
        super(MultiHeadPropertyClassificationModel, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Linear(core_graph_vector_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Create separate classification heads for each task
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, num_classes),
                nn.Softmax(dim=-1)
            ) for num_classes in num_classes_list
        ])

    def forward(self, Z_core: FloatTensor):
        shared_features = self.shared_mlp(Z_core)
        
        # Apply each classification head to the shared features
        outputs = [head(shared_features) for head in self.classification_heads]
        
        return outputs