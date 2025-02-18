from torch import FloatTensor
import torch.nn as nn

class TerminationPredictionModel(nn.Module):

    def __init__(self, 
                 core_graph_vector_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.):
        super(TerminationPredictionModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(core_graph_vector_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.activation = nn.Sigmoid()

    def forward(self,
                Z_core: FloatTensor,
                return_logit: bool = False):
        logit = self.mlp(Z_core).squeeze(-1)
        if return_logit:
            return logit
        else:
            return self.activation(logit)