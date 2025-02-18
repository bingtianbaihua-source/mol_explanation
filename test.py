import torch
import torch.nn as nn
from torch import FloatTensor

class SimpleEmbeddingModel(nn.Module):
    def __init__(self,
                 property_keys: list[str],
                 embedding_dims: dict[str, int],
                 hidden_dim: int = 128,
                 dropout: float = 0.):
        super(SimpleEmbeddingModel, self).__init__()
        
        self.property_keys = property_keys
        self.embeddings = nn.ModuleDict({
            key: nn.Embedding(num_embeddings, embedding_dim)
            for key, (num_embeddings, embedding_dim) in embedding_dims.items()
        })
        
        input_dim = sum(embedding_dim for _, embedding_dim in embedding_dims.values())

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),  # Assuming binary classification
            nn.Sigmoid()
        )

    def embedding_property(self, property: dict[str, int]):
        assert sorted(property.keys()) == sorted(self.property_keys), f'Invalid keys, Input: {set(property.keys())} Required {set(self.property_keys)}'
        
        embedded_properties = [self.embeddings[key](torch.tensor([value])) for key, value in property.items()]
        return torch.cat(embedded_properties, dim=-1)

    def forward(self, property: dict[str, int]):
        embedded_property = self.embedding_property(property)
        return self.mlp(embedded_property)

# Example usage
if __name__ == "__main__":
    property_keys = ['color', 'shape', 'size']
    embedding_dims = {
        'color': (10, 5),  # 10 possible colors, embedding dim 5
        'shape': (7, 3),   # 7 possible shapes, embedding dim 3
        'size': (5, 2)     # 5 possible sizes, embedding dim 2
    }
    
    model = SimpleEmbeddingModel(property_keys, embedding_dims, hidden_dim=128, dropout=0.5)
    
    # Dummy input
    property_input = {
        'color': 3,
        'shape': 1,
        'size': 4
    }
    
    output = model(property_input)
    print(output)