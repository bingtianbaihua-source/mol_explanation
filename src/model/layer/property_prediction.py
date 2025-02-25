import torch
from torch import FloatTensor, Tensor
import torch.nn as nn

class MultiHeadPropertyClassificationModel(nn.Module):
    def __init__(self, 
                 core_graph_vector_dim: int,
                 num_classes_list: list,
                 hidden_dim: int = 128,
                 dropout: float = 0.):
        super().__init__()

        # Shared feature extractor
        self.shared_mlp = nn.Sequential(
            nn.Linear(core_graph_vector_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Task-specific heads (output logits without Softmax)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)  # 移除Softmax
            for num_classes in num_classes_list
        ])

    def forward(self, Z_core: FloatTensor) -> list[Tensor]:
        shared_features = self.shared_mlp(Z_core)
        return [head(shared_features) for head in self.heads]

class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights: list[float] = None):
        super().__init__()
        self.task_weights = task_weights

    def forward(self, outputs: list[Tensor], targets: list[Tensor]) -> Tensor:
        """
        Args:
            outputs: List of task logits [ (B, C1), (B, C2), ... ]
            targets: List of task labels [ (B,), (B,), ... ]
        """
        total_loss = 0.0
        for i, (logits, y) in enumerate(zip(outputs, targets)):
            # 计算每个任务的交叉熵损失
            loss = nn.functional.cross_entropy(logits, y)
            
            # 应用任务权重
            weight = 1.0 if self.task_weights is None else self.task_weights[i]
            total_loss += weight * loss
            
        return total_loss