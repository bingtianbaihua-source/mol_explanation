import torch
from torch import FloatTensor, Tensor
import torch.nn as nn
from torch.nn import functional as F

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
    def __init__(self, 
                 task_weights: list[float] = None,
                 class_weights: list[Tensor] = None):
        super().__init__()
        self.task_weights = task_weights
        self.class_weights = class_weights

    def forward(self, 
                outputs: list[Tensor], 
                targets: list[Tensor],
                return_metrics: bool = False) -> Tensor:
        """
        Args:
            outputs: List of task logits [ (B, C1), (B, C2), ... ]
            targets: List of task labels [ (B,), (B,), ... ]
        """
        total_loss = 0.0
        task_losses = []
        metrics = {}

        for i, (logits, y) in enumerate(zip(outputs, targets)):
            
            weight = self.class_weights[i].to(y.device) if self.class_weights else None
            loss = F.cross_entropy(
                logits, y,
                weight=weight,
            )

            task_losses.append(loss.detach())
            metrics[f'task{i}/loss'] = loss.item()

            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            metrics[f'task{i}/acc'] = acc.item()

            task_weight = 1.0 if self.task_weights is None else self.task_weights[i]
            total_loss += task_weight * loss

        metrics['total_loss'] = total_loss.item()
        return (total_loss, metrics) if return_metrics else total_loss            
            
class DynamicWeightLoss(MultiTaskLoss):

    def forward(self, outputs:list[Tensor], targets: list[Tensor]):
        task_losses = []
        with torch.no_grad():
            for i, (logits, y) in enumerate(zip(outputs, targets)):
                loss = F.cross_entropy(logits, y, weight=self.class_weights[i] if self.class_weights else None)
                task_losses.append(loss)
        
        total_loss = sum(task_losses)
        weights = [loss/total_loss for loss in task_losses]

        total_loss = 0.0
        for i, (logits,y) in enumerate(zip(outputs, targets)):
            loss = F.cross_entropy(logits, y, weight=self.class_weights[i] if self.class_weights else None)
            total_loss += weights[i] * loss

        return total_loss