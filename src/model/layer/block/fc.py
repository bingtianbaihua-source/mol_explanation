import torch
from torch import nn

class Linear(nn.Sequential):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bias: bool = True,
            dropout = 0.,
    ):
        nonlinear_layer = nn.SiLU
        norm_layer = nn.LayerNorm
        super(Linear, self).__init__(
            nn.Linear(input_dim, output_dim, bias=bias),
            norm_layer,
            nonlinear_layer,
            nn.Dropout(p=dropout)
        )