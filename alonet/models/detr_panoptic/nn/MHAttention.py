# Taken from
# https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/segmentation.py
"""
Multi head attention used in PanopticHead.
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[Tensor] = None) -> torch.Tensor:
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)

        """ Einsum is not supported in tensorRT. Change original operation"""
        # qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        # kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        # weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads, 1, 1)
        kh = k.view(k.shape[0], 1, self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = (qh * kh * self.normalize_fact).sum(dim=3)  # Output shape = [bqnhw]

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights
