import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """Одно головое self-attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V."""

    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Q, K, V: (batch, seq_len, d_k)
        mask: (batch, seq_len, seq_len) — True там, где запрещено attendить
        Returns: (batch, seq_len, d_k)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, L, L)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn


class MultiHeadAttention(nn.Module):
    """Multi-head с GQA-совместимой структурой: n_heads, d_k на голову."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} не делится на n_heads={n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_k = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads  # сколько раз повторяем KV для GQA

        self.W_q = nn.Linear(d_model, self.n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(self.n_heads * self.d_k, d_model, bias=False)
        self.attn = ScaledDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((seq_len, seq_len), True, dtype=torch.bool, device=device), diagonal=1)
        return mask.unsqueeze(0)  # (1, L, L)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, L = x.shape[0], x.shape[1]

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)

        if self.n_rep > 1:
            K = K.repeat_interleave(self.n_rep, dim=1)
            V = V.repeat_interleave(self.n_rep, dim=1)

        if mask is None:
            mask = self._create_causal_mask(L, x.device)

        attn_output, attn_weights = self.attn(
            Q.reshape(-1, L, self.d_k),
            K.reshape(-1, L, self.d_k),
            V.reshape(-1, L, self.d_k),
            mask,
        )

        attn_output = attn_output.view(B, self.n_heads, L, self.d_k).transpose(1, 2).reshape(B, L, -1)
        attn_output = self.dropout(self.W_o(attn_output))

        if return_attn:
            attn_weights = attn_weights.view(B, self.n_heads, L, L)
            return attn_output, attn_weights
        return attn_output
