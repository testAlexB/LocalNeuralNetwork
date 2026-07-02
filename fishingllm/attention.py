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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn


class MultiHeadAttention(nn.Module):
    """Multi-head с GQA-совместимой структурой: n_heads, d_k на голову."""

    _mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, (
                f"n_heads={n_heads} не делится на n_kv_heads={n_kv_heads}"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_k = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads

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
        key = (seq_len, device)
        if key in MultiHeadAttention._mask_cache:
            return MultiHeadAttention._mask_cache[key]
        mask = torch.triu(torch.full((seq_len, seq_len), True, dtype=torch.bool, device=device), diagonal=1)
        mask = mask.unsqueeze(0)
        MultiHeadAttention._mask_cache[key] = mask
        return mask

    @staticmethod
    def clear_mask_cache():
        MultiHeadAttention._mask_cache.clear()

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads для GQA без копирования памяти (expand вместо repeat_interleave)."""
        if self.n_rep == 1:
            return x
        B, H, L, D = x.shape
        x = x.unsqueeze(2).expand(B, H, self.n_rep, L, D)
        return x.reshape(B, H * self.n_rep, L, D)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, L = x.shape[0], x.shape[1]

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)

        K = self._expand_kv(K)
        V = self._expand_kv(V)

        if mask is None:
            mask = self._create_causal_mask(L, x.device)
        elif mask.dim() == 3 and mask.shape[0] == B:
            mask = mask.unsqueeze(1).expand(B, self.n_heads, L, L).reshape(-1, L, L)

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
