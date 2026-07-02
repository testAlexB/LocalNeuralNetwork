import pytest
import torch
import math

from fishingllm.attention import ScaledDotProductAttention, MultiHeadAttention


class TestScaledDotProductAttention:
    @pytest.fixture
    def attn(self):
        return ScaledDotProductAttention(d_k=64)

    def test_output_shape(self, attn):
        B, L, d_k = 2, 8, 64
        Q = torch.randn(B, L, d_k)
        K = torch.randn(B, L, d_k)
        V = torch.randn(B, L, d_k)
        out, weights = attn(Q, K, V)
        assert out.shape == (B, L, d_k), f"Ожидал ({B},{L},{d_k}), получил {out.shape}"
        assert weights.shape == (B, L, L), f"Ожидал ({B},{L},{L}), получил {weights.shape}"

    def test_attention_weights_sum_to_one(self, attn):
        B, L, d_k = 1, 5, 64
        Q = torch.randn(B, L, d_k)
        K = torch.randn(B, L, d_k)
        V = torch.randn(B, L, d_k)
        _, weights = attn(Q, K, V)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B, L), atol=1e-6)

    def test_causal_mask_prevents_future_attention(self, attn):
        B, L, d_k = 1, 4, 64
        Q = torch.randn(B, L, d_k)
        K = torch.randn(B, L, d_k)
        V = torch.randn(B, L, d_k)
        mask = torch.triu(torch.full((L, L), True, dtype=torch.bool), diagonal=1).unsqueeze(0)
        out, weights = attn(Q, K, V, mask)
        for i in range(L):
            for j in range(i + 1, L):
                assert weights[0, i, j] == 0.0, f"Токен {i} не должен attendить к {j}"

    def test_identical_qkv_gives_diagonal_attention(self, attn):
        B, L, d_k = 1, 3, 64
        x = torch.randn(B, L, d_k)
        _, weights = attn(x, x, x)
        diag = torch.diagonal(weights[0])
        assert torch.all(diag > 0.5), "Ожидал, что диагональ доминирует при Q=K=V"

    def test_scaling_factor(self):
        d_k = 256
        attn = ScaledDotProductAttention(d_k)
        assert attn.scale == 1.0 / math.sqrt(d_k), f"scale={attn.scale}, ожидал {1/math.sqrt(d_k)}"
        L = 10
        Q = torch.ones(1, L, d_k) * 10.0
        out = attn(Q, Q, Q)
        assert not torch.any(torch.isnan(out[0])), "NaN при больших значениях — масштабирование не работает"


class TestMultiHeadAttention:
    @pytest.fixture
    def mha(self):
        return MultiHeadAttention(d_model=512, n_heads=8, n_kv_heads=4, dropout=0.0)

    def test_output_shape(self, mha):
        B, L = 2, 16
        x = torch.randn(B, L, 512)
        out = mha(x)
        assert out.shape == (B, L, 512), f"Ожидал ({B},{L},512), получил {out.shape}"

    def test_gqa_kv_repeat(self, mha):
        B, L = 1, 6
        x = torch.randn(B, L, 512)
        mha.eval()
        out, attn = mha(x, return_attn=True)
        assert attn.shape == (1, 8, 6, 6), f"Ожидал (1,8,6,6), получил {attn.shape}"
        assert mha.n_rep == 2

    def test_causal_mask_works(self, mha):
        B, L = 1, 10
        x = torch.randn(B, L, 512)
        out = mha(x)
        assert not torch.any(torch.isnan(out)), "NaN в выходе с маской"

    def test_without_gqa(self):
        mha = MultiHeadAttention(d_model=256, n_heads=8, dropout=0.0)
        assert mha.n_kv_heads == 8
        assert mha.n_rep == 1
        B, L = 2, 12
        x = torch.randn(B, L, 256)
        out = mha(x)
        assert out.shape == (B, L, 256)

    def test_residual_connection_not_broken(self, mha):
        B, L = 2, 8
        x = torch.randn(B, L, 512)
        out = mha(x)
        assert not torch.equal(out, torch.zeros_like(out)), "Весь выход — ноль (инициализация сломана)"
        assert not torch.allclose(out, x, atol=1e-3), "Выход = вход (проекции не обучаются)"

    def test_different_lengths(self, mha):
        B = 1
        x_short = torch.randn(B, 5, 512)
        x_long = torch.randn(B, 20, 512)
        out_short = mha(x_short)
        out_long = mha(x_long)
        assert out_short.shape == (1, 5, 512)
        assert out_long.shape == (1, 20, 512)

    def test_gqa_vs_vanilla_equivalence_single_token(self):
        """GQA и full MHA должны совпадать для одного токена (нет KV-повторов)."""
        torch.manual_seed(42)
        d_model, L = 128, 1
        x = torch.randn(2, L, d_model)
        gqa = MultiHeadAttention(d_model, n_heads=4, n_kv_heads=2, dropout=0.0)
        full = MultiHeadAttention(d_model, n_heads=4, n_kv_heads=4, dropout=0.0)
        with torch.no_grad():
            out_gqa = gqa(x)
            out_full = full(x)
        assert out_gqa.shape == out_full.shape, "Размеры GQA и full MHA не совпадают"
