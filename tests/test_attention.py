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
        assert out.shape == (B, L, d_k)
        assert weights.shape == (B, L, L)

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
        _, weights = attn(Q, K, V, mask)
        for i in range(L):
            for j in range(i + 1, L):
                assert weights[0, i, j] == 0.0, f"Токен {i} не должен attendить к {j}"

    def test_identical_qkv_no_nan(self, attn):
        B, L, d_k = 1, 5, 64
        x = torch.randn(B, L, d_k)
        out, weights = attn(x, x, x)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isnan(weights))
        assert weights.shape == (B, L, L)

    def test_fully_masked_row_no_nan(self, attn):
        B, L, d_k = 1, 4, 64
        Q = torch.randn(B, L, d_k)
        K = torch.randn(B, L, d_k)
        V = torch.randn(B, L, d_k)
        mask = torch.zeros(B, L, L, dtype=torch.bool)
        mask[:, 0, :] = True
        out, _ = attn(Q, K, V, mask)
        assert not torch.any(torch.isnan(out)), "Полностью замаскированная строка даёт NaN"

    def test_scaling_factor(self):
        d_k = 256
        attn = ScaledDotProductAttention(d_k)
        assert attn.scale == 1.0 / math.sqrt(d_k)
        L = 10
        Q = torch.ones(1, L, d_k) * 10.0
        out = attn(Q, Q, Q)
        assert not torch.any(torch.isnan(out[0])), "NaN при больших значениях"

    def test_gradients_flow(self, attn):
        B, L, d_k = 1, 3, 64
        Q = torch.randn(B, L, d_k, requires_grad=True)
        K = torch.randn(B, L, d_k, requires_grad=True)
        V = torch.randn(B, L, d_k, requires_grad=True)
        out, _ = attn(Q, K, V)
        loss = out.sum()
        loss.backward()
        assert Q.grad is not None and not torch.any(torch.isnan(Q.grad))
        assert K.grad is not None and not torch.any(torch.isnan(K.grad))
        assert V.grad is not None and not torch.any(torch.isnan(V.grad))


class TestMultiHeadAttention:
    @pytest.fixture
    def mha(self):
        return MultiHeadAttention(d_model=512, n_heads=8, n_kv_heads=4, dropout=0.0)

    def test_output_shape(self, mha):
        B, L = 2, 16
        x = torch.randn(B, L, 512)
        out = mha(x)
        assert out.shape == (B, L, 512)

    def test_gqa_kv_repeat(self, mha):
        B, L = 1, 6
        x = torch.randn(B, L, 512)
        mha.eval()
        _, attn = mha(x, return_attn=True)
        assert attn.shape == (1, 8, 6, 6)
        assert mha.n_rep == 2

    def test_causal_mask_works(self, mha):
        B, L = 1, 10
        x = torch.randn(B, L, 512)
        out = mha(x)
        assert not torch.any(torch.isnan(out))

    def test_without_gqa(self):
        mha = MultiHeadAttention(d_model=256, n_heads=8, dropout=0.0)
        assert mha.n_kv_heads == 8
        assert mha.n_rep == 1
        B, L = 2, 12
        x = torch.randn(B, L, 256)
        out = mha(x)
        assert out.shape == (B, L, 256)

    def test_invalid_gqa_config_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=512, n_heads=10, n_kv_heads=3)

    def test_projection_changes_representation(self, mha):
        B, L = 2, 8
        x = torch.randn(B, L, 512)
        out = mha(x)
        assert not torch.equal(out, torch.zeros_like(out)), "Весь выход — ноль"
        assert not torch.allclose(out, x, atol=1e-3), "Выход = вход"

    def test_different_lengths(self, mha):
        B = 1
        out_short = mha(torch.randn(B, 5, 512))
        out_long = mha(torch.randn(B, 20, 512))
        assert out_short.shape == (1, 5, 512)
        assert out_long.shape == (1, 20, 512)

    def test_return_attn_false(self, mha):
        B, L = 1, 4
        x = torch.randn(B, L, 512)
        out = mha(x, return_attn=False)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, L, 512)

    def test_user_mask_batch(self):
        mha = MultiHeadAttention(d_model=128, n_heads=4, dropout=0.0)
        B, L = 3, 6
        x = torch.randn(B, L, 128)
        mask = torch.triu(torch.full((B, L, L), True, dtype=torch.bool), diagonal=1)
        out = mha(x, mask=mask)
        assert out.shape == (B, L, 128)
        assert not torch.any(torch.isnan(out))

    def test_gqa_vs_vanilla_equivalence_with_shared_weights(self):
        """GQA и full MHA дают одинаковый выход при общих W_k, W_v и W_o."""
        torch.manual_seed(42)
        d_model, L = 128, 6
        d_k = d_model // 4
        x = torch.randn(2, L, d_model)

        gqa = MultiHeadAttention(d_model, n_heads=4, n_kv_heads=2, dropout=0.0)
        full = MultiHeadAttention(d_model, n_heads=4, n_kv_heads=4, dropout=0.0)

        with torch.no_grad():
            full.W_q.weight.copy_(gqa.W_q.weight)
            gqa_k = gqa.W_k.weight
            for h in range(full.n_heads):
                kv_idx = h // gqa.n_rep
                full.W_k.weight[h * d_k : (h + 1) * d_k].copy_(
                    gqa_k[kv_idx * d_k : (kv_idx + 1) * d_k]
                )
            gqa_v = gqa.W_v.weight
            for h in range(full.n_heads):
                kv_idx = h // gqa.n_rep
                full.W_v.weight[h * d_k : (h + 1) * d_k].copy_(
                    gqa_v[kv_idx * d_k : (kv_idx + 1) * d_k]
                )
            full.W_o.weight.copy_(gqa.W_o.weight)

            out_gqa = gqa(x)
            out_full = full(x)

        assert torch.allclose(out_gqa, out_full, atol=1e-5), (
            "GQA и full MHA с общими весами должны совпадать"
        )

    def test_gradients_flow(self, mha):
        B, L = 1, 4
        x = torch.randn(B, L, 512, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.any(torch.isnan(x.grad))
        for name, param in mha.named_parameters():
            assert param.grad is not None, f"Нет градиента у {name}"
            assert not torch.any(torch.isnan(param.grad)), f"NaN в градиенте {name}"
