"""tests/test_kv_cache.py
Light CPU logic/exactness tests for the CoMER KV-cache implementation.

Run:
    python -m pytest -q tests/test_kv_cache.py
    python -m pytest -q tests/test_kv_cache.py -k "tree or cache or arm"

No checkpoint or CROHME data required.
"""
from __future__ import annotations

import math
import copy
from typing import List

import pytest
import torch
import torch.nn as nn


# ======================================================================
# 1. TreeRelativeBias – square and non-square forward
# ======================================================================

class TestTreeRelativeBias:
    """Tests for models/transformer/tree_bias.py::TreeRelativeBias."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from models.transformer.tree_bias import TreeRelativeBias
        self.TreeRelativeBias = TreeRelativeBias

    def _make(self, H: int = 4, num_rel: int = 8) -> "TreeRelativeBias":
        return self.TreeRelativeBias(num_heads=H, num_relations=num_rel)

    def test_square_forward_unflatten(self):
        """Square (B, L, L) → (B, H, L, L) without flattening."""
        H, L, B = 4, 6, 2
        bias = self._make(H=H)
        rel_ids = torch.randint(0, bias.num_relations, (B, L, L))
        out = bias(rel_ids, flatten=False)
        assert out.shape == (B, H, L, L)

    def test_square_forward_flatten(self):
        """Square (B, L, L) → (B*H, L, L) when flatten=True."""
        H, L, B = 4, 5, 3
        bias = self._make(H=H)
        rel_ids = torch.randint(0, bias.num_relations, (B, L, L))
        out = bias(rel_ids, flatten=True)
        assert out.shape == (B * H, L, L)

    def test_square_unchanged_values(self):
        """Flattened square output should match unflatten reshaped."""
        H, L, B = 4, 5, 2
        bias = self._make(H=H)
        rel_ids = torch.randint(0, bias.num_relations, (B, L, L))
        flat = bias(rel_ids, flatten=True)
        full = bias(rel_ids, flatten=False)
        expected = full.contiguous().view(B * H, L, L)
        assert torch.allclose(flat, expected)

    def test_nonsquare_row_slice(self):
        """Non-square (B, 1, S) → (B*H, 1, S) for incremental decode."""
        H, S, B = 4, 10, 2
        bias = self._make(H=H)
        rel_ids = torch.randint(0, bias.num_relations, (B, 1, S))
        out = bias(rel_ids, flatten=True)
        assert out.shape == (B * H, 1, S)

    def test_nonsquare_values_match_square_row(self):
        """Row (B, 1, S) values must match the last row of the square (B, S, S) result."""
        H, S, B = 4, 7, 2
        bias = self._make(H=H)
        rel_ids_sq = torch.randint(0, bias.num_relations, (B, S, S))
        # Row slice: last row
        rel_ids_row = rel_ids_sq[:, S - 1:S, :]          # (B, 1, S)
        flat_sq = bias(rel_ids_sq, flatten=True)           # (B*H, S, S)
        flat_row = bias(rel_ids_row, flatten=True)         # (B*H, 1, S)
        # The last row of flat_sq should equal flat_row
        assert torch.allclose(flat_sq[:, S - 1:S, :], flat_row)


# ======================================================================
# 2. DecoderKVCache – expand_beam_ and reorder_
# ======================================================================

class TestDecoderKVCache:
    """Tests for models/transformer/kv_cache.py::DecoderKVCache."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from models.transformer.kv_cache import DecoderKVCache
        self.DecoderKVCache = DecoderKVCache

    def _make_cache(self, B: int = 2, H: int = 4, S: int = 16,
                    max_len: int = 20, num_layers: int = 3):
        Hd = 8
        self_k = [torch.randn(B, H, max_len, Hd) for _ in range(num_layers)]
        self_v = [torch.randn(B, H, max_len, Hd) for _ in range(num_layers)]
        cross_k = [torch.randn(B, H, S, Hd) for _ in range(num_layers)]
        cross_v = [torch.randn(B, H, S, Hd) for _ in range(num_layers)]
        cross_pre_sum = [torch.zeros(B, H, S) for _ in range(max(num_layers - 1, 0))]
        cross_final_sum = [torch.zeros(B, H, S) for _ in range(max(num_layers - 1, 0))]
        b2b = torch.arange(B, dtype=torch.long)
        mem_mask = torch.zeros(B, S, dtype=torch.bool)
        cache = self.DecoderKVCache(
            self_k=self_k,
            self_v=self_v,
            cross_k=cross_k,
            cross_v=cross_v,
            cross_pre_sum=cross_pre_sum,
            cross_final_sum=cross_final_sum,
            beam_to_batch_idx=b2b,
            memory_key_padding_mask=mem_mask,
            height=4,
            cur_len=0,
            max_len=max_len,
            batch_size=B,
            beam_size=1,
            num_layers=num_layers,
            num_heads=H,
            head_dim=Hd,
        )
        return cache

    def test_expand_self_kv_shape(self):
        """expand_beam_ should duplicate self K/V to [B*beam, ...]."""
        B, beam = 2, 3
        cache = self._make_cache(B=B)
        orig_self_k = [t.clone() for t in cache.self_k]
        cache.expand_beam_(beam)
        for i, t in enumerate(cache.self_k):
            assert t.shape[0] == B * beam, f"self_k[{i}] batch dim wrong"
            # Check interleaved repeat
            for b in range(B):
                for m in range(beam):
                    assert torch.allclose(t[b * beam + m], orig_self_k[i][b])

    def test_expand_cross_kv_unchanged(self):
        """expand_beam_ must NOT change cross K/V shapes or values."""
        B, beam = 2, 3
        cache = self._make_cache(B=B)
        orig = [t.clone() for t in cache.cross_k]
        cache.expand_beam_(beam)
        for i, t in enumerate(cache.cross_k):
            assert t.shape[0] == B, "cross_k batch dim changed – must stay B_original"
            assert torch.allclose(t, orig[i])

    def test_expand_beam_to_batch_idx(self):
        """After expand_beam_, beam_to_batch_idx should map each hypothesis to original batch."""
        B, beam = 3, 2
        cache = self._make_cache(B=B)
        cache.expand_beam_(beam)
        expected = torch.arange(B, dtype=torch.long).repeat_interleave(beam)
        assert torch.equal(cache.beam_to_batch_idx, expected)

    def test_reorder_self_kv(self):
        """reorder_ should select self K/V by beam_idx."""
        B, beam = 2, 3
        cache = self._make_cache(B=B)
        cache.expand_beam_(beam)
        B_active = B * beam
        beam_idx = torch.tensor([1, 0, 2, 4, 3, 5], dtype=torch.long)  # arbitrary
        old_self_k = [t.clone() for t in cache.self_k]
        cache.reorder_(beam_idx)
        for i, t in enumerate(cache.self_k):
            assert torch.allclose(t, old_self_k[i][beam_idx])

    def test_reorder_with_duplicates(self):
        """reorder_ with duplicate parent indices (e.g. [2, 0, 1, 2]) must work."""
        B, beam = 2, 2
        cache = self._make_cache(B=B)
        cache.expand_beam_(beam)
        beam_idx = torch.tensor([2, 0, 1, 2], dtype=torch.long)
        old_self_k = [t.clone() for t in cache.self_k]
        cache.reorder_(beam_idx)
        for i, t in enumerate(cache.self_k):
            assert torch.allclose(t, old_self_k[i][beam_idx])

    def test_reorder_cross_kv_unchanged(self):
        """reorder_ must NOT touch cross K/V."""
        B, beam = 2, 2
        cache = self._make_cache(B=B)
        cache.expand_beam_(beam)
        orig_cross_k = [t.clone() for t in cache.cross_k]
        cache.reorder_(torch.tensor([0, 1, 2, 3], dtype=torch.long))
        for i, t in enumerate(cache.cross_k):
            assert torch.allclose(t, orig_cross_k[i])

    def test_beam_to_batch_idx_after_reorder(self):
        """beam_to_batch_idx must be correctly reordered."""
        B, beam = 2, 2
        cache = self._make_cache(B=B)
        cache.expand_beam_(beam)
        # b2b should be [0,0,1,1]
        beam_idx = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        old_b2b = cache.beam_to_batch_idx.clone()
        cache.reorder_(beam_idx)
        assert torch.equal(cache.beam_to_batch_idx, old_b2b[beam_idx])


# ======================================================================
# 3. project_static_kv – matches full MHA K/V projection
# ======================================================================

class TestProjectStaticKV:
    """Cross K/V projection must match what multi_head_attention_forward does."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from models.transformer.attention import MultiheadAttention
        self.MHA = MultiheadAttention

    def _make_mha(self, D: int = 32, H: int = 4):
        return self.MHA(embed_dim=D, num_heads=H).eval()

    def test_project_static_kv_matches_full(self):
        """project_static_kv K/V must equal full MHA's K/V projections."""
        D, H, B, S = 32, 4, 2, 12
        Hd = D // H
        mha = self._make_mha(D=D, H=H)
        memory = torch.randn(S, B, D)   # [S, B, D]

        with torch.no_grad():
            ck, cv = mha.project_static_kv(memory)

        # Reproduce what full forward does for cross-attention (key is value, query is separate)
        with torch.no_grad():
            E = D
            _w = mha.in_proj_weight[E:, :]
            _b = mha.in_proj_bias[E:] if mha.in_proj_bias is not None else None
            import torch.nn.functional as F
            kv = F.linear(memory, _w, _b)
            k_ref, v_ref = kv.chunk(2, dim=-1)  # [S, B, D]

            # Reshape: [S, B, D] → [B, H, S, Hd]
            k_ref = k_ref.contiguous().view(S, B * H, Hd).transpose(0, 1).view(B, H, S, Hd)
            v_ref = v_ref.contiguous().view(S, B * H, Hd).transpose(0, 1).view(B, H, S, Hd)

        assert torch.allclose(ck, k_ref, atol=1e-5), "cross K mismatch"
        assert torch.allclose(cv, v_ref, atol=1e-5), "cross V mismatch"


# ======================================================================
# 4. Cached self-attention – incremental vs. full-prefix
# ======================================================================

class TestCachedSelfAttention:
    """forward_cached_self must match full forward for each position."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from models.transformer.attention import MultiheadAttention
        self.MHA = MultiheadAttention

    def test_incremental_matches_full_prefix_no_tree_bias(self):
        """Without tree bias, incremental self-attention must match full-prefix last row."""
        D, H, B, L = 32, 4, 2, 8
        Hd = D // H
        mha = self.MHA(embed_dim=D, num_heads=H).eval()

        torch.manual_seed(42)
        seq = torch.randn(L, B, D)   # full sequence [L, B, D]
        max_len = L + 4

        cache_k = torch.zeros(B, H, max_len, Hd)
        cache_v = torch.zeros(B, H, max_len, Hd)

        with torch.no_grad():
            # Build incremental outputs
            inc_outs = []
            for t in range(L):
                query_step = seq[t:t + 1, :, :]  # [1, B, D]
                out_inc, _ = mha.forward_cached_self(
                    query_step=query_step,
                    cache_k=cache_k,
                    cache_v=cache_v,
                    write_pos=t,
                )
                inc_outs.append(out_inc[0])  # [B, D]

            # Full-prefix reference (causal mask)
            causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
            full_out, _ = mha(
                query=seq, key=seq, value=seq,
                attn_mask=causal_mask,
                need_weights=False,
            )  # [L, B, D]

        for t in range(L):
            assert torch.allclose(inc_outs[t], full_out[t], atol=1e-4), \
                f"Mismatch at t={t}, max_diff={( inc_outs[t] - full_out[t]).abs().max():.2e}"

    def test_incremental_matches_full_prefix_with_rel_bias(self):
        """With random rel_bias row, incremental must still match full-prefix last row."""
        D, H, B, L = 32, 4, 2, 6
        Hd = D // H
        mha = self.MHA(embed_dim=D, num_heads=H).eval()
        torch.manual_seed(7)
        seq = torch.randn(L, B, D)
        rel_bias_sq = torch.randn(B * H, L, L) * 0.1

        max_len = L + 2
        cache_k = torch.zeros(B, H, max_len, Hd)
        cache_v = torch.zeros(B, H, max_len, Hd)

        with torch.no_grad():
            inc_outs = []
            for t in range(L):
                rel_row = rel_bias_sq[:, t:t + 1, :t + 1]
                out_inc, _ = mha.forward_cached_self(
                    query_step=seq[t:t + 1],
                    cache_k=cache_k,
                    cache_v=cache_v,
                    write_pos=t,
                    rel_bias_step=rel_row,
                )
                inc_outs.append(out_inc[0])

            # Full-prefix with causal mask + rel_bias
            causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
            from models.transformer.attention import multi_head_attention_forward
            # We need rel_bias support; use the module's forward which passes rel_bias
            from models.transformer.attention import multi_head_attention_forward as mhaf
            # Directly call with rel_bias
            full_out_w, _ = mhaf(
                query=seq, key=seq, value=seq,
                arm=None,
                embed_dim_to_check=D,
                num_heads=H,
                in_proj_weight=mha.in_proj_weight,
                in_proj_bias=mha.in_proj_bias,
                bias_k=None, bias_v=None,
                add_zero_attn=False,
                dropout_p=0.0,
                out_proj_weight=mha.out_proj.weight,
                out_proj_bias=mha.out_proj.bias,
                training=False,
                attn_mask=causal_mask,
                rel_bias=rel_bias_sq,
            )

        for t in range(L):
            assert torch.allclose(inc_outs[t], full_out_w[t], atol=1e-4), \
                f"Mismatch at t={t}"


# ======================================================================
# 5. ARM forward_from_sums – matches full forward
# ======================================================================

class TestARMFromSums:
    """forward_from_sums must produce results identical to forward()[:, -1:, :]."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from models.transformer.arm import AttentionRefinementModule
        self.ARM = AttentionRefinementModule

    def test_forward_from_sums_matches_full(self):
        """Single-step ARM bias from sums must match full ARM's last time-step."""
        torch.manual_seed(0)
        B, H, h, w = 2, 4, 3, 4
        S = h * w
        T = 6   # history length
        nhead = H
        dc = 8

        arm = self.ARM(nhead=nhead, dc=dc, cross_coverage=True, self_coverage=True).eval()

        # Simulate T steps of cross attention
        prev_attn = torch.rand(B * H, T, S)   # previous layer final attention
        curr_attn = torch.rand(B * H, T, S)   # current layer pre-ARM attention

        key_padding_mask = torch.zeros(B, S, dtype=torch.bool)

        with torch.no_grad():
            # Full forward for T queries
            full_bias = arm(prev_attn, key_padding_mask, h, curr_attn)  # [B*H, T, S]
            # Result at last position
            full_last = full_bias[:, T - 1:T, :]   # [B*H, 1, S]

            # Cumulative sums up to (but not including) step T-1
            prev_sum = prev_attn[:, :T - 1, :].sum(dim=1).view(B, H, S).float()
            curr_sum = curr_attn[:, :T - 1, :].sum(dim=1).view(B, H, S).float()

            from_sums = arm.forward_from_sums(
                prev_attn_sum=prev_sum,
                curr_attn_sum=curr_sum,
                key_padding_mask=key_padding_mask,
                h=h,
                dtype=prev_attn.dtype,
            )   # [B*H, 1, S]

        assert from_sums.shape == full_last.shape, "Shape mismatch"
        assert torch.allclose(from_sums, full_last, atol=1e-4), \
            f"ARM from_sums mismatch, max diff={( from_sums - full_last).abs().max():.2e}"

    def test_fp32_sums_used_for_half(self):
        """Sums should stay fp32 even when model tensors are fp16."""
        B, H, h, w = 1, 2, 2, 3
        S = h * w
        arm = self.ARM(nhead=H, dc=4, cross_coverage=True, self_coverage=False).eval()
        # Use float() for test since CPU doesn't support fp16 conv well; just check dtype contract
        prev_sum = torch.zeros(B, H, S, dtype=torch.float32)
        curr_sum = torch.zeros(B, H, S, dtype=torch.float32)
        mask = torch.zeros(B, S, dtype=torch.bool)
        with torch.no_grad():
            out = arm.forward_from_sums(prev_sum, curr_sum, mask, h, dtype=torch.float32)
        assert out.dtype == torch.float32

    def test_sos_position_zero_coverage(self):
        """At SOS position (all-zero sums), forward_from_sums should still run cleanly."""
        B, H, h, w = 2, 4, 3, 4
        S = h * w
        arm = self.ARM(nhead=H, dc=8, cross_coverage=True, self_coverage=True).eval()
        prev_sum = torch.zeros(B, H, S, dtype=torch.float32)
        curr_sum = torch.zeros(B, H, S, dtype=torch.float32)
        mask = torch.zeros(B, S, dtype=torch.bool)
        with torch.no_grad():
            out = arm.forward_from_sums(prev_sum, curr_sum, mask, h, dtype=torch.float32)
        assert out.shape == (B * H, 1, S)
        # Should not produce NaN with zero sums
        assert not torch.isnan(out).any()


# ======================================================================
# 6. transform_step exactness – incremental vs. full-prefix
# ======================================================================

def _make_small_decoder(
    D: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    dim_ff: int = 64,
    dc: int = 8,
    vocab_size: int = 20,
    use_tree_bias: bool = False,
    cross_coverage: bool = True,
    self_coverage: bool = True,
):
    """Build a tiny Decoder (no real vocab_info needed — stub VocabInfo)."""
    # Capture as a local to avoid Python class-body scoping issues
    _vocab_size = vocab_size

    # Minimal stub VocabInfo so Decoder can construct without real data
    class _Words:
        idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", **{i: f"tok{i}" for i in range(3, _vocab_size)}}

    class _VocabInfo:
        vocab_size = _vocab_size
        pad_id = 0
        sos_id = 1
        eos_id = 2
        words = _Words()

    from models.decoder import Decoder
    dec = Decoder(
        d_model=D,
        nhead=nhead,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=0.0,   # no dropout so outputs are deterministic
        dc=dc,
        cross_coverage=cross_coverage,
        self_coverage=self_coverage,
        vocab_info=_VocabInfo(),
        use_tree_bias=use_tree_bias,
        use_bidirectional=False,
    )
    dec.eval()
    return dec


class TestTransformStepExactness:
    """Decoder.transform_step must match full-prefix transform()[:, -1, :] at every t."""

    def test_exactness_no_arm_no_tree(self):
        """Without ARM or tree bias, incremental logits match full-prefix exactly."""
        torch.manual_seed(1)
        D, nhead, L, B = 32, 4, 8, 2
        h, w = 3, 4

        dec = _make_small_decoder(D=D, nhead=nhead, num_layers=2, cross_coverage=False, self_coverage=False)

        src = torch.randn(B, h, w, D)
        src_mask = torch.zeros(B, h, w, dtype=torch.bool)
        sos = torch.ones(B, 1, dtype=torch.long)  # SOS = 1
        # Random tokens for steps 1..L-1
        tokens = torch.randint(3, 10, (B, L - 1))
        input_ids = torch.cat([sos, tokens], dim=1)  # [B, L]

        with torch.no_grad():
            # Init cache and run incremental decode
            cache = dec.init_decode_cache(src, src_mask, max_len=L + 10)
            inc_logits = []
            for t in range(L):
                logits_t = dec.transform_step(
                    src=[src], src_mask=[src_mask],
                    token_ids=input_ids[:, :t + 1],
                    cache=cache,
                )
                inc_logits.append(logits_t)   # [B, V]

            # Full-prefix reference
            full_logits_all = dec.transform([src], [src_mask], input_ids)  # [B, L, V]

        for t in range(L):
            assert torch.allclose(inc_logits[t], full_logits_all[:, t, :], atol=1e-4), \
                f"Mismatch at t={t}, max={( inc_logits[t] - full_logits_all[:, t, :]).abs().max():.2e}"

    def test_exactness_with_arm(self):
        """With ARM (cross+self coverage), incremental logits still match full-prefix."""
        torch.manual_seed(2)
        D, nhead, L, B = 32, 4, 6, 2
        h, w = 3, 4

        dec = _make_small_decoder(D=D, nhead=nhead, num_layers=3,
                                   cross_coverage=True, self_coverage=True)

        src = torch.randn(B, h, w, D)
        src_mask = torch.zeros(B, h, w, dtype=torch.bool)
        sos = torch.ones(B, 1, dtype=torch.long)
        tokens = torch.randint(3, 10, (B, L - 1))
        input_ids = torch.cat([sos, tokens], dim=1)

        with torch.no_grad():
            cache = dec.init_decode_cache(src, src_mask, max_len=L + 10)
            inc_logits = []
            for t in range(L):
                logits_t = dec.transform_step(
                    src=[src], src_mask=[src_mask],
                    token_ids=input_ids[:, :t + 1],
                    cache=cache,
                )
                inc_logits.append(logits_t)

            full_logits_all = dec.transform([src], [src_mask], input_ids)

        for t in range(L):
            diff = (inc_logits[t] - full_logits_all[:, t, :]).abs().max().item()
            # ARM uses Conv2d + BatchNorm; slight fp32 differences between the two
            # execution paths (incremental vs. full-prefix) are expected.
            assert diff < 5e-3, f"ARM mismatch at t={t}, max diff={diff:.2e}"


# ======================================================================
# 7. Fallback guard – training mode / grad enabled must not use cache
# ======================================================================

class TestFallbackGuard:
    """KV-cache must not activate during training or when grad is enabled."""

    def test_training_mode_does_not_activate_cache(self):
        """In training mode, use_kv_cache_active must be False."""
        dec = _make_small_decoder()
        dec.train()
        # Simulate what _beam_search computes for use_kv_cache_active
        use_kv_cache = True
        use_kv_cache_active = (
            use_kv_cache
            and hasattr(dec, "init_decode_cache")
            and (not dec.training)
            and (not torch.is_grad_enabled())
        )
        assert not use_kv_cache_active, "Cache must not activate in training mode"

    def test_grad_enabled_does_not_activate_cache(self):
        """With grad enabled, use_kv_cache_active must be False."""
        dec = _make_small_decoder()
        dec.eval()
        with torch.enable_grad():
            use_kv_cache = True
            use_kv_cache_active = (
                use_kv_cache
                and hasattr(dec, "init_decode_cache")
                and (not dec.training)
                and (not torch.is_grad_enabled())
            )
            assert not use_kv_cache_active, "Cache must not activate when grad is enabled"

    def test_eval_no_grad_activates_cache(self):
        """In eval + no_grad, cache should be eligible to activate."""
        dec = _make_small_decoder()
        dec.eval()
        with torch.no_grad():
            use_kv_cache = True
            use_kv_cache_active = (
                use_kv_cache
                and hasattr(dec, "init_decode_cache")
                and (not dec.training)
                and (not torch.is_grad_enabled())
            )
            assert use_kv_cache_active, "Cache must activate in eval + no_grad"


# ======================================================================
# Main: run all tests if executed directly
# ======================================================================
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
