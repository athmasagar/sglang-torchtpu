"""Numerics of the multimodal_gen TPU splash attention backend.

Every dispatch branch of ``tpu_attention`` is checked against a float32
reference: the custom Pallas kernel (long KV, including ragged KV tails and a
short sequence-parallel query shard), stock ``make_splash_mha`` for short
aligned cross-attention and medium or causal shapes (including zero-padded
KV, whose padded keys must not take softmax weight), and the SDPA fallback.

The numeric tests need the Pallas kernels (TPU v7x, torch_tpu, jax) and are
skipped otherwise; each asserts which path ran, so a lane without the kernels
cannot pass by comparing SDPA with SDPA.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=150, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.test_utils import CustomTestCase


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


def _reference(q, k, v, scale, is_causal):
    """float32 attention on CPU, chunked over queries."""
    q, k, v = q.float(), k.float(), v.float()
    q_seq, kv_seq = q.shape[2], k.shape[2]
    out = torch.empty_like(q)
    for start in range(0, q_seq, 1024):
        stop = min(start + 1024, q_seq)
        scores = torch.matmul(q[:, :, start:stop], k.transpose(-2, -1)) * scale
        if is_causal:
            rows = torch.arange(start, stop)[:, None]
            cols = torch.arange(kv_seq)[None, :]
            scores = scores.masked_fill(cols > rows, float("-inf"))
        out[:, :, start:stop] = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out


def _rel_l2(got, want):
    return ((got.float() - want).norm() / want.norm()).item()


def _inputs(batch, heads, q_seq, kv_seq, *, q_gain=1.0, negative_scores=False):
    gen = torch.Generator().manual_seed(q_seq * 7919 + kv_seq)
    shape_q = (batch, heads, q_seq, 128)
    shape_kv = (batch, heads, kv_seq, 128)
    q = torch.randn(shape_q, generator=gen) * q_gain
    k = torch.randn(shape_kv, generator=gen)
    v = torch.randn(shape_kv, generator=gen)
    if negative_scores:
        # Every real logit is strongly negative (mean -64 / sqrt(128) = -5.7),
        # so a zero-padded key (logit 0) would dominate the softmax if it were
        # not masked.
        direction = torch.ones(128) / 128**0.5
        q = q * 0.1 + 8.0 * direction
        k = k * 0.1 - 8.0 * direction
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuSplashAttentionNumerics(CustomTestCase):
    def _check(
        self,
        batch,
        heads,
        q_seq,
        kv_seq,
        *,
        expect,
        is_causal=False,
        scale=128**-0.5,
        **kwargs,
    ):
        """``expect`` is the path that must run: "flash", "splash" or "sdpa"."""
        from sglang_tpu.multimodal_gen import tpu_splash_attn as tsa

        if not tsa._check_splash_available():
            self.skipTest("Pallas kernels unavailable (needs TPU v7x and jax)")
        tsa._register_ops()
        q, k, v = _inputs(batch, heads, q_seq, kv_seq, **kwargs)
        want = _reference(q, k, v, scale, is_causal)
        q, k, v = q.to("tpu"), k.to("tpu"), v.to("tpu")
        with (
            patch.object(tsa, "_FLASH_OP", wraps=tsa._FLASH_OP) as flash,
            patch.object(tsa, "_SPLASH_MHA_OP", wraps=tsa._SPLASH_MHA_OP) as splash,
        ):
            got = tsa.tpu_attention(q, k, v, scale=scale, is_causal=is_causal)
        got = got.to("cpu")
        ran = {"flash": (1, 0), "splash": (0, 1), "sdpa": (0, 0)}[expect]
        self.assertEqual((flash.call_count, splash.call_count), ran, expect)
        sdpa = F.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=is_causal
        ).to("cpu")
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(got).any().item())
        rel, sdpa_rel = _rel_l2(got, want), _rel_l2(sdpa, want)
        err = (got.float() - want).abs().max().item()
        sdpa_err = (sdpa.float() - want).abs().max().item()
        print(
            f"{self._testMethodName}: rel-L2 {rel:.3e} (sdpa {sdpa_rel:.3e}), "
            f"max abs {err:.3e} (sdpa {sdpa_err:.3e})"
        )
        # Relative L2 over the whole output: a padded key taking softmax weight
        # stays under a max-abs bound but moves rel-L2 by ~1e-1. bf16 inputs
        # bound the achievable accuracy, so allow twice bf16 SDPA's error.
        self.assertLess(rel, max(1e-2, 2 * sdpa_rel), "rel-L2")
        self.assertLess(err, max(2e-2, 2 * sdpa_err), "max abs")

    # Custom kernel: kv_seq >= 2048.
    def test_custom_two_blocks_and_a_ragged_tail(self):
        # 4608 = 2 * 2048 + 512.
        self._check(1, 4, 4608, 4608, expect="flash")

    def test_custom_whole_kv_blocks(self):
        # 4096 = 2 * 2048: no partial last block.
        self._check(1, 2, 4096, 4096, expect="flash")

    def test_custom_single_kv_block(self):
        # One KV block: the first, last and only grid step coincide.
        self._check(1, 2, 2048, 2048, expect="flash")

    def test_custom_softmax_scale(self):
        # A scale other than head_dim ** -0.5, which equals the default.
        self._check(1, 2, 4128, 4128, expect="flash", scale=0.05)

    def test_custom_ragged_kv_tail(self):
        # 4128 = 2 * 2048 + 32: the last KV block is a 32-row partial block.
        self._check(1, 2, 4128, 4128, expect="flash")

    def test_custom_odd_kv_tail(self):
        # 8191 = 3 * 2048 + 2047: an odd-length partial last block.
        self._check(1, 2, 8191, 8191, expect="flash")

    def test_custom_short_query_shard(self):
        # A sequence-parallel query shard attending the gathered KV.
        self._check(1, 2, 880, 3520, expect="flash")

    def test_custom_batched(self):
        # B=2, as with batched CFG.
        self._check(2, 2, 8190, 8190, expect="flash")

    def test_custom_high_logit_gain(self):
        # Large logits make the softmax peaked and amplify rounding.
        self._check(1, 2, 8190, 8190, expect="flash", q_gain=8.0)

    def test_custom_negative_scores(self):
        self._check(1, 2, 4128, 4128, expect="flash", negative_scores=True)

    # make_splash_mha, long query against short aligned KV.
    def test_cross_attention_aligned_kv(self):
        self._check(1, 4, 4096, 512, expect="splash")

    def test_cross_attention_kv_not_multiple_of_1024(self):
        self._check(1, 2, 4096, 1536, expect="splash", negative_scores=True)

    # make_splash_mha, medium and causal shapes.
    def test_medium_padded_kv_is_masked(self):
        # 1000 pads to 1024: 24 zero keys that must stay masked.
        self._check(1, 4, 1000, 1000, expect="splash", negative_scores=True)

    def test_medium_aligned(self):
        self._check(1, 2, 1024, 1024, expect="splash")

    def test_short_query_long_padded_kv(self):
        # q_seq < 256 with kv >= 2048: make_splash_mha with segment-id masking.
        self._check(1, 2, 200, 4128, expect="splash", negative_scores=True)

    def test_long_query_unaligned_kv(self):
        # 1024 < kv < 2048, not 128-aligned: KV padded to 1536, masked.
        self._check(1, 2, 4096, 1500, expect="splash", negative_scores=True)

    def test_causal_padded(self):
        self._check(1, 2, 1000, 1000, expect="splash", is_causal=True)

    # Short or unaligned-short KV stays on SDPA.
    def test_short_sequences(self):
        self._check(1, 4, 300, 77, expect="sdpa")

    def test_unaligned_short_kv(self):
        self._check(1, 2, 4096, 1000, expect="sdpa")

    def test_gqa(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            tpu_attention,
        )

        q, k, v = _inputs(1, 4, 4608, 4608)
        k, v = k[:, :2], v[:, :2]
        scale = 128**-0.5
        want = _reference(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), scale, False
        )
        got = tpu_attention(q.to("tpu"), k.to("tpu"), v.to("tpu"), scale=scale)
        self.assertLess((got.to("cpu").float() - want).abs().max().item(), 2e-2)

    def test_gqa_make_splash_mha(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            tpu_attention,
        )

        q, k, v = _inputs(1, 4, 4096, 512)
        k, v = k[:, :2], v[:, :2]
        scale = 128**-0.5
        want = _reference(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), scale, False
        )
        got = tpu_attention(q.to("tpu"), k.to("tpu"), v.to("tpu"), scale=scale)
        self.assertLess((got.to("cpu").float() - want).abs().max().item(), 2e-2)

    def test_varlen_segments_use_sdpa(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        lens = (3000, 2500)
        q, k, v = (
            torch.randn(sum(lens), 2, 128, dtype=torch.bfloat16) for _ in range(3)
        )
        cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32)
        with patch.object(
            tpu_splash_attn, "tpu_attention", side_effect=AssertionError("kernel")
        ):
            got = impl.forward_varlen(
                q.to("tpu"),
                k.to("tpu"),
                v.to("tpu"),
                cu_seqlens=cu.to("tpu"),
                max_seqlen=max(lens),
                cu_seqlens_host=(0, lens[0], sum(lens)),
            ).to("cpu")
        start = 0
        for n in lens:
            seg = slice(start, start + n)
            want = _reference(
                q[None, seg].transpose(1, 2),
                k[None, seg].transpose(1, 2),
                v[None, seg].transpose(1, 2),
                128**-0.5,
                False,
            ).transpose(1, 2)[0]
            self.assertLess((got[seg].float() - want).abs().max().item(), 2e-2)
            start += n

    def test_backend_impl_layout(self):
        # The attention layer calls the impl with [B, S, H, D] tensors.
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            TpuSplashAttentionImpl,
        )

        impl = TpuSplashAttentionImpl(
            num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        q, k, v = _inputs(1, 4, 4608, 4608)
        want = _reference(q, k, v, 128**-0.5, False).transpose(1, 2)
        got = impl.forward(
            q.transpose(1, 2).to("tpu"),
            k.transpose(1, 2).to("tpu"),
            v.transpose(1, 2).to("tpu"),
            None,
        ).to("cpu")
        self.assertEqual(got.shape, want.shape)
        self.assertLess((got.float() - want).abs().max().item(), 2e-2)

    def test_torch_compile_output_layout(self):
        # Dynamo traces the op through its fake impl, whose strides must match
        # the kernel's contiguous output for the layer's reshape to lower.
        import torch_tpu._internal.compile  # noqa: F401

        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            TpuSplashAttentionImpl,
        )

        impl = TpuSplashAttentionImpl(
            num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5
        )

        def attn(q, k, v):
            out = impl.forward(q, k, v, None)
            return out.reshape(out.shape[0], out.shape[1], -1)

        # Custom kernel, and make_splash_mha with a padded query.
        for seq in (4608, 1000):
            with self.subTest(seq=seq):
                q, k, v = _inputs(1, 4, seq, seq)
                want = _reference(q, k, v, 128**-0.5, False).transpose(1, 2)
                compiled = torch.compile(attn, backend="tpu", dynamic=False)
                # Contiguous [B, S, H, D], as the layer's projections produce.
                got = compiled(
                    *(t.transpose(1, 2).contiguous().to("tpu") for t in (q, k, v))
                ).to("cpu")
                self.assertEqual(got.shape, (1, seq, 4 * 128))
                err = (got.float() - want.reshape(1, seq, -1)).abs().max().item()
                self.assertLess(err, 2e-2)

    def test_float32_falls_back_to_sdpa(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            tpu_attention,
        )

        q, k, v = (t.float() for t in _inputs(1, 2, 2048, 2048))
        got = tpu_attention(q.to("tpu"), k.to("tpu"), v.to("tpu")).to("cpu")
        self.assertEqual(got.dtype, torch.float32)
        want = _reference(q, k, v, 128**-0.5, False)
        self.assertLess((got - want).abs().max().item(), 2e-2)


class TestTpuSplashDispatch(CustomTestCase):
    """Kernel selection logic; runs without a TPU."""

    def test_custom_kernel_selection(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            _use_custom_kernel,
        )

        self.assertTrue(_use_custom_kernel(75600, 75600, False))
        self.assertTrue(_use_custom_kernel(880, 3520, False))
        self.assertFalse(_use_custom_kernel(75600, 512, False))
        self.assertFalse(_use_custom_kernel(4096, 4096, True))

    def test_cross_attn_block_kv_divides_kv(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            _cross_attn_block_kv,
        )

        for kv_seq in range(128, 2048, 128):
            block_kv = _cross_attn_block_kv(kv_seq)
            self.assertEqual(kv_seq % block_kv, 0)
            self.assertLessEqual(block_kv, 1024)

    def test_splash_mha_blocks(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            _splash_mha_block_sizes,
        )

        self.assertEqual(_splash_mha_block_sizes(4096, 1536, False), (2048, 512))
        self.assertEqual(_splash_mha_block_sizes(4096, 4096, True), (1024, 1024))
        self.assertEqual(_splash_mha_block_sizes(1000, 1000, True), (512, 512))
        self.assertEqual(_splash_mha_block_sizes(200, 4128, False), (256, 256))

    def test_kernel_source_keys_the_compile_cache_once(self):
        import torch.compiler.config as compiler_config

        from sglang_tpu.multimodal_gen import tpu_splash_attn

        with patch.object(compiler_config, "cache_key_tag", "user-tag"):
            tpu_splash_attn._add_kernel_source_to_compile_cache_key()
            tpu_splash_attn._add_kernel_source_to_compile_cache_key()
            tags = compiler_config.cache_key_tag.split("|")
        self.assertEqual(tags[0], "user-tag")
        self.assertEqual(len(tags), 2)
        self.assertTrue(tags[1].startswith("diffusion-tpu-splash-attn:"))

    def test_sdpa_fallback_reasons(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        def reason(q_shape, kv_shape, *, is_causal=False, dtype=torch.bfloat16):
            q = torch.empty(q_shape, dtype=dtype, device="meta")
            k = torch.empty(kv_shape, dtype=dtype, device="meta")
            return tpu_splash_attn._sdpa_fallback_reason(q, k, k, is_causal)

        with patch.object(
            tpu_splash_attn, "_check_splash_available", return_value=True
        ):
            self.assertIsNone(reason((1, 4, 4608, 128), (1, 4, 4608, 128)))
            self.assertIsNone(reason((1, 4, 4096, 128), (1, 2, 512, 128)))
            self.assertEqual(reason((1, 4, 300, 128), (1, 4, 77, 128)), "short")
            self.assertEqual(reason((1, 4, 4096, 128), (1, 4, 1000, 128)), "short")
            self.assertEqual(
                reason((1, 4, 1000, 128), (1, 4, 2000, 128), is_causal=True),
                "ragged causal",
            )
            self.assertEqual(
                reason((1, 4, 4608, 64), (1, 4, 4608, 64)), "head_dim=64/64"
            )
            self.assertEqual(reason((1, 3, 4608, 128), (1, 2, 4608, 128)), "heads=3/2")
            self.assertEqual(
                reason((1, 4, 4608, 128), (1, 4, 4608, 128), dtype=torch.float32),
                "dtype=torch.float32/torch.float32/torch.float32",
            )
            self.assertEqual(reason((1, 4, 0, 128), (1, 4, 4608, 128)), "empty")
        with patch.object(
            tpu_splash_attn, "_check_splash_available", return_value=False
        ):
            self.assertEqual(
                reason((1, 4, 4608, 128), (1, 4, 4608, 128)), "pallas unavailable"
            )


if __name__ == "__main__":
    unittest.main()
