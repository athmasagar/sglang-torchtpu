"""TPU CI entry for the all-gather ring attention dispatch tests (CPU logic)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.server_args.server_args import (
    TPU_RING_CAPABLE_ATTENTION_BACKENDS,
)
from sglang.multimodal_gen.test.unit.test_usp_allgather_ring import (  # noqa: F401
    TestAllGatherRingAttention,
)
from sglang.multimodal_gen.test.unit.test_usp_ring_tail_pad import (  # noqa: F401
    TestRingTailPadDispatch,
)
from sglang_tpu.multimodal_gen import tpu_splash_attn
from sglang_tpu.multimodal_gen.tpu_splash_attn import (
    TpuSplashAttentionBackend,
    TpuSplashAttentionImpl,
)


class TestTpuSplashRingCapability(unittest.TestCase):
    def test_declares_ring_support(self):
        self.assertTrue(TpuSplashAttentionBackend.supports_ring_rotation())
        self.assertEqual(
            TPU_RING_CAPABLE_ATTENTION_BACKENDS,
            (TpuSplashAttentionBackend.get_enum().name.lower(),),
        )

    def test_only_the_tpu_impl_resumes_in_kernel(self):
        self.assertTrue(TpuSplashAttentionImpl.supports_ring_resume())
        self.assertFalse(SDPAImpl.supports_ring_resume())
        self.assertFalse(FlashAttentionBackend.get_impl_cls().supports_ring_resume())

    def test_auto_selected_ring_backend_serves_sdpa_layers(self):
        # Without --attention-backend a TPU ring launch selects tpu_splash_attn
        # implicitly; a DiT layer that lists TORCH_SDPA must still take it.
        from sglang.multimodal_gen.configs.pipeline_configs.base import (
            PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.layers.attention import selector
        from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
        from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
        from sglang.multimodal_gen.runtime.server_args import server_args
        from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

        with patch.object(PipelineConfig, "from_kwargs", return_value=PipelineConfig()):
            args = ServerArgs.from_dict({"model_path": "/fake"})
        args.num_gpus, args.ring_degree = 8, 2
        platforms = "sglang.multimodal_gen.runtime.platforms"
        with patch(f"{platforms}.current_platform.is_tpu", return_value=True):
            args._adjust_attention_backend()
        self.assertEqual(args.attention_backend, "tpu_splash_attn")
        self.assertFalse(args.is_arg_explicitly_set("attention_backend"))

        selector._cached_get_attn_backend.cache_clear()
        try:
            with (
                patch(f"{platforms}.current_platform", TpuPlatform()),
                patch.object(server_args, "_global_server_args", args),
            ):
                backend = selector.get_attn_backend(
                    128,
                    torch.bfloat16,
                    supported_attention_backends=set(
                        BaseDiT._supported_attention_backends
                    ),
                )
        finally:
            selector._cached_get_attn_backend.cache_clear()
        self.assertIs(backend, TpuSplashAttentionBackend)


class TestRingKernelFallback(unittest.TestCase):
    """Ring steps the Pallas kernel does not take (here: short shapes) run a
    chunked float32 reference."""

    def setUp(self):
        self.tsa = tpu_splash_attn
        torch.manual_seed(0)
        self.batch, self.heads, self.kv_heads, self.dim = 2, 4, 2, 64
        self.scale = self.dim**-0.5

    def _rows_per_chunk(self, rows, kv_len):
        # Bytes for `rows` query rows of float32 scores: forces several chunks.
        return patch.object(
            self.tsa,
            "_REFERENCE_SCORE_BYTES",
            4 * self.batch * self.heads * kv_len * rows,
        )

    def test_chained_steps_match_attention_over_the_real_keys(self):
        b, h, hkv, d = self.batch, self.heads, self.kv_heads, self.dim
        q_len, shard, pad = 48, 40, 7
        q = torch.randn(b, h, q_len, d)
        keys = torch.randn(2, b, hkv, shard, d)
        values = torch.randn(2, b, hkv, shard, d)
        # Padding past the real keys may hold anything, NaN included.
        keys[1, :, :, shard - pad :] = float("nan")
        values[1, :, :, shard - pad :] = float("nan")
        real = 2 * shard - pad
        carry = None
        with self._rows_per_chunk(5, shard):
            for i in range(2):
                info = torch.tensor(
                    [i * b * hkv, min(max(real - i * shard, 0), shard)],
                    dtype=torch.int32,
                )
                out, carry = self.tsa.tpu_ring_attention_step(
                    q, keys.flatten(0, 1), values.flatten(0, 1), info, carry, self.scale
                )
        k_all = keys.permute(1, 2, 0, 3, 4).reshape(b, hkv, 2 * shard, d)[:, :, :real]
        v_all = values.permute(1, 2, 0, 3, 4).reshape(b, hkv, 2 * shard, d)[:, :, :real]
        k_all = k_all.repeat_interleave(h // hkv, 1)
        v_all = v_all.repeat_interleave(h // hkv, 1)
        want = torch.softmax(q @ k_all.transpose(-1, -2) * self.scale, -1) @ v_all
        self.assertFalse(out.isnan().any())
        torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-5)

    def test_an_empty_shard_keeps_the_carry(self):
        b, hkv, d = self.batch, self.kv_heads, self.dim
        q = torch.randn(b, self.heads, 24, d)
        keys, values = torch.randn(b, hkv, 32, d), torch.randn(b, hkv, 32, d)
        full = torch.tensor([0, 32], dtype=torch.int32)
        out, carry = self.tsa.tpu_ring_attention_step(
            q, keys, values, full, None, self.scale
        )
        empty = torch.tensor([0, 0], dtype=torch.int32)
        again, _ = self.tsa.tpu_ring_attention_step(
            q, keys, values, empty, carry, self.scale
        )
        torch.testing.assert_close(again, out)

    def test_chunked_lse_attention_matches_full_scores(self):
        b, h, d = self.batch, self.heads, self.dim
        q, k, v = (torch.randn(b, h, 30, d) for _ in range(3))
        for is_causal in (False, True):
            with self.subTest(is_causal=is_causal):
                scores = q @ k.transpose(-1, -2) * self.scale
                if is_causal:
                    scores = scores.masked_fill(
                        torch.ones(30, 30, dtype=torch.bool).triu(1), float("-inf")
                    )
                with self._rows_per_chunk(4, 30):
                    out, lse = self.tsa.tpu_attention_with_lse(
                        q, k, v, self.scale, is_causal=is_causal
                    )
                torch.testing.assert_close(lse, torch.logsumexp(scores, -1))
                torch.testing.assert_close(out, torch.softmax(scores, -1) @ v)


if __name__ == "__main__":
    unittest.main()
