"""TPU CI entry for the all-gather ring attention tests: the dispatch and
reference logic, and the ring with the Pallas kernel on device."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers import usp
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


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


def _float32_attention(q, k, v):
    """float32 attention on CPU; [B, S, H, D] in and out."""
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    return (torch.softmax(scores, dim=-1) @ v).transpose(1, 2)


def _rel_l2(got, want):
    return ((got.float() - want).norm() / want.norm()).item()


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestAllGatherRingOnDevice(unittest.TestCase):
    """usp.allgather_kv_ring_attn with the Pallas impl on device, where the
    stacked gather layout, the device kv_info and the float32 carry meet the
    kernel. Ranks run one after another; the two collectives return what they
    would on a ring of that size."""

    # bf16 inputs; an all-gather ring matches unsharded attention to bf16.
    REL_L2_BOUND = 1e-2

    def setUp(self):
        if not tpu_splash_attn._check_splash_available():
            self.skipTest("Pallas kernels unavailable (needs TPU v7x and jax)")
        tpu_splash_attn._register_ops()
        self.impl = TpuSplashAttentionImpl(
            num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
        )

    def _ring(self, world_size, q, k, v, kv_real_len):
        """Each rank's [B, S_shard, H, D] output."""
        shards = [t.chunk(world_size, dim=1) for t in (q, k, v)]
        # all_gather_tensor's result on every rank: the heads-major K (then
        # V) shards stacked along dim 0.
        stacks = [
            torch.cat([s.transpose(1, 2).contiguous() for s in t_shards]).to("tpu")
            for t_shards in shards[1:]
        ]
        outputs = []
        for rank in range(world_size):
            gathered = iter(stacks)

            def all_gather_tensor(t, gather_dim, group, gathered=gathered):
                return next(gathered)

            def all_to_all_single(
                t, output_split_sizes, input_split_sizes, group, rank=rank
            ):
                # Rank r receives element r of every rank's [0, .., R-1].
                return torch.full_like(t, float(rank))

            with (
                patch.object(
                    usp,
                    "get_sp_group",
                    lambda: SimpleNamespace(ring_group=object()),
                ),
                patch.object(usp.dist, "get_world_size", lambda group=None: world_size),
                patch.object(usp.ft_c, "all_gather_tensor", all_gather_tensor),
                patch.object(usp.ft_c, "all_to_all_single", all_to_all_single),
                patch.object(
                    tpu_splash_attn,
                    "_FLASH_RING_OP",
                    wraps=tpu_splash_attn._FLASH_RING_OP,
                ) as ring,
            ):
                out = usp.allgather_kv_ring_attn(
                    *(t_shards[rank].to("tpu") for t_shards in shards),
                    attn_impl=self.impl,
                    kv_real_len=kv_real_len,
                )
            # The local shard, then one resumed step per peer, all on the
            # Pallas ring kernel.
            self.assertEqual(ring.call_count, world_size, f"rank {rank}")
            outputs.append(out.to("cpu"))
        return outputs

    def _check(self, world_size, shard_len, kv_real_len=None):
        gen = torch.Generator().manual_seed(world_size * shard_len)
        shape = (1, world_size * shard_len, 2, 128)
        q, k, v = (torch.randn(shape, generator=gen).bfloat16() for _ in range(3))
        real = kv_real_len or world_size * shard_len
        # Keys past the real count are padding and may hold anything.
        k[:, real:], v[:, real:] = float("nan"), float("nan")

        out = torch.cat(self._ring(world_size, q, k, v, kv_real_len), dim=1)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(out.isnan().any().item())
        # Query rows past the real count are padding: zeroed.
        self.assertTrue(torch.all(out[:, real:] == 0).item())
        out = out[:, :real]
        want = _float32_attention(q, k[:, :real], v[:, :real])[:, :real]
        # The same impl over the whole sequence, as without ring parallelism.
        unsharded = self.impl.forward(
            q.to("tpu"), k[:, :real].to("tpu"), v[:, :real].to("tpu"), None
        ).to("cpu")[:, :real]
        for name, reference in (
            ("float32 attention", want),
            ("unsharded impl", unsharded.float()),
        ):
            rel = _rel_l2(out, reference)
            print(f"{self._testMethodName} R={world_size}: rel-L2 vs {name} {rel:.3e}")
            self.assertLess(rel, self.REL_L2_BOUND, name)

    def test_ring_matches_attention_over_all_keys(self):
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                self._check(world_size, shard_len=1024)

    def test_tail_pad_leaves_fewer_than_128_keys_in_the_last_shard(self):
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                # The last shard holds 100 real keys and 924 padding keys.
                self._check(
                    world_size,
                    shard_len=1024,
                    kv_real_len=(world_size - 1) * 1024 + 100,
                )


if __name__ == "__main__":
    unittest.main()
