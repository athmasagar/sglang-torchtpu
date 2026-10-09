# SPDX-License-Identifier: Apache-2.0
"""On-device check that the TPU RoPE pair swap matches the strided form."""

import importlib.util
import unittest

import torch

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    apply_rotary_embedding_native,
    apply_rotary_embedding_pair_swap,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.utils import is_in_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)


class TestRopePairSwapTpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_tpu") is None:
            if is_in_ci():
                raise RuntimeError("TPU CI requires torch_tpu and a TPU device")
            raise unittest.SkipTest("torch_tpu is not installed")
        from sglang_tpu.env_override import apply_tpu_env_overrides

        apply_tpu_env_overrides()
        import torch_tpu  # noqa: F401

        # Do not turn initialization/lowering failures into a successful skip.
        torch.zeros(1, device="tpu").cpu()

    def test_bf16_eager_matches_the_strided_form_exactly(self):
        torch.manual_seed(0)
        for head_dim in (64, 128):
            for interleaved in (False, True):
                with self.subTest(head_dim=head_dim, interleaved=interleaved):
                    x = torch.randn(33, 4, head_dim).bfloat16().to("tpu")
                    angles = torch.randn(33, head_dim // 2)
                    cos, sin = angles.cos(), angles.sin()
                    if interleaved:
                        cos = cos.repeat_interleave(2, dim=-1)
                        sin = sin.repeat_interleave(2, dim=-1)
                    cos, sin = cos.to("tpu"), sin.to("tpu")
                    actual = apply_rotary_embedding_pair_swap(x, cos, sin, interleaved)
                    expected = apply_rotary_embedding_native(x, cos, sin, interleaved)
                    self.assertEqual(actual.device.type, "tpu")
                    torch.testing.assert_close(
                        actual.cpu(), expected.cpu(), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
