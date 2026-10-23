"""TPU compile policy shared by every TpuSpecWorker drafter."""

from __future__ import annotations

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecodeCompilePolicyMixin,
)


class TpuSpecDecodeCompilePolicyMixin(DFlashSpecDecodeCompilePolicyMixin):
    """Target warms TARGET_VERIFY and EXTEND(FULL); subclasses warm their draft forwards.

    Every drafter's verify is TpuSpecWorker's chain verify, whose padded-output
    handling the DFlash policy already owns; the drafters differ only on the draft runner.
    """

    @property
    def algo_label(self) -> str:
        return self.model_runner.spec_algorithm.name

    # TpuSpecWorker never sends the target DECODE (decode is TARGET_VERIFY) or a
    # capture-NULL EXTEND (extend always captures hidden for the draft).
    def _precompile_decode(self) -> list[PendingCall]:
        return []

    def _precompile_extend(self) -> list[PendingCall]:
        return []

    def make_null_spec_info(self) -> DFlashVerifyInput:
        return make_draft_block_spec_info(
            draft_token_num=self.draft_token_num, device=self.model_runner.device
        )

    def _make_dummy_forward_batch(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: DFlashVerifyInput | None = None,
    ) -> ForwardBatch:
        batch = super()._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        # The draft embeds input_ids in-graph with the attached embedding.
        if self.is_draft_worker:
            batch.input_embeds = None
        return batch

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        if is_dp_attention_enabled() and forward_batch.dp_padding_mode is None:
            # forward_draft_rows builds its TARGET_VERIFY batch directly at the
            # bucket without init_new, so stamp the bucket DP metadata as _run_one does.
            self._set_dp_attention_metadata(
                forward_batch,
                int(forward_batch.input_ids.shape[0]),
                forward_batch.batch_size,
                forward_batch.forward_mode,
            )
        super().prepare_forward_batch(forward_batch)
