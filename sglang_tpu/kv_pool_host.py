"""Host-side KV pool for TPU HiCache, backed by a TPU DMA backend's host memory.

``MHATokenToKVPoolHost`` (the CUDA host pool) allocates split ``[layer, token,
head, dim]`` K/V tensors and moves data with ``sgl_kernel`` kernels — neither
works on TPU. This subclass keeps the HiCache ``HostKVCache`` interface but
delegates storage and transfer to a ``TpuDMAKVTransfer`` engine (raiden unless
``--hicache-io-backend`` names another): the engine owns the host memory and
copies whole pages of the fused device buffer.

HiCache addresses the host pool at **token** granularity, but engines address
**pages**. A page is ``page_size`` consecutive tokens. To keep the two
consistent under fragmentation, this pool allocates and frees at page
granularity and only ever hands out whole, page-aligned token runs, so
``host_index // page_size`` is an exact engine page id.

Scope is L2 (TPU<->host) only; the L3 flat-page accessors raise so a future
storage-backend port has an explicit, loud extension point.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes
from sglang_tpu.tpu_dma_hicache import TransferEvent, create_tpu_dma_kv_transfer

if TYPE_CHECKING:
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

logger = logging.getLogger(__name__)

# Marketing name surfaced as ``layout`` for logs / compatibility reads.
TPU_DMA_LAYOUT = "fused_tpu_dma"


class TpuDMAHostKVPool(HostKVCache):
    """Page-granular host KV pool whose storage + transfer live in an engine.

    One instance mirrors one fused device pool: a hybrid sliding-window model
    builds two of these (one per ``SWAKVPool`` sub-pool), each owning an
    engine over just its sub-pool's layers and page-id space.
    """

    # The --hicache-io-backend engine; tpu_dma_host_kv_pool_cls binds another.
    io_backend = "raiden"

    device_pool: TpuFusedMHATokenToKVPool
    # The hybrid HostPoolGroup reads this directly off every entry; the JIT
    # write-back path is a CUDA kernel feature.
    can_use_write_back_jit = False

    def __init__(
        self,
        device_pool: TpuFusedMHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = False,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        pool_label: str = "kv",
        mtp_draft_device_pools: tuple = (),
    ):
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "MTP draft KV offload is not supported on --hicache-io-backend "
                f"{self.io_backend}."
            )
        # The engine owns the host storage, so skip the base allocator setup
        # but retain its sizing and host-memory admission check.
        self.device_pool = device_pool
        self.pool_label = pool_label
        self.page_size = page_size
        self.layout = TPU_DMA_LAYOUT
        self.pin_memory = pin_memory
        self.device = device

        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = device_pool.layer_num

        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align the pool up to a whole number of pages.
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        if self.size <= device_pool.size:
            logger.warning(
                "HiCache %s host pool (%d tokens) is smaller than the device "
                "pool (%d tokens); L2 cache effectiveness is reduced. "
                "Consider increasing --hicache-ratio (or --hicache-size) for "
                "higher L2 cache hit rate.",
                pool_label,
                self.size,
                device_pool.size,
            )

        requested_bytes = self.size * self.size_per_token
        available_bytes = host_memory_budget_bytes(requested_bytes)
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        logger.info(
            "Allocating %s hierarchical KV host pool: %d tokens, %.2f GB host memory.",
            pool_label,
            self.size,
            requested_bytes / 1e9,
        )

        self.transfer = create_tpu_dma_kv_transfer(
            self.io_backend, device_pool, self.page_num
        )

        self.lock = threading.RLock()
        self.clear()

    # ------------------------------------------------------------------ #
    # Sizing
    # ------------------------------------------------------------------ #

    def get_size_per_token(self) -> int:
        """Host bytes per cached token = fused device page bytes / page_size.

        Uses the *real* (padded) fused-page byte size so the host_size GB
        budget matches the engine's actual per-page allocation, not the logical
        ``2 * head_num * head_dim`` lower bound.
        """
        fused = self.device_pool.get_fused_kv_buffer(self.start_layer)
        page_bytes = int(np.prod(fused.shape[1:])) * fused.element_size()
        per_token = page_bytes // self.page_size
        return per_token * self.layer_num

    def init_kv_buffer(self):
        # The engine owns host memory; there is no torch host buffer.
        return None

    # ------------------------------------------------------------------ #
    # Page-granular allocation (token-addressed interface)
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        with self.lock:
            # Free list of host *pages*. Page 0..page_num-1.
            self.free_pages = deque(range(self.page_num))

    def available_size(self) -> int:
        return len(self.free_pages) * self.page_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate ``need_size`` token slots as whole pages.

        Returns page-aligned token indices (int64) so ``// page_size`` yields
        exact engine page ids, or ``None`` if the pool is full.
        """
        assert need_size % self.page_size == 0, (
            "host alloc size must be a multiple of page_size"
        )
        num_pages = need_size // self.page_size
        with self.lock:
            if num_pages > len(self.free_pages):
                return None
            pages = [self.free_pages.popleft() for _ in range(num_pages)]
        base = np.repeat(
            np.array(pages, dtype=np.int64) * self.page_size, self.page_size
        )
        offsets = np.tile(np.arange(self.page_size, dtype=np.int64), num_pages)
        return torch.from_numpy(base + offsets)

    def free(self, indices: torch.Tensor) -> int:
        page_ids = torch.unique(indices.cpu() // self.page_size).tolist()
        with self.lock:
            self.free_pages.extend(page_ids)
        return len(indices)

    # ------------------------------------------------------------------ #
    # Transfer (token indices -> engine page ids)
    # ------------------------------------------------------------------ #

    def _to_page_ids(self, token_indices: torch.Tensor) -> List[int]:
        """Whole-page token runs -> page ids, one per page, order preserved."""
        # Indices are host-resident (the transfer path runs with
        # SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1), so
        # this is a pure host slice+divide. Take the first token of each
        # page-sized run; the runs are page-aligned and contiguous within a page
        # by construction (see ``alloc``).
        return (token_indices[:: self.page_size] // self.page_size).tolist()

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> TransferEvent:
        """D2H for all layers in one engine call; returns its future-event."""
        device_pages = self._to_page_ids(device_indices)
        host_pages = self._to_page_ids(host_indices)
        return self.transfer.backup(device_pages, host_pages)

    def load_all_layer(
        self, device_pool, host_indices, device_indices
    ) -> TransferEvent:
        """H2D for all layers in one engine call; returns its future-event."""
        host_pages = self._to_page_ids(host_indices)
        device_pages = self._to_page_ids(device_indices)
        return self.transfer.load(host_pages, device_pages)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ) -> None:
        # The TPU controllers issue a single all-layer ``load_all_layer``;
        # the per-layer path must never be reached on this pool.
        raise NotImplementedError(
            "TpuDMAHostKVPool uses all-layer loads via load_all_layer(); "
            "load_to_device_per_layer should not be called on this path."
        )

    # ------------------------------------------------------------------ #
    # L3 storage seams (not yet supported on TPU)
    # ------------------------------------------------------------------ #

    def get_data_page(self, index, flat: bool = True):
        raise NotImplementedError(
            f"L3 storage is not supported on --hicache-io-backend {self.io_backend}"
        )

    def get_dummy_flat_data_page(self):
        raise NotImplementedError(
            f"L3 storage is not supported on --hicache-io-backend {self.io_backend}"
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError(
            f"L3 storage is not supported on --hicache-io-backend {self.io_backend}"
        )


@functools.cache
def tpu_dma_host_kv_pool_cls(io_backend: str) -> type[TpuDMAHostKVPool]:
    """The ``TpuDMAHostKVPool`` class whose instances copy with ``io_backend``."""
    if io_backend == TpuDMAHostKVPool.io_backend:
        return TpuDMAHostKVPool
    return type(
        f"TpuDMAHostKVPool[{io_backend}]",
        (TpuDMAHostKVPool,),
        {"io_backend": io_backend},
    )
