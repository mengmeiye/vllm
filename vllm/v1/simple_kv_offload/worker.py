# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

import queue
from collections import Counter
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.simple_kv_offload.copy_backend import (
    CompletionEvent,
    DmaCopyBackend,
)
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
    TierStoreTransfer,
)
from vllm.v1.simple_kv_offload.metrics import SimpleCPUOffloadStats
from vllm.v1.simple_kv_offload.tier import (
    CpuTierConfig,
    CpuTierId,
    DmaEmulationConfig,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class _CompletionQueue:
    """List-like producer interface backed by a thread-safe queue."""

    def __init__(self) -> None:
        self.queue: queue.SimpleQueue[tuple[int, CompletionEvent]] = queue.SimpleQueue()

    def append(self, item: tuple[int, CompletionEvent]) -> None:
        self.queue.put(item)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        kv_offload_backend: str = "cpu",
        disk_path: str | None = None,
        disk_capacity_bytes: int = 0,
        disk_buffer_slots: int = 2,
        use_page_cache: bool = False,
        fast_cpu_capacity_bytes: int = 0,
        slow_cpu_capacity_bytes: int | None = None,
        fast_dma: DmaEmulationConfig | None = None,
        slow_dma: DmaEmulationConfig | None = None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        slow_capacity = (
            cpu_capacity_bytes
            if slow_cpu_capacity_bytes is None
            else slow_cpu_capacity_bytes
        )
        self.cpu_capacity_bytes = slow_capacity
        self.disk_path = disk_path
        self.disk_capacity_bytes = disk_capacity_bytes
        self.disk_buffer_slots = disk_buffer_slots
        self.use_page_cache = use_page_cache
        self.disk_mode = kv_offload_backend == "disk"
        self.tier_configs = {
            int(tier.tier_id): tier
            for tier in (
                CpuTierConfig(
                    CpuTierId.FAST,
                    "FAST_CPU",
                    fast_cpu_capacity_bytes,
                    fast_dma or DmaEmulationConfig(),
                ),
                CpuTierConfig(
                    CpuTierId.SLOW,
                    "SLOW_CPU",
                    slow_capacity,
                    slow_dma or DmaEmulationConfig(),
                ),
            )
            if tier.capacity_bytes > 0
        }

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches_by_tier: dict[int, dict[str, torch.Tensor]] = {}
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None
        self.load_streams_by_tier: dict[int, torch.cuda.Stream] = {}
        self.store_streams_by_tier: dict[int, torch.cuda.Stream] = {}

        self._backend: DmaCopyBackend | DiskBackend | None = None
        self._backends: dict[int, DmaCopyBackend | DiskBackend] = {}

        self._load_completion_sink = _CompletionQueue()
        self._store_completion_sink = _CompletionQueue()
        self._load_events: list[tuple[int, CompletionEvent]] = []
        self._store_events: list[tuple[int, CompletionEvent]] = []
        self._load_expected_parts: dict[int, int] = {}
        self._store_expected_parts: dict[int, int] = {}
        self._load_completed_parts: Counter[int] = Counter()
        self._store_completed_parts: Counter[int] = Counter()
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Compute-done event recorded before each store; reused across steps
        # (get_finished runs once per step, copy queue is FIFO).
        self._store_compute_done: torch.Event | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}
        self._stats = SimpleCPUOffloadStats()

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.

        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        self.device = next(iter(kv_caches.values())).device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks
        assert self.kv_cache_config.kv_cache_tensors

        # The DMA backend copies blocks as base + block_id * stride(0), so every
        # region is a [num_blocks, block_bytes] view. Block bytes come from each
        # tensor's placement -- layers in one allocation can differ in page size.
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        seen: set[tuple[torch.device, int]] = set()
        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            name = next((n for n in kv_cache_tensor.layers if n in kv_caches), None)
            if name is None:
                continue
            num_layers = len(kv_cache_tensor.layers)
            block_bytes = kv_cache_tensor.block_stride
            blocks_span = num_blocks * block_bytes
            layer_outermost = kv_cache_tensor.layer_stride >= blocks_span
            if layer_outermost:
                # One region per layer, or per KV head group under LHBNC.
                start = kv_cache_tensor.offset
                span = num_layers * kv_cache_tensor.layer_stride
                layer_names = kv_cache_tensor.layers
            else:
                start = 0
                span = blocks_span
                layer_names = kv_cache_tensor.layers[:1]

            cache = kv_caches[name]
            raw = torch.empty(0, dtype=torch.int8, device=cache.device).set_(
                cache.untyped_storage()
            )
            assert raw.numel() >= start + span, (
                f"KV cache {name!r} storage has {raw.numel()} bytes, smaller than "
                f"the {span} bytes it places at offset {start}"
            )
            regions = raw[start : start + span].view(-1, num_blocks, block_bytes)
            groups_per_layer, remainder = divmod(len(regions), len(layer_names))
            assert remainder == 0, (
                f"KV cache {name!r} places {len(layer_names)} layers in "
                f"{len(regions)} block-strided regions"
            )
            for idx, region in enumerate(regions):
                key = (region.device, region.data_ptr())
                if key in seen:
                    continue
                seen.add(key)
                layer_name = layer_names[idx // groups_per_layer]
                key_name = (
                    layer_name
                    if groups_per_layer == 1
                    else f"{layer_name}.{idx % groups_per_layer}"
                )
                unique_gpu_caches[key_name] = region

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        self.gpu_kv_caches = unique_gpu_caches

        if self.disk_mode:
            # Use lowest priority so KV cache I/O yields to compute streams.
            low_pri, _ = torch.cuda.Stream.priority_range()
            self.load_stream = torch.cuda.Stream(priority=low_pri)
            self.store_stream = torch.cuda.Stream(priority=low_pri)
            self._init_disk_mode(unique_gpu_caches, total_bytes_per_block, self.device)
        else:
            self._init_cpu_mode(unique_gpu_caches, total_bytes_per_block, self.device)

    def _init_disk_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        num_disk_slots = max(1, self.disk_capacity_bytes // total_bytes_per_block)
        self.num_cpu_blocks = num_disk_slots

        logger.info(
            "SimpleCPUOffloadWorker [DISK]: %d tensors, %d disk slots (%.2f GB)",
            len(unique_gpu_caches),
            num_disk_slots,
            (num_disk_slots * total_bytes_per_block) / (1024**3),
        )

        assert self.disk_path is not None
        rank_path = f"{self.disk_path}.rank_{device.index or 0}"
        self._backend = DiskBackend()
        self._backend.init(
            unique_gpu_caches,
            device,
            self.load_stream,
            self.store_stream,
            rank_path,
            num_disk_slots,
            total_bytes_per_block,
            self.disk_buffer_slots,
            self.use_page_cache,
        )
        tier_id = int(CpuTierId.SLOW)
        self._backends[tier_id] = self._backend
        self.load_streams_by_tier[tier_id] = self.load_stream
        self.store_streams_by_tier[tier_id] = self.store_stream

    def _init_cpu_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        pin_memory = PIN_MEMORY
        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        low_pri, _ = torch.cuda.Stream.priority_range()
        for tier_id, tier_config in self.tier_configs.items():
            num_blocks = tier_config.capacity_bytes // total_bytes_per_block
            if num_blocks <= 0:
                raise ValueError(
                    f"{tier_config.name} capacity {tier_config.capacity_bytes} "
                    f"bytes cannot hold one KV block; need at least "
                    f"{total_bytes_per_block} bytes"
                )
            tier_caches: dict[str, torch.Tensor] = {}
            for name, gpu_tensor in unique_gpu_caches.items():
                cpu_shape = (num_blocks,) + gpu_tensor.shape[1:]
                # Avoid CUDACachingHostAllocator's power-of-two rounding.
                tensor = torch.zeros(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
                if pin_memory:
                    pin_tensor(tensor)
                tier_caches[name] = tensor

            load_stream = torch.cuda.Stream(priority=low_pri)
            store_stream = torch.cuda.Stream(priority=low_pri)
            backend = DmaCopyBackend(
                emulation=tier_config.dma, tier_name=tier_config.name
            )
            backend.init(
                unique_gpu_caches,
                tier_caches,
                device,
                load_stream,
                store_stream,
            )
            self.cpu_kv_caches_by_tier[tier_id] = tier_caches
            self.load_streams_by_tier[tier_id] = load_stream
            self.store_streams_by_tier[tier_id] = store_stream
            self._backends[tier_id] = backend
            logger.info(
                "SimpleCPUOffloadWorker [CPU]: %s=%d blocks (%.2f GB), "
                "DMA cap=%.2f GB/s + %.1f us",
                tier_config.name,
                num_blocks,
                (num_blocks * total_bytes_per_block) / (1024**3),
                tier_config.dma.bandwidth_gbps,
                tier_config.dma.latency_us,
            )

        default_tier_id = self._default_tier_id()
        self.cpu_kv_caches = self.cpu_kv_caches_by_tier[default_tier_id]
        self.num_cpu_blocks = next(iter(self.cpu_kv_caches.values())).shape[0]
        self._backend = self._backends[default_tier_id]
        self.load_stream = self.load_streams_by_tier[default_tier_id]
        self.store_stream = self.store_streams_by_tier[default_tier_id]

    def _default_tier_id(self) -> int:
        slow = int(CpuTierId.SLOW)
        if slow in self._backends or not self._backends:
            return slow
        return int(CpuTierId.FAST)

    def _backend_for_tier(self, tier_id: int) -> DmaCopyBackend | DiskBackend:
        backend = self._backends.get(tier_id, self._backend)
        assert backend is not None
        return backend

    def _record_transfer(
        self,
        tier_id: int,
        direction: str,
        num_blocks: int,
        backend: DmaCopyBackend | DiskBackend,
    ) -> None:
        if not isinstance(backend, DmaCopyBackend):
            return
        tier = self.tier_configs.get(tier_id)
        tier_name = tier.name if tier is not None else "SLOW_CPU"
        self._stats.record_transfer(
            tier_name,
            direction,
            num_blocks,
            num_blocks * backend.bytes_per_block,
        )

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
            self._load_event_to_reqs.update(metadata.load_event_to_reqs)
        if metadata.store_transfers:
            for transfer in metadata.store_transfers:
                self._pending_store_event_indices.add(transfer.event)
        elif metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # Invoked after the forward launch on steps without sync loads
        # (SchedulerOutput.has_sync_kv_loads), so the CPU-side block copy
        # op overhead (~5ms) stays hidden behind GPU compute. Stores are
        # issued in wait_for_save().
        metadata = self._connector_metadata
        if metadata is not None and metadata.load_cpu_blocks:
            tier_ids = metadata.load_cpu_tiers or [self._default_tier_id()] * len(
                metadata.load_cpu_blocks
            )
            grouped: dict[int, tuple[list[int], list[int]]] = {}
            for cpu_block, gpu_block, tier_id in zip(
                metadata.load_cpu_blocks,
                metadata.load_gpu_blocks,
                tier_ids,
            ):
                cpu_blocks, gpu_blocks = grouped.setdefault(tier_id, ([], []))
                cpu_blocks.append(cpu_block)
                gpu_blocks.append(gpu_block)

            self._load_expected_parts[metadata.load_event] = len(grouped)
            for tier_id, (cpu_blocks, gpu_blocks) in grouped.items():
                backend = self._backend_for_tier(tier_id)
                backend.launch_copy(
                    cpu_blocks,
                    gpu_blocks,
                    is_store=False,
                    event_idx=metadata.load_event,
                    events_list=self._load_completion_sink,
                )
                self._record_transfer(tier_id, "load", len(cpu_blocks), backend)

    def wait_for_save(self) -> None:
        """Submit async stores.

        Stores (GPU->CPU) read the live KV cache, which the compute stream may
        still be writing under v1 overlapped execution, so they are ordered
        after a compute-done event recorded on the current stream. See
        #45704 for the bug and #39306 for the srcAccessOrder rationale.
        """
        metadata = self._connector_metadata
        if metadata is not None:
            transfers = metadata.store_transfers
            if not transfers and metadata.store_gpu_blocks:
                transfers = [
                    TierStoreTransfer(
                        tier_id=self._default_tier_id(),
                        event=metadata.store_event,
                        gpu_blocks=metadata.store_gpu_blocks,
                        cpu_blocks=metadata.store_cpu_blocks,
                    )
                ]
            if not transfers:
                return
            if self._store_compute_done is None:
                self._store_compute_done = torch.Event()
            self._store_compute_done.record(torch.cuda.current_stream())
            for transfer in transfers:
                self._store_expected_parts[transfer.event] = (
                    self._store_expected_parts.get(transfer.event, 0) + 1
                )
                backend = self._backend_for_tier(transfer.tier_id)
                backend.launch_copy(
                    transfer.gpu_blocks,
                    transfer.cpu_blocks,
                    is_store=True,
                    event_idx=transfer.event,
                    events_list=self._store_completion_sink,
                    wait_event=self._store_compute_done,
                )
                self._record_transfer(
                    transfer.tier_id,
                    "store",
                    len(transfer.gpu_blocks),
                    backend,
                )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report completed transfer events to the scheduler.

        Loads are submitted in start_load_kv() and stores in wait_for_save().

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.

        """
        finished_recving: set[str] = set()

        if self._pending_load_event_indices:
            completed = self._poll_stream_events(is_store=False)
            for j in self._pending_load_event_indices.intersection(completed):
                self._pending_load_event_indices.discard(j)
                req_ids = self._load_event_to_reqs.pop(j, None)
                if req_ids:
                    finished_recving.update(req_ids)

        if self._pending_store_event_indices:
            completed = self._poll_stream_events(is_store=True)
            for j in self._pending_store_event_indices.intersection(completed):
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1

        return None, finished_recving or None

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed store events since the last call."""
        if not self._completed_store_events:
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta

    def get_kv_connector_stats(self) -> SimpleCPUOffloadStats | None:
        """Return transfer counters accumulated since the previous poll."""
        if self._stats.is_empty():
            return None
        return self._stats.clone_and_reset()

    def shutdown(self) -> None:
        """Stop all background copy threads."""
        try:
            self._flush_and_sync_all()
        finally:
            for backend in self._backends.values():
                backend.shutdown()
            if not self._backends and self._backend is not None:
                self._backend.shutdown()

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events."""
        backends = list(self._backends.values())
        if not backends and self._backend is not None:
            backends.append(self._backend)
        for backend in backends:
            if isinstance(backend, DmaCopyBackend):
                backend.flush()
        self._drain_completion_events(is_store=False)
        self._drain_completion_events(is_store=True)

        first_error: Exception | None = None
        for event_idx, event in self._load_events:
            try:
                event.synchronize()
                self._load_completed_parts[event_idx] += 1
            except Exception as error:
                first_error = first_error or error
        self._load_events.clear()

        for event_idx, event in self._store_events:
            try:
                event.synchronize()
                self._store_completed_parts[event_idx] += 1
            except Exception as error:
                first_error = first_error or error
        self._store_events.clear()
        if first_error is not None:
            raise first_error

    def _drain_completion_events(self, is_store: bool) -> None:
        sink = self._store_completion_sink if is_store else self._load_completion_sink
        events = self._store_events if is_store else self._load_events
        while True:
            try:
                events.append(sink.queue.get_nowait())
            except queue.Empty:
                return

    def _poll_stream_events(self, is_store: bool) -> set[int]:
        """Poll unordered tier events and return complete logical jobs."""
        self._drain_completion_events(is_store)
        events = self._store_events if is_store else self._load_events
        expected = self._store_expected_parts if is_store else self._load_expected_parts
        completed_parts = (
            self._store_completed_parts if is_store else self._load_completed_parts
        )
        remaining: list[tuple[int, CompletionEvent]] = []
        for event_idx, event in events:
            if event.query():
                completed_parts[event_idx] += 1
            else:
                remaining.append((event_idx, event))
        events[:] = remaining

        completed = {
            event_idx
            for event_idx, count in completed_parts.items()
            if count >= expected.get(event_idx, 1)
        }
        for event_idx in completed:
            completed_parts.pop(event_idx, None)
            expected.pop(event_idx, None)
        return completed
