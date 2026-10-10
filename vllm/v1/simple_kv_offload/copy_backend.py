# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading
import time
from typing import NoReturn, Protocol

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)
from vllm.v1.simple_kv_offload.tier import DmaEmulationConfig

logger = init_logger(__name__)


class CompletionEvent(Protocol):
    """Completion operations used by the offload worker."""

    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


class CompletionSink(Protocol):
    """Thread-safe destination for copy completion objects."""

    def append(self, item: tuple[int, CompletionEvent]) -> None: ...


class EmulatedTransferEvent:
    """Device completion gated by a configurable minimum service time."""

    def __init__(self, event: torch.Event, ready_at: float) -> None:
        self._event = event
        self._ready_at = ready_at

    def query(self) -> bool:
        return time.monotonic() >= self._ready_at and self._event.query()

    def synchronize(self) -> None:
        self._event.synchronize()
        remaining = self._ready_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


class FailedTransferEvent:
    """Completion object that surfaces a background submission failure."""

    def __init__(self, error: Exception, tier_name: str, is_store: bool) -> None:
        self._error = error
        self._tier_name = tier_name
        self._direction = "store" if is_store else "load"

    def _raise(self) -> NoReturn:
        raise RuntimeError(
            f"{self._tier_name} {self._direction} DMA submission failed"
        ) from self._error

    def query(self) -> bool:
        self._raise()

    def synchronize(self) -> None:
        self._raise()


class DmaCopyBackend:
    """cuMemcpyBatchAsync copy backend (background thread)."""

    def __init__(
        self,
        emulation: DmaEmulationConfig | None = None,
        tier_name: str = "CPU",
    ) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._emulation = emulation or DmaEmulationConfig()
        self._tier_name = tier_name
        self._bytes_per_block = 0
        self._emulated_available_at = 0.0

    @staticmethod
    def emulated_service_time_seconds(
        num_bytes: int, config: DmaEmulationConfig
    ) -> float:
        """Return the configured minimum transfer service time."""
        seconds = config.latency_us / 1_000_000.0
        if config.bandwidth_gbps > 0:
            seconds += num_bytes / (config.bandwidth_gbps * 1_000_000_000.0)
        return seconds

    def _reserve_emulated_completion(
        self, submitted_at: float, service_time: float
    ) -> float:
        service_start = max(submitted_at, self._emulated_available_at)
        self._emulated_available_at = service_start + service_time
        return self._emulated_available_at

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream

        # Stores read the live KV cache -> STREAM (paired with the compute-done
        # wait in get_finished); loads read stable pinned host memory -> ANY.
        self._store_params = build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )
        self._bytes_per_block = int(self._store_params.bpb.sum())

        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
        )
        self._thread.start()

    @property
    def bytes_per_block(self) -> int:
        """Return bytes copied for one logical KV block."""
        return self._bytes_per_block

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: CompletionSink,
        wait_event: torch.Event | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            )
        )

    def flush(self) -> None:
        """Wait until queued copies have published completion objects."""
        if self._queue is not None:
            self._queue.join()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.join()
            self._queue.put(None)
            self._queue.join()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Timed out stopping %s copy thread", self._tier_name)

    def _copy_loop(
        self,
        q: queue.Queue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            try:
                item = q.get()
                if item is None:
                    return
                (
                    src_blocks,
                    dst_blocks,
                    params,
                    is_store,
                    event_idx,
                    events_list,
                    wait_event,
                ) = item
                started_at = time.monotonic()
                try:
                    stream = store_stream if is_store else load_stream
                    if wait_event is not None:
                        stream.wait_event(wait_event)
                    copy_blocks(src_blocks, dst_blocks, params)
                    event = torch.Event()
                    event.record(stream)
                    completion: CompletionEvent = event
                    if self._emulation.enabled:
                        num_bytes = len(src_blocks) * int(params.bpb.sum())
                        service_time = self.emulated_service_time_seconds(
                            num_bytes, self._emulation
                        )
                        ready_at = self._reserve_emulated_completion(
                            started_at, service_time
                        )
                        completion = EmulatedTransferEvent(event, ready_at)
                except Exception as error:
                    logger.exception(
                        "%s %s DMA submission failed",
                        self._tier_name,
                        "store" if is_store else "load",
                    )
                    completion = FailedTransferEvent(error, self._tier_name, is_store)
                events_list.append((event_idx, completion))
            finally:
                q.task_done()
