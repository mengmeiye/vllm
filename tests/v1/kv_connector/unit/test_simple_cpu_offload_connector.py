# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector-level behavior."""

from __future__ import annotations

import pytest

from tests.v1.kv_connector.unit.utils import create_vllm_config
from tests.v1.simple_kv_offload.test_scheduler import (
    _BYTES_PER_BLOCK,
    BLOCK_SIZE,
    DTYPE,
    HEAD_SIZE,
    NUM_KV_HEADS,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.simple_kv_offload.copy_backend import (
    DmaCopyBackend,
    FailedTransferEvent,
)
from vllm.v1.simple_kv_offload.tier import CpuTierId, DmaEmulationConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    fa_layers = ["layer0"]
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    groups = [KVCacheGroupSpec(fa_layers, spec)]
    size = _BYTES_PER_BLOCK * num_blocks
    tensors = [
        KVCacheTensor(
            size=size,
            layers=fa_layers,
            layer_stride=size,
            block_stride=_BYTES_PER_BLOCK,
        )
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def _make_connector(
    extra_config: dict | None = None,
    num_gpu_blocks: int = 16,
    cpu_bytes: int = _BYTES_PER_BLOCK * 8,
) -> SimpleCPUOffloadConnector:
    kv_cache_config = _make_kv_cache_config(num_gpu_blocks)
    config_dict = {
        "cpu_bytes_to_use": cpu_bytes,
    }
    if extra_config:
        config_dict.update(extra_config)
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config=config_dict,
    )
    connector = SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=kv_cache_config,
    )
    return connector


def test_disk_mode_rejects_non_positive_capacity() -> None:
    """Disk mode with disk_capacity_bytes <= 0 raises ValueError.

    Without this guard, the worker operates in disk mode but the scheduler
    classifies events as MEDIUM_CPU instead of MEDIUM_STORAGE, silently
    mislabeling events.
    """
    with pytest.raises(ValueError, match="disk_capacity_bytes > 0"):
        _make_connector(
            extra_config={
                "kv_offload_backend": "disk",
                "disk_path": "/tmp/fake",
                "disk_capacity_bytes": 0,
            }
        )


def test_legacy_capacity_keeps_one_slow_pool() -> None:
    connector = _make_connector(cpu_bytes=_BYTES_PER_BLOCK * 8)

    scheduler = connector.scheduler_manager
    assert set(scheduler.cpu_tiers) == {int(CpuTierId.SLOW)}
    assert scheduler.cpu_tiers[int(CpuTierId.SLOW)].kv_cache_config.num_blocks == 8


def test_explicit_tiers_do_not_inherit_legacy_capacity() -> None:
    connector = _make_connector(
        extra_config={"fast_cpu_bytes": _BYTES_PER_BLOCK * 4},
        cpu_bytes=_BYTES_PER_BLOCK * 8,
    )

    scheduler = connector.scheduler_manager
    assert set(scheduler.cpu_tiers) == {int(CpuTierId.FAST)}
    assert scheduler.cpu_tiers[int(CpuTierId.FAST)].kv_cache_config.num_blocks == 4


def test_explicit_tier_configuration_reaches_scheduler() -> None:
    connector = _make_connector(
        extra_config={
            "fast_cpu_bytes": _BYTES_PER_BLOCK * 4,
            "slow_cpu_bytes": _BYTES_PER_BLOCK * 6,
            "priority_threshold": 3,
        }
    )

    scheduler = connector.scheduler_manager
    assert scheduler.priority_threshold == 3
    assert {
        tier_id: tier.kv_cache_config.num_blocks
        for tier_id, tier in scheduler.cpu_tiers.items()
    } == {int(CpuTierId.FAST): 4, int(CpuTierId.SLOW): 6}


@pytest.mark.parametrize(
    "extra_config",
    [
        {
            "fast_cpu_bytes": -1,
            "slow_cpu_bytes": _BYTES_PER_BLOCK * 4,
        },
        {
            "fast_cpu_bytes_per_rank": -1,
            "slow_cpu_bytes": _BYTES_PER_BLOCK * 4,
        },
    ],
)
def test_explicit_tiers_reject_negative_capacity(extra_config: dict) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        _make_connector(extra_config=extra_config)


def test_lazy_mode_rejects_multiple_cpu_tiers() -> None:
    with pytest.raises(ValueError, match="lazy_offload"):
        _make_connector(
            extra_config={
                "fast_cpu_bytes": _BYTES_PER_BLOCK * 4,
                "slow_cpu_bytes": _BYTES_PER_BLOCK * 6,
                "lazy_offload": True,
            }
        )


def test_dma_service_time_and_serialization() -> None:
    config = DmaEmulationConfig(bandwidth_gbps=2.0, latency_us=10.0)
    assert DmaCopyBackend.emulated_service_time_seconds(1_000_000_000, config) == (
        pytest.approx(0.50001)
    )

    backend = DmaCopyBackend(config)
    assert backend._reserve_emulated_completion(10.0, 2.0) == 12.0
    assert backend._reserve_emulated_completion(10.5, 1.0) == 13.0
    assert backend._reserve_emulated_completion(20.0, 1.0) == 21.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bandwidth_gbps": -1.0},
        {"latency_us": -1.0},
        {"bandwidth_gbps": float("inf")},
        {"latency_us": float("nan")},
    ],
)
def test_dma_emulation_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        DmaEmulationConfig(**kwargs)


def test_background_copy_failure_is_reported() -> None:
    event = FailedTransferEvent(ValueError("copy failed"), "FAST_CPU", True)

    with pytest.raises(RuntimeError, match="FAST_CPU store DMA submission failed"):
        event.query()
    with pytest.raises(RuntimeError, match="FAST_CPU store DMA submission failed"):
        event.synchronize()


def _make_ring_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    """A paged full-attention group beside a per-request ring group."""
    full = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    ring = CircularBufferSpec(
        block_size=6, num_kv_heads=1, head_size=32, head_size_v=0, dtype=DTYPE
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=_BYTES_PER_BLOCK * num_blocks,
                layers=["attn"],
                layer_stride=_BYTES_PER_BLOCK * num_blocks,
                block_stride=_BYTES_PER_BLOCK,
            ),
            KVCacheTensor(
                size=ring.page_size_bytes * num_blocks,
                layers=["ring"],
                layer_stride=ring.page_size_bytes * num_blocks,
                block_stride=ring.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], full),
            KVCacheGroupSpec(["ring"], ring),
        ],
    )


def test_ring_scratch_group_is_accepted_and_never_offloaded() -> None:
    """The ring group's capacity is not a token granularity, it holds no
    hashed blocks, and it never participates in CPU hits."""
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": _BYTES_PER_BLOCK * 8},
    )
    connector = SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=_make_ring_kv_cache_config(),
    )
    scheduler = connector.scheduler_manager
    assert scheduler.fa_gidx == 0
    cpu_groups = scheduler.cpu_kv_cache_config.kv_cache_groups
    assert [g.kv_cache_spec.prefix_cacheable for g in cpu_groups] == [True, False]
    # Only the paged group takes part in CPU-side hit lookup.
    attention_groups = scheduler.cpu_coordinator.attention_groups
    assert [g.group_ids for g in attention_groups] == [[0]]
