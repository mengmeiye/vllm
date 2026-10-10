# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared descriptions for CPU tiers in the hierarchy emulator."""

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite


class CpuTierId(IntEnum):
    """Stable identifiers carried between scheduler and worker."""

    FAST = 0
    SLOW = 1


@dataclass(frozen=True)
class DmaEmulationConfig:
    """Minimum service-time model applied around a real device copy."""

    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.bandwidth_gbps) or self.bandwidth_gbps < 0:
            raise ValueError("bandwidth_gbps must be finite and non-negative")
        if not isfinite(self.latency_us) or self.latency_us < 0:
            raise ValueError("latency_us must be finite and non-negative")

    @property
    def enabled(self) -> bool:
        return self.bandwidth_gbps > 0 or self.latency_us > 0


@dataclass(frozen=True)
class CpuTierConfig:
    """Capacity and copy behavior for one pinned-host pool."""

    tier_id: CpuTierId
    name: str
    capacity_bytes: int
    dma: DmaEmulationConfig = DmaEmulationConfig()

    def __post_init__(self) -> None:
        if self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
