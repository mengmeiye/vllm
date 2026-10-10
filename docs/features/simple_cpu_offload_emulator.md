# Simple CPU Offload Hierarchy Emulator

`SimpleCPUOffloadConnector` can split its pinned-host KV cache into fast and
slow pools. This is an experimental way to evaluate request placement and
different transfer profiles before secondary-memory hardware is available.
Both pools are real pinned host memory; the bandwidth and latency settings
only delay copy completion and do not turn either pool into physical CXL or
near-accelerator memory.

```bash
vllm serve <model> \
  --enable-prefix-caching \
  --scheduling-policy priority \
  --kv-transfer-config '{
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "fast_cpu_bytes": 8589934592,
      "slow_cpu_bytes": 34359738368,
      "priority_threshold": 1,
      "fast_cpu_bandwidth_gbps": 32,
      "fast_cpu_latency_us": 8,
      "slow_cpu_bandwidth_gbps": 12,
      "slow_cpu_latency_us": 18
    }
  }'
```

Lower priority numbers are more important. In this example, priority `0`
requests are stored in the fast pool and priority `1` or greater requests are
stored in the slow pool. Prefix lookup checks both pools, chooses the longest
contiguous match, and favors the fast pool when both matches are the same
length. A load is served entirely by one pool; matches are not joined across
the two pools.

The capacity values are totals for the vLLM instance and are divided by world
size. `fast_cpu_bytes_per_rank` and `slow_cpu_bytes_per_rank` can set per-rank
capacities explicitly. Configuring either fast/slow capacity key enables the
explicit-pool mode; an omitted pool then has zero capacity.

`bandwidth_gbps` uses decimal GB/s. A transfer's minimum emulated time is:

```text
latency_us / 1,000,000 + bytes / (bandwidth_gbps * 1,000,000,000)
```

Each pool has its own serialized timing line, copy backend, and CUDA streams.
The actual CUDA copy must also finish, so the emulator cannot make a transfer
faster than the host system. A zero bandwidth and zero latency disables added
delay for that pool.

Multi-pool operation supports eager offload. `lazy_offload` is rejected when
both pools are active. There is no promotion, demotion, replication, or direct
copy between host pools.

Existing configurations remain valid. `cpu_bytes_to_use` creates the original
single host pool when no fast/slow capacity key is present. The disk backend
also remains single-pool and cannot be combined with explicit fast/slow keys.

The `/metrics` endpoint exposes physical-copy counters:

- `vllm:simple_cpu_offload_bytes_total`
- `vllm:simple_cpu_offload_blocks_total`
- `vllm:simple_cpu_offload_operations_total`

Each counter has `tier` (`fast_cpu` or `slow_cpu`) and `direction` (`load` or
`store`) labels. With KV-cache events enabled, a two-pool configuration reports
the completed store or eviction medium as `FAST_CPU` or `SLOW_CPU`.
