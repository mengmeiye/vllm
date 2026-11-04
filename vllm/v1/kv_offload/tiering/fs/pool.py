# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attach side of an externally managed filesystem KV pool.

A pool manager outside vLLM publishes each pool generation as one directory,
built under a temporary name and renamed into place once complete:

    <root_dir>/
      pool.json   {"magic", "version", "generation", "state", "config"}
      lock        flock target: shared while attached, exclusive to destroy
      data/       block files, used as the filesystem tier's root

vLLM only attaches and detaches. Attaching takes a shared flock on ``lock``
for the lifetime of the tier, so the kernel releases it however the process
exits. The manager destroys a generation only after taking the exclusive lock
without blocking, so a destroy fails while any process is attached, and it
marks the generation ``DESTROYING`` before unlinking it, so a process that
locks a generation being destroyed rejects it. Detaching never removes data:
the pool and its blocks outlive every attached process.
"""

import contextlib
import errno
import fcntl
import json
import os
import re
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

POOL_MAGIC = "vllm-kv-pool"
POOL_FORMAT_VERSION = 1
POOL_STATE_READY = "READY"
POOL_METADATA_FILE = "pool.json"
POOL_LOCK_FILE = "lock"
POOL_DATA_DIR = "data"

_GENERATION_RE = re.compile(r"[0-9a-f]{32}")
_LOCK_POLL_INTERVAL_S = 0.05


class PoolAttachment:
    """A shared attachment to one generation of a managed pool."""

    def __init__(self, root_dir: str, generation: str, lock_fd: int) -> None:
        self.root_dir = root_dir
        self.generation = generation
        self.data_dir = os.path.join(root_dir, POOL_DATA_DIR)
        self._lock_fd: int | None = lock_fd

    @classmethod
    def attach(
        cls,
        root_dir: str,
        generation: str | None = None,
        lock_timeout_s: float = 30.0,
    ) -> "PoolAttachment":
        """Attach to the pool generation published at ``root_dir``.

        Args:
            root_dir: Directory of the published pool generation.
            generation: Expected generation (32 lowercase hex characters).
                When None, attach to whichever generation is published.
            lock_timeout_s: How long to wait for a concurrent destroy to
                release the exclusive lock.

        Raises:
            ValueError: If ``generation`` is malformed.
            RuntimeError: If no attachable generation is published, or it is
                not the expected one.

        """
        if generation is not None and not _GENERATION_RE.fullmatch(generation):
            raise ValueError(
                f"pool_generation must be 32 lowercase hex characters, "
                f"got {generation!r}"
            )
        try:
            root_fd = os.open(root_dir, os.O_RDONLY | os.O_DIRECTORY)
        except OSError as exc:
            raise RuntimeError(
                f"No managed KV pool is published at {root_dir!r}; the pool "
                "manager must create it before vLLM attaches"
            ) from exc
        try:
            try:
                lock_fd = os.open(POOL_LOCK_FILE, os.O_RDWR, dir_fd=root_fd)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Managed KV pool at {root_dir!r} was destroyed while attaching"
                ) from exc
            try:
                _lock_shared(lock_fd, lock_timeout_s)
                # Read through the locked directory: after a destroy renamed
                # it away, a fresh path lookup could reach a newer generation.
                metadata = _read_metadata(root_fd)
                attached = _validate(metadata, root_dir, generation)
                if not os.path.isdir(os.path.join(root_dir, POOL_DATA_DIR)):
                    raise RuntimeError(
                        f"Managed KV pool at {root_dir!r} has no "
                        f"{POOL_DATA_DIR!r} directory"
                    )
            except BaseException:
                os.close(lock_fd)
                raise
        finally:
            os.close(root_fd)
        logger.info(
            "Attached to managed KV pool %s (generation %s)", root_dir, attached
        )
        return cls(root_dir, attached, lock_fd)

    @property
    def attached(self) -> bool:
        return self._lock_fd is not None

    def detach(self) -> None:
        """Release the attachment. Published blocks are left in place."""
        if self._lock_fd is None:
            return
        os.close(self._lock_fd)
        self._lock_fd = None
        logger.info(
            "Detached from managed KV pool %s (generation %s)",
            self.root_dir,
            self.generation,
        )


def _lock_shared(lock_fd: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
        if time.monotonic() > deadline:
            raise RuntimeError(
                "Timed out waiting for the managed KV pool lock; a destroy may be stuck"
            )
        time.sleep(_LOCK_POLL_INTERVAL_S)


def _read_metadata(root_fd: int) -> dict:
    try:
        fd = os.open(POOL_METADATA_FILE, os.O_RDONLY, dir_fd=root_fd)
    except FileNotFoundError as exc:
        raise RuntimeError("Managed KV pool metadata is missing") from exc
    with os.fdopen(fd, "rb") as f:
        raw = f.read()
    with contextlib.suppress(ValueError):
        metadata = json.loads(raw)
        if isinstance(metadata, dict):
            return metadata
    raise RuntimeError("Managed KV pool metadata is not a JSON object")


def _validate(metadata: dict, root_dir: str, expected: str | None) -> str:
    if (
        metadata.get("magic") != POOL_MAGIC
        or metadata.get("version") != POOL_FORMAT_VERSION
    ):
        raise RuntimeError(
            f"{root_dir!r} is not a version-{POOL_FORMAT_VERSION} managed KV pool"
        )
    state = metadata.get("state")
    if state != POOL_STATE_READY:
        raise RuntimeError(
            f"Managed KV pool at {root_dir!r} is {state!r}, not {POOL_STATE_READY!r}"
        )
    generation = metadata.get("generation")
    if not isinstance(generation, str) or not _GENERATION_RE.fullmatch(generation):
        raise RuntimeError(f"Managed KV pool at {root_dir!r} has no valid generation")
    if expected is not None and generation != expected:
        raise RuntimeError(
            f"Managed KV pool at {root_dir!r} is generation {generation}, "
            f"expected {expected}"
        )
    return generation
