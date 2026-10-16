"""
Snapshot Providers for GKE Fast Pod Snapshotting.

This module provides GKESnapshotProvider for triggering gVisor sandbox checkpoints and clearing model weight caches.
"""

from __future__ import annotations

import errno
import logging
import os
import pathlib
import shutil
import sys
import time
from typing import Optional

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.snapshot.providers")
except ImportError:
    logger = logging.getLogger("snapshot.providers")
    if not logger.handlers:
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

GVISOR_CHECKPOINT_PATH = "/proc/gvisor/checkpoint"

__all__ = [
    "GKESnapshotProvider",
    "SnapshotError",
    "get_snapshot_provider",
]


def _is_eager_loading_configured() -> bool:
    """Check if '--safetensors-load-strategy eager' is configured via CLI arguments."""
    return any(
        (arg == "eager" and i > 0 and sys.argv[i - 1] == "--safetensors-load-strategy")
        or arg.startswith("--safetensors-load-strategy=eager")
        for i, arg in enumerate(sys.argv)
    )


class SnapshotError(RuntimeError):
    """Raised when container process snapshotting or cache clearing fails."""
    pass


class GKESnapshotProvider:
    """
    Snapshot provider for GKE Sandboxes using gVisor's procfs control interface.
    Clears cached weight files and triggers container checkpointing by writing to /proc/gvisor/checkpoint.
    """

    def __init__(
        self,
        proc_path: str = GVISOR_CHECKPOINT_PATH,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.proc_path = proc_path
        self.cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR")

    def clear_cache(self) -> None:
        """Clear out cached weight files and directories if a cache directory exists."""
        if not self.cache_dir:
            logger.info("MODEL_CACHE_DIR is not set; skipping disk weight cache purge.")
            return

        # Log a warning instead of raising an error if eager loading is not detected in sys.argv.
        # Clearing cached weights without eager loading can break memory-mapped file descriptors,
        # but we do not fail hard to avoid false positives for non-safetensors models.
        if not _is_eager_loading_configured():
            logger.warning(
                "Clearing model cache without '--safetensors-load-strategy eager' detected. "
                "Ensure eager loading is enabled for safetensors models so model weights are copied to memory rather than memory-mapped from disk."
            )

        target_path = pathlib.Path(os.path.expanduser(self.cache_dir))
        logger.info("Purging local model weight cache at '%s'...", target_path)
        t0 = time.monotonic()
        removed_count = 0
        try:
            if target_path.exists():
                if target_path.is_file() or target_path.is_symlink():
                    logger.info("Removing cache file/link: %s", target_path)
                    target_path.unlink()
                    removed_count = 1
                else:
                    for child in target_path.iterdir():
                        logger.info("Clearing cache entry: %s", child)
                        if child.is_file() or child.is_symlink():
                            child.unlink()
                        else:
                            shutil.rmtree(child)
                        removed_count += 1
                logger.info(
                    "Purged %d cache entry/entries from '%s' in %.2fs.",
                    removed_count,
                    target_path,
                    time.monotonic() - t0,
                )
            else:
                logger.info("Cache path '%s' does not exist; nothing to purge.", target_path)
        except OSError as e:
            raise SnapshotError(f"Could not delete locally stored weights at {target_path}: {e}") from e

    def is_available(self) -> bool:
        """Check if the gVisor checkpoint file interface is writable."""
        return os.access(self.proc_path, os.W_OK)

    def trigger(self) -> None:
        """Execute weight cache clearing followed by gVisor checkpoint triggering and barrier sync."""
        if not self.is_available():
            logger.warning("Pod snapshot trigger not available (checkpoint file '%s' is not writable).", self.proc_path)
            return

        self.clear_cache()

        logger.info("Opening gVisor checkpoint interface '%s' (pid=%d)...", self.proc_path, os.getpid())
        try:
            fd = os.open(self.proc_path, os.O_RDWR)
        except PermissionError as e:
            raise SnapshotError(f"gVisor checkpoint file is not writable: {self.proc_path}") from e
        except OSError as e:
            raise SnapshotError(f"Failed to open gVisor checkpoint file '{self.proc_path}': {e}") from e

        try:
            t0 = time.monotonic()
            logger.info("Writing trigger byte to '%s' to initiate GKE Pod Snapshot...", self.proc_path)
            try:
                os.write(fd, b"1")
            except OSError as e:
                raise SnapshotError(f"Failed to write to gVisor checkpoint file: {e}") from e

            logger.info("Waiting on '%s' barrier until checkpoint/restore completes...", self.proc_path)
            try:
                # Attempting to read 1 byte blocks until checkpoint/restore completes.
                # In gVisor, reading /proc/gvisor/checkpoint returns b"" (EOF) or b"r" (resumed) on success.
                res = os.read(fd, 1)
            except OSError as e:
                msg = (
                    f"Snapshot was never initiated: {e}"
                    if e.errno == errno.EINVAL
                    else f"gVisor checkpoint failed: {e}"
                )
                raise SnapshotError(msg) from e

            if res and res != b"r":
                raise SnapshotError(f"gVisor checkpoint returned unexpected status: {res!r}")

            logger.info(
                "gVisor checkpoint completed successfully (barrier unblocked in %.2fs, status=%r).",
                time.monotonic() - t0,
                res,
            )
        finally:
            os.close(fd)


def get_snapshot_provider() -> Optional[GKESnapshotProvider]:
    """
    Returns the appropriate snapshot provider instance based on SNAPSHOT_PROVIDER env var.
    Returns GKESnapshotProvider if SNAPSHOT_PROVIDER is set to 'gke_gvisor'.
    Returns None if SNAPSHOT_PROVIDER is unset, empty, or set to any other value.
    """
    provider_name = os.getenv("SNAPSHOT_PROVIDER", "").strip().lower()
    # Fetch the current snapshot provider, but if it's not GKE gVisor, reset to None.
    if provider_name == "gke_gvisor":
        provider = GKESnapshotProvider()
        logger.info(
            "Configured GKESnapshotProvider (proc_path='%s', cache_dir=%r).",
            provider.proc_path,
            provider.cache_dir,
        )
        return provider
    return None
