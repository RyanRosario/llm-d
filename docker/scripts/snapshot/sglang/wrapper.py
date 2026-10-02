"""
SGLang Wrapper Entrypoint for GKE Fast Pod Snapshotting (docker/scripts/snapshot/sglang/wrapper.py).

Note: Scope is single-rank deployments (DP/multi-rank barrier coordination is not covered).

Patches sglang.srt.entrypoints.http_server._wait_and_warmup to:
1. Run standard server warmup (captures CUDA graphs, pre-allocates VRAM, and freezes GC).
2. Release physical VRAM via tokenizer_manager.release_memory_occupation(tags=["weights", "kv_cache"]).
3. Trigger the snapshot checkpoint via snapshot_provider.trigger() (clearing model weights cache on disk).
4. Re-allocate physical VRAM via tokenizer_manager.resume_memory_occupation(tags=["weights", "kv_cache"]) upon restore.
5. Mark tokenizer_manager.server_status = ServerStatus.Up and invoke launch_callback to begin serving traffic.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Optional

from ..providers import (
    GKESnapshotProvider,
    get_snapshot_provider,
)


WRAPPER_IMPORT_TIME = time.monotonic()

logger = logging.getLogger("sglang.snapshot.wrapper")
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


def patch_sglang_wait_and_warmup(snapshot_provider: Optional[GKESnapshotProvider] = None):
    """
    Patches sglang's http_server._wait_and_warmup for GKE pod snapshotting (single-rank scope).

    Args:
        snapshot_provider: Snapshot provider instance. Defaults to the provider configured
            by the SNAPSHOT_PROVIDER environment variable (or None if unset/disabled).
    """
    from sglang.srt.entrypoints import http_server
    from sglang.srt.entrypoints.http_server import (
        ServerStatus,
        _execute_server_warmup,
        _freeze_gc_after_server_warmup,
        _wait_weights_ready,
        get_exec,
        get_model,
        get_observability,
        get_serving,
        kill_process_tree,
    )
    from sglang.srt.managers.io_struct import (
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
    )

    if snapshot_provider is None:
        # Get the configured snapshot provider, if any
        snapshot_provider = get_snapshot_provider()

    if snapshot_provider is None:
        logger.info(
            "No snapshot provider configured (SNAPSHOT_PROVIDER is unset or empty). Snapshotting is disabled."
        )
        return

    original_wait_and_warmup = http_server._wait_and_warmup

    def patched_wait_and_warmup(
        server_args,
        launch_callback=None,
        execute_warmup_func=_execute_server_warmup,
    ):
        t_enter = time.monotonic()
        logger.info(
            "[Control Plane] Entering patched _wait_and_warmup (pid=%d, provider=%s, proc_path='%s', cache_dir=%r, elapsed_since_start=%.2fs).",
            os.getpid(),
            type(snapshot_provider).__name__,
            getattr(snapshot_provider, "proc_path", "unknown"),
            getattr(snapshot_provider, "cache_dir", None),
            t_enter - WRAPPER_IMPORT_TIME,
        )
        # This is a blocking function not asynchronous context.
        if hasattr(snapshot_provider, "is_available") and not snapshot_provider.is_available():
            logger.warning(
                "Pod snapshot trigger not available (checkpoint file '%s' is not writable). Skipping snapshot.",
                getattr(snapshot_provider, "proc_path", "unknown"),
            )
            return original_wait_and_warmup(
                server_args,
                launch_callback=launch_callback,
                execute_warmup_func=execute_warmup_func,
            )

        logger.info("[Control Plane] Checking if model weights are ready in GPUs...")
        if get_model().checkpoint_engine_wait_weights_before_ready:
            t_weights = time.monotonic()
            logger.info("[Control Plane] Waiting for checkpoint engine weights to be ready in GPUs...")
            _wait_weights_ready()
            logger.info(
                "[Control Plane] Model weights are ready in GPUs (waited %.2fs).",
                time.monotonic() - t_weights,
            )

        # Joiner schedulers are served through the primary after adoption.
        skip_elastic_joiner_warmup = server_args.is_ep_scale_joiner
        if skip_elastic_joiner_warmup:
            logger.debug(
                "[Elastic EP] Skipping server warmup for elastic joiner (ep_join_mode=%s)",
                get_exec().moe.ep_join_mode,
            )

        # Warmup captures CUDA graphs and pre-allocates VRAM
        if not get_serving().skip_server_warmup and not skip_elastic_joiner_warmup:
            t_warmup = time.monotonic()
            logger.info("[Control Plane] Starting SGLang server warmup (capturing CUDA graphs & pre-allocating VRAM)...")
            if not execute_warmup_func(server_args):
                logger.error("[Control Plane] Server warmup failed; aborting before snapshot checkpoint.")
                return
            logger.info(
                "[Control Plane] Server warmup completed in %.2fs (cold-start elapsed_since_start=%.2fs).",
                time.monotonic() - t_warmup,
                time.monotonic() - WRAPPER_IMPORT_TIME,
            )
        else:
            logger.warning("[Control Plane] Warmup skipped.")

        t_gc = time.monotonic()
        logger.info("[Control Plane] Freezing Python garbage collection after server warmup...")
        _freeze_gc_after_server_warmup(server_args)
        logger.info(
            "[Control Plane] Froze Python garbage collection in %.2fs.",
            time.monotonic() - t_gc,
        )

        tokenizer_manager = http_server._global_state.tokenizer_manager
        tokenizer_manager.server_status = ServerStatus.Starting
        logger.info(
            "[Control Plane] Set tokenizer_manager.server_status = %s (keeping /health at 503 during sleep/checkpoint).",
            tokenizer_manager.server_status,
        )

        # SLEEP
        sleep_tags = ["weights", "kv_cache"]
        t_sleep = time.monotonic()
        logger.info(
            "[Control Plane] Sleep signal received. Releasing GPU memory occupation for tags=%s...",
            sleep_tags,
        )
        asyncio.run_coroutine_threadsafe(
            tokenizer_manager.release_memory_occupation(ReleaseMemoryOccupationReqInput(tags=sleep_tags)),
            tokenizer_manager.event_loop,
        ).result()
        logger.info(
            "[Control Plane] Released GPU memory occupation for tags=%s in %.2fs.",
            sleep_tags,
            time.monotonic() - t_sleep,
        )

        t_trigger = time.monotonic()
        logger.info("[Control Plane] Triggering snapshot checkpoint via %s...", type(snapshot_provider).__name__)
        try:
            snapshot_provider.trigger()
            logger.info(
                "[Control Plane] Snapshot checkpoint created / process restored from checkpoint (barrier elapsed=%.2fs).",
                time.monotonic() - t_trigger,
            )
        except Exception as e:
            logger.error("Snapshot checkpointing failed: %s. Resuming sglang service without checkpoint.", e, exc_info=True)

        # WAKE
        wake_tags = ["weights", "kv_cache"]
        t_wake = time.monotonic()
        logger.info(
            "[Control Plane] Wake signal received. Resuming GPU memory occupation for tags=%s...",
            wake_tags,
        )
        asyncio.run_coroutine_threadsafe(
            tokenizer_manager.resume_memory_occupation(
                ResumeMemoryOccupationReqInput(tags=wake_tags)),
            tokenizer_manager.event_loop,
        ).result()
        wake_elapsed = time.monotonic() - t_wake
        logger.info(
            "[Control Plane] Resumed GPU memory occupation for tags=%s in %.2fs.",
            wake_tags,
            wake_elapsed,
        )

        # The server is ready for requests
        # Only set the server to ready once it has woken up again. This satisfies the readiness probe
        tokenizer_manager.server_status = ServerStatus.Up
        logger.info(
            "[Control Plane] Set tokenizer_manager.server_status = %s (wake-to-ready=%.2fs). The server is fired up and ready to roll!",
            tokenizer_manager.server_status,
            time.monotonic() - t_wake,
        )

        if get_observability().debug_tensor_dump_input_file:
            logger.info("[Control Plane] debug_tensor_dump_input_file set; terminating process tree (pid=%d).", os.getpid())
            kill_process_tree(os.getpid())

        if launch_callback is not None:
            t_cb = time.monotonic()
            logger.info("[Control Plane] Invoking launch_callback...")
            launch_callback()
            logger.info("[Control Plane] launch_callback completed in %.2fs.", time.monotonic() - t_cb)

    http_server._wait_and_warmup = patched_wait_and_warmup
    logger.info("Successfully patched SGLang _wait_and_warmup for GKE snapshotting.")
