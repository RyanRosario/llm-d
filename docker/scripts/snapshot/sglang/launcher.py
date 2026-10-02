"""
Launcher for SGLang with GKE Fast Pod Snapshotting support.

Hooks sglang.srt.entrypoints.http_server._wait_and_warmup with
patch_sglang_wait_and_warmup, then delegates CLI invocation to sglang.launch_server.
"""

from __future__ import annotations

import logging
import os
import sys
import time

LAUNCHER_START_TIME = time.monotonic()

logger = logging.getLogger("sglang.snapshot.launcher")
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


def _hook_api_server() -> bool:
    """Hooks sglang.srt.entrypoints.http_server._wait_and_warmup for GKE pod snapshotting (single-rank scope)."""
    t0 = time.monotonic()
    try:
        from .wrapper import patch_sglang_wait_and_warmup

        logger.info(
            "Hooking SGLang HTTP server startup (_wait_and_warmup) in pid=%d (SNAPSHOT_PROVIDER=%r)...",
            os.getpid(),
            os.getenv("SNAPSHOT_PROVIDER"),
        )
        patch_sglang_wait_and_warmup()
        logger.info(
            "Hooked SGLang _wait_and_warmup in %.2fs.",
            time.monotonic() - t0,
        )
        return True
    except (ImportError, AttributeError) as err:
        logger.warning(
            "SGLang server is not importable (%s); no snapshot will be taken.",
            err,
        )
        return False


# Module level execution so child processes also inherit the patched _wait_and_warmup
_hook_api_server()


def main() -> None:
    try:
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import prepare_server_args
    except ImportError as err:
        raise RuntimeError(
            "sglang must be installed to run snapshot launcher (python3 -m docker.scripts.snapshot.sglang.launcher)"
        ) from err

    cli_args = sys.argv[1:]
    t_args = time.monotonic()
    logger.info("Preparing SGLang server arguments from CLI args: %s", cli_args)
    server_args = prepare_server_args(cli_args)
    logger.info(
        "Prepared SGLang server arguments in %.2fs. Launching SGLang server (pid=%d, elapsed since launcher import=%.2fs)...",
        time.monotonic() - t_args,
        os.getpid(),
        time.monotonic() - LAUNCHER_START_TIME,
    )
    launch_server(server_args)


if __name__ == "__main__":
    main()
