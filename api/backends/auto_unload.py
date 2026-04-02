# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Model Auto-Unload Manager.

Monitors model idle time and automatically unloads the model from VRAM/RAM
when it has been idle for longer than the configured threshold.
The model is automatically reloaded on the next inference request.

Configuration (environment variables):
  MODEL_IDLE_TIMEOUT   – idle duration before unload, e.g. "10m", "30s", "1h"
                         Supports: s (seconds), m (minutes), h (hours)
                         Set to "0" or leave empty to disable (default).
  MODEL_UNLOAD_CHECK_INTERVAL – how often to poll for idleness, e.g. "30s"
                                Default: "30s"
"""

import asyncio
import logging
import os
import re
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import TTSBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: human-readable duration parser
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(
    r"^\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m(?:in)?)?\s*(?:(\d+)\s*s(?:ec)?)?\s*$",
    re.IGNORECASE,
)


def parse_duration(value: str) -> int:
    """
    Parse a human-readable duration string into seconds.

    Supported formats:
      "0"     -> 0  (disabled)
      "30s"   -> 30
      "2m"    -> 120
      "1h"    -> 3600
      "1h30m" -> 5400
      "90"    -> 90  (bare integer treated as seconds)

    Raises:
        ValueError: if the string cannot be parsed.
    """
    value = value.strip()
    if not value or value == "0":
        return 0

    # Pure integer → seconds
    if value.isdigit():
        return int(value)

    m = _DURATION_RE.match(value)
    if not m:
        raise ValueError(
            f"Cannot parse duration '{value}'. "
            "Use formats like '30s', '2m', '1h', '1h30m'."
        )

    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    total = hours * 3600 + minutes * 60 + seconds
    if total == 0:
        raise ValueError(f"Duration '{value}' parsed to 0 seconds; use '0' to disable.")
    return total


def _fmt_seconds(seconds: int) -> str:
    """Format seconds as a compact human-readable string e.g. '1h30m' or '2m'."""
    parts = []
    if seconds >= 3600:
        parts.append(f"{seconds // 3600}h")
        seconds %= 3600
    if seconds >= 60:
        parts.append(f"{seconds // 60}m")
        seconds %= 60
    if seconds:
        parts.append(f"{seconds}s")
    return "".join(parts) or "0s"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ModelAutoUnloadManager:
    """
    Background manager that unloads the TTS model after a configurable idle
    period and transparently reloads it on the next inference request.

    Usage
    -----
    # At server startup (inside lifespan):
        manager = ModelAutoUnloadManager()
        manager.start(backend, timeout_seconds=300, check_interval=30)

    # Before each inference request (inside get_tts_backend):
        await manager.ensure_loaded()
        manager.touch()

    # At server shutdown:
        await manager.stop()
    """

    def __init__(self) -> None:
        self._backend: Optional["TTSBackend"] = None
        self._timeout: int = 0          # 0 = disabled
        self._check_interval: int = 30
        self._last_activity: float = time.monotonic()
        self._monitor_task: Optional[asyncio.Task] = None
        self._reload_lock: asyncio.Lock = asyncio.Lock()
        # Track the custom voices directory so we can reload custom voices too
        self._custom_voices_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        backend: "TTSBackend",
        timeout_seconds: int,
        check_interval: int = 30,
        custom_voices_dir: Optional[str] = None,
    ) -> None:
        """Start the background monitor task."""
        self._backend = backend
        self._timeout = timeout_seconds
        self._check_interval = max(1, check_interval)
        self._custom_voices_dir = custom_voices_dir
        self.touch()  # mark as active immediately

        if timeout_seconds <= 0:
            logger.info("Model auto-unload is DISABLED (MODEL_IDLE_TIMEOUT=0)")
            return

        logger.info(
            f"Model auto-unload enabled: idle timeout={_fmt_seconds(timeout_seconds)}, "
            f"check every {_fmt_seconds(check_interval)}"
        )
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="model-auto-unload"
        )

    async def stop(self) -> None:
        """Cancel the background monitor task (call on server shutdown)."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.debug("ModelAutoUnloadManager stopped")

    def touch(self) -> None:
        """Update the last-activity timestamp. Call this on every inference request."""
        self._last_activity = time.monotonic()

    async def ensure_loaded(self) -> None:
        """
        Ensure the backend model is loaded and ready.

        If the model was previously unloaded due to idleness, this method
        re-initialises it (with a lock to prevent concurrent re-loads).
        This is a no-op when auto-unload is disabled or the model is ready.
        """
        if self._backend is None:
            return
        if self._backend.is_ready():
            return

        async with self._reload_lock:
            # Double-check after acquiring lock (another coroutine may have loaded it)
            if self._backend.is_ready():
                return
            logger.info(
                "Model not ready — reloading before processing request..."
            )
            t0 = time.monotonic()
            await self._backend.initialize()
            # Restore custom voices if any
            if self._custom_voices_dir:
                try:
                    await self._backend.load_custom_voices(self._custom_voices_dir)
                except Exception as e:
                    logger.warning(f"Custom voice reload failed (non-critical): {e}")
            elapsed = time.monotonic() - t0
            logger.info(f"Model reloaded successfully in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Periodically check idle time and unload the model if it has expired."""
        try:
            while True:
                await asyncio.sleep(self._check_interval)
                if self._backend is None or not self._backend.is_ready():
                    continue

                idle = time.monotonic() - self._last_activity
                if idle >= self._timeout:
                    logger.info(
                        f"Model idle for {idle:.0f}s (threshold: "
                        f"{_fmt_seconds(self._timeout)}) — unloading..."
                    )
                    await self._unload()
        except asyncio.CancelledError:
            pass

    async def _unload(self) -> None:
        """Unload the model to free VRAM/RAM."""
        if self._backend is None:
            return
        try:
            await self._backend.unload()
            logger.info("Model unloaded successfully (VRAM/RAM freed)")
        except Exception as e:
            logger.error(f"Error during model unload: {e}")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager_instance: Optional[ModelAutoUnloadManager] = None


def get_auto_unload_manager() -> ModelAutoUnloadManager:
    """Return (or create) the global ModelAutoUnloadManager singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelAutoUnloadManager()
    return _manager_instance
