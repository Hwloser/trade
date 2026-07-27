"""Finite process-group execution for layout performance probes."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessOutcome:
    stdout: bytes
    stderr: bytes
    returncode: int
    duration_ms: float
    timed_out: bool
    cleanup_survivors: int


class PerformanceProcessError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def run_process(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: float,
    output_limit_bytes: int = 1024 * 1024,
    allowed_returncodes: frozenset[int] = frozenset({0}),
    env: dict[str, str] | None = None,
) -> ProcessOutcome:
    if not argv or timeout_seconds <= 0:
        raise ValueError("process argv and timeout must be finite")
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise PerformanceProcessError(
            "layout.performance.process_spawn",
            f"cannot start {argv[0]}: {exc}",
        ) from exc
    timed_out = False
    try:
        stdout, stderr, timed_out = _read_bounded(
            process,
            deadline=time.monotonic() + timeout_seconds,
            output_limit_bytes=output_limit_bytes,
        )
        if not timed_out and _group_exists(process.pid):
            _terminate_group(process)
            raise PerformanceProcessError(
                "layout.performance.process_survivor",
                f"{argv[0]} left a descendant process after exit",
            )
        survivors = 1 if _group_exists(process.pid) else 0
        returncode = process.returncode
        if returncode is None:
            raise PerformanceProcessError(
                "layout.performance.process_state",
                f"{argv[0]} did not report an exit code",
            )
        if not timed_out and returncode not in allowed_returncodes:
            detail = (stderr or stdout).decode("utf-8", "replace")[-2048:].strip()
            raise PerformanceProcessError(
                "layout.performance.process_exit",
                f"{argv[0]} exited with {returncode}: {detail}",
            )
        return ProcessOutcome(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            duration_ms=(time.monotonic() - started) * 1000,
            timed_out=timed_out,
            cleanup_survivors=survivors,
        )
    except BaseException:
        if process.poll() is None or _group_exists(process.pid):
            _terminate_group(process)
        raise


def _read_bounded(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
    output_limit_bytes: int,
) -> tuple[bytes, bytes, bool]:
    if process.stdout is None or process.stderr is None:
        raise PerformanceProcessError(
            "layout.performance.process_state",
            "captured process pipes are unavailable",
        )
    streams = {
        process.stdout.fileno(): ("stdout", bytearray()),
        process.stderr.fileno(): ("stderr", bytearray()),
    }
    selector = selectors.DefaultSelector()
    try:
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        timed_out = False
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _terminate_group(process)
                remaining = 0.1
            events = selector.select(timeout=min(max(remaining, 0.01), 0.1))
            parent_returncode = process.poll()
            if parent_returncode is not None and _group_exists(process.pid):
                _terminate_group(process)
                raise PerformanceProcessError(
                    "layout.performance.process_survivor",
                    "parent exited while a descendant retained the process group",
                )
            if not events and parent_returncode is not None:
                events = tuple(
                    (key, selectors.EVENT_READ) for key in selector.get_map().values()
                )
            for key, _mask in events:
                descriptor = int(key.fd)
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                name, output = streams[descriptor]
                output.extend(chunk)
                if len(output) > output_limit_bytes:
                    _terminate_group(process)
                    raise PerformanceProcessError(
                        "layout.performance.process_output",
                        f"{name} exceeded the {output_limit_bytes}-byte output limit",
                    )
            if timed_out and process.poll() is not None and not selector.get_map():
                break
        if process.poll() is None:
            process.wait(timeout=1)
        return bytes(streams[process.stdout.fileno()][1]), bytes(
            streams[process.stderr.fileno()][1]
        ), timed_out
    finally:
        selector.close()


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 0.5
    while _group_exists(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    if _group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise PerformanceProcessError(
            "layout.performance.process_cleanup",
            f"process group {process.pid} survived TERM-to-KILL cleanup",
        ) from exc
    deadline = time.monotonic() + 0.5
    while _group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _group_exists(process.pid):
        raise PerformanceProcessError(
            "layout.performance.process_cleanup",
            f"process group {process.pid} remains after cleanup",
        )


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = [
    "PerformanceProcessError",
    "ProcessOutcome",
    "run_process",
]
