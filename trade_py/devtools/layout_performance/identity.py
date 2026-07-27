"""Stable runner identity for package-layout performance evidence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from trade_py.devtools.layout_performance.capacity import detected_memory_bytes
from trade_py.devtools.layout_performance.models import RunnerIdentity
from trade_py.devtools.layout_performance.processes import run_process


def capture_runner_identity(repo_root: Path) -> RunnerIdentity:
    root = repo_root.resolve()
    payload = {
        "runner_image": _runner_image(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": max(1, os.cpu_count() or 1),
        "memory_limit_bytes": detected_memory_bytes(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable_digest": _file_digest(Path(sys.executable).resolve()),
        "uv_lock_digest": _optional_file_digest(root / "uv.lock"),
        "frontend_lock_digest": _optional_file_digest(
            root / "trade_web" / "frontend" / "package-lock.json"
        ),
        "node_version": _tool_version("node"),
        "npm_version": _tool_version("npm"),
    }
    identity_digest = _payload_digest(payload)
    return RunnerIdentity(identity_digest=identity_digest, **payload)


def source_commit(repo_root: Path) -> str:
    outcome = run_process(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_root.resolve(),
        timeout_seconds=5,
        output_limit_bytes=4096,
    )
    value = outcome.stdout.decode("ascii", "strict").strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("Git did not return a full lowercase source commit")
    return value


def _tool_version(name: str) -> str | None:
    executable = shutil.which(name)
    if executable is None:
        return None
    outcome = run_process(
        (executable, "--version"),
        cwd=Path.cwd(),
        timeout_seconds=5,
        output_limit_bytes=4096,
    )
    value = outcome.stdout.decode("utf-8", "replace").strip()
    return value[:128] or None


def _optional_file_digest(path: Path) -> str:
    return _file_digest(path) if path.is_file() else "unavailable"


def _runner_image() -> str:
    explicit = os.environ.get("TRADE_RUNNER_IMAGE", "").strip()
    if explicit:
        return explicit[:256]
    return f"local:{_optional_file_digest(Path('/etc/os-release'))}"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _payload_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = ["capture_runner_identity", "source_commit"]
