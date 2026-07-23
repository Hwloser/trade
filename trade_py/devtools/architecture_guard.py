"""Source-only architecture baseline validation.

This module deliberately operates below application startup.  It reads the
versioned baseline plus named repository source files through no-follow
descriptors, and uses the Git index only for the bounded warehouse-producer
inventory.  It must not import inspected application modules or inspect data.
"""

from __future__ import annotations

import ast
import os
import stat
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from trade_py.devtools.quality.toml_compat import tomllib

BASELINE_FILENAME = "architecture-baseline.toml"

CANONICAL_WAREHOUSE_MODULE = "trade_py.data.warehouse.io"
CANONICAL_WAREHOUSE_PACKAGE = "trade_py.data.warehouse"
CANONICAL_WRITERS = frozenset(
    {
        f"{CANONICAL_WAREHOUSE_MODULE}.write_table",
        f"{CANONICAL_WAREHOUSE_MODULE}.upsert_table",
    }
)
CANONICAL_LAYOUT = f"{CANONICAL_WAREHOUSE_MODULE}.WarehouseLayout"

PRODUCER_UNRESOLVED_IMPORT = "architecture.producer_discovery_unresolved_import"
PRODUCER_UNRESOLVED_LAYOUT = "architecture.producer_discovery_unresolved_layout"
PRODUCER_NONLITERAL_TARGET = "architecture.producer_discovery_nonliteral_target"
PRODUCER_UNDECLARED_WRITER = "architecture.producer_discovery_undeclared_writer"
PRODUCER_PATH_BUDGET = "architecture.producer_discovery_path_budget_exceeded"
PRODUCER_SOURCE_BUDGET = "architecture.producer_discovery_budget_exceeded"
PRODUCER_UNSAFE_SOURCE = "architecture.producer_discovery_unsafe_source"

_BASELINE_MALFORMED = "architecture.baseline_malformed"
_BASELINE_DUPLICATE = "architecture.baseline_duplicate_declaration"
_BASELINE_MISSING_SOURCE = "architecture.baseline_missing_source"
_BASELINE_UNSAFE_SOURCE = "architecture.baseline_unsafe_source"
_BASELINE_LITERAL_MISMATCH = "architecture.baseline_literal_mismatch"
_BASELINE_INCOMPLETE_PROVENANCE = "architecture.baseline_incomplete_provenance"
_BASELINE_CLASSIFICATION = "architecture.baseline_invalid_classification"
_BASELINE_NONAUTHORIZING = "architecture.baseline_non_authorizing_binding"
_BASELINE_MISSING_ARTIFACT = "architecture.baseline_missing_producer_artifact"
_BASELINE_INVALID_SOURCE = "architecture.baseline_invalid_source"

_PROVENANCE_ROLES = frozenset({"bootstrap", "migration", "alter", "data_transform"})
_CLASSIFICATIONS = frozenset({"candidate", "deferred", "approved_binding"})
_EXCLUDED_SOURCE_SEGMENTS = frozenset(
    {"vendor", "third_party", "generated", "cache", "__pycache__"}
)


@dataclass(frozen=True)
class DiscoveryLimits:
    """Hard ceilings for the source-only producer discovery pass."""

    max_raw_records: int = 1_024
    max_raw_path_bytes: int = 128 * 1024
    max_included_paths: int = 512
    max_included_path_bytes: int = 64 * 1024
    max_source_bytes: int = 32 * 1024 * 1024
    max_file_bytes: int = 1 * 1024 * 1024
    max_evidence_bytes: int = 1 * 1024 * 1024


DEFAULT_LIMITS = DiscoveryLimits()


@dataclass(frozen=True)
class ArchitectureFinding:
    rule_id: str
    path: str
    line: int | None
    message: str
    remediation: str


@dataclass(frozen=True)
class WarehouseProducer:
    source: str
    line: int
    writer: str
    layer: str
    table: str

    @property
    def artifact_key(self) -> str:
        return f"{self.layer}.{self.table}"


@dataclass(frozen=True)
class ArchitectureReport:
    findings: tuple[ArchitectureFinding, ...]
    producers: tuple[WarehouseProducer, ...]

    @property
    def ok(self) -> bool:
        return not self.findings


@dataclass(frozen=True)
class _SourceSignature:
    device: int
    inode: int
    size: int
    mtime_ns: int


class _GuardError(RuntimeError):
    def __init__(
        self,
        rule_id: str,
        path: str,
        message: str,
        remediation: str,
        *,
        line: int | None = None,
    ) -> None:
        super().__init__(message)
        self.finding = ArchitectureFinding(rule_id, path, line, message, remediation)


@dataclass(frozen=True)
class _Baseline:
    source_facts: tuple[Mapping[str, Any], ...]
    tables: tuple[Mapping[str, Any], ...]
    artifacts: tuple[Mapping[str, Any], ...]
    capture_risks: tuple[Mapping[str, Any], ...]
    interfaces: tuple[Mapping[str, Any], ...]
    native_bindings: tuple[Mapping[str, Any], ...]
    producers: tuple[Mapping[str, Any], ...]


def validate_architecture_baseline(
    repo_root: Path | str,
    *,
    baseline_name: str = BASELINE_FILENAME,
    limits: DiscoveryLimits = DEFAULT_LIMITS,
) -> ArchitectureReport:
    """Validate the architecture baseline without loading application code.

    A source-discovery failure intentionally returns no producers.  Consumers
    therefore cannot mistake a truncated inventory for an empty or complete one.
    """

    root = Path(repo_root)
    findings: list[ArchitectureFinding] = []
    try:
        baseline = _load_baseline(root, baseline_name, limits)
    except _GuardError as exc:
        return ArchitectureReport((exc.finding,), ())

    findings.extend(_validate_baseline_facts(root, baseline, limits))
    if findings:
        return ArchitectureReport(_ordered_findings(findings), ())

    findings.extend(_declared_producer_source_missing(root, baseline, limits))
    if findings:
        return ArchitectureReport(_ordered_findings(findings), ())

    try:
        producers = discover_warehouse_producers(root, limits=limits)
    except _GuardError as exc:
        return ArchitectureReport((exc.finding,), ())

    findings.extend(_validate_producer_declarations(root, baseline, producers, limits))
    return ArchitectureReport(_ordered_findings(findings), producers if not findings else ())


def discover_warehouse_producers(
    repo_root: Path | str,
    *,
    limits: DiscoveryLimits = DEFAULT_LIMITS,
) -> tuple[WarehouseProducer, ...]:
    """Return every bounded, tracked production call to a canonical writer."""

    root = Path(repo_root)
    sources: list[tuple[str, str]] = []
    raw_records = 0
    raw_path_bytes = 0
    included_path_bytes = 0
    included_paths: set[str] = set()
    aggregate_source_bytes = 0

    for mode, path in _iter_git_index(root):
        raw_records += 1
        raw_path_bytes += len(path.encode("utf-8"))
        if raw_records > limits.max_raw_records or raw_path_bytes > limits.max_raw_path_bytes:
            raise _producer_path_budget_error(
                path,
                "raw Git-index record or raw path-byte budget exceeded before AST parsing",
            )
        if path.startswith("trade_py/") and not _is_safe_relative_path(path):
            raise _unsafe_source_error(path, "Git index path escapes the repository")
        if not _is_production_python_path(path, mode):
            continue
        if path in included_paths:
            raise _GuardError(
                _BASELINE_DUPLICATE,
                path,
                "Git index produced a duplicate production source path",
                "Repair the index so each tracked source has one canonical path.",
            )
        included_paths.add(path)
        included_path_bytes += len(path.encode("utf-8"))
        if (
            len(included_paths) > limits.max_included_paths
            or included_path_bytes > limits.max_included_path_bytes
        ):
            raise _producer_path_budget_error(
                path,
                "included producer path-count or path-byte budget exceeded before AST parsing",
            )

        payload = _safe_read_relative(root, path, max_bytes=limits.max_file_bytes)
        aggregate_source_bytes += len(payload)
        if aggregate_source_bytes > limits.max_source_bytes:
            raise _producer_source_budget_error(
                path,
                "aggregate producer source-byte budget exceeded before AST parsing",
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _GuardError(
                PRODUCER_UNSAFE_SOURCE,
                path,
                "production source is not valid UTF-8",
                "Keep the inspected source as a stable UTF-8 regular file inside the repository.",
            ) from exc
        sources.append((path, text))

    producers: list[WarehouseProducer] = []
    findings: list[ArchitectureFinding] = []
    for path, text in sources:
        parsed, source_findings = _discover_in_source(path, text)
        producers.extend(parsed)
        findings.extend(source_findings)
    if findings:
        first = _ordered_findings(findings)[0]
        raise _GuardError(
            first.rule_id,
            first.path,
            first.message,
            first.remediation,
            line=first.line,
        )
    return tuple(sorted(producers, key=lambda item: (item.source, item.line, item.writer)))


def _load_baseline(root: Path, baseline_name: str, limits: DiscoveryLimits) -> _Baseline:
    if not _is_safe_relative_path(baseline_name):
        raise _GuardError(
            _BASELINE_MALFORMED,
            baseline_name,
            "baseline name must be a repository-relative file path",
            "Use a regular repository-relative baseline file.",
        )
    try:
        payload = _safe_read_relative(root, baseline_name, max_bytes=limits.max_evidence_bytes)
        parsed = tomllib.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise _GuardError(
            _BASELINE_MALFORMED,
            baseline_name,
            "baseline is not valid UTF-8 TOML",
            "Write the baseline as UTF-8 TOML.",
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise _GuardError(
            _BASELINE_MALFORMED,
            baseline_name,
            f"baseline TOML cannot be parsed: {exc}",
            "Repair the baseline TOML syntax and required declarations.",
        ) from exc
    except _GuardError:
        raise

    if not isinstance(parsed, dict):
        raise _GuardError(
            _BASELINE_MALFORMED,
            baseline_name,
            "baseline TOML root must be a table",
            "Use a TOML table with the required architecture declarations.",
        )
    if parsed.get("schema_version") != 1:
        raise _GuardError(
            _BASELINE_MALFORMED,
            baseline_name,
            "baseline schema_version must equal 1",
            "Use the supported architecture baseline schema version.",
        )
    target_source_root = _require_text(parsed, "target_source_root", baseline_name)
    target_import_root = _require_text(parsed, "target_import_root", baseline_name)
    if target_source_root != "src/trade" or target_import_root != "trade":
        raise _GuardError(
            _BASELINE_MALFORMED,
            baseline_name,
            "baseline must freeze target_source_root='src/trade' and target_import_root='trade'",
            "Keep the target filesystem and import roots distinct and explicit.",
        )
    _require_string_list(parsed, "legacy_package_roots", baseline_name)
    _require_string_list(parsed, "target_contexts", baseline_name)

    collections = {
        "source_facts": _read_table_array(parsed, "source_facts", baseline_name),
        "tables": _read_table_array(parsed, "tables", baseline_name),
        "artifacts": _read_table_array(parsed, "artifacts", baseline_name),
        "capture_risks": _read_table_array(parsed, "capture_risks", baseline_name),
        "interfaces": _read_table_array(parsed, "interfaces", baseline_name),
        "native_bindings": _read_table_array(parsed, "native_bindings", baseline_name),
        "producers": _read_table_array(parsed, "warehouse_producers", baseline_name),
    }
    for name, items in collections.items():
        if not items:
            raise _GuardError(
                _BASELINE_MALFORMED,
                baseline_name,
                f"baseline must declare at least one {name} entry",
                "Record the audited source-only facts for this required category.",
            )

    return _Baseline(**collections)


def _validate_baseline_facts(
    root: Path,
    baseline: _Baseline,
    limits: DiscoveryLimits,
) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    seen_ids: set[str] = set()
    artifact_ids: set[str] = set()

    for category, facts in (
        ("source_facts", baseline.source_facts),
        ("artifacts", baseline.artifacts),
        ("capture_risks", baseline.capture_risks),
        ("interfaces", baseline.interfaces),
        ("native_bindings", baseline.native_bindings),
        ("warehouse_producers", baseline.producers),
    ):
        for fact in facts:
            try:
                fact_id = _require_text(fact, "id", category)
                if fact_id in seen_ids:
                    raise _GuardError(
                        _BASELINE_DUPLICATE,
                        BASELINE_FILENAME,
                        f"duplicate baseline declaration id: {fact_id}",
                        "Give each declared source fact one stable, unique id.",
                    )
                seen_ids.add(fact_id)
                if category != "warehouse_producers":
                    _validate_common_fact(root, fact, category, limits)
                if category == "artifacts":
                    artifact_ids.add(fact_id)
                    _validate_classification(fact, category)
                    _require_text(fact, "role", category)
                elif category == "capture_risks":
                    _require_text(fact, "risk_kind", category)
                    _require_text(fact, "current_behavior", category)
                    _require_text(fact, "required_migration_proof", category)
                elif category == "interfaces":
                    _require_text(fact, "surface_kind", category)
                    _require_text(fact, "current_behavior", category)
                    _require_text(fact, "compatibility_owner", category)
                elif category == "native_bindings":
                    _require_text(fact, "current_binding", category)
                    _require_text(fact, "reserved_binding", category)
                elif category == "warehouse_producers":
                    _validate_classification(fact, category)
                    _require_text(fact, "current_owner", category)
                    _require_text(fact, "required_child", category)
                    _require_text(fact, "layer", category)
                    _require_text(fact, "table", category)
                    _require_text(fact, "path_role", category)
                    _require_text(fact, "artifact_id", category)
            except _GuardError as exc:
                findings.append(exc.finding)

    table_names: set[str] = set()
    for table in baseline.tables:
        try:
            name = _require_text(table, "logical_name", "tables")
            if name in table_names:
                raise _GuardError(
                    _BASELINE_DUPLICATE,
                    BASELINE_FILENAME,
                    f"duplicate logical table declaration: {name}",
                    "Declare each logical table exactly once and retain all provenance on it.",
                )
            table_names.add(name)
            _require_text(table, "current_owner", "tables")
            _require_text(table, "semantic_kind", "tables")
            _require_text(table, "reason", "tables")
            _require_text(table, "required_child", "tables")
            _validate_classification(table, "tables")
            provenance = table.get("provenance")
            if not isinstance(provenance, list) or not provenance:
                raise _GuardError(
                    _BASELINE_INCOMPLETE_PROVENANCE,
                    BASELINE_FILENAME,
                    f"logical table {name} has no source provenance",
                    "Record each bootstrap, migration, alter, or transform source that defines it.",
                )
            for item in provenance:
                if not isinstance(item, dict):
                    raise _GuardError(
                        _BASELINE_INCOMPLETE_PROVENANCE,
                        BASELINE_FILENAME,
                        f"logical table {name} has malformed provenance",
                        "Use table provenance records with source, literal, and role.",
                    )
                role = _require_text(item, "role", f"tables.{name}.provenance")
                if role not in _PROVENANCE_ROLES:
                    raise _GuardError(
                        _BASELINE_INCOMPLETE_PROVENANCE,
                        BASELINE_FILENAME,
                        f"logical table {name} has unsupported provenance role {role!r}",
                        "Use bootstrap, migration, alter, or data_transform provenance roles.",
                    )
                _validate_source_literal(root, item, limits)
        except _GuardError as exc:
            findings.append(exc.finding)

    for producer in baseline.producers:
        artifact_id = producer.get("artifact_id")
        if artifact_id not in artifact_ids:
            findings.append(
                ArchitectureFinding(
                    _BASELINE_MISSING_ARTIFACT,
                    BASELINE_FILENAME,
                    None,
                    f"warehouse producer {producer.get('id', '<unknown>')} references "
                    f"missing artifact declaration {artifact_id!r}",
                    "Add a reviewed source-only artifact declaration for the produced table.",
                )
            )
    return findings


def _declared_producer_source_missing(
    root: Path,
    baseline: _Baseline,
    limits: DiscoveryLimits,
) -> list[ArchitectureFinding]:
    """Classify only missing declared producer files before index scanning.

    A present-but-unsafe path remains the producer scanner's concern, preserving
    the dedicated unsafe-source diagnostic for symlinks and replacement races.
    """

    findings: list[ArchitectureFinding] = []
    for declaration in baseline.producers:
        source = declaration.get("source")
        if not isinstance(source, str) or not _is_safe_relative_path(source):
            continue
        try:
            _safe_read_relative(root, source, max_bytes=limits.max_file_bytes)
        except _GuardError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                findings.append(
                    ArchitectureFinding(
                        _BASELINE_MISSING_SOURCE,
                        source,
                        None,
                        "declared warehouse producer source does not exist",
                        "Update or remove the producer declaration with reviewed inventory evidence.",
                    )
                )
    return findings


def _validate_producer_declarations(
    root: Path,
    baseline: _Baseline,
    producers: Sequence[WarehouseProducer],
    limits: DiscoveryLimits,
) -> list[ArchitectureFinding]:
    findings: list[ArchitectureFinding] = []
    declarations: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for declaration in baseline.producers:
        try:
            _validate_source_literal(root, declaration, limits)
        except _GuardError as exc:
            findings.append(exc.finding)
        key = (
            str(declaration.get("source") or ""),
            str(declaration.get("layer") or ""),
            str(declaration.get("table") or ""),
        )
        if key in declarations:
            findings.append(
                ArchitectureFinding(
                    _BASELINE_DUPLICATE,
                    BASELINE_FILENAME,
                    None,
                    f"duplicate warehouse producer declaration for {key[0]} {key[1]}.{key[2]}",
                    "Keep one declaration per producer source and literal artifact coordinate.",
                )
            )
            continue
        declarations[key] = declaration

    for producer in producers:
        key = (producer.source, producer.layer, producer.table)
        if key not in declarations:
            findings.append(
                ArchitectureFinding(
                    PRODUCER_UNDECLARED_WRITER,
                    producer.source,
                    producer.line,
                    f"canonical writer {producer.writer} produces undeclared "
                    f"{producer.artifact_key}",
                    "Add a reviewed baseline declaration for this producer and its artifact.",
                )
            )
    return findings


def _validate_common_fact(
    root: Path,
    fact: Mapping[str, Any],
    category: str,
    limits: DiscoveryLimits,
) -> None:
    _require_text(fact, "current_owner", category)
    _require_text(fact, "required_child", category)
    _validate_source_literal(root, fact, limits)


def _validate_source_literal(
    root: Path,
    fact: Mapping[str, Any],
    limits: DiscoveryLimits,
) -> None:
    source = _require_text(fact, "source", "source fact")
    literal = _require_text(fact, "literal", "source fact")
    try:
        payload = _safe_read_relative(root, source, max_bytes=limits.max_evidence_bytes)
    except _GuardError as exc:
        if exc.__cause__ and isinstance(exc.__cause__, FileNotFoundError):
            raise _GuardError(
                _BASELINE_MISSING_SOURCE,
                source,
                "declared evidence source does not exist",
                "Update or remove the baseline declaration with the corresponding migration evidence.",
            ) from exc
        if exc.finding.rule_id == PRODUCER_UNSAFE_SOURCE:
            raise _GuardError(
                _BASELINE_UNSAFE_SOURCE,
                source,
                exc.finding.message,
                exc.finding.remediation,
            ) from exc
        raise
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _GuardError(
            _BASELINE_UNSAFE_SOURCE,
            source,
            "declared evidence source is not valid UTF-8",
            "Keep declared evidence as stable UTF-8 source inside the repository.",
        ) from exc
    offset = text.find(literal)
    if offset < 0:
        raise _GuardError(
            _BASELINE_LITERAL_MISMATCH,
            source,
            f"declared source literal is absent: {literal!r}",
            "Update the baseline fact and its migration evidence with the changed source literal.",
        )


def _validate_classification(fact: Mapping[str, Any], category: str) -> None:
    classification = _require_text(fact, "classification", category)
    if classification not in _CLASSIFICATIONS:
        raise _GuardError(
            _BASELINE_CLASSIFICATION,
            BASELINE_FILENAME,
            f"{category} has unsupported classification {classification!r}",
            "Use candidate, deferred, or a separately reviewed approved_binding.",
        )
    target_context = _require_text(fact, "target_context", category)
    if classification == "deferred":
        if target_context != "deferred":
            raise _GuardError(
                _BASELINE_CLASSIFICATION,
                BASELINE_FILENAME,
                f"{category} deferred declaration must target 'deferred'",
                "Keep unproven ownership explicitly deferred until its child proves an owner.",
            )
        _require_text(fact, "reason", category)
        return
    if target_context == "deferred":
        raise _GuardError(
            _BASELINE_CLASSIFICATION,
            BASELINE_FILENAME,
            f"{category} {classification} declaration cannot target 'deferred'",
            "Name the candidate or approved Context explicitly.",
        )
    if classification == "candidate":
        if "approved_binding" in fact or "adapter_scope" in fact:
            raise _GuardError(
                _BASELINE_NONAUTHORIZING,
                BASELINE_FILENAME,
                f"{category} candidate declaration must not contain a persistence binding",
                "Keep candidate facts audit-only until a separately reviewed approved binding exists.",
            )
        return
    for field in (
        "adapter_scope",
        "writer_evidence",
        "reader_evidence",
        "transaction_evidence",
        "compatibility_evidence",
    ):
        _require_text(fact, field, category)


def _read_table_array(
    parsed: Mapping[str, Any],
    key: str,
    baseline_name: str,
) -> tuple[Mapping[str, Any], ...]:
    value = parsed.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise _GuardError(
            _BASELINE_MALFORMED,
            baseline_name,
            f"{key} must be a TOML array of tables",
            "Declare the required source-only facts as TOML array tables.",
        )
    return tuple(value)


def _require_text(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _GuardError(
            _BASELINE_MALFORMED,
            BASELINE_FILENAME,
            f"{path} must declare non-empty {key}",
            "Fill every required baseline declaration field with an explicit value.",
        )
    return value


def _require_string_list(mapping: Mapping[str, Any], key: str, path: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise _GuardError(
            _BASELINE_MALFORMED,
            BASELINE_FILENAME,
            f"{path} must declare a non-empty {key} string list",
            "List the declared architecture roots explicitly.",
        )
    return tuple(value)


def _iter_git_index(root: Path) -> Iterator[tuple[str, str]]:
    try:
        process = subprocess.Popen(
            ["git", "ls-files", "-z", "--stage", "--", "trade_py"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise _GuardError(
            PRODUCER_UNSAFE_SOURCE,
            "trade_py",
            f"cannot start Git index discovery: {exc}",
            "Run the check in a repository with a readable Git index.",
        ) from exc
    assert process.stdout is not None
    buffer = b""
    while True:
        chunk = process.stdout.read(8_192)
        if not chunk:
            break
        buffer += chunk
        while b"\0" in buffer:
            raw, buffer = buffer.split(b"\0", 1)
            if not raw:
                continue
            yield _parse_index_record(raw)
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    return_code = process.wait()
    if buffer or return_code != 0:
        raise _GuardError(
            PRODUCER_UNSAFE_SOURCE,
            "trade_py",
            f"Git index discovery failed: {stderr.strip() or 'incomplete NUL record'}",
            "Repair the repository Git index before running source-only discovery.",
        )


def _parse_index_record(raw: bytes) -> tuple[str, str]:
    try:
        metadata, raw_path = raw.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0].decode("ascii")
        path = raw_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise _GuardError(
            PRODUCER_UNSAFE_SOURCE,
            "trade_py",
            "Git index contains an invalid source record",
            "Use valid UTF-8 repository paths and a regular Git index.",
        ) from exc
    return mode, path


def _is_production_python_path(path: str, mode: str) -> bool:
    if mode not in {"100644", "100755"}:
        return False
    if not _is_safe_relative_path(path):
        return False
    pure = PurePosixPath(path)
    parts = pure.parts
    if not parts or parts[0] != "trade_py" or not path.endswith(".py"):
        return False
    if any(part in {"test", "tests"} for part in parts):
        return False
    if any(part in _EXCLUDED_SOURCE_SEGMENTS for part in parts):
        return False
    name = pure.name
    return not name.startswith("test_") and not name.endswith("_test.py")


def _safe_read_relative(root: Path, relative: str, *, max_bytes: int) -> bytes:
    if not _is_safe_relative_path(relative):
        raise _GuardError(
            PRODUCER_UNSAFE_SOURCE,
            relative,
            "source path escapes or is not relative to the repository",
            "Use a stable, repository-confined regular source file.",
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag:
        raise _GuardError(
            PRODUCER_UNSAFE_SOURCE,
            relative,
            "the platform does not expose required no-follow descriptor primitives",
            "Run the check on a platform that supports no-follow regular-file reads.",
        )
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | directory_flag | nofollow
    file_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow
    try:
        root_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise _GuardError(
            PRODUCER_UNSAFE_SOURCE,
            relative,
            f"cannot open repository root safely: {exc}",
            "Use a readable, non-symlink repository root.",
        ) from exc
    try:
        descriptor = _open_relative_file(root_fd, relative, directory_flags, file_flags)
        try:
            before = _regular_signature(descriptor, relative)
            if before.size > max_bytes:
                raise _producer_source_budget_error(
                    relative,
                    f"source file exceeds the {max_bytes} byte read limit",
                )
            payload = _read_descriptor(descriptor, before.size, relative)
            after = _regular_signature(descriptor, relative)
            if before != after or len(payload) != before.size:
                raise _unsafe_source_error(relative, "source identity changed while it was read")
        finally:
            os.close(descriptor)
        verify_fd = _open_relative_file(root_fd, relative, directory_flags, file_flags)
        try:
            verified = _regular_signature(verify_fd, relative)
        finally:
            os.close(verify_fd)
        if verified != before:
            raise _unsafe_source_error(relative, "source identity changed after descriptor read")
        return payload
    except _GuardError:
        raise
    except FileNotFoundError as exc:
        raise _unsafe_source_error(relative, f"source does not exist: {relative}") from exc
    except OSError as exc:
        raise _unsafe_source_error(
            relative, f"cannot read repository source safely: {exc}"
        ) from exc
    finally:
        os.close(root_fd)


def _open_relative_file(
    root_fd: int,
    relative: str,
    directory_flags: int,
    file_flags: int,
) -> int:
    descriptor = os.dup(root_fd)
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(parts[-1], file_flags, dir_fd=descriptor)
        return file_descriptor
    finally:
        os.close(descriptor)


def _regular_signature(descriptor: int, path: str) -> _SourceSignature:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise _unsafe_source_error(path, "source is not a regular file")
    return _SourceSignature(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )


def _read_descriptor(descriptor: int, size: int, path: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) != size:
        raise _unsafe_source_error(path, "source could not be read completely from its descriptor")
    return payload


def _discover_in_source(
    path: str,
    text: str,
) -> tuple[list[WarehouseProducer], list[ArchitectureFinding]]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        return [], [
            ArchitectureFinding(
                _BASELINE_INVALID_SOURCE,
                path,
                exc.lineno,
                f"cannot parse production source for warehouse discovery: {exc.msg}",
                "Repair the source syntax before relying on its architecture inventory.",
            )
        ]

    aliases: dict[str, str] = {}
    layouts: set[str] = set()
    findings: list[ArchitectureFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                aliases[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                if module in {CANONICAL_WAREHOUSE_MODULE, CANONICAL_WAREHOUSE_PACKAGE}:
                    if alias.name in {"write_table", "upsert_table"}:
                        aliases[local] = f"{CANONICAL_WAREHOUSE_MODULE}.{alias.name}"
                    elif alias.name == "WarehouseLayout":
                        aliases[local] = CANONICAL_LAYOUT
                    elif alias.name == "io":
                        aliases[local] = CANONICAL_WAREHOUSE_MODULE
                    elif alias.name == "*":
                        findings.append(
                            _producer_finding(
                                PRODUCER_UNRESOLVED_IMPORT,
                                path,
                                node.lineno,
                                "star import from the warehouse boundary cannot be resolved",
                                "Import write_table or upsert_table explicitly from the canonical warehouse API.",
                            )
                        )
                    elif _writer_like(alias.name):
                        findings.append(
                            _producer_finding(
                                PRODUCER_UNRESOLVED_IMPORT,
                                path,
                                node.lineno,
                                f"warehouse writer-like import {alias.name!r} is not canonical",
                                "Import write_table or upsert_table from trade_py.data.warehouse.io "
                                "or the package re-export.",
                            )
                        )
                elif module.startswith(CANONICAL_WAREHOUSE_PACKAGE) and _writer_like(alias.name):
                    findings.append(
                        _producer_finding(
                            PRODUCER_UNRESOLVED_IMPORT,
                            path,
                            node.lineno,
                            f"warehouse writer-like import {module}.{alias.name} is not canonical",
                            "Import a canonical writer from trade_py.data.warehouse.io.",
                        )
                    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            resolved = _resolve_expression(node.value, aliases)
            names = _assignment_names(node.targets)
            if resolved in CANONICAL_WRITERS:
                for name in names:
                    aliases[name] = resolved
            if _is_layout_factory(node.value, aliases):
                layouts.update(names)
        elif isinstance(node, ast.AnnAssign):
            if _annotation_is_layout(node.annotation, aliases):
                layouts.update(_assignment_names((node.target,)))
            elif _is_layout_factory(node.value, aliases):
                layouts.update(_assignment_names((node.target,)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if _annotation_is_layout(argument.annotation, aliases):
                    layouts.add(argument.arg)

    producers: list[WarehouseProducer] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        writer = _resolve_expression(node.func, aliases)
        if writer not in CANONICAL_WRITERS:
            continue
        if not node.args or not _is_layout_expression(node.args[0], aliases, layouts):
            findings.append(
                _producer_finding(
                    PRODUCER_UNRESOLVED_LAYOUT,
                    path,
                    node.lineno,
                    f"canonical writer {writer} has no statically known WarehouseLayout first argument",
                    "Bind the first argument from WarehouseLayout or WarehouseLayout.from_data_root.",
                )
            )
            continue
        if len(node.args) < 3 or not all(
            isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            for argument in node.args[1:3]
        ):
            findings.append(
                _producer_finding(
                    PRODUCER_NONLITERAL_TARGET,
                    path,
                    node.lineno,
                    f"canonical writer {writer} has a nonliteral layer or table target",
                    "Use literal layer and table strings so the artifact declaration is auditable.",
                )
            )
            continue
        layer = node.args[1]
        table = node.args[2]
        assert isinstance(layer, ast.Constant) and isinstance(layer.value, str)
        assert isinstance(table, ast.Constant) and isinstance(table.value, str)
        producers.append(
            WarehouseProducer(
                source=path,
                line=node.lineno,
                writer=writer,
                layer=layer.value,
                table=table.value,
            )
        )
    return producers, findings


def _resolve_expression(node: ast.AST | None, aliases: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if not isinstance(node, ast.Attribute):
        return None
    base = _resolve_expression(node.value, aliases)
    if base is None and isinstance(node.value, ast.Name):
        base = aliases.get(node.value.id)
    if base is None:
        return None
    resolved = f"{base}.{node.attr}"
    if resolved in {
        f"{CANONICAL_WAREHOUSE_PACKAGE}.write_table",
        f"{CANONICAL_WAREHOUSE_PACKAGE}.upsert_table",
    }:
        return f"{CANONICAL_WAREHOUSE_MODULE}.{node.attr}"
    if resolved == f"{CANONICAL_WAREHOUSE_PACKAGE}.WarehouseLayout":
        return CANONICAL_LAYOUT
    return resolved


def _is_layout_factory(node: ast.AST | None, aliases: Mapping[str, str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    callee = _resolve_expression(node.func, aliases)
    return callee in {CANONICAL_LAYOUT, f"{CANONICAL_LAYOUT}.from_data_root"}


def _is_layout_expression(
    node: ast.AST,
    aliases: Mapping[str, str],
    layouts: set[str],
) -> bool:
    return isinstance(node, ast.Name) and node.id in layouts or _is_layout_factory(node, aliases)


def _annotation_is_layout(node: ast.AST | None, aliases: Mapping[str, str]) -> bool:
    return _resolve_expression(node, aliases) == CANONICAL_LAYOUT


def _assignment_names(targets: Sequence[ast.AST]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(_assignment_names(target.elts))
    return names


def _writer_like(name: str) -> bool:
    lowered = name.lower()
    return "write" in lowered or "upsert" in lowered or "writer" in lowered


def _producer_finding(
    rule_id: str,
    path: str,
    line: int,
    message: str,
    remediation: str,
) -> ArchitectureFinding:
    return ArchitectureFinding(rule_id, path, line, message, remediation)


def _producer_path_budget_error(path: str, message: str) -> _GuardError:
    return _GuardError(
        PRODUCER_PATH_BUDGET,
        path,
        message,
        "Reduce or split the source scope, or make a reviewed governed budget increase.",
    )


def _producer_source_budget_error(path: str, message: str) -> _GuardError:
    return _GuardError(
        PRODUCER_SOURCE_BUDGET,
        path,
        message,
        "Reduce or split the source scope, or make a reviewed governed budget increase.",
    )


def _unsafe_source_error(path: str, message: str) -> _GuardError:
    return _GuardError(
        PRODUCER_UNSAFE_SOURCE,
        path,
        message,
        "Keep the source regular, repository-confined, and stable for the check.",
    )


def _is_safe_relative_path(path: str) -> bool:
    if not path or "\\" in path:
        return False
    pure = PurePosixPath(path)
    return not pure.is_absolute() and all(part not in {"", ".", ".."} for part in pure.parts)


def _ordered_findings(
    findings: Sequence[ArchitectureFinding],
) -> tuple[ArchitectureFinding, ...]:
    return tuple(
        sorted(
            findings,
            key=lambda item: (
                item.path,
                item.line if item.line is not None else -1,
                item.rule_id,
                item.message,
            ),
        )
    )


__all__ = [
    "ArchitectureFinding",
    "ArchitectureReport",
    "DEFAULT_LIMITS",
    "DiscoveryLimits",
    "WarehouseProducer",
    "discover_warehouse_producers",
    "validate_architecture_baseline",
]
