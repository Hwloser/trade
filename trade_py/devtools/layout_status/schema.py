"""Typed immutable snapshots parsed from layout evidence records."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Final, NoReturn

from trade_py.devtools.layout_status.constraints import LayoutStatusAxes
from trade_py.devtools.layout_status.errors import invalid
from trade_py.devtools.layout_status.models import (
    LayoutSelectorSnapshotV1,
    MigrationEvidenceRef,
    OperationStatusSnapshotV1,
    PreparedEvidenceRef,
    ProcessSnapshot,
    RollbackSnapshot,
    SelectorObservation,
    ShutdownSnapshot,
)
from trade_py.devtools.layout_status.records import EvidenceRecord, canonical_json

_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}")
_IDENTITY_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SCOPE_RE: Final = re.compile(r"(?:python_deployment|web_build|native_capability:[a-z][a-z0-9_]*)")
_PHASES: Final = (
    "prepared",
    "selector_committed",
    "start_intent_recorded",
    "process_started",
    "verified",
    "failed",
    "rollback_prepared",
    "rollback_selector_committed",
    "rollback_verified",
)
_SHUTDOWN_STAGES: Final = frozenset({"not_started", "term", "kill", "complete"})
_ESCALATIONS: Final = frozenset({"none", "term", "term_kill"})


def parse_prepared(record: EvidenceRecord) -> PreparedEvidenceRef:
    _expect_type(record, "prepared_evidence")
    payload = _object(
        record.payload,
        {
            "operation_id",
            "attempt_id",
            "scope",
            "source_commit",
            "source_tree_digest",
            "policy_digest",
            "approved_design_digest",
            "activation_plan_digest",
            "current_composition_digest",
            "immutable_input_refs",
            "intended_target_generation",
            "expected_generation",
            "expected_revision",
            "expected_fence",
            "deployment_unit",
            "invocation_token",
            "command_digests",
            "prepared_at",
        },
        record,
    )
    return PreparedEvidenceRef(
        record_digest=record.record_digest,
        operation_id=_identity(payload, "operation_id", record),
        attempt_id=_identity(payload, "attempt_id", record),
        scope=_scope(payload, "scope", record),
        source_commit=_commit(payload, "source_commit", record),
        source_tree_digest=_digest(payload, "source_tree_digest", record),
        policy_digest=_digest(payload, "policy_digest", record),
        approved_design_digest=_digest(payload, "approved_design_digest", record),
        activation_plan_digest=_digest(payload, "activation_plan_digest", record),
        current_composition_digest=_digest(payload, "current_composition_digest", record),
        immutable_input_refs=_digest_list(payload, "immutable_input_refs", record),
        intended_target_generation=_identity(payload, "intended_target_generation", record),
        expected_generation=_identity(payload, "expected_generation", record),
        expected_revision=_integer(payload, "expected_revision", record),
        expected_fence=_integer(payload, "expected_fence", record),
        deployment_unit=_identity(payload, "deployment_unit", record),
        invocation_token=_identity(payload, "invocation_token", record),
        command_digests=_digest_list(payload, "command_digests", record),
        prepared_at=_utc_timestamp(payload, "prepared_at", record),
    )


def parse_selector(record: EvidenceRecord) -> LayoutSelectorSnapshotV1:
    _expect_type(record, "layout_selector_snapshot")
    payload = _object(
        record.payload,
        {
            "scope",
            "generation",
            "revision",
            "fence",
            "operation_id",
            "plan_digest",
            "prepared_evidence_ref",
            "predecessor_generation",
            "predecessor_revision",
            "selector_payload_digest",
        },
        record,
    )
    selector = LayoutSelectorSnapshotV1(
        record_digest=record.record_digest,
        scope=_scope(payload, "scope", record),
        generation=_identity(payload, "generation", record),
        revision=_integer(payload, "revision", record, minimum=1),
        fence=_integer(payload, "fence", record, minimum=1),
        operation_id=_identity(payload, "operation_id", record),
        plan_digest=_digest(payload, "plan_digest", record),
        prepared_evidence_ref=_digest(payload, "prepared_evidence_ref", record),
        predecessor_generation=_identity(payload, "predecessor_generation", record),
        predecessor_revision=_integer(payload, "predecessor_revision", record),
        selector_payload_digest=_digest(payload, "selector_payload_digest", record),
    )
    expected_digest = _content_digest(payload, "selector_payload_digest")
    if selector.selector_payload_digest != expected_digest:
        _fail(record, "layout.status.selector_digest", "Selector payload digest mismatches.")
    if selector.predecessor_revision + 1 != selector.revision:
        _fail(
            record,
            "layout.status.selector_revision",
            "Selector revision must be exactly one after its predecessor.",
        )
    if selector.generation == selector.predecessor_generation:
        _fail(
            record,
            "layout.status.selector_generation",
            "Selector generation must differ from its predecessor.",
        )
    return selector


def parse_operation(record: EvidenceRecord) -> OperationStatusSnapshotV1:
    _expect_type(record, "operation_status_snapshot")
    payload = _object(
        record.payload,
        {
            "operation_id",
            "attempt_id",
            "scope",
            "activation_phase",
            "states",
            "operator_action",
            "tool_exit_code",
            "partial_evidence_ref",
            "stopped_early",
            "stop_reason",
            "failure_detail",
            "degraded_components",
            "process",
            "shutdown",
            "rollback",
        },
        record,
    )
    phase = _string(payload, "activation_phase", record)
    if phase not in _PHASES:
        _fail(record, "layout.status.phase", "Operation uses an unknown activation phase.")
    action = _string(payload, "operator_action", record)
    process = _parse_process(_mapping(payload, "process", record), record)
    shutdown = _parse_shutdown(_mapping(payload, "shutdown", record), record)
    rollback = _parse_rollback(_mapping(payload, "rollback", record), record)
    degraded = _identity_list(payload, "degraded_components", record, maximum=32)
    return OperationStatusSnapshotV1(
        record_digest=record.record_digest,
        operation_id=_identity(payload, "operation_id", record),
        attempt_id=_identity(payload, "attempt_id", record),
        scope=_scope(payload, "scope", record),
        activation_phase=phase,
        axes=_axes(_mapping(payload, "states", record), record),
        operator_action=action,
        tool_exit_code=_optional_integer(payload, "tool_exit_code", record, minimum=0),
        partial_evidence_ref=_optional_digest(payload, "partial_evidence_ref", record),
        stopped_early=_boolean(payload, "stopped_early", record),
        stop_reason=_optional_identity(payload, "stop_reason", record),
        failure_detail=_optional_identity(payload, "failure_detail", record),
        degraded_components=degraded,
        process=process,
        shutdown=shutdown,
        rollback=rollback,
    )


def parse_migration(record: EvidenceRecord) -> MigrationEvidenceRef:
    _expect_type(record, "migration_evidence")
    payload = _object(
        record.payload,
        {
            "operation_id",
            "attempt_id",
            "scope",
            "source_commit",
            "source_tree_digest",
            "policy_digest",
            "approved_design_digest",
            "consumer_inventory_ref",
            "module_authority_ref",
            "artifact_refs",
            "activation_plan_digest",
            "prepared_evidence_ref",
            "selector_before",
            "selector_after",
            "phases",
            "terminal_outcome",
            "command_digests",
            "toolchain",
            "deadline_milliseconds",
            "typed_outcomes",
            "partial_evidence_refs",
            "report_entries_digest",
        },
        record,
    )
    phases = _phase_list(payload, record)
    terminal = _string(payload, "terminal_outcome", record)
    if terminal not in {"verified", "failed", "rolled_back"}:
        _fail(record, "layout.status.terminal_outcome", "Migration outcome is not terminal.")
    if terminal == "verified" and phases[-1] != "verified":
        _fail(record, "layout.status.phase_order", "Verified evidence must end at verified.")
    if terminal == "failed" and phases[-1] != "failed":
        _fail(record, "layout.status.phase_order", "Failed evidence must end at failed.")
    if terminal == "rolled_back" and phases[-1] != "rollback_verified":
        _fail(
            record,
            "layout.status.phase_order",
            "Rolled-back evidence must end at rollback_verified.",
        )
    return MigrationEvidenceRef(
        record_digest=record.record_digest,
        operation_id=_identity(payload, "operation_id", record),
        attempt_id=_identity(payload, "attempt_id", record),
        scope=_scope(payload, "scope", record),
        source_commit=_commit(payload, "source_commit", record),
        source_tree_digest=_digest(payload, "source_tree_digest", record),
        policy_digest=_digest(payload, "policy_digest", record),
        approved_design_digest=_digest(payload, "approved_design_digest", record),
        consumer_inventory_ref=_digest(payload, "consumer_inventory_ref", record),
        module_authority_ref=_digest(payload, "module_authority_ref", record),
        artifact_refs=_digest_list(payload, "artifact_refs", record),
        activation_plan_digest=_digest(payload, "activation_plan_digest", record),
        prepared_evidence_ref=_digest(payload, "prepared_evidence_ref", record),
        selector_before=_selector_observation(_mapping(payload, "selector_before", record), record),
        selector_after=_selector_observation(_mapping(payload, "selector_after", record), record),
        phases=phases,
        terminal_outcome=terminal,
        command_digests=_digest_list(payload, "command_digests", record),
        toolchain=_identity_list(payload, "toolchain", record, maximum=32),
        deadline_milliseconds=_integer(payload, "deadline_milliseconds", record, minimum=1),
        typed_outcomes=_axes(_mapping(payload, "typed_outcomes", record), record),
        partial_evidence_refs=_digest_list(payload, "partial_evidence_refs", record),
        report_entries_digest=_digest(payload, "report_entries_digest", record),
    )


def _parse_process(payload: dict[str, Any], record: EvidenceRecord) -> ProcessSnapshot:
    _exact(
        payload,
        {
            "deployment_unit",
            "invocation_token",
            "generation",
            "revision",
            "fence",
            "process_started_receipt",
            "matching_live_instances",
            "historical_process_started",
            "terminal_receipt",
            "terminal_identity_match",
            "zero_live_descendants",
            "teardown_receipt",
            "teardown_identity_match",
        },
        record,
    )
    return ProcessSnapshot(
        deployment_unit=_identity(payload, "deployment_unit", record),
        invocation_token=_identity(payload, "invocation_token", record),
        generation=_identity(payload, "generation", record),
        revision=_integer(payload, "revision", record),
        fence=_integer(payload, "fence", record),
        process_started_receipt=_boolean(payload, "process_started_receipt", record),
        matching_live_instances=_integer(payload, "matching_live_instances", record, maximum=2),
        historical_process_started=_boolean(payload, "historical_process_started", record),
        terminal_receipt=_boolean(payload, "terminal_receipt", record),
        terminal_identity_match=_boolean(payload, "terminal_identity_match", record),
        zero_live_descendants=_boolean(payload, "zero_live_descendants", record),
        teardown_receipt=_boolean(payload, "teardown_receipt", record),
        teardown_identity_match=_boolean(payload, "teardown_identity_match", record),
    )


def _parse_shutdown(payload: dict[str, Any], record: EvidenceRecord) -> ShutdownSnapshot:
    _exact(
        payload,
        {
            "stage",
            "signal_escalation",
            "residual_process_count",
            "residual_thread_count",
            "forced_exit_receipt",
            "complete",
        },
        record,
    )
    stage = _string(payload, "stage", record)
    escalation = _string(payload, "signal_escalation", record)
    if stage not in _SHUTDOWN_STAGES or escalation not in _ESCALATIONS:
        _fail(record, "layout.status.shutdown", "Shutdown state uses an unknown value.")
    return ShutdownSnapshot(
        stage=stage,
        signal_escalation=escalation,
        residual_process_count=_integer(payload, "residual_process_count", record),
        residual_thread_count=_integer(payload, "residual_thread_count", record),
        forced_exit_receipt=_boolean(payload, "forced_exit_receipt", record),
        complete=_boolean(payload, "complete", record),
    )


def _parse_rollback(payload: dict[str, Any], record: EvidenceRecord) -> RollbackSnapshot:
    _exact(
        payload,
        {
            "plan_scope",
            "target_generation",
            "later_accepted_slices",
            "target_is_historical_predecessor",
            "compensation_preserves_later_slices",
        },
        record,
    )
    return RollbackSnapshot(
        plan_scope=_scope(payload, "plan_scope", record),
        target_generation=_optional_identity(payload, "target_generation", record),
        later_accepted_slices=_boolean(payload, "later_accepted_slices", record),
        target_is_historical_predecessor=_boolean(
            payload, "target_is_historical_predecessor", record
        ),
        compensation_preserves_later_slices=_boolean(
            payload, "compensation_preserves_later_slices", record
        ),
    )


def _selector_observation(payload: dict[str, Any], record: EvidenceRecord) -> SelectorObservation:
    _exact(payload, {"generation", "revision", "fence"}, record)
    return SelectorObservation(
        generation=_identity(payload, "generation", record),
        revision=_integer(payload, "revision", record),
        fence=_integer(payload, "fence", record),
    )


def _axes(payload: dict[str, Any], record: EvidenceRecord) -> LayoutStatusAxes:
    keys = {
        "migration_state",
        "execution_state",
        "failure_class",
        "rollback_state",
        "startup_state",
        "reconciliation_state",
    }
    _exact(payload, keys, record)
    return LayoutStatusAxes(**{key: _string(payload, key, record) for key in keys})


def _phase_list(payload: dict[str, Any], record: EvidenceRecord) -> tuple[str, ...]:
    value = payload.get("phases")
    if not isinstance(value, list) or not value or len(value) > len(_PHASES):
        _fail(record, "layout.status.phase_order", "Migration phases are missing or unbounded.")
    assert isinstance(value, list)
    if not all(isinstance(item, str) and item in _PHASES for item in value):
        _fail(record, "layout.status.phase_order", "Migration phases contain unknown values.")
    phases = tuple(value)
    indexes = tuple(_PHASES.index(item) for item in phases)
    if phases[0] != "prepared" or indexes != tuple(sorted(set(indexes))):
        _fail(record, "layout.status.phase_order", "Migration phases are not strictly ordered.")
    return phases


def _object(payload: dict[str, Any], keys: set[str], record: EvidenceRecord) -> dict[str, Any]:
    _exact(payload, keys, record)
    return payload


def _mapping(payload: dict[str, Any], key: str, record: EvidenceRecord) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        _fail(record, "layout.status.evidence_shape", f"{key} must be an object.")
    return value


def _exact(payload: dict[str, Any], keys: set[str], record: EvidenceRecord) -> None:
    if set(payload) != keys:
        _fail(
            record,
            "layout.status.evidence_shape",
            "Evidence payload has missing or unknown fields.",
        )


def _expect_type(record: EvidenceRecord, expected: str) -> None:
    if record.record_type != expected:
        _fail(record, "layout.status.record_type", f"Expected {expected} evidence.")


def _string(
    payload: dict[str, Any], key: str, record: EvidenceRecord, *, maximum: int = 256
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or len(value) > maximum:
        _fail(record, "layout.status.evidence_shape", f"{key} must be a bounded string.")
    return value


def _optional_string(
    payload: dict[str, Any], key: str, record: EvidenceRecord, *, maximum: int
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return _string(payload, key, record, maximum=maximum)


def _identity(payload: dict[str, Any], key: str, record: EvidenceRecord) -> str:
    value = _string(payload, key, record)
    if not _IDENTITY_RE.fullmatch(value) or value in {"latest", "current"}:
        _fail(record, "layout.status.identity", f"{key} is not an immutable identity.")
    return value


def _optional_identity(payload: dict[str, Any], key: str, record: EvidenceRecord) -> str | None:
    if payload.get(key) is None:
        return None
    return _identity(payload, key, record)


def _identity_list(
    payload: dict[str, Any], key: str, record: EvidenceRecord, *, maximum: int
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) > maximum:
        _fail(record, "layout.status.evidence_shape", f"{key} must be a bounded list.")
    assert isinstance(value, list)
    result = tuple(_identity({key: item}, key, record) for item in value)
    if result != tuple(sorted(set(result))):
        _fail(record, "layout.status.evidence_shape", f"{key} must be sorted and unique.")
    return result


def _scope(payload: dict[str, Any], key: str, record: EvidenceRecord) -> str:
    value = _string(payload, key, record)
    if not _SCOPE_RE.fullmatch(value):
        _fail(record, "layout.status.scope", f"{key} is not a closed selector scope.")
    return value


def _digest(payload: dict[str, Any], key: str, record: EvidenceRecord) -> str:
    value = _string(payload, key, record)
    if not _DIGEST_RE.fullmatch(value):
        _fail(record, "layout.status.digest", f"{key} is not a complete SHA-256 digest.")
    return value


def _optional_digest(payload: dict[str, Any], key: str, record: EvidenceRecord) -> str | None:
    if payload.get(key) is None:
        return None
    return _digest(payload, key, record)


def _digest_list(payload: dict[str, Any], key: str, record: EvidenceRecord) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) > 64:
        _fail(record, "layout.status.evidence_shape", f"{key} must be a bounded list.")
    assert isinstance(value, list)
    result = tuple(_digest({key: item}, key, record) for item in value)
    if result != tuple(sorted(set(result))):
        _fail(record, "layout.status.evidence_shape", f"{key} must be sorted and unique.")
    return result


def _commit(payload: dict[str, Any], key: str, record: EvidenceRecord) -> str:
    value = _string(payload, key, record)
    if not _COMMIT_RE.fullmatch(value):
        _fail(record, "layout.status.commit", f"{key} is not a full commit identity.")
    return value


def _integer(
    payload: dict[str, Any],
    key: str,
    record: EvidenceRecord,
    *,
    minimum: int = 0,
    maximum: int = 2**63 - 1,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        _fail(record, "layout.status.evidence_shape", f"{key} is outside its integer bound.")
    return value


def _optional_integer(
    payload: dict[str, Any],
    key: str,
    record: EvidenceRecord,
    *,
    minimum: int,
) -> int | None:
    if payload.get(key) is None:
        return None
    return _integer(payload, key, record, minimum=minimum)


def _boolean(payload: dict[str, Any], key: str, record: EvidenceRecord) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        _fail(record, "layout.status.evidence_shape", f"{key} must be boolean.")
    return value


def _utc_timestamp(payload: dict[str, Any], key: str, record: EvidenceRecord) -> str:
    value = _string(payload, key, record)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise invalid(
            "layout.status.timestamp",
            f"{key} must be an ISO-8601 UTC timestamp.",
            record=record.path,
        ) from exc
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
        _fail(record, "layout.status.timestamp", f"{key} must be an explicit UTC timestamp.")
    return value


def _content_digest(payload: dict[str, Any], digest_key: str) -> str:
    content = dict(payload)
    content.pop(digest_key)
    return "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()


def _fail(record: EvidenceRecord, code: str, message: str) -> NoReturn:
    raise invalid(code, message, record=record.path)


__all__ = [
    "LayoutSelectorSnapshotV1",
    "MigrationEvidenceRef",
    "OperationStatusSnapshotV1",
    "PreparedEvidenceRef",
    "ProcessSnapshot",
    "RollbackSnapshot",
    "SelectorObservation",
    "ShutdownSnapshot",
    "parse_migration",
    "parse_operation",
    "parse_prepared",
    "parse_selector",
]
