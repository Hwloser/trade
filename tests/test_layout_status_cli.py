from __future__ import annotations

import ast
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from trade_py.cli import dev
from trade_py.devtools.layout_status.errors import LayoutStatusInvalid
from trade_py.devtools.layout_status.records import (
    ExplicitRecordReader,
    ReaderLimits,
    canonical_json,
    canonical_record_digest,
)
from trade_py.devtools.layout_status.validation import validate_graph

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
COMMIT = "d" * 40
REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(record_type: str, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    record = {
        "schema_version": "trade.layout.record.v1",
        "record_type": record_type,
        "record_id": record_id,
        "references": [],
        "payload": payload,
    }
    record["record_digest"] = canonical_record_digest(record)
    return record


def _write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(record) + b"\n")


def _selector_payload() -> dict[str, Any]:
    payload = {
        "scope": "python_deployment",
        "generation": "generation-2",
        "revision": 2,
        "fence": 2,
        "operation_id": "operation-1",
        "plan_digest": DIGEST_A,
        "prepared_evidence_ref": "",
        "predecessor_generation": "generation-1",
        "predecessor_revision": 1,
    }
    payload["selector_payload_digest"] = (
        "sha256:" + __import__("hashlib").sha256(canonical_json(payload)).hexdigest()
    )
    return payload


def _prepared_payload() -> dict[str, Any]:
    return {
        "operation_id": "operation-1",
        "attempt_id": "attempt-1",
        "scope": "python_deployment",
        "source_commit": COMMIT,
        "source_tree_digest": DIGEST_B,
        "policy_digest": DIGEST_C,
        "approved_design_digest": DIGEST_A,
        "activation_plan_digest": DIGEST_A,
        "current_composition_digest": DIGEST_B,
        "immutable_input_refs": [DIGEST_A, DIGEST_B],
        "intended_target_generation": "generation-2",
        "expected_generation": "generation-1",
        "expected_revision": 1,
        "expected_fence": 1,
        "deployment_unit": "trade-api",
        "invocation_token": "invocation-1",
        "command_digests": [DIGEST_A],
        "prepared_at": "2026-07-27T12:00:00Z",
    }


def _states(
    *,
    migration: str = "retireable",
    execution: str = "passed",
    failure: str = "none",
    rollback: str = "not_required",
    startup: str = "started_healthy",
    reconciliation: str = "not_required",
) -> dict[str, str]:
    return {
        "migration_state": migration,
        "execution_state": execution,
        "failure_class": failure,
        "rollback_state": rollback,
        "startup_state": startup,
        "reconciliation_state": reconciliation,
    }


def _operation_payload(
    *,
    states: dict[str, str] | None = None,
    action: str = "none",
    phase: str = "verified",
) -> dict[str, Any]:
    return {
        "operation_id": "operation-1",
        "attempt_id": "attempt-1",
        "scope": "python_deployment",
        "activation_phase": phase,
        "states": states or _states(),
        "operator_action": action,
        "tool_exit_code": 0,
        "partial_evidence_ref": None,
        "stopped_early": False,
        "stop_reason": None,
        "failure_detail": None,
        "degraded_components": [],
        "process": {
            "deployment_unit": "trade-api",
            "invocation_token": "invocation-1",
            "generation": "generation-2",
            "revision": 2,
            "fence": 2,
            "process_started_receipt": True,
            "matching_live_instances": 1,
            "historical_process_started": False,
            "terminal_receipt": False,
            "terminal_identity_match": False,
            "zero_live_descendants": False,
            "teardown_receipt": False,
            "teardown_identity_match": False,
        },
        "shutdown": {
            "stage": "not_started",
            "signal_escalation": "none",
            "residual_process_count": 0,
            "residual_thread_count": 0,
            "forced_exit_receipt": False,
            "complete": False,
        },
        "rollback": {
            "plan_scope": "python_deployment",
            "target_generation": None,
            "later_accepted_slices": False,
            "target_is_historical_predecessor": False,
            "compensation_preserves_later_slices": False,
        },
    }


def _migration_payload(
    operation: dict[str, Any],
    prepared_digest: str,
) -> dict[str, Any]:
    phase = operation["activation_phase"]
    if phase == "verified":
        phases = [
            "prepared",
            "selector_committed",
            "start_intent_recorded",
            "process_started",
            "verified",
        ]
        terminal_outcome = "verified"
    elif phase == "failed":
        phases = [
            "prepared",
            "selector_committed",
            "start_intent_recorded",
            "process_started",
            "failed",
        ]
        terminal_outcome = "failed"
    elif phase == "rollback_verified":
        phases = [
            "prepared",
            "selector_committed",
            "start_intent_recorded",
            "process_started",
            "failed",
            "rollback_prepared",
            "rollback_selector_committed",
            "rollback_verified",
        ]
        terminal_outcome = "rolled_back"
    else:
        raise ValueError(f"Cannot build terminal evidence for phase: {phase}")
    return {
        "operation_id": "operation-1",
        "attempt_id": "attempt-1",
        "scope": "python_deployment",
        "source_commit": COMMIT,
        "source_tree_digest": DIGEST_B,
        "policy_digest": DIGEST_C,
        "approved_design_digest": DIGEST_A,
        "consumer_inventory_ref": DIGEST_A,
        "module_authority_ref": DIGEST_B,
        "artifact_refs": [DIGEST_C],
        "activation_plan_digest": DIGEST_A,
        "prepared_evidence_ref": prepared_digest,
        "selector_before": {
            "generation": "generation-1",
            "revision": 1,
            "fence": 1,
        },
        "selector_after": {
            "generation": "generation-2",
            "revision": 2,
            "fence": 2,
        },
        "phases": phases,
        "terminal_outcome": terminal_outcome,
        "command_digests": [DIGEST_A],
        "toolchain": ["python-3.10"],
        "deadline_milliseconds": 5000,
        "typed_outcomes": operation["states"],
        "partial_evidence_refs": [],
        "report_entries_digest": DIGEST_C,
    }


def _manifest_payload(
    selector_digest: str,
    operation_digest: str,
    prepared_digest: str,
    migration_digest: str | None,
) -> dict[str, Any]:
    return {
        "scope": "python_deployment",
        "selected_generation": "generation-2",
        "prior_generation": "generation-1",
        "selected_revision": 2,
        "selected_fence": 2,
        "wheel_digest": DIGEST_A,
        "wheel_member_count": 42,
        "legacy_module_origin": "trade_py",
        "target_module_origin": "trade",
        "selected_authority": "trade",
        "source_commit": COMMIT,
        "source_tree_digest": DIGEST_B,
        "inventory_scanner_digest": DIGEST_A,
        "inventory_rules_digest": DIGEST_B,
        "inventory_completeness": "complete",
        "inventory_age_seconds": 60,
        "missing_consumers": 0,
        "duplicate_consumers": 0,
        "reverse_dependencies": 0,
        "unclassified_consumers": 0,
        "root_console_parity": "passed",
        "asgi_import_state": "passed",
        "reload_child_import_state": "passed",
        "route_parity_state": "passed",
        "openapi_parity_state": "passed",
        "sse_parity_state": "passed",
        "capability_parity_state": "passed",
        "web_build_digest": DIGEST_C,
        "web_missing_asset_count": 0,
        "native_capability_state": "passed",
        "native_build_state": "passed",
        "native_differential_state": "passed",
        "notebook_state": "passed",
        "bridge_owner": "interfaces",
        "bridge_population_digest": DIGEST_C,
        "bridge_coverage_state": "complete",
        "bridge_age_seconds": 60,
        "bridge_last_observed_use": "2026-07-27T11:59:00Z",
        "bridge_deadline": "2026-10-27T12:00:00Z",
        "selector_ref": selector_digest,
        "operation_ref": operation_digest,
        "prepared_evidence_ref": prepared_digest,
        "migration_evidence_ref": migration_digest,
    }


def _fixture(
    tmp_path: Path,
    *,
    operation_payload: dict[str, Any] | None = None,
    manifest_changes: dict[str, Any] | None = None,
    include_migration: bool = True,
) -> Path:
    root = tmp_path / "layout-control"
    operation_payload = deepcopy(operation_payload or _operation_payload())
    rollback_succeeded = operation_payload["states"]["rollback_state"] == "succeeded"
    prepared_payload = _prepared_payload()
    selector_payload = _selector_payload()
    if rollback_succeeded:
        target = operation_payload["rollback"]["target_generation"]
        assert isinstance(target, str)
        prepared_payload.update(
            {
                "intended_target_generation": target,
                "expected_generation": "generation-2",
                "expected_revision": 2,
                "expected_fence": 2,
            }
        )
        selector_payload.update(
            {
                "generation": target,
                "revision": 3,
                "fence": 3,
                "predecessor_generation": "generation-2",
                "predecessor_revision": 2,
            }
        )
        operation_payload["process"].update(
            {
                "generation": target,
                "revision": 3,
                "fence": 3,
            }
        )
    prepared = _record("prepared_evidence", "prepared-1", prepared_payload)
    selector_payload["prepared_evidence_ref"] = prepared["record_digest"]
    selector_payload["selector_payload_digest"] = (
        "sha256:"
        + __import__("hashlib")
        .sha256(
            canonical_json(
                {
                    key: value
                    for key, value in selector_payload.items()
                    if key != "selector_payload_digest"
                }
            )
        )
        .hexdigest()
    )
    selector = _record("layout_selector_snapshot", "selector-1", selector_payload)
    operation = _record("operation_status_snapshot", "operation-1", operation_payload)
    migration = (
        _record(
            "migration_evidence",
            "migration-1",
            _migration_payload(operation_payload, prepared["record_digest"]),
        )
        if include_migration
        else None
    )
    if migration is not None and rollback_succeeded:
        migration["payload"]["selector_before"] = {
            "generation": "generation-2",
            "revision": 2,
            "fence": 2,
        }
        migration["payload"]["selector_after"] = {
            "generation": selector_payload["generation"],
            "revision": selector_payload["revision"],
            "fence": selector_payload["fence"],
        }
        migration["record_digest"] = canonical_record_digest(migration)
    for name, record in (
        ("records/prepared.json", prepared),
        ("records/selector.json", selector),
        ("records/operation.json", operation),
    ):
        _write(root / name, record)
    if migration is not None:
        _write(root / "records/migration.json", migration)

    manifest_payload = _manifest_payload(
        selector["record_digest"],
        operation["record_digest"],
        prepared["record_digest"],
        migration["record_digest"] if migration is not None else None,
    )
    if rollback_succeeded:
        manifest_payload.update(
            {
                "selected_generation": selector_payload["generation"],
                "prior_generation": selector_payload["predecessor_generation"],
                "selected_revision": selector_payload["revision"],
                "selected_fence": selector_payload["fence"],
            }
        )
    if manifest_changes:
        manifest_payload.update(manifest_changes)
    manifest = _record("layout_status_manifest", "status-1", manifest_payload)
    manifest["references"] = [
        {
            "name": "operation",
            "path": "records/operation.json",
            "digest": operation["record_digest"],
        },
        {
            "name": "prepared",
            "path": "records/prepared.json",
            "digest": prepared["record_digest"],
        },
        {
            "name": "selector",
            "path": "records/selector.json",
            "digest": selector["record_digest"],
        },
    ]
    if migration is not None:
        manifest["references"].append(
            {
                "name": "migration",
                "path": "records/migration.json",
                "digest": migration["record_digest"],
            }
        )
    manifest["record_digest"] = canonical_record_digest(manifest)
    path = root / "status.json"
    _write(path, manifest)
    return path


def test_parser_contract_and_unset_root_fail_before_business_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = dev.make_parser().parse_args(["layout-status", "--json"])
    assert args.cmd == "layout-status"
    assert args.as_json is True

    monkeypatch.delenv("TRADE_LAYOUT_STATUS_MANIFEST", raising=False)
    code = dev.main(["layout-status", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert payload["error"]["code"] == "layout.status.manifest_unset"
    assert payload["summary"] is None


def test_json_and_human_views_share_enum_action_and_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _fixture(tmp_path)
    monkeypatch.setenv("TRADE_LAYOUT_STATUS_MANIFEST", str(manifest))

    json_code = dev.main(["layout-status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    human_code = dev.main(["layout-status"])
    human = capsys.readouterr().out

    assert json_code == human_code == 0
    assert payload["status"] == "HEALTHY"
    assert payload["operation"]["states"]["reconciliation_state"] == "not_required"
    assert payload["operation"]["operator_action"] == "none"
    assert payload["validation"]["record_count"] == 5
    assert "Layout status: HEALTHY (exit 0)" in human
    assert "reconciliation_state=not_required" in human
    assert "action=none" in human
    assert "revision=2 fence=2" in human
    assert "residual_processes=0 residual_threads=0" in human


@pytest.mark.parametrize(
    ("states", "action", "phase", "process_changes", "shutdown_changes"),
    [
        (
            _states(
                migration="prepared",
                execution="running",
                startup="starting",
                reconciliation="pending",
            ),
            "resume_reconciliation",
            "start_intent_recorded",
            {"process_started_receipt": False, "matching_live_instances": 0},
            {},
        ),
        (
            _states(
                migration="prepared",
                execution="stopped",
                startup="stopped",
                reconciliation="absence_proved",
            ),
            "retry_identical_invocation",
            "process_started",
            {
                "matching_live_instances": 0,
                "historical_process_started": True,
                "terminal_receipt": True,
                "terminal_identity_match": True,
                "zero_live_descendants": True,
            },
            {},
        ),
        (
            _states(
                migration="prepared",
                execution="stopped",
                rollback="ready",
                startup="stopped",
                reconciliation="fenced_teardown",
            ),
            "execute_reviewed_rollback",
            "failed",
            {
                "matching_live_instances": 0,
                "zero_live_descendants": True,
                "teardown_receipt": True,
                "teardown_identity_match": True,
            },
            {"stage": "complete", "complete": True},
        ),
    ],
)
def test_reconciliation_attention_goldens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    states: dict[str, str],
    action: str,
    phase: str,
    process_changes: dict[str, Any],
    shutdown_changes: dict[str, Any],
) -> None:
    operation = _operation_payload(states=states, action=action, phase=phase)
    operation["process"].update(process_changes)
    operation["shutdown"].update(shutdown_changes)
    if states["rollback_state"] != "not_required":
        operation["rollback"]["target_generation"] = "generation-compensation-3"
    manifest = _fixture(
        tmp_path,
        operation_payload=operation,
        include_migration=phase in {"failed", "rollback_verified"},
    )
    monkeypatch.setenv("TRADE_LAYOUT_STATUS_MANIFEST", str(manifest))

    code = dev.main(["layout-status", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["status"] == "ATTENTION"
    assert payload["operation"]["operator_action"] == action
    assert payload["operation"]["states"]["reconciliation_state"] == states["reconciliation_state"]


@pytest.mark.parametrize(
    ("mutate", "violation"),
    [
        (
            lambda operation: (
                operation.update(
                    {
                        "states": _states(
                            migration="target_authoritative",
                            reconciliation="adopted",
                        ),
                        "operator_action": "continue_validation",
                    }
                ),
                operation["process"].update(
                    {"matching_live_instances": 0, "process_started_receipt": False}
                ),
            ),
            "layout.status.receipt_invalid",
        ),
        (
            lambda operation: operation["process"].update({"revision": 1}),
            "layout.status.identity_mismatch",
        ),
        (
            lambda operation: operation.update({"operator_action": "wait"}),
            "layout.status.action_mismatch",
        ),
    ],
)
def test_invalid_receipt_identity_and_action_return_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutate: Any,
    violation: str,
) -> None:
    operation = _operation_payload()
    mutate(operation)
    manifest = _fixture(tmp_path, operation_payload=operation)
    monkeypatch.setenv("TRADE_LAYOUT_STATUS_MANIFEST", str(manifest))

    code = dev.main(["layout-status", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert violation in payload["validation"]["violations"]


def test_historical_start_requires_exact_terminal_absence_receipt(tmp_path: Path) -> None:
    operation = _operation_payload(
        states=_states(
            migration="prepared",
            execution="stopped",
            startup="stopped",
            reconciliation="absence_proved",
        ),
        action="retry_identical_invocation",
        phase="process_started",
    )
    operation["process"].update(
        {
            "matching_live_instances": 0,
            "historical_process_started": True,
            "terminal_receipt": True,
            "terminal_identity_match": False,
            "zero_live_descendants": True,
        }
    )
    status = validate_graph(
        ExplicitRecordReader(
            _fixture(tmp_path, operation_payload=operation, include_migration=False)
        ).read()
    )

    assert status.constraints.exit_code == 2
    assert "layout.status.receipt_invalid" in status.constraints.violations


def test_non_lifo_historical_rollback_target_is_invalid(tmp_path: Path) -> None:
    operation = _operation_payload(
        states=_states(
            migration="prepared",
            execution="failed",
            failure="contract_mismatch",
            rollback="ready",
            startup="failed",
        ),
        action="execute_reviewed_rollback",
        phase="failed",
    )
    operation["rollback"].update(
        {
            "target_generation": "generation-1",
            "later_accepted_slices": True,
            "target_is_historical_predecessor": True,
            "compensation_preserves_later_slices": False,
        }
    )
    status = validate_graph(
        ExplicitRecordReader(
            _fixture(tmp_path, operation_payload=operation, include_migration=False)
        ).read()
    )

    assert status.constraints.exit_code == 2
    assert "layout.status.rollback_target_invalid" in status.constraints.violations


def test_timeout_retains_failure_after_successful_forward_rollback(tmp_path: Path) -> None:
    operation = _operation_payload(
        states=_states(
            migration="prepared",
            execution="stopped",
            failure="timeout",
            rollback="succeeded",
            startup="stopped",
        ),
        action="investigate",
        phase="rollback_verified",
    )
    operation["tool_exit_code"] = 124
    operation["stopped_early"] = True
    operation["stop_reason"] = "deadline_exceeded"
    operation["failure_detail"] = "subprocess_timeout"
    operation["rollback"].update(
        {
            "target_generation": "generation-compensation-3",
            "later_accepted_slices": True,
            "target_is_historical_predecessor": False,
            "compensation_preserves_later_slices": True,
        }
    )
    operation["shutdown"].update(
        {
            "stage": "complete",
            "signal_escalation": "term_kill",
            "forced_exit_receipt": True,
            "complete": True,
        }
    )
    operation["process"].update(
        {
            "matching_live_instances": 0,
            "terminal_receipt": True,
            "terminal_identity_match": True,
            "zero_live_descendants": True,
        }
    )
    status = validate_graph(
        ExplicitRecordReader(_fixture(tmp_path, operation_payload=operation)).read()
    )

    assert status.constraints.exit_code == 1
    assert status.constraints.derived_action == "investigate"
    assert status.operation.axes.failure_class == "timeout"
    assert status.operation.axes.rollback_state == "succeeded"
    assert status.selector.generation == "generation-compensation-3"
    assert status.selector.revision == 3
    assert status.selector.fence == 3
    assert status.operation.rollback.target_generation == status.selector.generation
    assert status.operation.shutdown.signal_escalation == "term_kill"
    assert status.operation.shutdown.forced_exit_receipt is True


def test_cleanup_residue_is_attention_not_success(tmp_path: Path) -> None:
    operation = _operation_payload(
        states=_states(
            migration="prepared",
            execution="stopped",
            failure="process_cleanup_incomplete",
            rollback="ready",
            startup="stopped",
            reconciliation="required",
        ),
        action="investigate",
        phase="failed",
    )
    operation["rollback"]["target_generation"] = "generation-compensation-3"
    operation["shutdown"].update(
        {
            "stage": "kill",
            "signal_escalation": "term_kill",
            "residual_process_count": 1,
            "residual_thread_count": 2,
            "forced_exit_receipt": True,
            "complete": False,
        }
    )
    status = validate_graph(
        ExplicitRecordReader(_fixture(tmp_path, operation_payload=operation)).read()
    )

    assert status.constraints.exit_code == 1
    assert status.constraints.derived_action == "investigate"
    assert status.operation.shutdown.residual_process_count == 1
    assert status.operation.shutdown.residual_thread_count == 2


def test_tool_failure_retains_partial_evidence_and_early_stop(tmp_path: Path) -> None:
    operation = _operation_payload(
        states=_states(
            migration="prepared",
            execution="failed",
            failure="tool_failure",
            startup="failed",
        ),
        action="investigate",
        phase="failed",
    )
    operation["tool_exit_code"] = 70
    operation["partial_evidence_ref"] = DIGEST_C
    operation["stopped_early"] = True
    operation["stop_reason"] = "tool_failed"
    operation["failure_detail"] = "validator_exit_70"
    status = validate_graph(
        ExplicitRecordReader(_fixture(tmp_path, operation_payload=operation)).read()
    )

    assert status.constraints.exit_code == 1
    assert status.operation.partial_evidence_ref == DIGEST_C
    assert status.operation.stopped_early is True
    assert status.operation.stop_reason == "tool_failed"


def test_unknown_enum_phase_order_and_operation_conflict_return_two(tmp_path: Path) -> None:
    unknown = _operation_payload()
    unknown["states"]["startup_state"] = "future_state"
    unknown_status = validate_graph(
        ExplicitRecordReader(_fixture(tmp_path / "unknown", operation_payload=unknown)).read()
    )
    assert unknown_status.constraints.exit_code == 2
    assert "layout.status.unknown_state" in unknown_status.constraints.violations

    operation = _operation_payload()
    manifest = _fixture(tmp_path / "phase", operation_payload=operation)
    migration_path = manifest.parent / "records" / "migration.json"
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    migration["payload"]["phases"] = [
        "prepared",
        "process_started",
        "selector_committed",
        "verified",
    ]
    migration["record_digest"] = canonical_record_digest(migration)
    _write(migration_path, migration)
    manifest_record = json.loads(manifest.read_text(encoding="utf-8"))
    migration_ref = next(
        item for item in manifest_record["references"] if item["name"] == "migration"
    )
    migration_ref["digest"] = migration["record_digest"]
    manifest_record["payload"]["migration_evidence_ref"] = migration["record_digest"]
    manifest_record["record_digest"] = canonical_record_digest(manifest_record)
    _write(manifest, manifest_record)
    with pytest.raises(LayoutStatusInvalid) as phase:
        validate_graph(ExplicitRecordReader(manifest).read())
    assert phase.value.error.code == "layout.status.phase_order"

    conflict = _operation_payload()
    conflict["operation_id"] = "operation-conflict"
    conflict_status = validate_graph(
        ExplicitRecordReader(_fixture(tmp_path / "conflict", operation_payload=conflict)).read()
    )
    assert conflict_status.constraints.exit_code == 2
    assert "layout.status.identity_mismatch" in conflict_status.constraints.violations


def test_partial_bridge_coverage_is_attention_with_continue_validation(
    tmp_path: Path,
) -> None:
    operation = _operation_payload(action="continue_validation")
    status = validate_graph(
        ExplicitRecordReader(
            _fixture(
                tmp_path,
                operation_payload=operation,
                manifest_changes={"bridge_coverage_state": "partial"},
            )
        ).read()
    )

    assert status.constraints.exit_code == 1
    assert status.constraints.derived_action == "continue_validation"
    assert dict(status.constraints.axis_classifications)["bridge_coverage"] == "valid_attention"


def test_inventory_over_age_or_incomplete_is_invalid(tmp_path: Path) -> None:
    for index, changes in enumerate(
        ({"inventory_age_seconds": 86_401}, {"inventory_completeness": "tool_failed"})
    ):
        status = validate_graph(
            ExplicitRecordReader(_fixture(tmp_path / str(index), manifest_changes=changes)).read()
        )
        assert status.constraints.exit_code == 2
        assert "layout.status.additional_invalid" in status.constraints.violations


def test_reader_rejects_symlink_corrupt_digest_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    manifest = _fixture(tmp_path / "symlink")
    operation = manifest.parent / "records" / "operation.json"
    real = operation.with_name("real-operation.json")
    operation.rename(real)
    operation.symlink_to(real.name)
    with pytest.raises(LayoutStatusInvalid) as symlink:
        ExplicitRecordReader(manifest).read()
    assert symlink.value.error.code == "layout.status.record_open"

    manifest = _fixture(tmp_path / "digest")
    operation = manifest.parent / "records" / "operation.json"
    payload = bytearray(operation.read_bytes())
    payload[-3] = ord("0") if payload[-3] != ord("0") else ord("1")
    operation.write_bytes(bytes(payload))
    with pytest.raises(LayoutStatusInvalid) as digest:
        ExplicitRecordReader(manifest).read()
    assert digest.value.error.code in {
        "layout.status.record_digest",
        "layout.status.record_json",
        "layout.status.reference_digest",
    }

    manifest = _fixture(tmp_path / "pretty")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    with pytest.raises(LayoutStatusInvalid) as noncanonical:
        ExplicitRecordReader(manifest).read()
    assert noncanonical.value.error.code == "layout.status.record_canonical"


def test_reader_enforces_count_size_depth_and_deadline(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path / "count")
    with pytest.raises(LayoutStatusInvalid) as count:
        ExplicitRecordReader(manifest, limits=ReaderLimits(max_records=4)).read()
    assert count.value.error.code == "layout.status.record_count"

    manifest = _fixture(tmp_path / "size")
    with pytest.raises(LayoutStatusInvalid) as size:
        ExplicitRecordReader(manifest, limits=ReaderLimits(max_record_bytes=128)).read()
    assert size.value.error.code == "layout.status.record_size"

    manifest = _fixture(tmp_path / "aggregate")
    with pytest.raises(LayoutStatusInvalid) as aggregate:
        ExplicitRecordReader(
            manifest,
            limits=ReaderLimits(max_aggregate_bytes=1024),
        ).read()
    assert aggregate.value.error.code == "layout.status.aggregate_size"

    manifest = _fixture(tmp_path / "depth")
    with pytest.raises(LayoutStatusInvalid) as depth:
        ExplicitRecordReader(manifest, limits=ReaderLimits(max_depth=0)).read()
    assert depth.value.error.code == "layout.status.reference_depth"

    ticks = iter((0.0, 0.0, 6.0))
    with pytest.raises(LayoutStatusInvalid) as deadline:
        ExplicitRecordReader(manifest, monotonic=lambda: next(ticks)).read()
    assert deadline.value.error.code == "layout.status.deadline"


def test_absolute_manifest_and_relative_references_cannot_escape(tmp_path: Path) -> None:
    with pytest.raises(LayoutStatusInvalid) as relative:
        ExplicitRecordReader(Path("status.json"))
    assert relative.value.error.code == "layout.status.manifest_not_absolute"

    manifest = _fixture(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["references"][0]["path"] = "../operation.json"
    record["record_digest"] = canonical_record_digest(record)
    _write(manifest, record)
    with pytest.raises(LayoutStatusInvalid) as escape:
        ExplicitRecordReader(manifest).read()
    assert escape.value.error.code == "layout.status.reference_path"


def test_cli_does_not_invoke_git_process_or_business_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _fixture(tmp_path)
    monkeypatch.setenv("TRADE_LAYOUT_STATUS_MANIFEST", str(manifest))
    calls: list[tuple[Any, ...]] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append((*args, kwargs))
        raise AssertionError("layout-status must not launch a process")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    assert dev.main(["layout-status", "--json"]) == 0
    assert calls == []


def test_shell_route_is_frozen_no_sync_and_help_lists_status(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    routed = subprocess.run(
        [str(REPO_ROOT / "trade"), "dev", "layout-status", "--json"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    help_result = subprocess.run(
        [str(REPO_ROOT / "trade"), "help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    lines = routed.stdout.splitlines()
    assert lines[:4] == ["run", "--frozen", "--no-sync", "python"]
    assert lines[-3:] == ["dev", "layout-status", "--json"]
    assert "./trade dev layout-status" in help_result.stdout


def test_layout_status_source_has_no_runtime_or_mutation_dependencies() -> None:
    forbidden_imports = (
        "trade_py.data",
        "trade_py.db",
        "trade_py.event",
        "trade_py.infra",
        "trade_py.jobs",
        "trade_web",
    )
    forbidden_calls = {
        "glob",
        "rglob",
        "walk",
        "Popen",
        "run",
        "system",
        "fork",
        "kill",
    }
    for path in (REPO_ROOT / "trade_py" / "devtools" / "layout_status").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported: list[str] = []
        calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
        assert not any(
            module == prefix or module.startswith(f"{prefix}.")
            for module in imported
            for prefix in forbidden_imports
        ), path
        assert not forbidden_calls.intersection(calls), path
        assert "/proc" not in source
        assert "systemctl" not in source
        assert "service-manager" not in source
        assert "git " not in source.lower()
