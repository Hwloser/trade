"""Immutable read models for package-layout evidence."""

from __future__ import annotations

from dataclasses import dataclass

from trade_py.devtools.layout_status.constraints import LayoutStatusAxes


@dataclass(frozen=True)
class PreparedEvidenceRef:
    record_digest: str
    operation_id: str
    attempt_id: str
    scope: str
    source_commit: str
    source_tree_digest: str
    policy_digest: str
    approved_design_digest: str
    activation_plan_digest: str
    current_composition_digest: str
    immutable_input_refs: tuple[str, ...]
    intended_target_generation: str
    expected_generation: str
    expected_revision: int
    expected_fence: int
    deployment_unit: str
    invocation_token: str
    command_digests: tuple[str, ...]
    prepared_at: str


@dataclass(frozen=True)
class LayoutSelectorSnapshotV1:
    record_digest: str
    scope: str
    generation: str
    revision: int
    fence: int
    operation_id: str
    plan_digest: str
    prepared_evidence_ref: str
    predecessor_generation: str
    predecessor_revision: int
    selector_payload_digest: str


@dataclass(frozen=True)
class ProcessSnapshot:
    deployment_unit: str
    invocation_token: str
    generation: str
    revision: int
    fence: int
    process_started_receipt: bool
    matching_live_instances: int
    historical_process_started: bool
    terminal_receipt: bool
    terminal_identity_match: bool
    zero_live_descendants: bool
    teardown_receipt: bool
    teardown_identity_match: bool


@dataclass(frozen=True)
class ShutdownSnapshot:
    stage: str
    signal_escalation: str
    residual_process_count: int
    residual_thread_count: int
    forced_exit_receipt: bool
    complete: bool


@dataclass(frozen=True)
class RollbackSnapshot:
    plan_scope: str
    target_generation: str | None
    later_accepted_slices: bool
    target_is_historical_predecessor: bool
    compensation_preserves_later_slices: bool


@dataclass(frozen=True)
class OperationStatusSnapshotV1:
    record_digest: str
    operation_id: str
    attempt_id: str
    scope: str
    activation_phase: str
    axes: LayoutStatusAxes
    operator_action: str
    tool_exit_code: int | None
    partial_evidence_ref: str | None
    stopped_early: bool
    stop_reason: str | None
    failure_detail: str | None
    degraded_components: tuple[str, ...]
    process: ProcessSnapshot
    shutdown: ShutdownSnapshot
    rollback: RollbackSnapshot


@dataclass(frozen=True)
class SelectorObservation:
    generation: str
    revision: int
    fence: int


@dataclass(frozen=True)
class MigrationEvidenceRef:
    record_digest: str
    operation_id: str
    attempt_id: str
    scope: str
    source_commit: str
    source_tree_digest: str
    policy_digest: str
    approved_design_digest: str
    consumer_inventory_ref: str
    module_authority_ref: str
    artifact_refs: tuple[str, ...]
    activation_plan_digest: str
    prepared_evidence_ref: str
    selector_before: SelectorObservation
    selector_after: SelectorObservation
    phases: tuple[str, ...]
    terminal_outcome: str
    command_digests: tuple[str, ...]
    toolchain: tuple[str, ...]
    deadline_milliseconds: int
    typed_outcomes: LayoutStatusAxes
    partial_evidence_refs: tuple[str, ...]
    report_entries_digest: str


__all__ = [
    "LayoutSelectorSnapshotV1",
    "MigrationEvidenceRef",
    "OperationStatusSnapshotV1",
    "PreparedEvidenceRef",
    "ProcessSnapshot",
    "RollbackSnapshot",
    "SelectorObservation",
    "ShutdownSnapshot",
]
