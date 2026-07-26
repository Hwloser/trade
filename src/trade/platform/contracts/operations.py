from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from trade.kernel.ids import IdNamespace, OpaqueId
from trade.kernel.time import UtcInstant
from trade.platform.contracts.actor import ActorContext
from trade.platform.contracts.messages import (
    FingerprintDomain,
    FingerprintV1,
)

__all__ = [
    "ADMISSION_OUTCOME_COUNTER_NAME",
    "AdmissionOutcome",
    "AdmissionOutcomeLabels",
    "AdmissionRefusalEventV1",
    "AdmissionRefusalAuditV1",
    "DeadlineTerminalEvidence",
    "OperationReceipt",
    "OperationState",
    "validate_operation_receipt_transition",
    "validate_operation_transition",
]

_LOWER_TOKEN_PATTERN = re.compile(r"[a-z0-9._:-]{1,96}", re.ASCII)
_REASON_PATTERN = re.compile(r"[A-Z0-9._-]{1,96}", re.ASCII)
_MAX_INT32 = 2_147_483_647
_MAX_INT64 = 9_223_372_036_854_775_807
ADMISSION_OUTCOME_COUNTER_NAME = "platform_idempotency_admission_outcomes_total"
_TERMINAL_STATES = frozenset(
    {
        "completed",
        "compensated",
        "failed",
        "cancelled",
        "deadline_exceeded",
    }
)


class OperationState(str, Enum):
    REQUESTED = "requested"
    ACCEPTED = "accepted"
    RUNNING = "running"
    WAITING = "waiting"
    RETRY_SCHEDULED = "retry_scheduled"
    COMPENSATION_PENDING = "compensation_pending"
    COMPLETED = "completed"
    COMPENSATED = "compensated"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"

    @property
    def is_terminal(self) -> bool:
        return self.value in _TERMINAL_STATES


_OPERATION_TRANSITIONS = {
    OperationState.REQUESTED: frozenset(
        {
            OperationState.ACCEPTED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.DEADLINE_EXCEEDED,
        }
    ),
    OperationState.ACCEPTED: frozenset(
        {
            OperationState.RUNNING,
            OperationState.WAITING,
            OperationState.RETRY_SCHEDULED,
            OperationState.BLOCKED,
            OperationState.COMPLETED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.DEADLINE_EXCEEDED,
        }
    ),
    OperationState.RUNNING: frozenset(
        {
            OperationState.WAITING,
            OperationState.RETRY_SCHEDULED,
            OperationState.COMPENSATION_PENDING,
            OperationState.BLOCKED,
            OperationState.COMPLETED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.DEADLINE_EXCEEDED,
        }
    ),
    OperationState.WAITING: frozenset(
        {
            OperationState.RUNNING,
            OperationState.RETRY_SCHEDULED,
            OperationState.BLOCKED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.DEADLINE_EXCEEDED,
        }
    ),
    OperationState.RETRY_SCHEDULED: frozenset(
        {
            OperationState.RUNNING,
            OperationState.WAITING,
            OperationState.BLOCKED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.DEADLINE_EXCEEDED,
        }
    ),
    OperationState.COMPENSATION_PENDING: frozenset(
        {
            OperationState.COMPENSATED,
            OperationState.BLOCKED,
            OperationState.FAILED,
            OperationState.DEADLINE_EXCEEDED,
        }
    ),
    OperationState.BLOCKED: frozenset(
        {
            OperationState.RUNNING,
            OperationState.RETRY_SCHEDULED,
            OperationState.COMPENSATION_PENDING,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.DEADLINE_EXCEEDED,
        }
    ),
    OperationState.COMPLETED: frozenset(),
    OperationState.COMPENSATED: frozenset(),
    OperationState.FAILED: frozenset(),
    OperationState.CANCELLED: frozenset(),
    OperationState.DEADLINE_EXCEEDED: frozenset(),
}


class AdmissionOutcome(str, Enum):
    CREATED = "created"
    REPLAYED = "replayed"
    COMMAND_CONFLICT = "command_conflict"
    CLAIM_CORRUPT = "claim_corrupt"
    KEYSET_CONTENTION = "keyset_contention"
    AUDIT_UNAVAILABLE = "audit_unavailable"


_REFUSAL_OUTCOMES = {
    "IDEMPOTENCY_COMMAND_CONFLICT": AdmissionOutcome.COMMAND_CONFLICT,
    "IDEMPOTENCY_CLAIM_CORRUPT": AdmissionOutcome.CLAIM_CORRUPT,
    "IDEMPOTENCY_KEYSET_CONTENTION": AdmissionOutcome.KEYSET_CONTENTION,
    "IDEMPOTENCY_AUDIT_UNAVAILABLE": AdmissionOutcome.AUDIT_UNAVAILABLE,
}


@dataclass(frozen=True, slots=True)
class AdmissionOutcomeLabels:
    owner_namespace: IdNamespace
    outcome: AdmissionOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.owner_namespace, IdNamespace):
            raise TypeError("owner_namespace must be IdNamespace")
        if not isinstance(self.outcome, AdmissionOutcome):
            raise TypeError("outcome must be AdmissionOutcome")


@dataclass(frozen=True, slots=True)
class AdmissionRefusalEventV1:
    schema_name: str
    schema_version: int
    reason_code: str
    outcome: AdmissionOutcome
    request_message_id: OpaqueId
    correlation_id: OpaqueId
    causation_id: OpaqueId | None
    owner_namespace: IdNamespace
    matched_key_versions: tuple[int, ...]
    key_set_generation: int
    attempt_count: int

    def __post_init__(self) -> None:
        if self.schema_name != "trade.idempotency_admission_refusal":
            raise ValueError(
                "AdmissionRefusalEventV1 schema_name must be "
                "'trade.idempotency_admission_refusal'"
            )
        if self.schema_version != 1:
            raise ValueError("AdmissionRefusalEventV1 schema_version must be 1")
        expected_outcome = _REFUSAL_OUTCOMES.get(self.reason_code)
        if expected_outcome is None:
            raise ValueError("refusal event reason is not a closed terminal outcome")
        if self.outcome is not expected_outcome:
            raise ValueError("refusal event outcome does not match its reason")
        _validate_refusal_evidence(
            request_message_id=self.request_message_id,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            owner_namespace=self.owner_namespace,
            matched_key_versions=self.matched_key_versions,
            key_set_generation=self.key_set_generation,
            attempt_count=self.attempt_count,
        )
        _validate_matched_versions_for_reason(
            reason_code=self.reason_code,
            matched_key_versions=self.matched_key_versions,
            audit_required=False,
        )


class DeadlineTerminalEvidence(str, Enum):
    ALL_OWNED_WORKERS_EXITED = "all_owned_workers_exited"
    RESIDUAL_WRITES_DURABLY_FENCED = "residual_writes_durably_fenced"


@dataclass(frozen=True, slots=True)
class OperationReceipt:
    schema_name: str
    schema_version: int
    operation_id: OpaqueId
    operation_kind: str
    command_name: str
    command_fingerprint: FingerprintV1
    actor: ActorContext
    request_message_id: OpaqueId
    correlation_id: OpaqueId
    causation_id: OpaqueId | None
    idempotency_scope: str
    idempotency_fingerprint: FingerprintV1
    state: OperationState
    reason_code: str
    accepted_at: UtcInstant
    updated_at: UtcInstant
    terminal_at: UtcInstant | None
    process_id: OpaqueId | None

    def __post_init__(self) -> None:
        if self.schema_name != "trade.operation_receipt":
            raise ValueError(
                "OperationReceipt schema_name must be 'trade.operation_receipt'"
            )
        if self.schema_version != 1:
            raise ValueError("OperationReceipt schema_version must be 1")
        if not isinstance(self.operation_id, OpaqueId):
            raise TypeError("operation_id must be OpaqueId")
        _validate_lower_token(self.operation_kind, field_name="operation_kind")
        _validate_lower_token(self.command_name, field_name="command_name")
        if (
            not isinstance(self.command_fingerprint, FingerprintV1)
            or self.command_fingerprint.domain is not FingerprintDomain.COMMAND
        ):
            raise ValueError("command_fingerprint must use the command fingerprint domain")
        if not isinstance(self.actor, ActorContext):
            raise TypeError("actor must be ActorContext")
        if not self.actor.can_submit_mutation:
            raise ValueError("operation receipt actor must retain verified mutation authority")
        _validate_causal_tuple(
            request_message_id=self.request_message_id,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
        )
        _validate_lower_token(self.idempotency_scope, field_name="idempotency_scope")
        if (
            not isinstance(self.idempotency_fingerprint, FingerprintV1)
            or self.idempotency_fingerprint.domain is not FingerprintDomain.IDEMPOTENCY
        ):
            raise ValueError(
                "idempotency_fingerprint must use the idempotency fingerprint domain"
            )
        if not isinstance(self.state, OperationState):
            raise TypeError("state must be OperationState")
        _validate_reason(self.reason_code)
        if not isinstance(self.accepted_at, UtcInstant):
            raise TypeError("accepted_at must be UtcInstant")
        if not isinstance(self.updated_at, UtcInstant):
            raise TypeError("updated_at must be UtcInstant")
        if self.updated_at.value < self.accepted_at.value:
            raise ValueError("updated_at cannot precede accepted_at")
        if self.terminal_at is not None and not isinstance(self.terminal_at, UtcInstant):
            raise TypeError("terminal_at must be UtcInstant or None")
        if self.state.is_terminal:
            if self.terminal_at is None:
                raise ValueError("terminal operation state requires terminal_at")
            if not self.accepted_at.value <= self.terminal_at.value <= self.updated_at.value:
                raise ValueError(
                    "terminal_at must be between accepted_at and updated_at"
                )
        elif self.terminal_at is not None:
            raise ValueError("non-terminal operation state forbids terminal_at")
        if self.process_id is not None and not isinstance(self.process_id, OpaqueId):
            raise TypeError("process_id must be OpaqueId or None")


@dataclass(frozen=True, slots=True)
class AdmissionRefusalAuditV1:
    schema_name: str
    schema_version: int
    reason_code: str
    request_message_id: OpaqueId
    correlation_id: OpaqueId
    causation_id: OpaqueId | None
    owner_namespace: IdNamespace
    matched_key_versions: tuple[int, ...]
    key_set_generation: int
    attempt_count: int
    occurred_at: UtcInstant

    def __post_init__(self) -> None:
        if self.schema_name != "trade.idempotency_refusal_audit":
            raise ValueError(
                "AdmissionRefusalAuditV1 schema_name must be "
                "'trade.idempotency_refusal_audit'"
            )
        if self.schema_version != 1:
            raise ValueError("AdmissionRefusalAuditV1 schema_version must be 1")
        if self.reason_code not in {
            "IDEMPOTENCY_COMMAND_CONFLICT",
            "IDEMPOTENCY_CLAIM_CORRUPT",
            "IDEMPOTENCY_KEYSET_CONTENTION",
        }:
            raise ValueError("refusal audit reason is not a durable terminal outcome")
        _validate_refusal_evidence(
            request_message_id=self.request_message_id,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            owner_namespace=self.owner_namespace,
            matched_key_versions=self.matched_key_versions,
            key_set_generation=self.key_set_generation,
            attempt_count=self.attempt_count,
        )
        if not isinstance(self.occurred_at, UtcInstant):
            raise TypeError("occurred_at must be UtcInstant")
        _validate_matched_versions_for_reason(
            reason_code=self.reason_code,
            matched_key_versions=self.matched_key_versions,
            audit_required=True,
        )


def _validate_refusal_evidence(
    *,
    request_message_id: OpaqueId,
    correlation_id: OpaqueId,
    causation_id: OpaqueId | None,
    owner_namespace: IdNamespace,
    matched_key_versions: tuple[int, ...],
    key_set_generation: int,
    attempt_count: int,
) -> None:
    _validate_causal_tuple(
        request_message_id=request_message_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )
    if not isinstance(owner_namespace, IdNamespace):
        raise TypeError("owner_namespace must be IdNamespace")
    if not isinstance(matched_key_versions, tuple):
        raise TypeError("matched_key_versions must be a tuple")
    if len(matched_key_versions) > 4:
        raise ValueError("matched_key_versions must contain at most four entries")
    for version in matched_key_versions:
        _validate_int(
            version,
            field_name="matched key version",
            minimum=1,
            maximum=_MAX_INT32,
        )
    if matched_key_versions != tuple(sorted(set(matched_key_versions))):
        raise ValueError("matched_key_versions must be sorted and unique")
    _validate_int(
        key_set_generation,
        field_name="key_set_generation",
        minimum=1,
        maximum=_MAX_INT64,
    )
    _validate_int(
        attempt_count,
        field_name="attempt_count",
        minimum=1,
        maximum=3,
    )


def _validate_matched_versions_for_reason(
    *,
    reason_code: str,
    matched_key_versions: tuple[int, ...],
    audit_required: bool,
) -> None:
    if reason_code == "IDEMPOTENCY_COMMAND_CONFLICT":
        if len(matched_key_versions) != 1:
            raise ValueError("command conflict evidence requires one matched key version")
    elif reason_code == "IDEMPOTENCY_CLAIM_CORRUPT":
        if not matched_key_versions:
            raise ValueError("claim corruption evidence requires matched key versions")
    elif reason_code == "IDEMPOTENCY_KEYSET_CONTENTION" and matched_key_versions:
        raise ValueError("key-set contention evidence forbids matched key versions")
    elif audit_required and reason_code == "IDEMPOTENCY_AUDIT_UNAVAILABLE":
        raise ValueError("audit-unavailable cannot be represented as a committed audit")


def validate_operation_transition(
    previous: OperationState,
    current: OperationState,
    *,
    deadline_evidence: DeadlineTerminalEvidence | None = None,
) -> None:
    if not isinstance(previous, OperationState):
        raise TypeError("previous must be OperationState")
    if not isinstance(current, OperationState):
        raise TypeError("current must be OperationState")
    if current is previous:
        if deadline_evidence is not None:
            raise ValueError("idempotent state observation forbids new deadline evidence")
        return
    if current not in _OPERATION_TRANSITIONS[previous]:
        raise ValueError(
            f"operation transition {previous.value} -> {current.value} is not allowed"
        )
    _validate_deadline_evidence(current, deadline_evidence)


def validate_operation_receipt_transition(
    previous: OperationReceipt,
    current: OperationReceipt,
    *,
    deadline_evidence: DeadlineTerminalEvidence | None = None,
) -> None:
    if not isinstance(previous, OperationReceipt):
        raise TypeError("previous must be OperationReceipt")
    if not isinstance(current, OperationReceipt):
        raise TypeError("current must be OperationReceipt")
    immutable_fields = (
        "schema_name",
        "schema_version",
        "operation_id",
        "operation_kind",
        "command_name",
        "command_fingerprint",
        "actor",
        "request_message_id",
        "correlation_id",
        "causation_id",
        "idempotency_scope",
        "idempotency_fingerprint",
        "accepted_at",
    )
    for field_name in immutable_fields:
        if getattr(previous, field_name) != getattr(current, field_name):
            raise ValueError(f"operation admission field {field_name} is immutable")
    validate_operation_transition(
        previous.state,
        current.state,
        deadline_evidence=deadline_evidence,
    )
    if current.updated_at.value < previous.updated_at.value:
        raise ValueError("operation receipt updated_at cannot move backwards")
    if previous.process_id is not None and current.process_id != previous.process_id:
        raise ValueError("operation process link cannot be replaced or removed")
    if previous.state.is_terminal and current != previous:
        raise ValueError("terminal operation receipt is immutable")
    if previous.terminal_at is not None and current.terminal_at != previous.terminal_at:
        raise ValueError("operation terminal_at is immutable once established")


def _validate_lower_token(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _LOWER_TOKEN_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be 1-96 ASCII lower-case letters, digits, '.', "
            "'_', ':' or '-'"
        )
    return value


def _validate_reason(value: str) -> str:
    if not isinstance(value, str) or _REASON_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "reason_code must be 1-96 ASCII upper-case letters, digits, '.', '_' or '-'"
        )
    return value


def _validate_causal_tuple(
    *,
    request_message_id: OpaqueId,
    correlation_id: OpaqueId,
    causation_id: OpaqueId | None,
) -> None:
    if not isinstance(request_message_id, OpaqueId):
        raise TypeError("request_message_id must be OpaqueId")
    if not isinstance(correlation_id, OpaqueId):
        raise TypeError("correlation_id must be OpaqueId")
    if causation_id is not None and not isinstance(causation_id, OpaqueId):
        raise TypeError("causation_id must be OpaqueId or None")
    if causation_id is None and correlation_id != request_message_id:
        raise ValueError("root operation correlation identity must equal request identity")
    if causation_id is not None and request_message_id in {correlation_id, causation_id}:
        raise ValueError("child operation request identity must be new")


def _validate_int(
    value: int,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be in {minimum}..{maximum}")
    return value


def _validate_deadline_evidence(
    current: OperationState,
    evidence: DeadlineTerminalEvidence | None,
) -> None:
    if current is OperationState.DEADLINE_EXCEEDED:
        if not isinstance(evidence, DeadlineTerminalEvidence):
            raise ValueError(
                "owner deadline_exceeded requires worker-exit or durable-fence evidence"
            )
    elif evidence is not None:
        raise ValueError("deadline terminal evidence is valid only for deadline_exceeded")
