from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, is_dataclass
from enum import Enum
from typing import Generic, TypeVar

from trade.kernel.digest import ContentDigest
from trade.kernel.envelope import EnvelopeMeta
from trade.kernel.ids import IdNamespace, OpaqueId
from trade.kernel.time import Deadline
from trade.platform.contracts.actor import (
    ActorAssurance,
    ActorContext,
    IdempotencySubjectV1,
    PolicyRefV1,
)

__all__ = [
    "CodecContentPolicy",
    "CommandEnvelope",
    "FingerprintDomain",
    "FingerprintKeySetV1",
    "FingerprintV1",
    "FrozenCodecRegistry",
    "OwnerCodecDescriptor",
    "PayloadPurpose",
    "QueryEnvelope",
    "ReplayAdmissionV1",
    "ReplayContextV1",
    "canonical_idempotency_subject_bytes",
    "derive_command_fingerprint",
    "derive_idempotency_fingerprint",
    "frame_hmac_components",
]

_SCHEMA_NAME_PATTERN = re.compile(r"[a-z0-9._-]{1,96}", re.ASCII)
_TOKEN_PATTERN = re.compile(r"[a-z0-9._:-]{1,96}", re.ASCII)
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
_MAX_SCHEMA_VERSION = 2_147_483_647
_MAX_KEY_SET_GENERATION = 9_223_372_036_854_775_807
_MAX_CANONICAL_BYTES = 65_536
_MIN_SECRET_BYTES = 32
_MAX_RETAINED_KEY_VERSIONS = 3

T = TypeVar("T")


class PayloadPurpose(str, Enum):
    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"
    REFERENCE = "reference"
    PROJECTION = "projection"


class CodecContentPolicy(str, Enum):
    INLINE_CONTRACT = "inline_contract"
    IMMUTABLE_REF_ONLY = "immutable_ref_only"


class FingerprintDomain(str, Enum):
    COMMAND = "trade.command.v1"
    IDEMPOTENCY = "trade.idempotency.v1"


@dataclass(frozen=True, slots=True)
class OwnerCodecDescriptor:
    owner_namespace: IdNamespace
    schema_name: str
    schema_version: int
    payload_purpose: PayloadPurpose
    max_canonical_bytes: int
    content_policy: CodecContentPolicy
    codec_identity: ContentDigest

    def __post_init__(self) -> None:
        if not isinstance(self.owner_namespace, IdNamespace):
            raise TypeError("owner_namespace must be IdNamespace")
        if (
            not isinstance(self.schema_name, str)
            or _SCHEMA_NAME_PATTERN.fullmatch(self.schema_name) is None
        ):
            raise ValueError(
                "schema_name must be 1-96 ASCII lower-case letters, digits, '.', '_' or '-'"
            )
        _positive_version(self.schema_version, field_name="schema_version")
        if not isinstance(self.payload_purpose, PayloadPurpose):
            raise TypeError("payload_purpose must be PayloadPurpose")
        if not isinstance(self.max_canonical_bytes, int) or isinstance(
            self.max_canonical_bytes, bool
        ):
            raise TypeError("max_canonical_bytes must be an integer")
        if not 1 <= self.max_canonical_bytes <= _MAX_CANONICAL_BYTES:
            raise ValueError("max_canonical_bytes must be in 1..65,536")
        if not isinstance(self.content_policy, CodecContentPolicy):
            raise TypeError("content_policy must be CodecContentPolicy")
        if not isinstance(self.codec_identity, ContentDigest):
            raise TypeError("codec_identity must be ContentDigest")

    @property
    def registry_key(self) -> tuple[str, str, int, str]:
        return (
            self.owner_namespace.value,
            self.schema_name,
            self.schema_version,
            self.payload_purpose.value,
        )


@dataclass(frozen=True, slots=True)
class FrozenCodecRegistry:
    descriptors: tuple[OwnerCodecDescriptor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptors, tuple):
            raise TypeError("descriptors must be a tuple")
        if any(not isinstance(item, OwnerCodecDescriptor) for item in self.descriptors):
            raise TypeError("registry entries must be OwnerCodecDescriptor")
        keys = tuple(item.registry_key for item in self.descriptors)
        if len(keys) != len(set(keys)):
            raise ValueError("owner codec registry contains a duplicate descriptor key")
        if keys != tuple(sorted(keys)):
            raise ValueError("owner codec descriptors must be sorted by registry key before freeze")

    @classmethod
    def freeze(cls, descriptors: tuple[OwnerCodecDescriptor, ...]) -> FrozenCodecRegistry:
        if not isinstance(descriptors, tuple):
            raise TypeError("descriptors must be a tuple")
        return cls(tuple(sorted(descriptors, key=lambda item: item.registry_key)))

    def resolve(
        self,
        *,
        owner_namespace: IdNamespace,
        schema_name: str,
        schema_version: int,
        payload_purpose: PayloadPurpose,
    ) -> OwnerCodecDescriptor | None:
        key = (
            owner_namespace.value,
            schema_name,
            schema_version,
            payload_purpose.value,
        )
        return next((item for item in self.descriptors if item.registry_key == key), None)


@dataclass(frozen=True, slots=True)
class FingerprintV1:
    algorithm: str
    domain: FingerprintDomain
    key_version: int
    value: str

    def __post_init__(self) -> None:
        if self.algorithm != "hmac-sha256-v1":
            raise ValueError("fingerprint algorithm must be 'hmac-sha256-v1'")
        if not isinstance(self.domain, FingerprintDomain):
            raise TypeError("fingerprint domain must be FingerprintDomain")
        _positive_version(self.key_version, field_name="key_version")
        if not isinstance(self.value, str) or _FINGERPRINT_PATTERN.fullmatch(self.value) is None:
            raise ValueError("fingerprint value must be 64 lower-case hexadecimal characters")


@dataclass(frozen=True, slots=True)
class FingerprintKeySetV1:
    key_set_generation: int
    active_write_version: int
    retained_read_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key_set_generation, int) or isinstance(
            self.key_set_generation, bool
        ):
            raise TypeError("key_set_generation must be an integer")
        if not 1 <= self.key_set_generation <= _MAX_KEY_SET_GENERATION:
            raise ValueError("key_set_generation is outside the v1 range")
        _positive_version(self.active_write_version, field_name="active_write_version")
        if not isinstance(self.retained_read_versions, tuple):
            raise TypeError("retained_read_versions must be a tuple")
        if len(self.retained_read_versions) > _MAX_RETAINED_KEY_VERSIONS:
            raise ValueError("at most three retained read-only key versions are allowed")
        for version in self.retained_read_versions:
            _positive_version(version, field_name="retained key version")
        if self.retained_read_versions != tuple(sorted(set(self.retained_read_versions))):
            raise ValueError("retained key versions must be sorted and unique")
        if self.active_write_version in self.retained_read_versions:
            raise ValueError("active write version cannot also be retained read-only")

    @property
    def candidate_versions(self) -> tuple[int, ...]:
        return (self.active_write_version, *self.retained_read_versions)


@dataclass(frozen=True, slots=True)
class CommandEnvelope(Generic[T]):
    meta: EnvelopeMeta
    actor: ActorContext
    deadline: Deadline
    codec: OwnerCodecDescriptor
    payload: T
    canonical_payload: bytes

    def __post_init__(self) -> None:
        _validate_typed_envelope(
            meta=self.meta,
            actor=self.actor,
            deadline=self.deadline,
            codec=self.codec,
            payload=self.payload,
            canonical_payload=self.canonical_payload,
            expected_purpose=PayloadPurpose.COMMAND,
        )
        if not self.actor.can_submit_mutation:
            raise ValueError("command envelope requires a verified authenticated or system actor")


@dataclass(frozen=True, slots=True)
class QueryEnvelope(Generic[T]):
    meta: EnvelopeMeta
    actor: ActorContext
    deadline: Deadline
    codec: OwnerCodecDescriptor
    payload: T
    canonical_payload: bytes

    def __post_init__(self) -> None:
        _validate_typed_envelope(
            meta=self.meta,
            actor=self.actor,
            deadline=self.deadline,
            codec=self.codec,
            payload=self.payload,
            canonical_payload=self.canonical_payload,
            expected_purpose=PayloadPurpose.QUERY,
        )


@dataclass(frozen=True, slots=True)
class ReplayContextV1:
    schema_version: int
    replay_request_message_id: OpaqueId
    replay_request_correlation_id: OpaqueId
    replay_request_causation_id: OpaqueId | None
    historical_message_id: OpaqueId
    historical_actor: ActorContext
    replay_initiator: ActorContext
    replay_policy: PolicyRefV1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ReplayContextV1 schema_version must be 1")
        _validate_causal_tuple(
            message_id=self.replay_request_message_id,
            correlation_id=self.replay_request_correlation_id,
            causation_id=self.replay_request_causation_id,
            field_prefix="replay request",
        )
        if not isinstance(self.historical_message_id, OpaqueId):
            raise TypeError("historical_message_id must be OpaqueId")
        if not isinstance(self.historical_actor, ActorContext):
            raise TypeError("historical_actor must be ActorContext")
        if self.historical_actor.assurance is not ActorAssurance.UNVERIFIED:
            raise ValueError("historical replay actor must remain unverified attribution")
        if not isinstance(self.replay_initiator, ActorContext):
            raise TypeError("replay_initiator must be ActorContext")
        if not self.replay_initiator.can_submit_mutation:
            raise ValueError("replay initiator must be a verified authenticated or system actor")
        if not isinstance(self.replay_policy, PolicyRefV1):
            raise TypeError("replay_policy must be PolicyRefV1")


@dataclass(frozen=True, slots=True)
class ReplayAdmissionV1:
    schema_version: int
    context: ReplayContextV1
    historical_message_id: OpaqueId
    historical_correlation_id: OpaqueId
    historical_causation_id: OpaqueId | None
    historical_envelope_digest: ContentDigest
    historical_operation_id: OpaqueId
    current_subject: IdempotencySubjectV1
    deadline: Deadline

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ReplayAdmissionV1 schema_version must be 1")
        if not isinstance(self.context, ReplayContextV1):
            raise TypeError("context must be ReplayContextV1")
        _validate_causal_tuple(
            message_id=self.historical_message_id,
            correlation_id=self.historical_correlation_id,
            causation_id=self.historical_causation_id,
            field_prefix="historical envelope",
        )
        if self.historical_message_id != self.context.historical_message_id:
            raise ValueError("historical message identity must equal ReplayContextV1 attribution")
        if not isinstance(self.historical_envelope_digest, ContentDigest):
            raise TypeError("historical_envelope_digest must be ContentDigest")
        if not isinstance(self.historical_operation_id, OpaqueId):
            raise TypeError("historical_operation_id must be OpaqueId")
        if not isinstance(self.current_subject, IdempotencySubjectV1):
            raise TypeError("current_subject must be IdempotencySubjectV1")
        initiator = self.context.replay_initiator
        if (
            self.current_subject.principal_kind is not initiator.principal_kind
            or self.current_subject.principal_id != initiator.principal_id
        ):
            raise ValueError("current subject must identify the verified replay initiator")
        if not isinstance(self.deadline, Deadline):
            raise TypeError("deadline must be Deadline")


def frame_hmac_components(components: tuple[bytes, ...]) -> bytes:
    if not isinstance(components, tuple):
        raise TypeError("HMAC components must be a tuple")
    framed = bytearray()
    for component in components:
        if not isinstance(component, bytes):
            raise TypeError("HMAC components must be bytes")
        if len(component) > 0xFFFFFFFF:
            raise ValueError("HMAC component exceeds unsigned four-byte length")
        framed.extend(len(component).to_bytes(4, "big"))
        framed.extend(component)
    return bytes(framed)


def derive_command_fingerprint(
    *,
    secret: bytes,
    key_version: int,
    canonical_payload: bytes,
) -> FingerprintV1:
    return _derive_fingerprint(
        secret=secret,
        domain=FingerprintDomain.COMMAND,
        key_version=key_version,
        components=(FingerprintDomain.COMMAND.value.encode("ascii"), canonical_payload),
    )


def derive_idempotency_fingerprint(
    *,
    secret: bytes,
    key_version: int,
    subject: IdempotencySubjectV1,
    command_scope: str,
    raw_key: bytes,
) -> FingerprintV1:
    if not isinstance(subject, IdempotencySubjectV1):
        raise TypeError("subject must be IdempotencySubjectV1")
    if not isinstance(command_scope, str) or _TOKEN_PATTERN.fullmatch(command_scope) is None:
        raise ValueError("command_scope must be a bounded lower-case contract token")
    if not isinstance(raw_key, bytes):
        raise TypeError("raw_key must be bytes")
    return _derive_fingerprint(
        secret=secret,
        domain=FingerprintDomain.IDEMPOTENCY,
        key_version=key_version,
        components=(
            FingerprintDomain.IDEMPOTENCY.value.encode("ascii"),
            canonical_idempotency_subject_bytes(subject),
            command_scope.encode("utf-8"),
            raw_key,
        ),
    )


def canonical_idempotency_subject_bytes(subject: IdempotencySubjectV1) -> bytes:
    if not isinstance(subject, IdempotencySubjectV1):
        raise TypeError("subject must be IdempotencySubjectV1")

    def identity(value: OpaqueId) -> dict[str, str]:
        return {"namespace": value.namespace.value, "value": value.value}

    payload: dict[str, object] = {
        "delegated_subject_ids": [identity(item) for item in subject.delegated_subject_ids],
        "owner_namespace": subject.owner_namespace.value,
        "principal_id": identity(subject.principal_id),
        "principal_kind": subject.principal_kind.value,
        "schema_version": subject.schema_version,
        "tenant_id": identity(subject.tenant_id),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _derive_fingerprint(
    *,
    secret: bytes,
    domain: FingerprintDomain,
    key_version: int,
    components: tuple[bytes, ...],
) -> FingerprintV1:
    if not isinstance(secret, bytes):
        raise TypeError("fingerprint secret must be bytes")
    if len(secret) < _MIN_SECRET_BYTES:
        raise ValueError("fingerprint secret must contain at least 256 bits")
    _positive_version(key_version, field_name="key_version")
    value = hmac.new(secret, frame_hmac_components(components), hashlib.sha256).hexdigest()
    return FingerprintV1(
        algorithm="hmac-sha256-v1",
        domain=domain,
        key_version=key_version,
        value=value,
    )


def _validate_typed_envelope(
    *,
    meta: EnvelopeMeta,
    actor: ActorContext,
    deadline: Deadline,
    codec: OwnerCodecDescriptor,
    payload: object,
    canonical_payload: bytes,
    expected_purpose: PayloadPurpose,
) -> None:
    if not isinstance(meta, EnvelopeMeta):
        raise TypeError("meta must be EnvelopeMeta")
    if not isinstance(actor, ActorContext):
        raise TypeError("actor must be ActorContext")
    if not isinstance(deadline, Deadline):
        raise TypeError("deadline must be Deadline")
    if not isinstance(codec, OwnerCodecDescriptor):
        raise TypeError("codec must be OwnerCodecDescriptor")
    if codec.payload_purpose is not expected_purpose:
        raise ValueError(f"codec payload purpose must be {expected_purpose.value}")
    if meta.schema_name != codec.schema_name or meta.schema_version != codec.schema_version:
        raise ValueError("envelope schema identity must match the owner codec descriptor")
    parameters = getattr(type(payload), "__dataclass_params__", None)
    if not is_dataclass(payload) or parameters is None or not parameters.frozen:
        raise TypeError("envelope payload must be a frozen typed owner dataclass")
    if not isinstance(canonical_payload, bytes):
        raise TypeError("canonical_payload must be bytes")
    if len(canonical_payload) > codec.max_canonical_bytes:
        raise ValueError("canonical payload exceeds the owner codec byte limit")
    if not canonical_payload:
        raise ValueError("canonical payload must not be empty")


def _validate_causal_tuple(
    *,
    message_id: OpaqueId,
    correlation_id: OpaqueId,
    causation_id: OpaqueId | None,
    field_prefix: str,
) -> None:
    if not isinstance(message_id, OpaqueId):
        raise TypeError(f"{field_prefix} message_id must be OpaqueId")
    if not isinstance(correlation_id, OpaqueId):
        raise TypeError(f"{field_prefix} correlation_id must be OpaqueId")
    if causation_id is not None and not isinstance(causation_id, OpaqueId):
        raise TypeError(f"{field_prefix} causation_id must be OpaqueId or None")
    if causation_id is None and correlation_id != message_id:
        raise ValueError(f"root {field_prefix} correlation identity must equal message identity")
    if causation_id is not None and message_id in {correlation_id, causation_id}:
        raise ValueError(f"child {field_prefix} message identity must be new")


def _positive_version(value: int, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if not 1 <= value <= _MAX_SCHEMA_VERSION:
        raise ValueError(f"{field_name} must be in 1..2,147,483,647")
    return value
