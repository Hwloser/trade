## ADDED Requirements

### Requirement: Actor context SHALL be established from trusted adapter evidence

`ActorContext` SHALL identify origin, principal kind and identifier, authority
scopes, bounded delegation, assurance, provenance and establishment time.
Mutation authority SHALL be established only from adapter-controlled CLI,
authenticated HTTP/SDK, registered scheduler, verified parent-event or
Bootstrap system evidence. Caller payload fields SHALL NOT establish identity
or authority. A decoded wire actor SHALL remain unverified until re-established
against trusted local evidence.

Provenance SHALL be a bounded `ActorProvenanceRef`, not an arbitrary mapping.
It SHALL contain a closed provenance type, bounded verifier namespace, opaque
evidence ID, establishment time, optional expiry and stable reason code, and
its canonical form SHALL be at most 1,024 bytes. It SHALL contain no credential,
raw claim set, environment dump or parent payload.
Authority scopes SHALL be 1-96 ASCII lower-case letters/digits plus `._:-`,
sorted and unique, with at most 32 entries. Delegation SHALL contain at most
eight provenance references.

#### Scenario: A payload claims an administrator actor
- **WHEN** a command body contains actor, user, principal or authority fields that are not established by the adapter
- **THEN** those fields confer no authority and the mutation is denied or attributed to the separately established actor

#### Scenario: A verified scheduler submits a command
- **WHEN** Platform Scheduling creates a command from a registered schedule and lease
- **THEN** it establishes a system actor whose provenance links to that schedule identity without embedding credentials

#### Scenario: An anonymous query is allowed
- **WHEN** an interface policy explicitly permits an anonymous read-only query
- **THEN** the actor is marked `anonymous` with `anonymous_allowed` assurance and receives only the declared query scope

### Requirement: Command and query envelopes SHALL use typed owner payloads

Command and query envelopes SHALL compose Kernel metadata, trusted
`ActorContext`, a finite deadline and one typed owner DTO. They SHALL expose a
canonical payload fingerprint but SHALL NOT expose raw idempotency secrets,
live callbacks, framework request objects or arbitrary payload dictionaries.

Public command and idempotency fingerprints SHALL use exact
`FingerprintV1 {algorithm, domain, key_version, value}` values. `algorithm`
SHALL be `hmac-sha256-v1`; domains SHALL be distinct
`trade.command.v1` and `trade.idempotency.v1`; value SHALL be lower-case
64-character hexadecimal; and key version SHALL be an integer
1-2,147,483,647. The secret SHALL contain at least 256 CSPRNG-generated bits,
remain behind a Platform ingress secret-store port and be absent from every
wire/log value.

The key set SHALL have exactly one active write version, at most three retained
read-only versions and a monotonically increasing `key_set_generation` in the
range 1-9,223,372,036,854,775,807. Rotation and admission SHALL use the same
owner-local CAS/lock; rotation SHALL atomically advance generation with the key
set. One command admission SHALL make at most three total claim attempts,
including the initial attempt. Each attempt SHALL start at most one claim transaction,
read one generation, derive at most four HMAC candidates and perform one
candidate query. A changed generation SHALL end and roll back that attempt;
re-derivation SHALL occur only in the next attempt. Claim attempts,
transaction/CAS acquisition, any contention backoff, one optional refusal-audit
transaction and telemetry SHALL consume one `CommandEnvelope` monotonic
remaining deadline.

In one owner-local admission transaction, ingress SHALL read generation, derive
and query candidate idempotency fingerprints for every active or retained key
before creating a claim, and apply this exact result product: zero matches
revalidates unchanged generation then creates one active-version claim; one
match validates the command identity then returns the original receipt; more
than one match always returns `IDEMPOTENCY_CLAIM_CORRUPT` and creates nothing,
even if rows point to one operation. A one-match claim SHALL bind `command_name`,
`operation_kind` and its original canonical command fingerprint. Ingress SHALL
recompute the current command fingerprint with that claim's recorded key
version. Any mismatch SHALL return stable `IDEMPOTENCY_COMMAND_CONFLICT`,
without returning the old receipt or creating a new operation.

If generation changes before zero-match insertion, admission SHALL roll back and
end the current attempt. A later attempt MAY re-derive every candidate from the
new key set within the same deadline. If the third claim attempt or shared
deadline is exhausted before a stable result, admission SHALL form a
provisional `IDEMPOTENCY_KEYSET_CONTENTION` refusal, create no claim, operation
or receipt, and start no background continuation. It SHALL return that reason
only after its refusal audit commits within the remaining shared deadline. If
no time remains to start/commit that audit, or the audit transaction fails,
admission SHALL return `IDEMPOTENCY_AUDIT_UNAVAILABLE` instead. Existing
receipts SHALL retain their original version and SHALL NOT be rewritten during
rotation. Each key SHALL be retained for at least the owner operation-retention
horizon; retirement SHALL fail if safe retention cannot fit the four-version
bound. A retired unknown version SHALL report unavailable rather than recompute
with the current key. An unkeyed digest SHALL NOT be published for low-entropy
idempotency keys or commands.

#### Scenario: A command is retried after transport timeout
- **WHEN** the same actor scope, canonical command and idempotency identity are resubmitted
- **THEN** command ingress can resolve the same digest and existing operation without depending on transport retry metadata

#### Scenario: A retry crosses an HMAC key rotation
- **WHEN** a raw idempotency identity was admitted with a retained key and is retried after a new write key becomes active
- **THEN** the same transaction finds the retained-version claim, verifies the same command identity and returns the original receipt rather than creating an active-version duplicate

#### Scenario: An idempotency identity is reused for a different command
- **WHEN** one claim matches but command name, operation kind or the recomputed canonical command fingerprint differs
- **THEN** ingress returns `IDEMPOTENCY_COMMAND_CONFLICT` without the old receipt and creates no operation

#### Scenario: Retained versions resolve multiple claims
- **WHEN** candidate fingerprints for one raw identity match more than one durable claim, including duplicate rows linked to one operation
- **THEN** ingress returns `IDEMPOTENCY_CLAIM_CORRUPT` and creates no new operation

#### Scenario: Rotation races a paused zero-match admission
- **WHEN** an admission reads generation N and pauses before insert while rotation and another admission commit generation N+1
- **THEN** the paused admission fails generation revalidation, rolls back and re-derives candidates so exactly one durable claim exists

#### Scenario: Rotation contention exhausts admission
- **WHEN** key-set generation changes through the third admission attempt and the contention refusal audit commits within the remaining shared deadline
- **THEN** ingress returns `IDEMPOTENCY_KEYSET_CONTENTION` without a claim, operation, receipt or background continuation
- **AND THEN** one admission has performed no more than twelve HMAC derivations, three candidate queries, three claim transaction/CAS acquisitions and one refusal-audit transaction

#### Scenario: Deadline expires before contention audit
- **WHEN** claim contention leaves no remaining time to start and commit its refusal audit
- **THEN** ingress returns `IDEMPOTENCY_AUDIT_UNAVAILABLE` without starting another transaction or reporting contention as durably recorded

#### Scenario: A framework object enters a command
- **WHEN** a DTO contains a FastAPI request, Pydantic model, ORM object, DataFrame, connection, filesystem path or service object
- **THEN** contract validation or the architecture guard rejects it before serialization

#### Scenario: A public observer guesses a low-entropy key
- **WHEN** an observer has a receipt fingerprint and guesses a likely raw idempotency key
- **THEN** the receipt exposes no unkeyed digest with which that guess can be verified offline

### Requirement: Operation receipts SHALL report durable admission and terminal state truthfully

An `OperationReceipt` SHALL contain version, operation identity and kind,
command fingerprint, trusted actor, correlation/causation identities, scoped
idempotency fingerprint, closed operation state, safe reason, timestamps and
optional process linkage. A receipt SHALL exist only after durable admission
identity is created. Only a command-equivalent duplicate admission SHALL return
the existing receipt; an idempotency identity reused for another command SHALL
return the conflict error above without that receipt. Raw payloads and raw
idempotency keys SHALL NOT be exposed.
Platform command ingress SHALL be the sole future authority and writer for
idempotency claims, operation identity and every initial, intermediate and
terminal `OperationReceipt`. Processes SHALL own Process Manager workflow state
and `ProcessView` only; it SHALL NOT create, rewrite or become a second source
of truth for an OperationReceipt. Platform and Processes SHALL link by opaque
identity plus durable command/event handoff, not a shared transaction.
The Platform-owned optional process link SHALL be `OpaqueId | None`; Platform
SHALL NOT import the Processes-owned `ProcessId`.

Before a Platform or Processes child creates a repository, the parent
architecture ownership matrix SHALL be governedly clarified to split Platform
operation-receipt ownership from Processes workflow ownership. Until then, this
child SHALL remain contract-only and SHALL authorize no durable writer.

The exact allowed state transitions SHALL be:
`requested -> accepted|failed|cancelled|deadline_exceeded`;
`accepted -> running|waiting|retry_scheduled|blocked|completed|failed|cancelled|deadline_exceeded`;
`running -> waiting|retry_scheduled|compensation_pending|blocked|completed|failed|cancelled|deadline_exceeded`;
`waiting -> running|retry_scheduled|blocked|failed|cancelled|deadline_exceeded`;
`retry_scheduled -> running|waiting|blocked|failed|cancelled|deadline_exceeded`;
`compensation_pending -> compensated|blocked|failed|deadline_exceeded`;
`blocked -> running|retry_scheduled|compensation_pending|failed|cancelled|deadline_exceeded`.
Terminal states SHALL have no outgoing transition. An identical state may be
re-observed only without changing terminal facts.

The exact Process state relation SHALL omit `accepted`:
`requested -> running|waiting|failed|cancelled|deadline_exceeded`; every other
non-terminal relation is the corresponding Operation relation above with no
edge to/from `accepted`. `terminal_at` SHALL be absent for non-terminal states,
required exactly once for terminal states and ordered after start/acceptance
and no later than `updated_at`.
An owner `deadline_exceeded` terminal state SHALL require either exit of every
owned worker or durable fencing that prevents every recorded residual worker
from writing. A caller observation timeout SHALL NOT transition an operation or
process to owner `deadline_exceeded`.

#### Scenario: Persistence fails before operation identity is committed
- **WHEN** command ingress cannot durably claim an operation
- **THEN** it returns an `ErrorEnvelope` without fabricating an operation receipt or operation ID

#### Scenario: A duplicate accepted command is observed
- **WHEN** the same scoped idempotency identity is admitted again
- **THEN** the existing operation receipt is returned and no second owner transaction is created

#### Scenario: An operation reaches a terminal state
- **WHEN** an operation becomes completed, compensated, failed, cancelled or deadline-exceeded
- **THEN** its terminal timestamp is set once and no later transition returns it to running or waiting

#### Scenario: Caller observation expires while the owner still runs
- **WHEN** the observation deadline expires before a newer owner state and the worker has neither exited nor been durably write-fenced
- **THEN** the query reports `not_observed` while owner state remains non-terminal

#### Scenario: A Process Manager advances its workflow
- **WHEN** Processes records a new current step or terminal Process state
- **THEN** it updates only its Process Manager record and ProcessView, and any OperationReceipt change requires a separate Platform-owned transition through durable handoff

### Requirement: Process views SHALL be bounded read-only recovery projections

A `ProcessView` SHALL expose process identity/type, correlation/causation,
idempotency fingerprint, closed process and observation states, current step,
top-level reason code, retry/limit/next-attempt, deadline, last safe error, compensation and
dead-letter states, no more than 50 ordered transitions, permitted recovery
actions and timestamps. It SHALL NOT expose raw command/event payload, SQL,
credentials, business-table rows, artifact bytes or traceback. Querying the
view SHALL NOT execute a recovery action.

The top-level reason code SHALL be required for `blocked`, `retry_scheduled`,
`failed`, `cancelled` and `deadline_exceeded`; optional for
`compensation_pending`; and forbidden for `requested`, `running`, `waiting`,
`completed` and `compensated`. A matching safe error may add details but SHALL
NOT replace the process-level reason.

Process history SHALL return the latest at most 50 transitions ordered by
strictly increasing owner sequence and SHALL include `total_count`,
`returned_count`, `first_sequence`, `last_sequence` and
`omitted_before_count`. Counts SHALL be integers 0-2,147,483,647 and sequences
0-9,223,372,036,854,775,807. Empty and non-empty window invariants SHALL be
validated exactly; duplicate/decreasing sequences SHALL be rejected.

Recovery capability SHALL use at most 16 Processes-owned
`RecoveryActionDescriptor` values containing owner, closed action, opaque
target, policy namespace/version, reason, expiry and required actor scope.
Closed generic actions SHALL distinguish `redeliver_message`,
`replay_immutable_input` and `request_new_external_interaction`; the generic
word `redrive` SHALL NOT combine these semantics. Owner namespace and later
owner commands supply Capture/event business meaning. A descriptor SHALL
neither authorize nor execute its action.

#### Scenario: A process is blocked
- **WHEN** an operator queries a blocked process
- **THEN** the view reports the blocking reason, deadline, last observed step and permitted recovery actions without retrying, redriving or mutating the process

#### Scenario: Process history exceeds the wire budget
- **WHEN** more than 50 transitions exist
- **THEN** the owner returns a deterministic bounded window plus truncation metadata and retains full durable history outside the public DTO

#### Scenario: A process cannot be observed
- **WHEN** the process query store is unavailable or no observation completes before the caller deadline
- **THEN** the view or error reports `unavailable` or `not_observed` and does not represent the result as empty, healthy or completed

### Requirement: Status families SHALL remain orthogonal and closed

The system SHALL represent admission, operation, process, observation,
query-condition, control and shutdown states as separate closed enums.
`unknown`, `not_observed` and
`unavailable` SHALL remain distinct from `empty`, `partial`, `stale`,
`quarantined`, `blocked`, success and terminal error. Adding an enum value SHALL
require a new wire-schema version and compatibility review.

`QueryStatus` SHALL be a tagged union with this exact state product:
`observed/present` and `observed/empty` forbid an error;
`observed/partial`, `observed/stale`, `observed/quarantined` and
`observed/blocked` require errors categorized respectively as `unavailable`,
`stale`, `quarantined` and `blocked`; `not_observed`, `unavailable` and
`unknown` forbid a condition and require errors categorized respectively as
`timeout`, `unavailable` and `internal`. Every error's observation state SHALL
match the QueryStatus observation. Every unlisted combination SHALL be
rejected.

#### Scenario: A query returns no business rows
- **WHEN** the owner successfully observes a valid empty result
- **THEN** it reports observation `observed` and condition `empty`, not `unknown` or `unavailable`

#### Scenario: A legacy status is unknown
- **WHEN** a mapper encounters a status not covered by its reviewed source version
- **THEN** it fails closed to an unknown observation plus a safe error and never maps the status to success

#### Scenario: A legacy process is terminated
- **WHEN** a legacy row says `terminated` without durable cancellation intent and terminal cancellation evidence
- **THEN** the target contract does not claim `cancelled` and reports conservative failed or unknown terminal observation

### Requirement: Error envelopes SHALL be safe, versioned and transport-neutral

`ErrorEnvelope` SHALL contain exact schema name/version, stable reason code,
closed category and observation state, retryability and optional bounded
retry-after, correlation/operation/process links, occurrence time, safe message
and safe recovery hint. It SHALL NOT contain arbitrary extra fields, raw
exception text, traceback, credential, SQL, path or raw payload. HTTP status,
CLI exit code and SSE framing SHALL remain interface-adapter mappings.

Reason codes SHALL be 1-96 ASCII upper-case letters/digits plus `._-`;
`retry_after_ms` SHALL be absent or an integer 0-86,400,000; safe message and
recovery hint SHALL each be at most 1,024 UTF-8 bytes.

In addition to required schema/version/reason/occurred-at/safe-message fields,
idempotency admission SHALL use this exact closed product for category,
observation, retry fields, public links and safe recovery hint. Every unlisted
field combination SHALL be rejected:

- `IDEMPOTENCY_COMMAND_CONFLICT`: category `conflict`, observation `observed`,
  `retryable=false`, no retry-after, correlation ID only, no operation/process
  ID, and a safe hint to submit the original command or a new idempotency
  identity.
- `IDEMPOTENCY_CLAIM_CORRUPT`: category `internal`, observation `observed`,
  `retryable=false`, no retry-after, correlation ID only, no operation/process
  ID, and a safe hint requiring audited operator inspection.
- `IDEMPOTENCY_KEYSET_CONTENTION`: category `unavailable`, observation
  `unavailable`, `retryable=true`, required `retry_after_ms` in 1-1,000,
  correlation ID only, no operation/process ID, and a safe hint to retry the
  same command/identity with a fresh finite deadline.
- `IDEMPOTENCY_AUDIT_UNAVAILABLE`: category `unavailable`, observation
  `unavailable`, `retryable=true`, required `retry_after_ms` in 1-1,000,
  correlation ID only, no operation/process ID, and a safe hint to inspect
  Platform persistence and retry after recovery.

These errors SHALL contain no old receipt, raw key, command payload, claim
internals, actor identifier or fingerprint. The future Platform persistence
owner SHALL record one immutable admission-audit fact for every terminal
conflict, corruption and contention outcome before returning that exact
outcome. The fact SHALL use one optional repository transaction after the claim
attempt ends, SHALL have canonical size at most 2,048 bytes, and SHALL contain
only schema/version, reason, correlation ID, static owner namespace, sorted
matched key versions (at most four), key-set generation, attempt count and
occurred-at. It SHALL contain no raw key, command DTO/payload, provider/source
body, artifact bytes, actor identifier, credential, path or fingerprint. It
SHALL be visible only through an authorized Platform operator audit query and
SHALL NOT enter an `ErrorEnvelope`, HTTP/SDK response or cross-Context
projection. The audit transaction SHALL NOT create or reopen a claim,
operation, receipt or dispatch. If the audit fact cannot start or commit within
the remaining deadline, admission SHALL return
`IDEMPOTENCY_AUDIT_UNAVAILABLE` instead and SHALL still create no claim,
operation or receipt. The provisional conflict, corruption or contention reason
SHALL NOT be returned as durably recorded in that case. One admission SHALL
therefore start at most three claim
transactions plus one refusal-audit transaction. The owner SHALL expose only the
low-cardinality counter `platform_idempotency_admission_outcomes_total` with
`owner_namespace` and the closed outcome
`created|replayed|command_conflict|claim_corrupt|keyset_retry|keyset_contention|audit_unavailable`
as labels. The counter, one bounded structured event and, for corruption, one
integrity alert SHALL each be attempted at most once, synchronously under the
same remaining deadline, with no retry or background continuation. The event
SHALL contain only reason, correlation, owner, sorted matched key versions (at
most four), key-set generation and attempt count. Telemetry emission failure
SHALL surface once to the owner health/audit channel and SHALL NOT convert a
refusal to success or permit claim creation; the owning child SHALL preserve
these bounds before adoption.

#### Scenario: An Observatory error is mapped
- **WHEN** a legacy `ObservatoryError` contains reason, message, evidence refs and arbitrary extra fields
- **THEN** the mapper emits only whitelisted safe fields and uses a stable generic message if the original message is not proven safe

#### Scenario: A retry-after value is present
- **WHEN** a retryable saturation or availability failure has an owner-approved finite delay
- **THEN** the envelope carries bounded integer milliseconds and the HTTP adapter may preserve its existing `Retry-After` representation

#### Scenario: Idempotency errors are encoded
- **WHEN** admission reports a command conflict, corrupt claim set, exhausted key-set contention or unavailable refusal audit
- **THEN** the exact reason/category/observation/retry/link/recovery product above is emitted and every forbidden combination is rejected

#### Scenario: Claim corruption is observed
- **WHEN** more than one candidate claim matches
- **THEN** the owner records one 2,048-byte-bounded allowlisted terminal refusal and attempts one closed low-cardinality counter, one redacted bounded event and one integrity alert under the same remaining deadline
- **AND THEN** telemetry is neither retried nor continued in a background task

#### Scenario: Terminal refusal audit cannot commit
- **WHEN** the owner cannot durably record the conflict, corruption or contention refusal before the shared deadline
- **THEN** admission returns `IDEMPOTENCY_AUDIT_UNAVAILABLE` with no claim, operation or receipt and surfaces the audit failure without reporting the underlying refusal as durably recorded

#### Scenario: Refusal audit outcome priority is deterministic
- **WHEN** three contention attempts finish and audit either commits, has no remaining start budget, or fails within its remaining budget
- **THEN** golden fixtures return respectively `IDEMPOTENCY_KEYSET_CONTENTION`, `IDEMPOTENCY_AUDIT_UNAVAILABLE`, and `IDEMPOTENCY_AUDIT_UNAVAILABLE`

### Requirement: Cancellation SHALL distinguish request acceptance from terminal cancellation

A cancellation API SHALL return a `ControlReceipt` with control identity,
operation/process linkage, actor, request time, finite control deadline and
closed disposition. `accepted` SHALL mean only that cancellation intent was
durably admitted. An operation or process SHALL enter `cancelled` only after
the owner records terminal cancellation evidence. Signal delivery, caller
disconnect or observation timeout alone SHALL NOT prove cancellation.

#### Scenario: A cancel request is accepted while work is still running
- **WHEN** the owner durably records cancellation intent before work exits
- **THEN** the control receipt is accepted while the operation remains running or waiting until a terminal owner receipt is observed

#### Scenario: Cancellation control times out
- **WHEN** intent cannot be admitted or applied before the finite control deadline
- **THEN** the response reports deadline-exceeded or unavailable with the last operation/process link and does not claim the work was cancelled

### Requirement: Shutdown SHALL have one finite deadline and explicit residual ownership

A shutdown API SHALL close admission and return a `ShutdownReceipt` containing
owner namespace/instance identity, fence generation, control/correlation/
causation identities, trusted initiator, optional operation/process links, request/deadline/
finished time, closed shutdown state/current stage/reason, graceful/forced
termination counts, bounded residual owners, owner-scoped recovery descriptors
and a safe error. `completed` SHALL require stage `done`, zero live owned work,
durable terminal audit, released non-reentrant resources and release of only
the matching generation fence. Deadline-exceeded, incomplete or failed
shutdown SHALL retain owner fencing and report stage/reason/residual/recovery
facts.

The trusted initiator SHALL be the bounded credential-free `ActorContext` that
requested shutdown. `control_id` SHALL resolve for the receipt retention period
to an immutable actor-bearing `ControlReceipt` with the same initiator,
correlation and causation. A mismatch SHALL be corruption. The optional Platform
process link SHALL be `OpaqueId | None` and SHALL NOT require a Platform import
of Processes contracts.

Fence generation SHALL be an integer 1-9,223,372,036,854,775,807 durably
claimed before admission. Every write or terminal receipt from an older
generation SHALL be rejected. Crash takeover may claim only the next generation
after durable proof that the prior lease expired or was revoked, and SHALL
record takeover causation; restart SHALL NOT reuse owner instance or generation.

Residual ownership SHALL use only `process_group`, `executor_task`,
`python_thread`, `persistence_audit`, `writer_lease` and `inflight_start`, with
at most 16 entries. Each entry SHALL contain count 1-2,147,483,647 and a
non-secret `OpaqueId` inspection selector. Platform-owned
`ShutdownRecoveryAction` SHALL use at most 16 entries and only
`inspect_residual`, `retry_terminal_audit`, `terminate_process_group`,
`retry_shutdown_with_deadline`, `revoke_expired_writer_lease` or
`operator_intervention`, plus target/reason/expiry/scope. It SHALL be
informational and SHALL NOT depend on Processes contracts. Shutdown stages
SHALL be `close_admission`, `request_graceful`, `force_process_tree`,
`drain_delivery`, `commit_terminal_audit`, `release_resources`,
`release_fence` or `done`.

The shutdown state product SHALL be exact. `completed` requires `done`,
`SHUTDOWN_COMPLETED`, no safe error, zero residual/recovery entries, durable
terminal audit, released resources and release of only the matching fence.
`deadline_exceeded` requires a non-`done` stage,
`SHUTDOWN_DEADLINE_EXCEEDED`, an observed timeout error, at least one residual
and applicable recovery action, and a retained fence. `incomplete` requires a
non-`done` stage, a stable non-deadline reason, an observed blocked or
unavailable error, at least one residual and applicable recovery action, and a
retained fence. `failed` requires a non-`done` stage, a stable failure reason,
an observed internal or unavailable error, at least one residual (including
writer/audit ownership when appropriate), an applicable recovery action and a
retained fence. Every unlisted state/stage/reason/error/residual/recovery/fence
combination SHALL be rejected.

Every potentially blocking shutdown stage, including lock acquisition,
persistence retry, queue drain, thread/future/executor/subprocess wait and
process-tree termination, SHALL consume the remaining shared deadline. A
bounded public API SHALL NOT perform an unbounded join after its deadline.

#### Scenario: A child process ignores graceful termination
- **WHEN** the child remains alive after the graceful portion of the shared deadline
- **THEN** the process owner applies its reviewed forced process-tree policy within the same deadline and records the outcome and residual ownership

#### Scenario: An executor task cannot be interrupted
- **WHEN** executor shutdown would wait beyond the shared deadline
- **THEN** shutdown returns incomplete or deadline-exceeded without an unbounded `shutdown(wait=True)`, keeps the ownership fence closed and reports the residual task

#### Scenario: Multiple callers request shutdown
- **WHEN** concurrent callers stop the same owner
- **THEN** they observe the same bounded stop attempt or its terminal receipt and no caller waits indefinitely on another caller

#### Scenario: Shutdown completes
- **WHEN** all owned work is terminal, durable receipts are committed and non-reentrant resources are released before the deadline
- **THEN** exactly one completed shutdown receipt reports zero residual owners

#### Scenario: A stale owner writes after takeover
- **WHEN** generation N+1 has durably taken over and generation N attempts a state or terminal-audit write
- **THEN** the repository rejects the stale writer and the N+1 receipt retains takeover causation

#### Scenario: A runtime adopts the control contract
- **WHEN** EventBus, Web resources, RuntimeCommandRunner or FastAPI lifespan would route through these contracts
- **THEN** adoption is blocked until `runtime-owner-shutdown-and-recovery-hardening-v1` has strict approval and passing persistence-retry, concurrent-stop, startup-cleanup, executor-tail, monotonic-wait, crash-takeover and real signal-to-reap fixtures
