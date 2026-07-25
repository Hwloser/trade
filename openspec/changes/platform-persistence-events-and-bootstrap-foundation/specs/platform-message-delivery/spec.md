## ADDED Requirements

### Requirement: Command ingress SHALL durably admit one operation identity

Platform command ingress SHALL accept only the approved framework-free Kernel
`CommandEnvelope[T]` carrying a trusted `ActorContext`, correlation and causation
identities, canonical command fingerprint, scoped idempotency fingerprint and finite
monotonic deadline. The owning ingress repository SHALL use the approved bounded
key-generation admission algorithm from strict-approved Kernel artifact
`sha256:1d06f033c231ce22d6abe164a1ed1f8fc553de54762f33a46882f4b4391b1f4f`.
One admission SHALL share the command's remaining monotonic deadline across at most
three claim attempts, each with at most one claim transaction, plus at most one
separate refusal-audit transaction after the final claim attempt ends. A generation
change SHALL roll back and end the current claim attempt; candidate re-derivation
SHALL occur only in a later attempt. A stable claim transaction SHALL either:

- create one immutable operation identity plus initial `OperationReceipt` and owner
  command outbox record;
- return the existing receipt for a command-equivalent duplicate; or
- produce a provisional conflict, corrupt-claim or contention refusal without
  fabricating an operation.

Ingress SHALL return a provisional refusal only after its bounded refusal audit
commits within the same remaining deadline. When no audit-start/commit budget remains
or that transaction fails, `IDEMPOTENCY_AUDIT_UNAVAILABLE` SHALL take precedence.
Neither path SHALL create a claim, operation, receipt, dispatch, retry or background
continuation. Refusal telemetry SHALL retain the Kernel one-shot/no-background bounds.

Ingress SHALL NOT synchronously execute a Context use case, call a provider, dispatch
an in-memory handler, expose raw command payload/idempotency secret or keep an HTTP,
CLI or scheduler caller attached to the eventual workflow. A formal
`OperationReceipt` SHALL not be derived from legacy `job_runs` or PID admission.

#### Scenario: The same command is admitted twice
- **WHEN** an actor repeats an equivalent command under the same scoped idempotency identity
- **THEN** ingress returns the first durable `OperationReceipt`, emits no second command outbox record and performs no second Context transition

#### Scenario: The identity is reused for another command
- **WHEN** the scoped idempotency identity resolves to a different canonical command fingerprint
- **THEN** ingress returns `IDEMPOTENCY_COMMAND_CONFLICT` without an operation identity, command payload or raw key

#### Scenario: The caller deadline expires before admission commits
- **WHEN** the shared monotonic deadline is exhausted before durable admission and required refusal audit complete
- **THEN** ingress returns the approved unavailable/deadline error, rolls back the claim attempt and leaves no operation, receipt or dispatch

#### Scenario: Rotation changes generation in every attempt
- **WHEN** generation changes during each of the three claim transactions and the separate contention refusal audit commits within the remaining deadline
- **THEN** ingress returns `IDEMPOTENCY_KEYSET_CONTENTION` after no more than three claim transactions plus one refusal-audit transaction and creates no operation, receipt or background continuation

#### Scenario: Refusal audit has no remaining budget
- **WHEN** a conflict, corruption or contention claim attempt ends but the shared deadline cannot start and commit its audit
- **THEN** ingress returns `IDEMPOTENCY_AUDIT_UNAVAILABLE`, starts no additional transaction and does not report the provisional reason as durably recorded

### Requirement: Outbox dispatch SHALL use durable leases and bounded outcomes

Every outbox record SHALL contain envelope schema/version, message identity, owner
namespace, message kind and payload digest, correlation/causation, operation/process
links, ordering declaration, priority, not-before time, finite deadline, retry policy
reference, created time and immutable payload bytes or an owner-verifiable immutable
payload reference. A dispatcher SHALL claim a bounded batch using an owner instance,
fence generation, lease token and expiry.

Delivery SHALL transition through the closed technical states `pending`, `leased`,
`delivered`, `retry_scheduled`, `dead_lettered` and `cancelled`. Lease renewal,
acknowledgement, retry scheduling and terminalization SHALL compare the lease token
and fence. A crash before acknowledgement SHALL make the record reclaimable after
proven lease expiry. Retry count, backoff, deadline and maximum elapsed age SHALL be
finite. No transient persistence error SHALL create an unbounded retry loop inside a
worker or shutdown call; unresolved terminal persistence SHALL retain a durable
technical owner/residual state for later recovery.

#### Scenario: A dispatcher crashes after consumer execution starts
- **WHEN** the delivery lease expires without a committed acknowledgement
- **THEN** another fenced dispatcher may reclaim the same immutable envelope and inbox deduplication prevents a second effective consumer transition

#### Scenario: Retry budget is exhausted
- **WHEN** the next attempt would exceed attempts, elapsed age or envelope deadline
- **THEN** Platform records one dead-letter outcome with safe failure facts and does not silently drop, continue or relabel the message as delivered

#### Scenario: Terminal persistence remains unavailable
- **WHEN** the dispatcher cannot commit ack, retry or dead-letter state within its remaining deadline
- **THEN** it stops synchronous retry, preserves lease/residual ownership evidence and returns an explicit unavailable or incomplete result for reconciliation

### Requirement: Inbox consumption SHALL make duplicate delivery ineffective

Before invoking an owner use case, a consumer adapter SHALL validate the envelope
schema, payload digest, target owner, message kind and deadline. The owner local
transaction SHALL insert or compare a durable inbox identity and atomically commit
the effective owner transition, owner audit/receipt, ordered-consumer head and
outgoing outbox. The inbox identity SHALL be `(consumer_effect_namespace,
message_id)`, where `consumer_effect_namespace` is an immutable owner-qualified
identity for one effective transition and is not an implementation, deployment or
schema version. It SHALL store target owner, message kind, schema, immutable payload
digest, first consumer contract version, effect-contract digest,
compatibility-policy digest and first effective receipt. Contract version and
compatibility-policy digest SHALL NOT participate in uniqueness.

An exact duplicate under the same contract version, or a duplicate whose current
consumer version declares the recorded version compatible through the pinned
owner compatibility policy and unchanged effect-contract digest, SHALL return the
existing receipt without invoking the transition. A duplicate identity with an
incompatible consumer version, another effect-contract digest, payload digest,
target, message kind or schema SHALL be quarantined as corruption. A compatible
binary upgrade SHALL preserve the effect namespace. An intentionally different
effect SHALL use a new owner-defined effect namespace and an explicit owner/process
migration; incrementing a version SHALL never reapply old messages. Handler code
SHALL not acknowledge outside the owner transaction and SHALL not embed
cross-Context orchestration.

#### Scenario: An exact message is delivered twice
- **WHEN** the same consumer receives the same message ID and payload digest after the first local commit
- **THEN** the second delivery returns the recorded inbox receipt and performs no owner write or child emission

#### Scenario: A compatible consumer binary is deployed
- **WHEN** a newer consumer contract version receives a message already applied by a recorded compatible version under the same effect namespace, effect digest and pinned compatibility policy
- **THEN** it returns the first receipt and performs no owner write even though the implementation version changed

#### Scenario: A consumer changes effective semantics
- **WHEN** a consumer version is not compatible with the recorded effect contract or attempts to reuse its effect namespace for another effect digest
- **THEN** delivery quarantines the mismatch and does not treat a version change as permission to invoke the owner again

#### Scenario: A duplicate identity has different bytes
- **WHEN** a message ID already exists for the consumer but schema, target or payload digest differs
- **THEN** consumption fails closed to quarantine/dead-letter evidence and does not call the owner use case

#### Scenario: The owner transaction rolls back
- **WHEN** the Context invariant or outbox insertion fails before commit
- **THEN** inbox acknowledgement is absent, the delivery remains retryable under policy and no partial business state is visible

### Requirement: Ordering SHALL be explicit, durable and bounded

Every message SHALL declare either `unordered` or an `OrderingContractRef`. An
`OrderingContract` SHALL contain contract version/digest, producer namespace,
ordering scope/key digest, producer fence epoch, transactionally assigned positive
sequence, consumer expected sequence, duplicate/stale policy, finite gap timeout,
maximum buffered gap count/bytes and head-of-line failure policy.

The producer SHALL allocate sequence in the same local transaction as its outbox
record. An ordered consumer SHALL atomically compare/update its expected sequence with
the inbox receipt and owner transition. It SHALL not apply N+1 before required
handling of N. Duplicate/stale, gap, epoch regression and head-of-line expiry SHALL
produce explicit receipt, retry, quarantine, reconciliation or dead-letter outcomes.
No implementation SHALL keep the only gap state in process memory or allow one key
to monopolize all dispatcher capacity.

Dead-lettering sequence N SHALL retain the durable consumer head at N and SHALL NOT
make N+1 eligible. Redelivery of N SHALL create a new delivery attempt generation
while preserving N's original producer epoch and sequence. The head SHALL advance
only when N commits effectively or an authorized owner issues
`ResolveOrderingGap(expected_head=N, dead_letter_generation, ordering_contract_digest,
reason, deadline)`. That command SHALL be allowed only when the pinned
head-of-line policy explicitly permits `skip_with_tombstone`; it SHALL compare-and-swap
the expected head and dead-letter generation and atomically append an immutable
resolution/tombstone receipt before setting expected sequence to N+1. Platform SHALL
reject the command when skip is forbidden, evidence changed, authorization is absent
or another resolution won. It SHALL never infer skip from timeout, DLQ presence or
operator query.

#### Scenario: N+1 arrives before N after restart
- **WHEN** durable expected sequence is N and N+1 is delivered first
- **THEN** the consumer records a bounded gap outcome, does not invoke the owner transition for N+1 and waits/reconciles only until the contract's finite limit

#### Scenario: A producer restarts with a stale epoch
- **WHEN** an outbox append presents an epoch lower than the durable producer fence
- **THEN** the append is rolled back and no duplicate sequence enters delivery

#### Scenario: A gap exhausts its policy
- **WHEN** N never arrives before the finite gap timeout or buffer budget
- **THEN** the head-of-line policy deterministically dead-letters N or requests audited reconciliation, retains expected sequence N and does not silently apply N+1

#### Scenario: An ordered dead letter is redelivered
- **WHEN** an operator redelivers dead-lettered sequence N
- **THEN** a new attempt uses the original producer epoch and sequence N, and N+1 remains blocked until N commits effectively

#### Scenario: An owner resolves an allowed permanent gap
- **WHEN** the pinned ordering contract permits `skip_with_tombstone` and an authorized command matches expected sequence N and the current dead-letter generation
- **THEN** Platform atomically records the immutable resolution, advances expected sequence to N+1 once and exposes the skipped position in audit/status

### Requirement: Dead-letter and redelivery SHALL be explicit operator controls

A dead-letter record SHALL preserve message/envelope identity, owner and consumer,
correlation/causation, operation/process links, payload digest only, ordering
position, attempt/lease history summary, policy reference, terminal safe reason,
dead-letter time and eligible recovery actions. Payload bytes, credentials,
tracebacks and raw idempotency keys SHALL not enter list/status projections.

Redelivery SHALL require a new audited `RedeliverMessage` command with trusted actor,
reason, exact dead-letter identity, expected dead-letter generation and finite
deadline. It SHALL create a new delivery attempt linked to the immutable original;
it SHALL NOT modify the original payload, rewind a business aggregate, refetch an
external source or combine `redeliver_message` with
`replay_immutable_input`/`request_new_external_interaction`.
For an ordered message, redelivery SHALL preserve the original epoch and sequence and
SHALL NOT itself resolve the head. `ResolveOrderingGap` SHALL be a separate audited
owner-authorized command with the ordering restrictions above; generic DLQ operators
SHALL not receive implicit skip authority.

#### Scenario: An operator inspects the DLQ
- **WHEN** an authorized Platform query lists dead letters
- **THEN** it returns bounded safe metadata and explicit recovery eligibility without payload content or automatic side effects

#### Scenario: Two operators redeliver concurrently
- **WHEN** both commands target the same dead-letter generation
- **THEN** compare-and-swap admits at most one new attempt and the other receives the existing receipt or a stable generation conflict

#### Scenario: A source must be contacted again
- **WHEN** recovery requires a fresh provider interaction rather than delivery of the original immutable message
- **THEN** Platform rejects `RedeliverMessage` for that purpose and requires the owning Process/Capture command path

### Requirement: Compatibility EventBus SHALL remain a bounded bridge

The implementation SHALL introduce Platform delivery beside the current
`trade_py.bus.EventBus`; it SHALL not reinterpret legacy `event_log` and
`event_handler_runs` rows as complete formal outbox/inbox/ordering records. A
compatibility adapter MAY map proven legacy admission/status facts lossily, delegate
selected legacy topics, or mirror a message only with a durable one-way bridge
identity and payload-digest comparison.

Current topic names, CLI behavior, Web event status and replay behavior SHALL remain
unchanged until their named interface/process children pass snapshots. Business
`Topic` constants, DAG handler factories and `pipeline_dag` orchestration SHALL not
move into Platform Events. Adoption by current runtime owners SHALL remain blocked on
`runtime-owner-shutdown-and-recovery-hardening-v1`.

#### Scenario: A legacy event is mirrored
- **WHEN** an approved bridge maps one legacy event to a Platform envelope
- **THEN** it records one bridge identity and digest so restart cannot create another effective Platform message

#### Scenario: A legacy row lacks formal metadata
- **WHEN** actor, canonical command fingerprint, ordering or payload-proof facts are absent
- **THEN** the adapter exposes a legacy observation or unknown field and does not fabricate a formal receipt or ordering guarantee

#### Scenario: EventBus shutdown is still on the legacy runtime
- **WHEN** the Platform delivery implementation exists but shutdown hardening is not strictly approved and implemented
- **THEN** current EventBus/Web/CLI owners remain on their legacy behavior and the foundation is not reported as a shutdown fix
