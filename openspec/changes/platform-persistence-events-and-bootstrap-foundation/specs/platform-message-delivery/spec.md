## ADDED Requirements

### Requirement: Command ingress SHALL durably admit one operation identity

Platform command ingress SHALL accept only the approved framework-free Kernel
`CommandEnvelope[T]` carrying a trusted `ActorContext`, correlation and causation
identities, canonical command fingerprint, scoped idempotency fingerprint and finite
monotonic deadline. The owning ingress repository SHALL use the approved bounded
key-generation admission algorithm and one local transaction to either:

- create one immutable operation identity plus initial `OperationReceipt` and owner
  command outbox record;
- return the existing receipt for a command-equivalent duplicate; or
- return the approved conflict, corrupt-claim, contention or audit-unavailable
  `ErrorEnvelope` without fabricating an operation.

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
outgoing outbox. The inbox identity SHALL be `(consumer_namespace,
consumer_contract_version, message_id)` and SHALL store the immutable payload digest
and first effective receipt.

An exact duplicate SHALL return the existing receipt without invoking the transition.
A duplicate identity with another payload digest, target or schema SHALL be
quarantined as corruption. Handler code SHALL not acknowledge outside the owner
transaction and SHALL not embed cross-Context orchestration.

#### Scenario: An exact message is delivered twice
- **WHEN** the same consumer receives the same message ID and payload digest after the first local commit
- **THEN** the second delivery returns the recorded inbox receipt and performs no owner write or child emission

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

#### Scenario: N+1 arrives before N after restart
- **WHEN** durable expected sequence is N and N+1 is delivered first
- **THEN** the consumer records a bounded gap outcome, does not invoke the owner transition for N+1 and waits/reconciles only until the contract's finite limit

#### Scenario: A producer restarts with a stale epoch
- **WHEN** an outbox append presents an epoch lower than the durable producer fence
- **THEN** the append is rolled back and no duplicate sequence enters delivery

#### Scenario: A gap exhausts its policy
- **WHEN** N never arrives before the finite gap timeout or buffer budget
- **THEN** the head-of-line policy deterministically dead-letters or requests audited reconciliation without silently applying N+1

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
