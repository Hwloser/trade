## ADDED Requirements

### Requirement: DatabaseRuntime SHALL separate observation, writing and migration authority

Platform Persistence SHALL expose one framework-free `DatabaseRuntime` capability
with closed startup modes `read_only`, `compatible_writer` and
`migration_leader`. `read_only` SHALL open an existing database without creating a
directory, database, table, index, migration row, default setting or seed and SHALL
enforce query-only behavior at the connection and transaction boundary.
`compatible_writer` SHALL require a durable writer lease and a schema capability
generation inside its declared read/write range but SHALL NOT execute DDL or seed
state. `migration_leader` SHALL require an exclusive migration lease, a monotonically
increasing fence generation and an explicit `MigrationPlan`.

Every runtime handle SHALL identify database identity, owner namespace, startup mode,
schema generation, minimum/maximum readable and writable generations, owner instance,
fence generation, lease expiry and capability digest. A handle SHALL NOT expose a
raw connection through a public contract. A stale, expired or mismatched writer fence
SHALL be rejected in the same transaction before an owner write.

#### Scenario: A status command opens a database
- **WHEN** a query-only CLI or HTTP capability starts in `read_only` mode
- **THEN** no filesystem or schema state is created or changed and an absent database is reported as explicitly unavailable rather than initialized

#### Scenario: A supported writer starts
- **WHEN** the current schema generation is within the binary's declared write range and the runtime acquires the current writer fence
- **THEN** `compatible_writer` returns an owner-bound handle without running migrations or seeds

#### Scenario: A stale process writes after a fence changes
- **WHEN** a transaction presents writer generation N after generation N+1 has taken over
- **THEN** the transaction is rolled back with a stable stale-writer reason and no state, audit, inbox or outbox row is committed

### Requirement: Owner transactions SHALL be local and atomically durable

Platform Persistence SHALL provide an owner-bound local transaction primitive.
Within one physical database transaction, one owning Context or Platform component
MAY commit only:

1. its own aggregate or technical state transition;
2. its own immutable transition receipt or audit record;
3. a consumer inbox receipt and ordered-consumer head when it is consuming a
   message; and
4. zero or more Platform outbox records emitted by that same transition.

The transaction SHALL carry owner namespace, transaction identity, operation,
correlation and causation identities, writer fence generation, finite monotonic
deadline and an explicit outcome. The same transaction SHALL verify every referenced
outbox payload digest before commit. Platform SHALL provide connection, transaction,
lock and rollback primitives only; business SQL, repositories, aggregate invariants
and context migrations SHALL remain with the owner. No transaction SHALL write two
business Contexts, call a provider, wait on message delivery or span another
database.

#### Scenario: A Capture transition emits an event
- **WHEN** a future Capture repository commits a capture receipt and
  `CaptureCommitted`
- **THEN** Capture state, its owner receipt and the immutable Platform outbox record become visible together or all remain absent

#### Scenario: A cross-Context write is requested
- **WHEN** one transaction attempts to update both a Datasets table and a Studies table
- **THEN** owner-scope validation rejects the transaction rather than expanding the Platform primitive into a shared business unit of work

#### Scenario: The process dies after commit
- **WHEN** the owner transaction commits and the process exits before a dispatcher observes its outbox row
- **THEN** the committed owner state and outbox remain recoverable and no caller must repeat the owner transition to recreate the message

### Requirement: MigrationCoordinator SHALL run owner registrations under durable capability fencing

`MigrationCoordinator` SHALL accept versioned `MigrationRegistration` values from
Platform and each extracted Context. A registration SHALL identify owner, migration
ID and digest, from/to schema capability, dependencies, additive or new-generation
strategy, backward/forward compatibility, checkpoint store, idempotency key,
validation, cutover and rollback entrypoints. Platform SHALL order registrations by
declared dependencies and SHALL NOT contain their business SQL.

The coordinator SHALL persist migration lease, plan digest, current step,
checkpoint, capability-before/after, compatibility range, status, safe error,
timestamps and audit receipt. An interrupted migration SHALL resume only when its
registration digest and checkpoint agree. A changed registration under the same
identity, missing dependency, unknown generation, failed validation or incompatible
writer SHALL fail closed.

While any writer that does not participate in `DatabaseRuntime` fencing can access
the database, an incompatible migration SHALL be prohibited. Only an additive
backward/forward-compatible migration or an externally proven exclusive maintenance
window MAY proceed. A capability table alone SHALL NOT be treated as proof that an
unaware legacy binary is fenced.

#### Scenario: Two participating binary generations start concurrently
- **WHEN** two Platform-aware binaries request migration leadership for one database
- **THEN** exactly one fence generation owns the migration lease, the follower starts in a compatible mode or is rejected, and no migration step runs twice

#### Scenario: An old unaware writer may still be running
- **WHEN** the coordinator cannot prove that every legacy writer has stopped or joined the writer-fence protocol
- **THEN** it blocks an incompatible schema cutover with `legacy_unfenced_writer` and leaves the previous generation writable

#### Scenario: A migration crashes after a checkpoint
- **WHEN** a registration stops after checkpoint K but before capability activation
- **THEN** restart verifies the registration digest, preserves the old compatible reader and resumes idempotently from K without repeating destructive work

### Requirement: Legacy schema bootstrap SHALL be isolated and temporary

The first implementation SHALL place the only target-to-legacy schema bridge at
`src/trade/platform/persistence/adapters/legacy_schema_bootstrap.py`. The bridge
SHALL expose only `LegacySchemaBootstrapAdapter`, SHALL be callable only by target
Bootstrap in `migration_leader` mode, and SHALL import only the exact legacy
bootstrap symbols approved in `architecture-baseline.toml`. Its implementation child
SHALL add source-verified writer, transaction and compatibility proof before the
guard accepts the bridge.

The adapter SHALL delegate the existing `TradeDB` schema/bootstrap history, close
all temporary resources, validate the expected legacy migration generation and
record one compatibility receipt. It SHALL NOT become a query repository, general
database facade or import path for a Context or Interface. Each owner-transition
child SHALL replace only its own legacy registration after its repository, old/new
reader comparison and rollback source exist. The bridge SHALL be removed after the
last registration satisfies its retirement window.

#### Scenario: A new database is explicitly initialized
- **WHEN** Bootstrap selects migration mode with a reviewed legacy bootstrap plan
- **THEN** only the bridge invokes the legacy initializer, records the observed legacy capability and returns no business repository

#### Scenario: A query path attempts bootstrap
- **WHEN** an HTTP BFF, CLI status command or Context use case imports or invokes the bridge
- **THEN** architecture and runtime capability checks reject the call and require a read-only or owner repository handle

#### Scenario: One Context migrates its table
- **WHEN** a later owner child passes dual-read and rollback gates for one legacy table
- **THEN** only that registration stops delegating to the legacy bridge while unrelated legacy tables and readers remain unchanged

### Requirement: Platform persistence metadata SHALL have one logical owner

The eventual additive Platform schema SHALL make Platform Persistence the sole
writer of schema capabilities, migration runs/checkpoints, writer leases and
operation/idempotency admission metadata. Platform Events SHALL be the sole writer
of outbox routing/delivery, inbox receipt, ordering and dead-letter metadata.
Platform Backup SHALL be the sole writer of backup certification, restore operation
and activation-journal metadata.

Existing `event_log`, `event_handler_runs`, `schema_migrations`,
`backup_snapshots`, `settings`, `job_runs` and `pipeline_dag` SHALL remain legacy
compatibility state until separately cut over. This child SHALL NOT claim mixed
business rows in `settings`, `job_runs` or `pipeline_dag` merely because the physical
tables are shared. Other Contexts and Interfaces SHALL consume Platform queries,
receipts or projections and SHALL NOT query Platform tables or `_conn` directly.

#### Scenario: A Web route needs delivery status
- **WHEN** an Operations BFF requests event backlog and dead-letter status
- **THEN** it uses a bounded Platform query contract and does not read `event_log`, delivery or inbox tables directly

#### Scenario: A business setting is encountered
- **WHEN** a legacy `settings` row carries risk, Capture source or model semantics
- **THEN** Platform does not assume semantic ownership; the row stays behind the compatibility adapter until its owning child classifies and migrates it

#### Scenario: Physical SQLite remains shared
- **WHEN** Platform and a Context temporarily use the same SQLite file
- **THEN** each table and transaction still has one logical owner and no shared physical file is interpreted as cross-owner write authority
