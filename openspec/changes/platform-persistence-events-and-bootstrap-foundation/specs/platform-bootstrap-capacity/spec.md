## ADDED Requirements

### Requirement: Target Bootstrap SHALL be the sole production composition root

The target architecture SHALL have one `trade.bootstrap` package as the only
production module allowed to assemble concrete Platform implementations, owner
repositories/adapters, Context use cases, Process managers and Interface
handlers. It SHALL expose explicit application profiles for `cli`, `http`,
`worker`, `scheduler` and `migration`; each profile SHALL declare required and
optional capabilities, database startup mode, owner instance/fence, lifecycle
order and compatibility selections.

Bootstrap SHALL compose through framework-free contracts and shall not contain
business rules, provider interaction, SQL, artifact parsing, process flow or
HTTP/CLI response shaping. Contexts, Processes and Interfaces SHALL not
instantiate `TradeDB`, EventBus, provider clients, native bindings, scheduler
loops or Platform adapters directly. Platform SHALL not import Bootstrap.

#### Scenario: The HTTP profile starts
- **WHEN** an HTTP application requests its Bootstrap profile
- **THEN** Bootstrap constructs one owned capability graph and Interface handlers receive only query/use-case/process/status handles

#### Scenario: A CLI query needs read-only state
- **WHEN** a status/show command selects the CLI read profile
- **THEN** Bootstrap supplies a read-only database/query capability and does not run migrations, seeds, EventBus or a worker runtime

#### Scenario: An Interface constructs an adapter
- **WHEN** a target HTTP, CLI, SDK, schedule or event module instantiates a database, EventBus or provider implementation
- **THEN** architecture checks reject the second composition root and require an injected Bootstrap handle

### Requirement: Bootstrap lifecycle SHALL be explicit, fenced and truthful

Every Bootstrap profile SHALL have a closed lifecycle `new`, `starting`,
`ready`, `stopping`, `stopped` or `failed` and one immutable generation. Startup
SHALL acquire database/runtime capabilities before constructing dependents,
record each acquired owner and unwind only those resources in reverse order
after a partial failure. Readiness SHALL require every mandatory capability and
shall identify optional degraded capabilities separately.

Shutdown adoption SHALL use the approved Kernel control/deadline/receipt
contracts only after `runtime-owner-shutdown-and-recovery-hardening-v1` is
strictly approved and implemented. Until then, Bootstrap compatibility profiles
SHALL delegate current lifecycle behavior without mapping free-form exceptions
to fabricated `ShutdownReceipt` facts. A profile SHALL not report `stopped`
while it retains a live process group, executor task, Python thread,
persistence audit, writer lease or inflight start.

#### Scenario: Startup fails after database acquisition
- **WHEN** a later mandatory capability fails during construction
- **THEN** Bootstrap closes admission and unwinds only the acquired generation in reverse order, preserving explicit residual ownership if cleanup cannot finish

#### Scenario: A shutdown deadline expires
- **WHEN** an adopted owner cannot drain, persist terminal state or release its fence before the shared deadline
- **THEN** Bootstrap returns incomplete/deadline-exceeded with retained fence and residual owner rather than closing the database or claiming success

#### Scenario: Shutdown hardening is not ready
- **WHEN** a current Web, EventBus, command-runner or daemon profile is requested before the hardening child completes
- **THEN** Bootstrap keeps the legacy compatibility lifecycle selected and does not advertise formal shutdown guarantees

### Requirement: Compatibility shims SHALL delegate one selected profile

Existing `trade_py` and `trade_web` entrypoints SHALL remain public compatibility
surfaces. Each migrated entrypoint SHALL delegate to exactly one named Bootstrap
profile through a thin shim while preserving existing arguments, environment
selection, exit/status semantics, route/payload/SSE shape and lifecycle timing.
An entrypoint that has not passed its compatibility child SHALL remain on the
legacy path.

There SHALL be no simultaneous independent Web/CLI/scheduler containers for one
logical owner generation. A global legacy EventBus facade MAY point to the
Bootstrap-owned bus during the compatibility window but SHALL not create or
replace it. Removal requires a consumer inventory, compatibility snapshots,
minimum 30-day window and a rollback selection.

#### Scenario: FastAPI delegates after approval
- **WHEN** the HTTP compatibility slice passes route/OpenAPI/SSE and lifecycle fixtures
- **THEN** `create_app` obtains its runtime handles from one Bootstrap HTTP profile without changing existing public responses

#### Scenario: A CLI command is not migrated
- **WHEN** its argument/exit/data compatibility evidence is incomplete
- **THEN** the command keeps its current construction path and cannot partially mix legacy and target owners

#### Scenario: Two shims request the same owner
- **WHEN** concurrent entrypoints would create separate writable runtimes for one data root
- **THEN** owner fencing admits one generation and returns an explicit conflict/unavailable result to the other

### Requirement: CapacityEnvelope SHALL make every 1x and 10x gate comparable

Platform SHALL define a versioned, deterministic `CapacityEnvelope` for
measurement results. It SHALL contain fixture identity/digest, scenario,
cardinality and duration, source/credential/stream shape when applicable,
binary/config generation, concurrency, worker and queue limits, runner CPU/
memory/disk profile, monotonic start/finish/duration, latency percentiles,
admission/rejection/dead-letter counts, SQLite transaction/lock/write time,
scan bytes/files, CPU/memory/disk peaks, backlog/recovery time, threshold policy
reference and closed result `pass`, `defer` or `overload`.

Each child SHALL reserve named finite limits before activating a capability.
Illustrative parent values, host CPU count or existing ThreadPool defaults SHALL
not silently become production limits. The envelope SHALL record measurement
failure as unavailable rather than a zero. Sampling, if required, SHALL be
deterministic and digest-bound.

#### Scenario: A 10x delivery run completes
- **WHEN** Platform Events processes the declared 10x fixture
- **THEN** the result records the exact backlog, worker/queue limit, SQLite contention, latency, resource peaks and recovery outcome under one comparable envelope

#### Scenario: Resource telemetry is unavailable
- **WHEN** peak memory or disk measurement cannot be observed
- **THEN** the envelope reports the field unavailable and the gate defers rather than filling zero or passing on incomplete evidence

#### Scenario: A child proposes a larger limit
- **WHEN** Capture, Datasets or an Interface changes a reserved queue, batch or concurrency limit
- **THEN** it supplies new 1x/10x envelopes and a reviewed threshold policy before cutover

### Requirement: Platform capabilities SHALL remain technical and bounded

The Bootstrap capability registry SHALL contain explicit technical capabilities
for persistence, events, execution, scheduling, settings, backup and status
queries. Platform source and contracts SHALL not introduce BTC, Kline, provider,
Dataset, Study, recommendation, portfolio, news or other business aggregate
vocabulary. Business command/event payloads remain owner contracts and are
opaque to generic routing except for schema identity and digest verification.

Queues, worker pools, query windows and status histories SHALL be finite.
Capability queries SHALL distinguish empty, partial, stale, blocked,
not-observed, unavailable and unknown and SHALL not start work or repair state.
Execution adapters MAY own subprocess/process-tree mechanics only behind an
approved Platform port; business use cases and Interfaces SHALL not create
processes directly.

#### Scenario: Platform routes a Capture event
- **WHEN** a future `CaptureCommitted` envelope enters delivery
- **THEN** Platform validates generic envelope metadata/digest and routes it without importing Capture domain or interpreting source semantics

#### Scenario: Operations queries capacity
- **WHEN** an Interface requests Platform status
- **THEN** the bounded query returns technical lifecycle/backlog/capacity evidence and performs no replay, migration, restore or data repair

#### Scenario: A business name enters Platform
- **WHEN** a proposed Platform module or table encodes a BTC-, Dataset- or Study-specific rule
- **THEN** architecture review rejects it and moves the rule to its owner Context or Process contract

### Requirement: Foundation activation SHALL require explicit prerequisites

Foundation implementation SHALL begin only after this change has current
digest-bound six-role approval and strict design-check, and after the strict-approved
`kernel-and-public-contracts` change is merged or otherwise present at the exact
reviewed artifact generation. Architecture guardrails SHALL be updated with the
exact target legacy bridge and Platform table bindings before those paths write.

No current runtime may adopt formal owner shutdown/control behavior before
`runtime-owner-shutdown-and-recovery-hardening-v1` passes strict approval and its
persistence-retry, concurrent-stop, startup-cleanup, executor-tail, monotonic,
crash-takeover and real signal-to-reap implementation fixtures. No Context may
emit formal outbox or accept formal ingress until Platform transaction, inbox,
lease recovery and duplicate/crash fixtures pass.

#### Scenario: Kernel contracts are not merged
- **WHEN** an implementation PR cannot consume the exact approved Kernel contracts
- **THEN** it remains blocked and does not recreate local IDs, envelopes, actors, receipts or errors

#### Scenario: A Context wants to migrate first
- **WHEN** Capture or another Context proposes a durable command/outbox before the Platform foundation passes its implementation gates
- **THEN** the Context child is blocked rather than inventing another transaction, event or runtime substrate

#### Scenario: Design artifacts drift
- **WHEN** implementation changes a reviewed contract, state machine, table owner or capacity policy
- **THEN** the design digest is invalidated, the affected roles re-review it and strict approval is regained before code continues
