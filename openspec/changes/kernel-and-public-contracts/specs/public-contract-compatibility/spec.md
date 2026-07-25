## ADDED Requirements

### Requirement: Public contract serialization SHALL be deterministic, exact and bounded

Each public wire DTO SHALL declare schema name and positive integer schema
version. Canonical JSON SHALL use UTF-8, sorted keys, compact separators,
finite values, UTC `Z` instants and explicit enum strings. Version 1 decoders
SHALL accept exactly version 1 and the exact declared field set. Unknown fields,
unknown versions, excessive depth/items/string/bytes or unregistered owner
payloads SHALL fail before object construction or digest calculation.

Version 1 SHALL limit raw UTF-8 input before parsing and canonical output each
to 65,536 bytes; one string/key to 2,048 UTF-8 bytes; every container to 100
items/members; aggregate array items plus object members to 1,024; total value
nodes to 2,048; nested-container depth to 8; actor scopes to 32; delegation hops
to 8; process history to 50; recovery descriptors to 16; and each safe error
message/hint to 1,024 UTF-8 bytes. Integer tokens SHALL be canonical
non-negative decimal `0` or `[1-9][0-9]{0,18}` before field-specific limits.

Before JSON materialization, a deterministic single-pass lexical/structural
scanner SHALL validate strict UTF-8/JSON grammar and enforce BOM, duplicate-key,
surrogate, float/non-finite, depth, decoded string/key, integer-token,
per-container, aggregate-member and value-node limits. Only accepted input MAY
reach the JSON decoder. A bounded `parse_int` hook SHALL repeat the 19-digit
limit before integer construction. Root scalar depth is zero, root container
depth is one, and each nested container increments depth by one. A value node
is the root or an object value/array element; keys consume string/member budgets
but are not extra value nodes. Post-materialization traversal SHALL use a
bounded explicit worklist rather than unbounded recursion. Python 3.10 and the
highest supported version SHALL emit the same structural error for the reviewed
deep, integer and duplicate-key fixtures.

#### Scenario: The same DTO is serialized twice
- **WHEN** code, contract version and values are identical
- **THEN** canonical bytes and digest are identical across source, editable-install and wheel-install environments

#### Scenario: A producer sends an additive unknown field
- **WHEN** a version 1 consumer receives a field absent from the exact version 1 schema
- **THEN** it rejects the payload and requires explicit version negotiation rather than silently discarding the field

#### Scenario: A payload exceeds a limit
- **WHEN** a payload exceeds any byte, depth, string, collection or history bound
- **THEN** decoding fails with a bounded structural error and no partial object is returned

#### Scenario: Input expands or aliases during parsing
- **WHEN** raw input is 65,537 bytes, repeats an object key, contains an invalid surrogate or exceeds the aggregate node/member budget while each local container is valid
- **THEN** decoding fails deterministically before DTO construction or fingerprint calculation

#### Scenario: Parser materialization would amplify input
- **WHEN** a payload contains depth 9, depth 1,500 or a within-byte-budget overlong integer token
- **THEN** the pre-materialization scanner rejects it with the same bounded structural error on Python 3.10 and the highest supported version, without `RecursionError` or large-integer construction

### Requirement: Source, editable and wheel installations SHALL expose compatible packages

Implementation SHALL prove an additive package-discovery configuration that
keeps distribution `trade-py`, the root `./trade` facade, `trade-py` console
entry and installed `trade_py` imports while adding installed `trade` contract
imports. The proof SHALL cover source tree, editable install and a clean wheel
installation. A root shim, symlink or test/notebook `sys.path` mutation SHALL
NOT be used.

The proof SHALL build one wheel, inspect that artifact's members and install
the same artifact into an isolated temporary environment with dependency
installation disabled and network access denied. The clean-wheel interpreter
SHALL run outside the repository cwd. Source/editable checks SHALL reuse the
locked development environment and SHALL NOT reinstall the full dependency
graph per mode. This additive proof SHALL NOT move legacy packages or replace
the later canonical package-layout child.

#### Scenario: Dual-root packaging cannot be proven
- **WHEN** the current build backend cannot produce both installed packages without broader migration or import ambiguity
- **THEN** this child stops before contract implementation and promotes the package-transition ADR rather than shipping source-only imports

#### Scenario: A wheel is installed in a clean environment
- **WHEN** the built wheel is installed without the repository working directory on `sys.path`
- **THEN** `--no-deps` offline installation succeeds, wheel members include both intended package roots, legacy `trade_py` compatibility imports and new `trade` contract imports succeed, and the console entry remains unchanged

### Requirement: Legacy mappings SHALL be explicit, one-way and conservative

Each compatibility mapper SHALL name source owner/version, target schema
version, lossiness, snapshot, retirement condition and refusal behavior. The
legacy layer MAY import target contracts; Kernel, Platform contracts and
Processes contracts SHALL NOT import `trade_py` or `trade_web`. Mappers SHALL
be pure and SHALL NOT call repository mutation APIs or SQL mutation
primitives, access `db._conn`, call providers, move pointers, signal processes
or repair state.

Mappers SHALL be split by legacy owner (`bus_contracts`,
`job_run_contracts`, `observatory_contracts`, `runtime_contracts`). No
`public_contracts` aggregate facade, package-level mapper re-export or
cross-owner mapper import SHALL be introduced.

The reviewed outcome matrix SHALL be exhaustive. EventBus `accepted`,
`saturated`, `shutting_down` and `submission_failed` all retain the durable
legacy event identity while describing handler-admission facts; none proves
handler/process completion. Web command `saturated` and pre-persistence
`stopping` have no operation receipt; `persistence_failed` has no durable
operation ID; post-persistence `stopping` and `spawn_failed` may carry a legacy
`run_id` but require separately observed job-row evidence for a target terminal
state. Outcome spelling, process exit, PID or exception text alone SHALL NOT
prove cancellation or completion.

The EventBus mapper SHALL retain total, accepted, saturated, shutting-down and
submission-failed handler counts plus a deterministic latest-at-most-50
bounded summary with returned/omitted metadata. A mixed handler set SHALL remain
explicitly mixed and SHALL NOT be flattened to the aggregate precedence value.
If the target schema cannot preserve these facts, the canonical mapping SHALL
be refused while the unchanged legacy result remains available.

Current `/api/run` accepted output SHALL map only to an owner-local legacy run
admission observation. A durable `run_id` alone SHALL NOT fabricate a formal
`OperationReceipt`: trusted actor, correlation/causation and versioned command/
idempotency fingerprints must come from the same future admission boundary.

#### Scenario: A legacy value is lossy
- **WHEN** a legacy event, artifact, error or status contains fields that are unsafe or have no target semantics
- **THEN** the mapper records the loss, excludes those fields and never broadens authority or certainty

#### Scenario: A target package is imported
- **WHEN** tests import all public target modules in a clean interpreter
- **THEN** no legacy implementation, Web, database, provider, pandas or native module is imported as a side effect

#### Scenario: A compatibility package is imported
- **WHEN** an owner mapper or its package is imported
- **THEN** only that owner mapper's reviewed target dependencies are visible and no aggregate mapper facade is exported

#### Scenario: The Web runner stops at different admission stages
- **WHEN** two `stopping` results differ because one has no `run_id` and one has a durably created `run_id`
- **THEN** the mapper produces no receipt for the first and requires separately observed durable state for the second rather than treating both as cancelled

#### Scenario: Event handlers have mixed admission outcomes
- **WHEN** one durable event has accepted, saturated and submission-failed handler results
- **THEN** the canonical observation retains each count and bounded summary or refuses mapping; it does not report all handlers as the aggregate precedence outcome

### Requirement: Current mutation and status surfaces SHALL retain snapshots until owning migration

This child SHALL preserve current EventBus admission behavior, `job_runs`
status rows, `/api/run` routes/status/headers/payloads, CLI event/run output and
exit behavior, Observatory errors/artifacts, SSE, database schema, artifact
layout and shutdown exceptions. New DTOs or mappers SHALL NOT reroute a current
interface. A later owning child SHALL provide a mutation ledger, old/new
snapshot parity and rollback before delegation.

#### Scenario: Web run is accepted
- **WHEN** the existing `/api/run` route accepts a command
- **THEN** its current 200 payload including accepted, target, limit, PID, run ID and running status remains byte/field compatible while the mapper yields only a legacy admission observation and no fabricated formal receipt

#### Scenario: Web run is saturated
- **WHEN** current command admission returns saturated
- **THEN** the existing 503 body and `Retry-After` header remain compatible and the canonical mapper does not advise duplicate work

#### Scenario: CLI event observation times out
- **WHEN** the admitted event does not settle within the configured legacy wait
- **THEN** existing output and exit behavior remain unchanged in this child while the canonical mapping records accepted command plus not-observed completion, not completed success

### Requirement: Compatibility shall not invent formal owner facts

Compatibility mapping SHALL NOT convert a legacy path-bearing artifact into a
formal Capture/Dataset reference, convert a job run into a Process Manager
record, infer trusted actor from payload, infer cancellation from process exit,
infer availability from an empty list, or infer error safety from free-form
exception text.

#### Scenario: A legacy terminated row lacks intent evidence
- **WHEN** only status `terminated` and a process exit are available
- **THEN** the mapper preserves the legacy value and produces conservative failed/unknown target observation without claiming actor-authorized cancellation

#### Scenario: A runtime shutdown error is free-form
- **WHEN** an existing exception says shutdown is incomplete but exposes no structured residual counts
- **THEN** the mapper emits a safe incomplete/unknown receipt or error and does not parse text into fabricated ownership evidence

#### Scenario: A legacy artifact has a path and digest
- **WHEN** Observatory supplies a relative path-bearing artifact, including traversal/absolute paths or a one-byte digest mismatch fixture
- **THEN** mapping yields only a closed `LegacyArtifactObservation` resolution/verification state and never existence, authorization, publication, quality, PIT or formal-reference proof

### Requirement: Public wait and control compatibility SHALL expose finite observation

New SDK and future Interface consumers SHALL observe operations through finite
receipt/process queries rather than holding a CLI, Web router, scheduler or
event handler for the full business workflow. Legacy synchronous waits may
remain only behind documented compatibility adapters until their owning
interface child migrates them. They SHALL NOT become defaults for new APIs.

#### Scenario: A new interface submits long-running work
- **WHEN** the command is durably accepted
- **THEN** the interface returns an OperationReceipt promptly and offers a finite ProcessView observation path instead of synchronously owning the whole workflow

#### Scenario: A non-returning worker exceeds observation deadline
- **WHEN** a worker or review task does not produce a newer state before the caller deadline
- **THEN** the caller receives not-observed/unavailable state and retains a correlation/operation/process link; the worker does not retain the caller indefinitely

### Requirement: Runtime contract adoption SHALL be separately hardened

This child SHALL NOT reroute EventBus, Web resources, RuntimeCommandRunner,
FastAPI lifespan, CLI or HTTP behavior through the new shutdown/control
contracts. Adoption SHALL require the independently reviewed
`runtime-owner-shutdown-and-recovery-hardening-v1` child. That child SHALL
preserve current CLI output/exit and HTTP payload/status snapshots while proving
bounded terminal-persistence retry, monotonic observation, concurrent stop,
startup cleanup, executor tail, owner-generation takeover and real Uvicorn
signal-to-process-tree reap.

#### Scenario: Contract DTOs exist before runtime hardening
- **WHEN** this child has shipped but the named runtime hardening child has not passed its gates
- **THEN** current runtime code remains on its compatibility path and architecture guards reject premature adoption
