## Why

Trade currently has useful SQLite event persistence, handler leases, bounded
channel admission, process ownership and backup commands, but their ownership is
spread across `TradeDB`, `trade_py.bus`, Web runtime containers, CLI entrypoints
and `scripts/backup.py`. In particular, constructing `TradeDB` changes schema and
seeds state, event completion persistence retries without a terminal bound,
multiple entrypoints assemble their own runtime graph, and restore extracts an
archive before manifest/member verification. Extracting Capture or another
Context on top of these paths would either duplicate these failure-prone
mechanics or create a new global database facade.

This child therefore freezes the generic Platform foundation required by the
strict-approved `restructure-trade-architecture-v1` design before any Context
owns a new durable command, outbox or cross-Context delivery. It is a
Non-trivial architecture change governed by the public-contract, storage,
migration and runtime-concurrency design-quality profiles.

## What Changes

- Define a framework-free Platform persistence API with explicit read-only,
  compatible-writer and migration-leader startup modes. Context repositories
  retain their own SQL, tables and migrations; Platform supplies only
  owner-bound connections, transactions, fencing and migration coordination.
- Define the same-database local transaction boundary that can atomically commit
  one Context transition, immutable owner audit/receipt, inbox acknowledgement
  and outbox records without exposing a cross-Context transaction.
- Define durable command ingress against the approved Kernel
  `CommandEnvelope`, `ActorContext`, idempotency fingerprint,
  `OperationReceipt` and `ErrorEnvelope` contracts, including its bounded
  three-claim-plus-one-audit transaction product and refusal outcome priority.
- Define generic outbox/inbox delivery, lease/ack recovery, bounded retry,
  dead-letter/redelivery, explicit unordered delivery and a durable
  `OrderingContract` for ordered streams. Stable consumer effect identity
  survives compatible binary upgrades; ordered dead letters cannot implicitly
  advance the head.
- Define an additive `DatabaseRuntime`, `MigrationCoordinator` and exact
  `LegacySchemaBootstrapAdapter` bridge so schema changes no longer occur as an
  implicit query-side effect. Existing `TradeDB` history remains intact until
  each owner migration replaces its registration.
- Define trust-policy-signed backup certification and the bounded streaming,
  staged, writer-fenced, generation-CAS `RestoreOperation` state machine.
  Untrusted, unsafe, corrupt, resource-exhausting or incompatible archives fail
  before fencing or activation.
- Define one target `trade.bootstrap` composition root and a versioned
  `CapacityEnvelope` so CLI, HTTP, worker, scheduler and later Context children
  share runtime capabilities and comparable 1x/10x evidence.
- Preserve all current CLI, HTTP, Web, EventBus, database, Parquet, SDK,
  notebook, scheduler and C++ behavior while this design is reviewed. Existing
  constructors remain compatibility shims until a later implementation slice
  passes its contract snapshots and cutover gate.
- Require the independent
  `runtime-owner-shutdown-and-recovery-hardening-v1` strict gate before current
  EventBus, Web resources, `RuntimeCommandRunner`, FastAPI lifespan or CLI
  daemons adopt the new shutdown/control contracts. This proposal does not
  claim the audited close-hang paths are fixed.

No production code, test code, runtime configuration, database, schema, data,
artifact, import path or execution behavior changes in this design round. The
eventual implementation is additive, uses temporary-root migration/restore
fixtures, and may not access real data except through a separately approved,
explicitly read-only probe.

## Capabilities

### New Capabilities

- `platform-persistence-runtime`: Owner-bound database runtime, local
  transaction/outbox primitive, migration coordination, schema capability
  fencing and the legacy schema bridge.
- `platform-message-delivery`: Durable command ingress, outbox/inbox,
  lease/ack, ordered delivery, bounded retry, dead-letter and audited redelivery.
- `platform-backup-restore`: Certified backup manifests, safe staged restore,
  activation/rebind health verification and crash reconciliation.
- `platform-bootstrap-capacity`: Sole target composition root, compatibility
  delegation and comparable bounded capacity evidence.

### Modified Capabilities

None. The parent architecture requirements remain authoritative; this child
adds implementation-ready capability specifications without modifying archived
or current production contracts.

## Impact

- Intended target paths for later implementation:
  `src/trade/platform/{contracts,api,persistence,events,backup,execution,settings}`,
  `src/trade/bootstrap/`, owner repository adapters and focused tests under
  `tests/{unit,integration,contract,architecture,golden}`.
- Audited compatibility sources:
  `trade_py/db/trade_db.py`, `trade_py/db/migrations.py`,
  `trade_py/bus/`, `trade_web/backend/runtime/`,
  `trade_web/backend/app.py`, CLI start/run/event/config/backup paths and
  `scripts/backup.py`.
- Affected future contracts: Platform startup capability, transaction/outbox,
  command admission, operation receipt, event delivery, ordering, dead-letter,
  migration, backup/restore, bootstrap capability and capacity-result schemas.
- Current public CLI/API payloads, database files/tables, artifact layouts and
  C++ interfaces remain unchanged. Compatibility adapters are removed only
  after named consumers pass a minimum 30-day compatibility window.
- Eventual schema work is additive and reversible: preserve the legacy
  `schema_migrations` history and prior active generation, use checkpointed
  replay or shadow copy, dual-read comparisons, verified backup, readiness
  gates and generation rollback. No table deletion or bulk data migration is
  authorized by this proposal.
- Implementation prerequisites are the merged strict-approved
  `kernel-and-public-contracts` artifact at portable digest
  `sha256:1d06f033c231ce22d6abe164a1ed1f8fc553de54762f33a46882f4b4391b1f4f`
  and the already merged architecture guardrails. Any Kernel artifact drift
  requires renewed compatibility review. Production runtime activation
  additionally requires the strict shutdown/recovery hardening child.
