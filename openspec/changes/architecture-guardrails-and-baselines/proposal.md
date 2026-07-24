## Why

The approved `restructure-trade-architecture-v1` design cannot be implemented
incrementally unless new target modules are prevented from recreating the
current cross-domain import and database-access patterns. Current code facts
include a central `TradeDB` schema facade, direct legacy imports from CLI/Web
surfaces, a legacy C++ module named `trade_py`, and a BTC compatibility pointer.

This child establishes a small, source-only safety net before any Kernel,
Platform, Context, package-layout, or schema work starts. It records the
audited legacy facts without treating them as target conformance, so later
children can replace one owner at a time with evidence and rollback.

## What Changes

- This delivery completes Task 2.1 only: add a versioned
  `architecture-baseline.toml` and a standard-library, directly callable,
  source-only baseline validator. It records current Python
  package roots, multi-source schema provenance, table classifications,
  warehouse artifact/pointer/receipt facts, Capture migration facts, native
  binding facts, source-derived CLI/HTTP/OpenAPI/SSE compatibility facts, and
  compatibility-pointer facts.
- Validate the baseline against source text only; it will not open a database,
  inspect a parquet file, import an application module, or mutate a data root.
- Add focused architecture tests with temporary source trees and a current-tree
  baseline validation fixture. The fixtures deny database/parquet connections,
  direct in-repository data/artifact reads, and out-of-repository reads while
  the checker runs.
- Keep Capture-risk and dynamic-DDL inventories closed to their reviewed Task
  2.1 bindings; a later owning child must extend the binding and review
  evidence before making a new temporal or dynamic-SQL claim.
- Task 2.2 will add prospective `src/trade` dependency and ownership checks.
- Task 2.3 will extend the shared quality-scope contract so the planner preserves
  canonical unfiltered delta, rename, and requested-filter metadata. The
  architecture contributor will use that metadata, rather than rediscovering Git
  state, to emit a fail-closed `architecture.partial_scope` quality result.

There are no breaking user-facing CLI, HTTP, Web, SDK, notebook, database,
artifact, C++ ABI, runtime, or existing developer-command behavior changes.
Task 2.1 is library-only: callers can invoke the baseline validator directly.
`trade dev check` integration and its deterministic architecture quality
failure are deferred to Task 2.3.

The target filesystem and import namespaces are independently frozen for this
guard: `target_source_root = "src/trade"` and `target_import_root = "trade"`.
The AST checker does not import that package; the package-layout child remains
responsible for distribution metadata and console-entry compatibility.

## Capabilities

### New Capabilities

- `architecture-static-guardrails`: Source-only enforcement for future target
  module imports, contract/domain boundaries, table ownership, read-only
  interface behavior, Platform business-vocabulary isolation, legacy escape
  prevention, and native loading boundaries.
- `architecture-baseline-inventory`: A versioned, auditable legacy inventory of
  package, schema-source, table, artifact, receipt, pointer, Capture-risk,
  native-binding, and compatibility facts used to scope future migration
  changes.

### Modified Capabilities

- None.

## Impact

This delivery affects the source-only validator, neutral TOML compatibility
reader, focused pytest coverage, and the versioned baseline file. Task 2.1
does not alter the shared quality scope/model/planner contract, contributor
registry, `trade dev check` plan/report output, or filtered `--path` behavior.
Task 2.3 will add one read-only architecture step when its scoped trigger and
canonical delta design are implemented; it will fail closed when `--path`
excludes an architecture-sensitive delta.

Design-quality governance applies. Task 2.1 provides deterministic direct
validator findings and bounded source inspection. The planned Task 2.3
developer-command contract will add scope metadata, a diagnostic envelope,
failure modes, remediation paths, and bounded executor batches. Application
runtime concurrency does not change. Persistent writes, schema work,
point-in-time semantics, predictive behavior, and external-event ingestion do
not apply: this child uses only repository source text and temporary test
trees.
