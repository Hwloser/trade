## Context

`restructure-trade-architecture-v1` is strictly approved as a design-only
parent. Its first implementation prerequisite is a guard that protects
future target modules before any extraction starts. The current repository has
not yet adopted `src/trade`: Python behavior remains under `trade_py/`, Web
behavior under `trade_web/backend/`, and the central `TradeDB` constructor and
DDL sources define many current tables. Direct `TradeDB` imports remain in
CLI, jobs, and Web runtime code. These facts are intentionally not treated as
the target architecture.

The audit also found a native-binding collision risk: `engine/cmake/python_bindings.cmake`
defines a module named `trade_py`, while `trade_py/__init__.py` attempts to
import that name as its C++ probe. The BTC store retains a verified
compatibility pointer at `market/crypto/btc_current.json` alongside
`market/crypto/btc.parquet`. Both are current compatibility facts, not a
license to create further unowned pointers.

The source audit also identifies warehouse Parquet families in
`trade_py/data/warehouse/materialize.py`, Crypto ADS pointer and completion
receipt conventions in `trade_py/data/warehouse/crypto_store.py`, and the
Kline reconciliation `current.json` convention in
`trade_py/data/operations/checks.py`. These are migration inputs, not
artifacts inspected by this child. Capture migration inputs are likewise
source-only: `RawRecord` has one `published_at` field, while RSS, GDELT, and
warehouse RSS paths substitute collector/fetch time when provider publication
time is unavailable. Archive and date-only feeds infer noon timestamps;
GDELT streaming re-fetches a provider while reading/writing local state, and
WAL recovery has a distinct legacy meaning. Semantic quarantine also occurs in
the warehouse transformation rather than at transport admission. RSS catalogs
can be selected through environment overrides, while rights-policy evidence is
not currently present. The Capture child, not this guard, owns the correction.

The audit found further independent persistence and projection declarations
outside the initial central schema sources: `trade_py/intelligence/schema.py`
defines `feed_scores` and `source_configs`, while
`trade_py/observatory/catalog/store.py` defines the rebuildable
`catalog.sqlite` projection, `generation.json` pointer, and `catalog_meta`,
`runs`, and `releases` tables. Warehouse materialization writes
`ads_warehouse_validation_report` even though that table is not in the
hand-maintained required-table list. The source-only baseline must inventory
these facts by source producer without treating them as ownership approval.

The parent migration matrix assigns this first child bounded CLI, HTTP, OpenAPI,
and SSE baselines. The audit identifies the root `trade` facade and
`trade_py/cli/main.py` domain registry, `trade_web/backend/app.py`,
`trade_web/backend/runtime/router.py`, Observatory routers, and existing
CLI/Web contract tests as the source evidence. This child records those
surfaces only; the later `cli-http-sdk-compatibility` child remains responsible
for behavior snapshots, adapter delegation, and retirement decisions.

This child is Non-trivial because it changes the repository-wide developer
quality path and sets future architecture enforcement across tooling, tests,
and new target module roots. It performs no runtime/domain implementation,
module move, external request, database open, artifact read, artifact write,
or ownership transition.

## Goals / Non-Goals

**Goals:**

- Make the parent design's prospective dependency rules executable for new
  `src/trade` Python files without first repairing the entire legacy tree.
- Detect forbidden concrete Context imports, invalid internal Cell direction,
  contract type leakage, private database or artifact-client escape, direct
  interface SQL, Platform business vocabulary, legacy namespace, dynamic
  execution, process-spawn escapes, unauthorized table access, and native
  imports outside a Context `adapters/native` boundary.
- Record auditable current facts for Python package roots, schema-definition
  sources, physical table classifications, independent projection declarations,
  warehouse artifacts, pointers, receipts, Capture migration risks,
  CLI/HTTP/OpenAPI/SSE source surfaces, C++ binding targets, and the BTC
  compatibility pointer.
- Keep guard execution deterministic, offline, bounded, and free of runtime
  imports or data-root access.
- Give later child changes a stable baseline and a small prerequisite rather
  than authorizing a broad directory move.
- Specify the canonical scope metadata that Task 2.3 must add so filtered
  architecture checks can later fail closed without a second Git discovery
  path. Task 2.1 does not modify quality scope, planner, or contributor code.

**Non-Goals:**

- Moving `trade_py/` to `src/trade/`, extracting any Context, adding Kernel
  types, changing CLI/HTTP/Web behavior, or changing import paths.
- Replacing `TradeDB`, editing existing DDL, opening a SQLite
  database, reading parquet/raw artifacts, or assigning final ownership to
  deferred KG, causal, factor, or legacy recommendation records.
- Implementing Capture clocks, SourceManifest rights, provider-free replay,
  quality/quarantine semantics, a plugin system, a remote worker, or a native
  binding. This child records source facts and prohibits ungoverned loading;
  their implementation remains separately governed.
- Changing CLI/HTTP/Web/SDK/Notebook behavior, generating a behavior snapshot,
  adding an interface compatibility adapter, or delegating an existing route.
  This child records source-level interface inventory only.
- Building a generalized lint framework, adding a third-party dependency, or
  enforcing current `trade_py` imports as if legacy code already met the
  target graph.
- Enabling, renaming, or linking a native Python extension.

## Design Quality Brief

### Requirements and acceptance

Task 2.1 delivers a directly callable source-only baseline validator. It reads
only the reviewed baseline, declared source evidence, and bounded regular
production-Python descriptors; it returns deterministic path, line, rule,
remediation, ordered findings, and no partial producer inventory. It does not
import inspected application modules, generate a quality envelope, or register
a `trade dev check` contributor.

Acceptance for Task 2.1 requires baseline source/table/pointer claims to match
the current repository text; all mandatory Capture-risk and dynamic-DDL
inventories to be closed and non-authorizing; approved table-binding proof to
be static, exact, receiver-matched, and bound to one unique undecorated
module-level callable with no competing module binding. Schema v1 is closed:
only its reviewed top-level declarations are admitted, and non-table source
facts, Capture risks, dynamic limitations, interface records, and native
bindings cannot carry classification, adapter, or proof fields. The static SQL
must be the first positional argument of the recognized persistence call;
writer and transaction proof must be table-specific writes, while reader and
compatibility proof must be one read-only `SELECT` statement; writer and
transaction proof must be one write statement. The declared proof literal must
exactly equal that direct SQL argument, not merely occur elsewhere in the
adapter. A transaction receiver or explicit alias remains valid only until a
direct-scope binding or object-namespace mutation, or an unmodelled dynamic
call. An alias must be one local name; a receiver root or alias declared
`global` or `nonlocal` anywhere in the callable is not local and cannot
authorize proof. An external `as` alias invalidates the entire transaction
context, so the otherwise-local receiver root cannot supply substitute proof.
Imports and non-local assignment or deletion targets in the transaction block
invalidate proof. A proof is admitted
only from its straight-line callable prefix and a direct transaction `with`
block in that prefix; nested `with` blocks, branches, loops, exception
handlers, assertions, matches, and deferred comprehensions never authorize
persistence. Any module-scope import or executable call is a competing
callable-binding risk. A class declaration or PEP 695 type alias is also
rejected because its body, bases, decorators, metaclass, or type metadata can
execute or rebind at module definition time; an unrelated function declaration
is admitted only without definition-time metadata such as decorators, defaults,
annotations, or type parameters. The only other admitted top-level declarations
are plain literal constants and a module docstring; assignments with lambdas,
attribute/subscript targets, control flow, context managers, and every other
module execution form are rejected. SQL read matching recognizes `FROM` and
`JOIN` tokens outside comments, strings, and quoted alias text. The proof is bounded by
persistence-operation, retained-SQL-byte, AST-node, and AST-depth limits, and
every rejected relationship has a focused failing fixture.

Task 2.3 will make `trade dev check` architecture-aware. It will trigger for
the baseline, `src/trade/**/*.py`, declared evidence, guard/contributor/registry
paths, native bindings, and interface sources. It will add
`trade.architecture.guard.v1` output, `ScopeSelection` canonical delta/filter
metadata, bounded `ProducerDiscoverySelection`, partial-scope refusal, ordered
producer-prefilter/baseline/target batches, and a normal legacy-only
non-triggering path. Those command, planner, batching, and executor behaviors
are deliberately not asserted as delivered by Task 2.1.

### Ownership and boundaries

`trade_py/devtools/architecture_guard.py` owns parsing, validation, bounded
direct-validator findings, and baseline fact semantics for Task 2.1.
`trade_py/devtools/toml_compat.py` is its neutral framework-free TOML reader;
the compatibility re-export keeps existing quality imports stable without
loading the quality runner. Task 2.3 will assign
`trade_py/devtools/quality/models.py` and `scope.py` the additive canonical
unfiltered delta/filter and `ProducerDiscoverySelection` contracts; `planner.py`
will own conversion of excluded architecture-sensitive delta facts, including
producer signals, into a fail-closed plan issue. The future
`trade_py/devtools/quality/contributors/architecture.py` receives those
canonical selections and only constructs deterministic bounded producer-prefilter,
baseline, and target subprocess steps; it must not rediscover Git state. The
existing quality registry will own contributor registration.
`architecture-baseline.toml` is the authoritative declaration of audited source
facts. It separately freezes `target_source_root = "src/trade"` and
`target_import_root = "trade"`; the guard uses the latter for absolute and
relative import resolution without requiring target-package installation.
`tests/test_architecture_guard.py`, `tests/test_architecture_contributor.py`,
and focused scope/planner extensions own target-fixture coverage; no fixture
opens the application or a data root.

The guard has a deliberately narrow prospective boundary. It applies to
`src/trade` once a later child creates that root. It does not validate all
legacy `trade_py` dependencies. A later extraction must add its own target file
and satisfy this guard before it becomes a new architectural dependency.

### Data and state invariants

The baseline is source metadata, not a runtime catalog, a schema authority, or
an ownership-transfer mechanism. Each logical table records a current owner,
one-or-more source facts with `bootstrap`, `migration`, `alter`, or
`data_transform` role, an audit-only `candidate` or `deferred` classification,
semantic kind, target Context/defer reason, and required child. Neither
`candidate` nor `deferred` authorizes persistence access. Only a later
implementation child may add an explicit table `approved_binding` that names
one Context and one non-empty, dot-delimited persistence-adapter scope using
identifier segments beneath `<context>.adapters.` after proving writer, reader,
transaction, compatibility, and owner behavior. Artifacts and warehouse
producers remain candidate or deferred until a future child defines their
separate authorization contract. Each table proof is a distinct
`{ source, literal, callable }` record verified through the same
repository-confined source-only reader. Every proof is located in the named
`src/trade/<adapter_scope path>.py` implementation and in its named module-level
callable's direct lexical scope; nested function, async-function, lambda, and
class bodies do not count. Writer proof must be a static table-specific write
passed as the persistence call's first positional argument; reader and
compatibility proof must be one read-only `SELECT` passed in that same position.
The declared source literal must exactly equal that static SQL argument and a
mismatch points to the named adapter callable line.
Transaction proof must be a static table-specific write inside one direct
transaction `with` block using the same receiver or that context manager's
explicit local-name `as` alias, and that receiver identity and object namespace
must remain unmodified from transaction entry to the call. A receiver root or
alias declared `global` or `nonlocal` anywhere in that callable is rejected,
because its lifetime can outlive the proof's lexical scope; an external `as`
alias invalidates the full transaction context rather than leaving the receiver
root available for a substitute operation. Imports and non-local assignment or
deletion targets, as well as any unmodelled direct-scope call, invalidate
transaction receiver proof rather than being interpreted dynamically. Writer
and transaction proof reject a semicolon-separated second statement after
comments and values are masked. SQL table identifiers match at supported
statement positions with identifier boundaries, not by substring; read proof
tokenization ignores comments, strings, and quoted alias text before
recognizing `FROM` or `JOIN`. Proof collection accepts only a
straight-line callable prefix and a direct transaction `with` block, stopping
before a nested `with`, conditional, loop, exception, assertion, match, or
deferred comprehension. It rejects a decorated, duplicate, deleted, assigned,
imported, wildcard-imported, type-aliased, namespace-mutated, dynamically
executed, or otherwise rebound callable name; an approved-proof adapter rejects
every module-scope import and executable call because their namespace effects
cannot be proven safely, including definition-time expressions in unrelated
declarations. Its module body is an explicit admission whitelist: plain
functions without definition-time metadata, inert literal constant declarations,
and a docstring only. It fails closed above its callable operation,
retained-SQL-byte, AST-node, or AST-depth budget. Thus a valid but unrelated
legacy literal, parameter value, suffix-named table, quoted alias,
cross-adapter proof, uncalled nested helper, top-level constant, dead proof,
rebound transaction receiver, or unrelated transaction receiver cannot
authorize the table. Prose, comments, stale literals, unnamed or malformed
adapter scopes, and data/artifact paths cannot authorize a binding.

Task 2.1 deliberately proves only lexical receiver identity, not the concrete
runtime type or provenance of an `execute`-like receiver. There are no
production `approved_binding` records in the current baseline. Before a Context
child adds its first non-fixture table approval, the
`persistence-receiver-provenance` child must define and validate an explicit
adapter-local persistence Port or admitted receiver provenance. Artifact,
provider, stream, object-store, vector, and other non-SQL resources remain
candidate or deferred: before any such resource can be authorized, the owning
Capture or Dataset child must introduce a separately governed
`approved_capability` contract with resource identity, operation set,
adapter-local callable proof, replay/compatibility evidence, and a
capability-specific matcher. It must not reuse a table `approved_binding`.

Static provenance is intentionally a bounded, named audit inventory, not a
claim to discover or semantically normalize every legacy SQL statement. The
guard fails closed when any required table/source/literal/role record for the
inventory changes, including required catalog projection table declarations
and the governed `pipeline_dag` and `asset_registry` records. The three
reviewed f-string construction sites for `Recommendation`,
`RecommendationTrace`, and `factor_registry` are recorded separately as
non-authorizing `dynamic_sql_limitations`, bound to their logical table,
construction-site literal, limitation kind, owning child, exact rationale, and
`non_authorizing = true` marker. A limitation cannot attach to an
`approved_binding` table. It remains a limitation until an owning migration
child introduces an AST-aware SQL-normalization or runtime migration-evidence
design. This closed Task 2.1 inventory admits only the bounded reviewed
`dynamic_ddl` sites and rejects an unreviewed addition; it does not claim to
discover or govern dynamic DML or data transforms, which require a separate
owning child.

The checker reads UTF-8 text through repository-confined no-follow descriptors
and rejects malformed TOML, missing sources, unsafe relative paths, symlinks,
non-regular files, source identity or bounded reread-content drift, duplicate
declarations, and source facts that no longer match an executable source
literal. Per invocation, it
caches each decoded and comment/inert-string-masked evidence source, batches
all pending literals for one source into one deterministic Aho-Corasick
multi-pattern source scan, and so does not rescan the whole source for every
baseline fact or build a backtracking regular expression. Per-source literal
count and literal-byte budgets reject pathological declaration groups before
the automaton is built. The Python masking pass advances monotonically through
sorted inert-string spans and translates AST UTF-8 columns through one compact
per-non-ASCII-line map. Before Python AST construction, a streamed token
admission budget rejects high-cardinality evidence. Successful and terminally
failed transformations are both memoized, so repeated facts do not repeat
bounded work or change the failure outcome. Python comments and standalone
string, bytes, or f-string expressions, and admitted shell/CMake/C-family
comments, cannot satisfy evidence. It does not load modules, initialize `TradeDB`, read an
artifact directory, or accept arbitrary paths outside the repository. The
focused source-only fixture permits reads only of the baseline, declared source
evidence, and verified regular descriptors in the bounded production-Python
producer-discovery universe; it denies
`sqlite3.connect`, `duckdb.connect`, `pandas.read_parquet`, all reads of
in-repository `data/**`, `warehouse/**`, `market/**`, SQLite, Parquet,
manifest, pointer, and receipt sentinels, and all out-of-repository paths.

### Contracts and compatibility

The user-facing `trade` command, existing CLI command names, HTTP routes,
OpenAPI output, SSE semantics, Web payloads, SDK imports, notebook behavior,
table readers, BTC pointer format, and C++ ABI remain unchanged. The bounded
baseline records where these CLI/HTTP/OpenAPI/SSE contracts are defined and
tested, but it does not snapshot or alter their behavior. Task 2.1 leaves the
developer-facing `trade dev check` contract unchanged. Its direct validator
findings identify a rule ID, source location, and remediation direction; Task
2.3 will expose those facts through a versioned command envelope and
partial-scope policy. Repository consumers can continue to use legacy import
paths until their individual compatibility child supplies a replacement.

The architecture baseline records, rather than replaces, the `trade_py`
package discovery and the `trade_py` CMake binding target. It reserves
`_trade_native` as the future native name but does not claim that the binding
exists. The later package-layout child owns the actual transition and must
prove source, editable, and wheel compatibility.

All target business Contexts can import only `trade.platform.contracts` or
`trade.platform.api`, never a concrete Platform adapter. Bootstrap is the
only normal composition root for concrete adapters. The sole future legacy
exception is a specifically declared Platform persistence adapter, imported
only by `trade.bootstrap`, which exposes
`LegacySchemaBootstrapAdapter` through a narrow schema-bootstrap allowlist and
removal condition. Every other target `trade_py.*` or `trade_web.*` import is
denied.

### Persistent-write safety

This child is source-only: `architecture-baseline.toml` is a reviewed
declaration of legacy table and artifact facts, not a durable runtime writer,
and `architecture_guard.py` reads repository text and temporary test fixtures
only. The existing `TradeDB`, pipeline databases, catalog SQLite projection,
Parquet artifacts, and their runtime transaction boundaries remain
authoritative and unchanged. No guard result grants a table write: a
`candidate` or `deferred` record is non-authorizing, while a future table
`approved_binding` needs table-specific, adapter-local writer, reader,
transaction, and compatibility proof. Dynamic SQL is recorded only as an
explicit non-authorizing limitation.

Before a future owner migration can write a new schema generation, its child
must name the authoritative writer, deterministic idempotency key, local
concurrency control, staged validation, atomic visibility point, crash and
corrupt-predecessor handling, partial-result state, reader consistency, backup
or hash verification, sample proof, rollback, and audit trail. This child
supplies the source inventory and rejects stale provenance so those later
claims cannot silently erase known bootstrap, alter, transform, projection, or
Capture-risk evidence. Its only failure output is a deterministic developer
diagnostic; it does not open a data root, create a database connection, or
write artifacts.

### Schema migration compatibility

The governed baseline records legacy schema evolution facts and the required
child responsible for each eventual ownership transfer. It does not execute,
generate, or authorize a migration. Future migrations must be additive
versioned changes that retain the old schema generation, support backward and
forward compatible readers and writers through a compatibility window, and
use an idempotent checkpointed replay or shadow copy. Cutover requires a
dual-read comparison or a readiness-gated pointer switch, a verified prior
generation or backup snapshot for restoration, and no destructive retirement
until the published compatibility window closes.

For this change, rollback is a reviewable correction to source metadata and
guard logic before any migration child consumes it; runtime databases,
artifacts, readers, and migrations remain untouched. The focused fixtures
mutate table provenance, roles, dynamic limitation markers, and Capture-risk
semantics to prove that an incomplete baseline blocks the developer gate
rather than being treated as a schema transition. A later child owns real
temporary-root migration rehearsal, row/artifact reconciliation, and
compatibility cutover evidence.

### Failure and recovery

Task 2.1 fails closed for malformed or unsafe baseline/source evidence,
closed-inventory mutation, exceeded source/proof budget, missing provenance,
unreachable or receiver-mismatched approved-binding proof, and producer
discovery failure. It leaves the source tree and local data unchanged. A
developer corrects the baseline or source evidence through a reviewed child
change. It does not automatically rewrite imports, infer table ownership, or
silently ignore an unknown file.

If the guard reports baseline or producer source identity/content drift, stop
concurrent source mutation and rerun after the worktree stabilizes; do not
bypass the refusal. A Git index subprocess that exceeds its stderr diagnostic
budget is likewise terminated and must be repaired or receive a reviewed
governed budget increase.

Task 2.4 will provide the developer runbook keyed by `architecture.*`,
`dependency.*`, `persistence.*`, `artifacts.*`, and `execution.*` rule IDs,
including `trade dev check --show-plan` and JSON-report commands. That runbook
is developer tooling documentation, not an application operations system.

Rollback removes the Task 2.1 guard, neutral TOML helper, baseline, and focused tests together.
Because the child does not alter runtime behavior, database content, artifact
content, or interface payloads, the previous quality plan remains usable
immediately. A bad baseline fact is corrected in a small documentation/tooling
commit before any owner migration consumes it.

### Performance and capacity

Task 2.1 has no contributor, target-batch, quality-executor, or structured
output-envelope lifecycle. It reads only the finite set of baseline-declared
source files and one bounded producer-inventory pass. The pass uses a bounded
internal Git-index subprocess with a 30-second deadline, a 4 KiB hard stderr
retention ceiling plus one overflow-probe byte, and process-group cleanup; that
probe stops discovery rather than draining stderr until timeout. It is not a
Task 2.3 quality-executor lifecycle. No
network access, package installation, database scan, artifact hash, or
recursive full-repository AST scan is permitted, except for the separately
bounded producer-inventory pass below. Baseline validation groups declared
evidence literals by source and scans each transformed source once for its
pending literals using a deterministic multi-pattern automaton. It rejects a
source with more than 256 distinct literals or more than 64 KiB of literal
bytes before matching, and rejects a callable proof above 256 persistence
operations, 64 KiB of retained SQL text, or its guarded AST node/depth budget.
It caches terminal proof failures, so a repeated proof cannot expand work or
turn a failure into partial authorization.

Task 2.3 will add explicit contributor triggers, `batched_paths()`,
deterministic producer-prefilter/baseline/target batch identifiers, target
scope limits, executor worker-wave bounds, a 30-second timeout, a 32 KiB
output envelope, and truncation semantics. These planned command integration
and executor capacity claims require their own scope/planner/contributor tests
before they can be considered delivered.

Warehouse producer inventory is a distinct source-only baseline pass, not a
target-Context dependency scan. Its initial universe is every Git-tracked
first-party production Python source beneath `trade_py/`, excluding test-only,
generated, vendor, cache, non-source data assets, and artifact paths; Python
production modules under `trade_py/data/` remain in scope. The pass is allowed
to walk that one declared root because a finite baseline-declared source list
cannot prove there is no unknown current writer. Git index enumeration is
NUL-delimited, streamed, path-validated, and stopped at 1,024 raw records or
128 KiB raw path bytes before it can accumulate an unbounded list. The
inclusion predicate accepts only unique regular `100644`/`100755` `trade_py/**.py`
entries with no `test`/`tests` component or `test_`/`_test.py` basename and
outside `vendor`, `third_party`, `generated`, `cache`, and `__pycache__` path
segments; at reviewed commit `5537e99f3ba4` it includes 306 files and 3,102,353
source bytes, with zero exclusions. Included paths are additionally bounded to
512 entries and 64 KiB
path bytes, each source is bounded to 1 MiB, and aggregate logical source
payload is bounded to 32 MiB. Each admitted source is reopened and reread once
for same-identity payload verification, so producer descriptor-read I/O is
bounded to 64 MiB per invocation, excluding separately bounded baseline and
evidence reads.
Repository-confined directory-descriptor traversal and `O_NOFOLLOW`-equivalent
file reads reject symlinks, non-regular entries, escapes, unsupported safe-read
primitives, pre-read/post-read/post-path identity drift, and a reopened
same-identity payload mismatch as `architecture.producer_discovery_unsafe_source`;
the identity tuple is device, inode, size, and nanosecond mtime and ctime.
Raw/included path overflow reports
`architecture.producer_discovery_path_budget_exceeded`; source overflow reports
`architecture.producer_discovery_budget_exceeded`. All failures occur before
an incomplete inventory can be emitted. It resolves only
`trade_py.data.warehouse.io.write_table` and `upsert_table`, direct/module
imports, aliases, and the `trade_py.data.warehouse` package re-exports. The
functions are module-level and receive `WarehouseLayout` as their first
argument; `WarehouseLayout` does not own writer methods. A candidate call with
an unresolved canonical import, layout binding, or literal layer/table is a
failure, not an excluded producer.

For future Task 2.3 ongoing changes, canonical unfiltered modified, added, rename-source,
rename-target, and untracked metadata identify production Python files before
ordinary path filters apply. `ScopeSelection` supplies them as the bounded
canonical `ProducerDiscoverySelection`; the contributor will transport openable
candidate paths through deterministic `batched_paths()` prefilter steps that
honor the 65,536-byte argv limit, retaining rename/delete endpoint metadata
without Git rescan or a temporary manifest. A single untransportable path or
an over-budget selection fails before legacy-only classification. Deleted and
rename-source paths are not opened and instead are compared with declared
producer sources. A canonical writer, an unresolved writer-like import, or a
changed/deleted declared producer source triggers baseline validation; an
unknown discovered producer fails until it is declared. Filtering out any
producer signal yields `architecture.partial_scope` before acceptance. This
targeted delta pass is bounded by the same path, file, and byte limits and
never repeats a generic repository AST scan. The source-only test suite proves
direct imports, package re-exports, aliases, a new production writer,
declared-writer rename/deletion, test-only writer exclusion, streamed
enumeration overflow, Git-index and worktree symlink/non-regular rejection,
and source replacement during read. Task 2.3 owns long argv path transport,
over-cap delta selection, and all partial-scope tests.

Existing scope discovery and before/after full-worktree fingerprinting remain
an owned quality-platform performance debt. The child records the named
`quality-scope-capacity-baseline` follow-up, owned by
developer-experience/quality-platform, with measured representative scope,
bounded byte/path work, and a documented fail-safe exit criterion before
repository-wide target adoption. This child does not add another full-tree
walk.

### Observability and operations

Task 2.1 returns direct `ArchitectureReport` findings only; it creates no
`trade.architecture.guard.v1` envelope or terminal quality report. Task 2.3
will require that every `trade.architecture.guard.v1` JSON envelope has
required
`schema_version`, `status`, `scope`, `partial_scope`, `findings`, `counts`,
`emitted_count`, and `omitted_count` fields. `status` is `pass`, `fail`, or
`invalid`; `scope` identifies the baseline or deterministic target batch; and
counts are consistent with emitted plus omitted findings. Each finding contains
`rule_id`, repository-relative `path`, positive `line`, bounded `message`, and
bounded `remediation`. JSON and terminal detail derive from the same ordered
finding set. An unrecognized schema, missing/unsafe field, inconsistent count,
or overflow is an infrastructure result rather than a pass.

Each failure reports a deterministic rule name such as
`dependency.context_implementation`, `cell.use_case_adapter`,
`contracts.implementation_type`, `database.owner_escape`,
`database.foreign_table_owner`, `persistence.unapproved_client`,
`artifacts.direct_access`, `interfaces.direct_sql`,
`dependency.legacy_namespace`, `dependency.dynamic_loading`,
`execution.direct_process_creation`, `platform.business_vocabulary`, or
`native.boundary`. Producer discovery additionally names
`architecture.producer_discovery_unresolved_import`,
`architecture.producer_discovery_unresolved_layout`,
`architecture.producer_discovery_nonliteral_target`,
`architecture.producer_discovery_undeclared_writer`,
`architecture.producer_discovery_path_budget_exceeded`,
`architecture.producer_discovery_budget_exceeded`, and
`architecture.producer_discovery_unsafe_source`. Diagnostics include the
repository-relative path and source line where the offending import,
annotation, attribute, SQL literal, direct artifact client, dynamic loader,
process spawn, vocabulary, or producer source occurs. They sort by path, line,
rule, and message; truncation reports the emitted and omitted counts rather
than hiding findings, with the omitted count equal to the number of hidden
real findings. Task 2.1 already returns ordered bounded findings for baseline
and producer failures; dependency, Cell, interface, execution, and envelope
findings are Task 2.2 or 2.3 behavior.

Task 2.3 will make `trade dev check --show-plan` show the architecture step
when triggered. Until then, a clean legacy-only implementation change receives
no new quality-plan behavior. The Task 2.1 baseline is format-checked by the
neutral TOML reader and semantically checked only through direct validator
invocation.

### Validation strategy

Focused pytest uses temporary repositories and source files to prove Task 2.1
baseline parsing, multi-source table provenance, closed Capture-risk and
dynamic-DDL inventories, independent intelligence/projection declarations,
producer-derived warehouse artifacts, missing/deleted/unsafe source evidence,
direct-scope reachable/receiver-matched/exact-identifier approved-binding
proofs, SQL-only-in-parameter rejection, mutable reader/compatibility rejection,
transaction root, local alias, global/nonlocal receiver-or-alias rejection,
including receiver-root operations beneath an external alias, import, and
assignment-target invalidation,
wildcard and dynamic
module-rebinding rejection, exact operation-literal provenance and adapter-line
diagnostics, proof budgets, Git-index non-regular rejection, neutral cold
import, negative-I/O admission, and the current-tree legacy baseline.

Task 2.2 will test the prospective target dependency graph. Task 2.3 will test
triggers, canonical deltas, partial-scope failures, batching, executor
capacity, timeout/output behavior, and check-mode non-mutation. Task 2.4 will
test the runbook. The Task 2.1 delivery runs focused tests, shared Python
compile validation, direct design diagnostics, and `git diff --check`; it does
not claim a new `trade dev check` contributor, C++ build, frontend build, API
smoke, or live-data validation.

### Alternatives and trade-offs

**Enforce the full current `trade_py` tree immediately:** rejected because the
audit already shows direct facade imports across many legacy paths. Treating
those facts as immediate failures would force a broad rewrite and obscure the
incremental owner migration required by the parent design.

**Use a third-party import linter:** rejected for this first slice because the
needed checks include Cell direction, AST annotation leakage, source-declared
table facts, pointer evidence, and native boundary policy. A small
standard-library implementation keeps the policy local, deterministic, and
testable without expanding packaging risk.

**Store the inventory in comments or a Markdown table:** rejected because
later child tooling needs parseable source locations and classifications. TOML
keeps the baseline versioned and machine-verifiable while the parent OpenSpec
continues to explain target ownership.

**Make the baseline final ownership authority:** rejected because several
legacy records require row-level analysis. `candidate` and `deferred` are both
audit-only; an owning Context child must prove its authoritative writer,
readers, transaction boundary, compatibility plan, and named persistence
adapter before adding an explicit `approved_binding`.

### Rollout and rollback

Task 2.1 follows this isolated rollout:

1. Add and approve this child design before code implementation.
2. Add the baseline parser/validator and fixture tests without registering it.
3. Commit the baseline/guard unit, then run the focused and unified quality
   gates.
4. Run the six-role implementation review against the frozen diff and resolve
   P0 findings before push/PR.

The guard is prospective: later children create `src/trade` modules only after
their approved designs name an owner and compatibility bridge. A failing
transition is rolled back by retaining the legacy path and removing the new
target module from the child branch; this guard itself has no runtime state to
restore.

Task 2.3 will separately register the contributor and verify changed-scope
planning. Its implementation, capacity evidence, approval, and rollback are
not part of this Task 2.1 delivery.

## Decisions

### Use an AST guard with a declarative scope

`architecture-baseline.toml` declares distinct `target_source_root` and
`target_import_root`, approved Context names, legacy source facts, table
provenance and approval state, warehouse/pointer/receipt/Capture facts,
CLI/HTTP/OpenAPI/SSE source facts, and native facts. The checker parses source
via `ast.parse`, resolves absolute and relative imports under the declared
import root, and applies the parent dependency graph only under the declared
target root. It rejects `src.trade.*` as a filesystem-path import rather than
mistaking it for a Context namespace. This catches the relationships that
matter at the time a new Context file is introduced without importing code or
relying on fragile text-only import searches.

The allowed graph is:

```text
kernel -> kernel
capture -> kernel, platform.contracts/api
datasets -> kernel, platform.contracts/api, capture.contracts
studies -> kernel, platform.contracts/api, datasets.contracts
decision_support -> kernel, platform.contracts/api, datasets.contracts, studies.contracts
processes -> kernel, platform.contracts/api, all business contracts
interfaces -> kernel, platform.contracts/api, processes, context contracts/use_cases
bootstrap -> all target modules
platform -> kernel, platform contracts/api
```

Within a Context Cell, `contracts` and `domain` only receive Kernel and own
types under their approved rule; `ports` receives own domain/contracts;
`use_cases` receives own domain/ports/contracts plus upstream contracts; and
`adapters` receives own ports/domain/contracts plus external libraries. A
Context or Interface file cannot directly import unapproved database or
artifact clients, legacy namespaces, or access private connection attributes. A
Context may access literal SQL only within a baseline-authorized persistence
adapter for its approved table. A target Platform file cannot contain declared
business aggregate vocabulary. A native extension is importable only from a
Context `adapters/native` module. Dynamic module/file/native loading, direct
process creation, shell execution, and process pools are forbidden until a
later separately approved plugin/worker or Platform execution contract.

### Validate baseline facts, not final Context ownership

The initial baseline uses the observed DDL locations in
`trade_py/db/trade_db.py`, `trade_py/db/migrations.py`,
`trade_py/db/pipeline_db.py`, and `trade_py/intelligence/schema.py`; it also
records the Observatory catalog projection declarations in
`trade_py/observatory/catalog/store.py`. It records one-or-more provenance
facts for each logical table and names current code owners plus target
classification. Warehouse artifact entries are derived from every production
call resolving to the canonical module-level
`trade_py.data.warehouse.io.write_table` or
`trade_py.data.warehouse.io.upsert_table` function, each with a
`WarehouseLayout` first argument. Direct imports, module aliases, and
`trade_py.data.warehouse` package re-exports resolve to those canonical
symbols. The bounded initial discovery pass includes the producers in
`trade_py/data/warehouse/materialize.py`, including the validation report, and
the standalone CLI fetch producers in `trade_py/cli/data.py` for
`dim.dim_data_source` and `ods.ods_fetch_attempt`; the ongoing bounded delta
prefilter prevents a new production writer from being classified legacy-only.
Test fixtures are excluded because they do not produce repository artifacts.
The classification is intentionally `candidate` for obvious families and
`deferred` where the parent design requires later file/row analysis. Both are
non-authorizing. This avoids making the guard another global database facade or
pretending exact future table names already exist.

The baseline pins every source literal in its bounded, governed static
bootstrap/migration/alter/data-transform inventory for legacy records whose
schema history is directly recoverable from executable source. It deliberately
does not claim source-wide discovery, complete SQL-statement semantics, or
literal-complete provenance for arbitrary legacy DDL/DML. The reviewed
f-string DDL construction sites in `Recommendation`, `RecommendationTrace`,
and `factor_registry` have mandatory non-authorizing limitation records rather
than invented normalized SQL facts. Each record has `non_authorizing = true`,
cannot authorize an approved table, and is owned by the corresponding migration
child. Their owning migration children need a reviewed AST-aware
SQL-normalization or runtime migration-evidence design before broadening the
claim.

The Capture-risk inventory is a bounded registry of named audited source facts,
including the EastMoney timezone/precision overwrite. It is intentionally not a
claim that every provider or temporal path has been exhaustively discovered.
Any child migrating a Capture-related adapter must perform a child-local audit
and add its own reviewed risk declaration before asserting broader coverage.

`BtcRunStore.current_path`, `compatibility_path`, and
`engine/cmake/python_bindings.cmake` are pinned as source facts, alongside
warehouse layout/materialization, Crypto ADS pointer/receipt, Kline
reconciliation, precise Capture time/catalog/replay/quarantine facts, and root
CLI/FastAPI/OpenAPI/SSE sources. This informs future Capture, Datasets,
interface, and package changes of known compatibility and recovery edges while
making no production change to those edges.

The one future legacy schema bridge is deliberately placed at
`trade.platform.persistence.adapters.legacy_schema_bootstrap`.
`LegacySchemaBootstrapAdapter` is an implementation in that Platform adapter;
only `trade.bootstrap` may import it. The baseline declaration must name the
adapter path and every legacy schema-bootstrap symbol it imports, and no
business Context, Process, or Interface may import it. This preserves the
parent's Bootstrap-only composition rule and avoids an ambiguous
`trade.bootstrap.compat` ownership boundary.

### Plan Task 2.3 integration through the existing quality contributor seam

The existing `DesignQualityContributor` shows the project convention for
scope-aware quality checks. In Task 2.3, a sibling `ArchitectureContributor`
will supply one baseline check plus explicitly batched read-only target checks;
normal provider ownership will remain unchanged. That contributor must never
invoke the guard in fix mode or make the architecture checker a catch-all
quality provider. Before contributor planning, the additive `ScopeSelection`
fields will preserve canonical unfiltered delta/filter facts. The shared
planner will compute
`architecture.partial_scope` as an ordinary failed plan issue, allowing the
existing runner and JSON/text reports to expose it consistently while
independent checks remain observable. This preserves the quality runner's
source-protection guarantee and avoids duplicate Git traversal.

## Risks / Trade-offs

- **Rule false positives from partial Context layouts** -> The checker applies
  only when a file has a declared target Context and Cell path. Unknown target
  layouts fail with an explicit rule rather than being guessed; each child
  introduces files in the canonical Cell shape.
- **Legacy baseline becomes stale** -> The baseline validates source literals
  and source paths on every evidence/native/target/guard-triggered run,
  including rename/deletion; a child must update it in the same reviewed change
  when audited facts legitimately move.
- **Repeated evidence or pathological inert strings exhaust review capacity** ->
  Descriptor-verified executable text and terminal transformation failures are
  memoized once per source per guard invocation, and pending baseline literals
  are grouped into one deterministic multi-pattern source scan rather than one
  scan per fact. A 256-literal and 64 KiB literal-byte budget per source rejects
  oversized declaration groups before matching; the automaton never builds a
  backtracking alternation or pairwise literal-coverage map. Python token
  masking advances through sorted inert spans with a monotonic cursor while
  reusing one UTF-8 byte-to-character map per non-ASCII physical line. A
  streamed pre-parse token ceiling rejects high-cardinality inputs before
  AST/span allocation. Focused regressions pin successful and failed
  transformation reuse, one grouped literal scan per evidence source,
  overlapping-prefix matcher parity, per-source literal-budget refusal, and
  large line-separated, same-line, and token-over-budget inert-string fixtures
  without adding runtime application I/O.
- **Approved-binding proof needs control-flow interpretation** -> The guard
  deliberately authorizes only direct calls in a unique, undecorated,
  unrebound module-level callable's straight-line prefix and directly nested
  transaction `with` blocks. It does not infer reachability through branches,
  loops, exceptions, assertions, matches, or comprehensions; an owning child
  must expose a small explicit persistence operation or propose a separate
  reviewed CFG-proof expansion before relying on such a path.
- **A familiar persistence method name does not prove an admitted runtime
  receiver** -> Task 2.1 rejects parameter-only SQL, lexical or dynamic
  receiver rebinding, nested control-flow proof, and dynamic callable
  replacement, but it intentionally does not resolve concrete receiver type or
  Port provenance. The current baseline has no production `approved_binding`;
  `persistence-receiver-provenance` must be completed by the owning Context
  before its first production table binding.
- **A transaction alias escapes its callable lexical scope** -> A receiver root
  or `as` alias declared `global` or `nonlocal` cannot prove a local
  transaction operation, even when its syntactic identity matches. An external
  `as` alias invalidates the whole `with` context, including a subsequent call
  expressed through the transaction receiver root. Task 2.1 collects
  declarations only from the proof callable's direct lexical scope, skips
  nested closure bodies, and rejects those contexts before adding transaction
  receivers. The cause-specific remediation is emitted only when its exact
  declared transaction SQL is inside that rejected single-item direct context
  and would otherwise be admitted with a callable-local alias; unrelated,
  nested, multi-item, or independently invalid contexts retain the generic
  proof diagnostic. A receiver root sharing the external alias name remains
  external in the counterfactual and retains the generic diagnostic. Since
  direct proof collection stops at the first rejected statement, any
  cause-specific diagnostic is deterministically tied to that first rejected
  direct transaction block. Focused temporary-source fixtures cover global
  receiver, global alias/root operation, synchronous and asynchronous alias
  diagnostics, nested nonlocal alias/root-operation, same-name
  receiver/alias, and unrelated external-alias diagnostic suppression.
- **An adapter-wide literal is mistaken for a callable proof** -> Every
  approved-binding literal must exactly equal the static first SQL argument
  captured from its named direct-scope operation. A mismatch reports the
  adapter callable line; later source-fact audit records remain intentionally
  separate non-authorizing lexical evidence.
- **A compound SQL script is mistaken for a table-local write proof** -> Writer
  and transaction evidence admit only one statement after comments and values
  are masked. A second statement, including one passed to `executescript`, is
  non-authorizing.
- **An import or newer binding form mutates an approved callable at module
  initialization** -> Approved-proof adapters reject every module-scope import,
  every executable call, class declaration, PEP 695 type alias, lambda,
  attribute/subscript store, and protocol-triggering control-flow statement.
  Their module body admits only plain functions, inert literal declarations,
  and a docstring. A later child must define a reviewed import-purity or richer
  callable-provenance contract before relaxing this rule.
- **A table SQL binding cannot authorize heterogeneous resources** -> Schema v1
  rejects unknown top-level resource declarations and classification/proof
  fields on every unclassifiable source-only record. Artifact, provider,
  stream, object-store, vector, and unstructured payload access stay candidate
  or deferred. The owning Capture or Dataset child must first define a
  fail-closed `approved_capability` schema and capability-specific static
  matcher; it cannot repurpose `approved_binding` or add a broad approval
  escape.
- **Many distinct approved proof callables increase retained review state** ->
  Per-callable operation, SQL-byte, and AST budgets prevent individual growth,
  and the aggregate source/baseline input budgets constrain the current
  surface. A later `callable-proof-capacity` follow-up must introduce an
  aggregate summary/operation/retained-SQL budget and callable index before a
  child admits more than eight distinct approved proof callables or a second
  resource-family authorization.
- **A source-text rule cannot prove runtime ownership** -> The guard blocks
  direct architectural bypasses and fail-closes unknown table/artifact,
  dynamic-loading, and process-spawn paths, but does not claim dynamic behavior
  proof. Later Context and Platform children retain focused
  integration/contract fixtures and six-role review obligations.
- **Direct SQL detection has lexical limits** -> It is intentionally scoped to
  target Interfaces and Context paths. Literal SQL is allowed only in an
  explicit `approved_binding`; dynamic SQL needs a later parser/allowlist
  design and cannot bypass the first guard.
- **Dynamic historical DDL lacks stable literals** -> Task 2.1 has explicit,
  fail-closed, non-authorizing limitation records for the reviewed f-string
  `Recommendation`, `RecommendationTrace`, and `factor_registry` construction
  sites. `non_authorizing = true` is a required machine-checked field and a
  limitation cannot authorize an approved table. The guard does not invent
  normalized provenance, govern dynamic DML/data transforms, or claim
  source-wide static DDL/DML completeness. Their owning migration children
  must add AST-aware normalized or runtime-backed evidence before broadening
  schema-provenance claims.
- **Guard volume or noisy diagnostics** -> Per-file, per-batch, total-scope,
  and concurrent-wave source-byte budgets; argv limits; timeout; versioned
  bounded envelope; stable sort; and explicit count-only truncation make
  failures predictable. Existing quality scope/fingerprint costs remain
  separately tracked through `quality-scope-capacity-baseline` and are not
  worsened by this child.
- **A new legacy Warehouse writer evades a finite evidence list** -> The
  initial bounded `trade_py/` production-source discovery and every-delta
  bounded canonical-writer prefilter detect direct imports, aliases, and
  package re-exports. Exceeded discovery budgets, unresolved canonical writer
  bindings, and undeclared producers fail closed rather than silently reducing
  inventory coverage.
- **Producer discovery reads an unsafe or changing source** -> The streamed
  index admission accepts only regular tracked entries, and repository-confined
  no-follow descriptor reads verify identity before and after each source.
  Symlinks, non-regular entries, path escapes, unavailable safe-read support,
  and replacement races emit `architecture.producer_discovery_unsafe_source`
  without a partial inventory.
- **Producer discovery consumes unbounded paths or loses an excluded delta** ->
  Raw/included path-count and byte budgets stop the streamed initial census;
  `ProducerDiscoverySelection` bounds canonical delta transport through
  deterministic argv-safe batches. An excluded modified, added, deleted,
  renamed, or untracked producer signal is `architecture.partial_scope`, never
  a legacy-only acceptance.
- **Source-only baseline reads data by accident** -> The validator allowlists
  only its baseline/evidence source files in a temporary fixture and explicitly
  denies in-repository data/artifact sentinels as well as external paths. A
  source fact cannot justify physical artifact inspection. The current
  negative-I/O fixture audits the guard's admitted descriptor reader and
  patches known database/parquet entry points; it is not a process-wide
  interception of every Python file API. A later `guard-io-admission` follow-up
  must add boundary instrumentation that permits only the baseline, declared
  evidence, and verified production-source descriptors and rejects every other
  file-open attempt.
- **Lexical source facts could be mistaken for runtime behavior proof** ->
  Artifact, pointer, interface, and Capture-risk facts are bounded,
  source-verified, non-authorizing migration inputs. Their exact literals and
  pinned descriptions do not prove the surrounding runtime branch, provider
  interaction, or data state. A later `source-fact-semantic-evidence` child or
  owning Context child must add AST-backed reachability and fact-specific
  semantic binding before a legacy source fact can support authorization,
  ownership, or behavior claims.
- **Future Capture risks drift into incompatible vocabulary** -> The closed
  Task 2.1 Capture-risk records remain exactly pinned. Before a Capture child
  adds a new provider, stream, unstructured payload, redaction, revision, or
  ordering risk, `capture-risk-taxonomy` must define versioned dimensions,
  per-record unknown-key closure, and backward-compatible mappings for the
  existing risk IDs. That child also owns parameterized allow/deny coverage for
  unknown keys in non-table source facts, Capture risks, dynamic SQL
  limitations, interfaces, and native bindings; Task 2.1 intentionally keeps
  the current closed top-level schema and authorization-field rejection without
  widening this source-only parser change.
- **First-child interface scope expands into a compatibility migration** ->
  This child inventories definition/test sources only. It does not generate
  behavioral snapshots, delegate a route, or alter a response form; those remain
  exit criteria of `cli-http-sdk-compatibility`.
- **Native boundary policy precedes a real binding** -> The baseline records
  the current collision and prohibits future direct imports; the package/native
  child will add CMake linkage and differential checks before enabling a
  binding.

## Migration Plan

The approved child roadmap is additive and order-preserving. Only phase 1 is
delivered by Task 2.1; later phases require their own implementation, tests,
review, and strict approval:

1. **Task 2.1:** add the baseline and parser tests while all application paths remain in
   their current locations.
2. **Task 2.3:** add the additive shared scope/model/planner metadata and focused
   modified/delete/rename/untracked filtered-scope tests.
3. **Tasks 2.2 and 2.3:** add the scoped target guard and contributor
   registration, with tests that prove
   no legacy-only source is newly audited.
4. **Later architecture children:** freeze the baseline as the first architecture
   migration input.
5. **`kernel-and-public-contracts`:** introduce any first `src/trade`
   paths under this guard.
6. **`persistence-receiver-provenance`:** precede the first production table
   `approved_binding` with an admitted adapter receiver/Port proof.
7. **Capture/Dataset capability child:** precede the first non-SQL resource
   authorization with the separate `approved_capability` contract.

Task 2.1 rollback removes the direct validator, neutral TOML helper, baseline
declaration, and focused tests, leaving source, database, artifact, native, and
interface behavior untouched. A later Task 2.3 rollback would separately remove
its additive scope metadata, planner architecture-only plan issue, and
contributor registration, restoring the prior filtered selection behavior.
Later child rollbacks retain legacy modules and update or restore baseline
source facts before attempting another extraction.

## Open Questions

- The final mapping for KG, causal, factor, and historical recommendation
  records remains deferred to their owning Dataset, Study, or Decision Support
  child; no ambiguity is resolved by this guard.
- The package-layout child must decide the exact distribution/console bridge
  and whether `_trade_native` is a separately installed extension before CMake
  linkage changes are designed.
