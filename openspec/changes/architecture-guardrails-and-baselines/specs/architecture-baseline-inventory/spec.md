## ADDED Requirements

### Requirement: The legacy architecture baseline SHALL be source-verified and non-authoritative

The repository SHALL maintain a versioned architecture baseline that records
current package roots, independently declared target source/import roots,
schema-definition sources, physical table classifications, artifact/pointer/
receipt facts, Capture migration facts, native binding facts, source-derived
CLI/HTTP/OpenAPI/SSE compatibility facts, and compatibility-pointer facts. The
quality gate SHALL validate each entry against source text only. The baseline
SHALL not initialize an application, open a database, read an artifact, or
claim final runtime ownership.

Every table declaration SHALL record a logical table name, current owner,
one-or-more provenance records, and a classification. A provenance record
SHALL name a repository source, an exact source literal, and one of
`bootstrap`, `migration`, `alter`, or `data_transform`; a single bootstrap DDL
location SHALL not be treated as complete physical-schema evidence. The
governed multi-source provenance inventory is a finite, named table/source/
literal/role set. Within that inventory, each declaration SHALL record every
required exact source/literal/role fact and validation SHALL fail if any is
removed or changed. This source-only check does not discover arbitrary legacy
DDL/DML or prove the complete semantics of a SQL expression; a later owning
migration child needs AST-aware normalization or runtime-backed evidence before
making that broader claim.

The bounded reviewed dynamic SQL/f-string-derived DDL inventory is outside the
static-literal evidence contract and SHALL be recorded as a non-authorizing
`dynamic_sql_limitations` record rather than represented by an invented
normalized literal. Each Task 2.1 limitation SHALL name a unique `id`, existing
`logical_name`, repository `source`, executable construction-site `literal`,
the `dynamic_ddl` limitation kind, the table's `owning_child`, and an explicit
limitation rationale, and SHALL declare `non_authorizing = true`. A limitation
SHALL never be attached to an `approved_binding` table and cannot authorize a
table, persistence adapter, SQL statement, or schema shape. The mandatory
audited dynamic-DDL set is the `Recommendation`,
`RecommendationTrace`, and `factor_registry` construction sites in
`trade_py/db/migrations.py`; their declaration, limitation kind, owning child,
non-authorizing marker, and rationale SHALL match the governed binding. A
dynamic limitation neither authorizes persistence access nor proves
physical-schema completeness. The Task 2.1 limitation set is closed: an
additional limitation record SHALL fail validation rather than implicitly
govern another dynamic SQL site. Dynamic DML/data-transform discovery and
governance are outside this bounded Task 2.1 inventory and require a later
owning child with an explicit source scope and reviewed binding.

The validator SHALL transform executable source evidence, including a terminal
transformation failure, at most once per source per invocation. It SHALL batch
the pending baseline literal matches for one transformed source into one
deterministic multi-pattern source scan before individual-fact validation. It
SHALL fail closed before scanning when a source exceeds the governed
per-source literal-count or literal-byte budget. Python inert-string masking SHALL
advance through ordered spans without a repeated whole-span scan and reuse one
UTF-8 byte-to-character mapping per non-ASCII physical line. Before
constructing a Python AST, the validator SHALL stream and enforce the governed
Python evidence-token budget, failing closed when the limit is exceeded.
classification SHALL carry a semantic kind, target Context or `deferred`,
reason, required owning child, and explicit activation state. `candidate` and
`deferred` are audit-only classifications and SHALL never authorize target
persistence access. Only a later, separately reviewed table `approved_binding`
with one target Context and a non-empty, dot-delimited persistence-adapter scope
matching `<context>.adapters.<identifier>[.<identifier>...]` authorizes a
literal SQL table reference. Artifact and warehouse-producer declarations SHALL
remain candidate or deferred until their separate authorization contracts
exist. A table `approved_binding` SHALL also declare separate
`writer_evidence`, `reader_evidence`, `transaction_evidence`, and
`compatibility_evidence` records; each record SHALL contain a repository source
an executable source literal, and one named module-level Python `callable`
accepted by the same source-only admission and no-follow validation as other
baseline evidence. Every proof SHALL be in the one named target adapter module
`src/trade/<adapter_scope path>.py` and within the named callable's direct
lexical scope; nested function, async-function, lambda, and class bodies SHALL
not satisfy a proof. The callable SHALL be exactly one undecorated top-level
function or async function and SHALL have no other executable module binding,
including a duplicate definition, assignment, deletion, import, or nested
control-flow rebind, wildcard import, module-namespace mutation, or dynamic
execution. An adapter containing an approved proof SHALL reject every
additional module-scope executable call because its namespace effect cannot be
proven safely. It SHALL also reject every class declaration and every function
declaration with definition-time metadata, including decorators, defaults,
annotations, or type parameters, because those forms can execute before the
module binding is admitted. It SHALL reject every PEP 695 type-alias declaration
and every module-scope import until a reviewed import-purity or callable-provenance
contract exists. The module body SHALL otherwise admit only plain functions
without definition-time metadata, inert literal constant declarations, and a
module docstring; lambdas, attribute or subscript assignments, context managers,
control-flow statements, and every other executable declaration SHALL be
rejected. Writer and transaction evidence SHALL each occur as one static
table-specific write statement passed as the first positional argument to a
persistence call; reader and compatibility evidence SHALL occur as one static read-only `SELECT`
statement in that same position. A string only in a persistence parameter or
keyword SHALL NOT authorize a proof. Transaction evidence SHALL occur inside a
direct transaction `with` block containing a static table-specific write on that
transaction receiver or its explicit `as` alias. No direct-scope binding
mutation or unmodelled dynamic call may alter that receiver root or alias before
the operation.
The declared proof literal SHALL exactly equal the static first SQL argument
captured from the named callable, rather than merely occurring elsewhere in the
adapter; an operation mismatch SHALL identify the adapter callable line.
Every accepted SQL table identifier SHALL match at a supported statement
position with identifier boundaries, not as a substring. Reader and
compatibility matching SHALL ignore SQL comments, string values, and quoted
alias text before recognizing a `FROM` or `JOIN` table identifier. Proof
collection SHALL retain only direct calls in the callable's straight-line
statement prefix and one direct transaction `with` block. It SHALL stop before
a nested `with`, conditional, loop, exception handler, assertion, match, or
deferred comprehension rather than inferring a control-flow path through that
construct. It SHALL fail closed before retaining more than the governed callable
proof operation, SQL-byte, AST-node, or AST-depth budgets.
`candidate` and `deferred` declarations SHALL reject every persistence-binding
field.

Task 2.1 SHALL NOT authorize a non-table artifact, provider, stream, object
store, vector index, or other heterogeneous resource. Before an owning Context
authorizes one, its separately reviewed child SHALL define an
`approved_capability` contract with a stable resource identity and namespace,
owner Context, adapter scope, operation set, source-local callable proof,
replay/compatibility evidence, and capability-specific static matcher. It SHALL
reject an unrecognized resource kind or cross-provider adapter scope and SHALL
NOT reuse table `approved_binding`.
Schema version 1 SHALL reject every unknown top-level declaration, including a
provider, stream, object-store, vector, or unstructured-resource array, and
every source fact, Capture risk, dynamic SQL limitation, interface, or native
binding SHALL reject classification, target-context, adapter, and persistence
proof fields.

The artifact, pointer, interface, and Capture-risk records are non-authorizing
lexical migration inputs. Their source/literal bindings and pinned descriptive
fields do not prove the runtime branch, provider interaction, or data state
surrounding a fact. A later `source-fact-semantic-evidence` child or owning
Context child SHALL add AST-backed reachability and fact-specific semantic
binding before a legacy source fact supports authorization, ownership, or a
runtime-behavior claim.

#### Scenario: A baseline table source changes

- **WHEN** a child moves or rewrites a source-defined table declaration
- **THEN** the baseline validation fails until that child updates the declared
  provenance fact or facts, classification, compatibility note, and focused
  migration evidence in the same reviewed change

#### Scenario: A table requires further classification

- **WHEN** the current source inventory identifies a KG, causal, factor, or
  historical recommendation record whose target Context is not yet proven
- **THEN** the baseline marks it `deferred`, records its reason and required
  owning child, and no target module treats the declaration as authority to
  read or write that record

#### Scenario: A candidate table lacks an approved binding

- **WHEN** a baseline entry names Datasets as a candidate target Context but
  does not contain an explicit approved persistence-adapter binding
- **THEN** a target Datasets adapter cannot query that table; the guard fails
  closed until its owning child adds the approved binding and focused owner,
  transaction, reader, and compatibility evidence

#### Scenario: An approved binding has unverified proof

- **WHEN** an owning child supplies prose, a comment-only literal, an unsafe
  path, a stale literal, a suffix-named table, a proof from another adapter or
  callable, an uncalled nested function/lambda/class helper, a top-level
  constant, a writer/reader/compatibility literal that does not identify the
  declared logical table, a decorated, duplicate, rebound, or imported proof
  callable, a proof in a branch, loop, exception handler, assertion, match, or
  deferred comprehension, a proof after an unconditional terminal statement, a
  proof above the governed operation, SQL-byte, AST-node, or AST-depth budget,
  SQL only in a persistence parameter or keyword, mutable or multi-statement
  reader/compatibility SQL, a quoted alias or comment/string pseudo-table
  reference, a transaction-only read, a wildcard/dynamic proof-callable rebind,
  compound writer or transaction SQL, any module-scope import or executable
  call, a PEP 695 type alias, lambda or non-inert module declaration, a rebound
  or dynamically mutated transaction receiver or alias, a class or
  definition-time metadata that can execute while defining an unrelated
  module-level declaration, a literal that is present only elsewhere in the
  adapter or differs from the named operation's first SQL
  argument, a nested `with` proof, or a transaction context with an unrelated
  receiver
- **THEN** baseline validation fails and the declaration does not authorize
  persistence access

#### Scenario: A non-SQL resource asks to reuse a table approval

- **WHEN** a Capture or Dataset child attempts to add an unknown provider,
  stream, object-store, vector, or unstructured-resource declaration, or adds
  classification/proof fields to a non-table source-only declaration
- **THEN** baseline validation rejects it until the owning child adds its
  separately reviewed new schema version and `approved_capability` contract
  with focused resource-specific authorization tests

#### Scenario: A dynamic SQL limitation is absent, misbound, or unreviewed

- **WHEN** the baseline omits or changes the reviewed dynamic-DDL limitation
  for `Recommendation`, `RecommendationTrace`, or `factor_registry`, or adds
  any other Task 2.1 limitation record
- **THEN** baseline validation fails until the exact construction site,
  limitation kind, owning child, explicit `non_authorizing = true` marker, and
  non-authorizing rationale are restored or the later owning child governs the
  added site; no normalized static provenance is inferred

#### Scenario: Historical DDL has two provenance roles

- **WHEN** a logical record is created by bootstrap SQL and later has a
  `migration` or `alter` provenance source
- **THEN** the baseline records both provenance facts with their distinct roles
  and does not claim either source alone describes the complete physical schema

#### Scenario: An independent schema or projection declaration is encountered

- **WHEN** current source declares `feed_scores`, `source_configs`,
  `catalog_meta`, `runs`, `releases`, `catalog.sqlite`, or `generation.json`
  outside the central TradeDB schema paths
- **THEN** the baseline records the exact source literal, current owner,
  projection or authoritative role, candidate/deferred target classification,
  and required child without treating the record as an approved persistence
  binding

### Requirement: Source-only artifact, pointer, receipt, and Capture-risk facts SHALL be complete enough to govern later children

The baseline SHALL record known source-defined warehouse Parquet artifact
families, the Crypto ADS current pointer and completion-receipt convention, the
BTC compatibility pointer, and the Kline reconciliation `current.json` fact.
It SHALL record each fact's repository source, exact literal, current code
owner, compatibility or recovery role, candidate target Context or deferred
state, and required owning child. The mandatory inventory includes the
warehouse Parquet family, catalog SQLite projection and generation pointer,
Crypto ADS current pointer and validation receipt, BTC compatibility pointer,
and both Kline reconciliation pointers. The validator SHALL fail if any
mandatory fact is absent or its source, literal, role, classification, target
Context/deferred state, or required child differs from the governed binding.
These are source-only migration inputs, not runtime artifact inspection or
release authorization.

The baseline SHALL record a bounded, mandatory Capture-risk inventory for the
named audited legacy `RawRecord` temporal model; RSS, GDELT, warehouse, archive,
and date-only publication-time fallbacks; the EastMoney provider
timezone/precision overwrite; and the `InfluenceSignal` runtime evaluation-time
substitution for `published_at`. This Task 2.1 inventory is closed: an
additional Capture-risk declaration SHALL fail validation rather than silently
asserting an unreviewed temporal, news, provider, capture, or replay behavior.
This is not a claim that every provider or temporal behavior in the repository
has been exhaustively discovered. Each future child that migrates a
Capture-related adapter SHALL audit its own source scope and add a separately
reviewed binding before asserting broader coverage.
Every Capture-risk record SHALL state its repository source,
exact literal, risk kind, current behavior, required child, and required
migration proof. The governed binding SHALL pin every one of those fields, so
weakening either prose field fails validation rather than merely satisfying a
non-empty-field check. Required risk kinds include provider timestamp
absence/substitution, date-only inferred precision, provider timezone and
precision overwrite, catalog/environment override and absent rights-policy
evidence, provider-refetch versus local artifact replay versus WAL recovery,
transport/integrity failure versus downstream semantic quarantine, and
runtime-evaluation-time substituted for publication time. `capture-boundary`
and `study-boundary` SHALL treat the bounded declarations as mandatory inputs
and prove independent provider/observed/received/available/revision/finality
clocks, SourceManifest rights enforcement, provider-free replay, and the
Capture transport-versus-Datasets semantic quarantine split before migrating
the corresponding audited news or NLP adapter.

The Task 2.1 negative-I/O fixture audits the guard's admitted descriptor reader
and selected database/parquet boundaries; it is not process-wide interception
of every Python file API. A later `guard-io-admission` child SHALL instrument
the file-open boundary so that only the baseline, declared evidence, and
verified production-Python discovery descriptors are admitted, and every other
in-repository, data/artifact, or out-of-repository open attempt fails.

All source literals SHALL occur in executable/admitted source content. Python
comments and standalone string, bytes, or f-string expressions, shell/CMake
`#` comments, and C-family line or block comments SHALL NOT satisfy an evidence
literal.

The warehouse artifact inventory SHALL be producer-driven. Its only canonical
writer targets are the module-level functions
`trade_py.data.warehouse.io.write_table` and
`trade_py.data.warehouse.io.upsert_table`, each of which accepts a
`WarehouseLayout` value as its first `layout` argument. The source-only
resolver SHALL recognize direct and module imports, local aliases, and the
`trade_py.data.warehouse` package re-exports of those functions; it SHALL NOT
invent nonexistent `WarehouseLayout` instance methods. A call counts as a
producer when its callee resolves to one canonical writer and its first
argument is a statically known `WarehouseLayout` binding. A nonliteral
layer/table, unresolved warehouse-writer import, or unresolved layout binding
in a candidate call SHALL fail closed rather than be omitted from the
inventory. The following stable finding IDs SHALL include the repository path,
positive line where applicable, bounded remediation, and one of the following
conditions: `architecture.producer_discovery_unresolved_import` for an
unresolved writer-like import; `architecture.producer_discovery_unresolved_layout`
for an unresolved first layout argument; `architecture.producer_discovery_nonliteral_target`
for a nonliteral layer or table; and
`architecture.producer_discovery_undeclared_writer` when a resolved literal
producer has no declaration. Their remediations respectively require a
canonical import, a statically traceable `WarehouseLayout` binding, literal
artifact coordinates, or a reviewed baseline declaration. A budget overflow,
unsafe source, or path-transport failure SHALL instead use the dedicated
finding IDs below.

The initial inventory pass SHALL parse the complete, bounded universe of Git
tracked first-party production Python sources below `trade_py/`. It excludes
test-only paths and files, generated, vendor, cache, non-source data assets,
and artifact paths; production modules such as `trade_py/data/**.py` remain in
scope. It never imports code or reads a database or artifact. This is the sole
narrow exception to the rule against recursive full-repository AST scanning: it
is limited to the declared `trade_py/` production universe. The exact
inclusion predicate is a unique NUL-delimited Git index path that begins
`trade_py/`, ends `.py`, has a regular `100644` or `100755` mode, has neither
`test` nor `tests` as a path component, has no basename beginning `test_` or
ending `_test.py`, and is not inside a `vendor`, `third_party`, `generated`,
`cache`, or `__pycache__` path segment; `trade_py/data/**.py` remains
production source.
Any Git-index entry whose path otherwise meets the production-Python path
predicate but whose mode is not regular `100644` or `100755` SHALL fail with
`architecture.producer_discovery_unsafe_source`; it SHALL not be silently
excluded from the producer universe.
The scanner SHALL consume the Git path stream incrementally, validate and
deduplicate each record as received, and stop before materializing an
unbounded path list. It SHALL refuse more than 1,024 raw index records or
128 KiB of raw path bytes, more than 512 included paths or 64 KiB of included
path bytes, more than 32 MiB aggregate source, or one source file over 1 MiB.
`architecture.producer_discovery_path_budget_exceeded` SHALL fail the scan
before AST parsing when a raw or included path budget is exceeded;
`architecture.producer_discovery_budget_exceeded` SHALL fail it before
emitting an incomplete inventory when a file or source-byte budget is
exceeded. Both remediations require reducing/splitting the source scope or a
reviewed increase to the governed budget.

Every included path SHALL be opened only through repository-confined directory
descriptors with no-follow semantics for every path component and file. The
scanner SHALL reject a symlink, non-regular file, path escape, unavailable
no-follow primitive, unreadable source, or any pre-read/post-read/post-path
identity mismatch with `architecture.producer_discovery_unsafe_source`; its
remediation requires a regular, repository-confined source stable for the
check. The identity SHALL include device, inode, size, and nanosecond mtime,
and the source contents SHALL be read only from the verified descriptor. This
prevents a symlink or replacement from becoming source evidence or an
inconsistent producer inventory.

The current capacity measurement is 306 included paths and 3,102,353 source
bytes from that exact predicate at reviewed commit `5537e99f3ba4`; it excludes
zero current paths. It is capacity evidence, not an authorization to raise the
limits. Reproduce the fixed-commit measurement from Git blobs, not the current
worktree, with:

```sh
git ls-tree -r -l -z 5537e99f3ba4 -- trade_py |
  uv run python -c '
import sys

excluded = {"vendor", "third_party", "generated", "cache", "__pycache__"}
paths = 0
source_bytes = 0
for entry in filter(None, sys.stdin.buffer.read().split(b"\0")):
    metadata, raw_path = entry.split(b"\t", 1)
    mode, kind, _object_id, size = metadata.split()
    path = raw_path.decode("utf-8")
    parts = path.split("/")
    name = parts[-1]
    if (
        mode in {b"100644", b"100755"}
        and kind == b"blob"
        and parts[0] == "trade_py"
        and path.endswith(".py")
        and not any(part in {"test", "tests"} | excluded for part in parts)
        and not name.startswith("test_")
        and not name.endswith("_test.py")
    ):
        paths += 1
        source_bytes += int(size)
print(f"included_paths={paths} source_bytes={source_bytes}")
'
```

The output SHALL be `included_paths=306 source_bytes=3102353`; a changed
reviewed measurement requires a governed evidence update and review.

### Task 2.3 future requirement: scoped producer delta admission

When Task 2.3 is implemented, every modified, added, renamed, or untracked
production Python file SHALL receive the same bounded AST import/call prefilter
before the planner classifies the delta as legacy-only. `ScopeSelection` SHALL
carry this canonical, unfiltered producer-candidate set and rename/delete
endpoints as an immutable `ProducerDiscoverySelection`, rather than permitting
the contributor to rediscover Git. The future selector SHALL enforce the same
512-path, 64-KiB-path, 32-MiB-source, and 1-MiB-file limits before AST work.
The future contributor SHALL transport existing candidates in deterministic
`batched_paths()` producer-prefilter steps within the existing 65,536-byte argv
limit; it SHALL create no manifest, cache, or mutable application state. A
candidate set whose encoded argv batch would exceed that limit SHALL split
deterministically, while a single untransportable path fails with
`architecture.producer_discovery_path_budget_exceeded`. Deleted paths and
rename-source endpoints are not opened; they are matched against declared
producer sources and produce a baseline-staleness failure until reconciled.

A detected canonical writer, a changed or deleted declared producer source, or
an unresolved warehouse-writer import SHALL force baseline validation when
Task 2.3 is implemented. Those producer signals are architecture-sensitive:
if `--path` excludes any modified, added, deleted, rename-source, rename-target,
or untracked signal, the future planner SHALL emit
`architecture.partial_scope` with the excluded paths before any acceptance
result. A noncandidate legacy-only delta remains non-triggering. The direct
Task 2.1 validator fails until the baseline declares every producer discovered
by its initial inventory; it does not accept a hand-maintained required-table
list or one materialization module as proof of completeness. Test fixtures are
not production artifact producers. Each declaration SHALL name the producer
source, exact call literal, layer, table, path role, and target/deferred
classification. This includes
`ads_warehouse_validation_report` where the producer exists even when a
validation-required table list omits it, and the CLI fetch producers
`dim.dim_data_source` and `ods.ods_fetch_attempt`.

#### Scenario: A production warehouse writer is outside the materializer

- **WHEN** `trade_py/cli/data.py` resolves a call imported from
  `trade_py.data.warehouse.write_table` for `dim.dim_data_source` or a call
  imported from `trade_py.data.warehouse.upsert_table` for
  `ods.ods_fetch_attempt`
- **THEN** the baseline declares both producer facts and validation fails for an
  undeclared production writer even when `_REQUIRED_TABLES` or
  `materialize.py` does not list it

#### Scenario: An alias or package re-export invokes a canonical writer

- **WHEN** a production source aliases a direct/module import or imports
  `write_table` or `upsert_table` through the `trade_py.data.warehouse`
  package re-export
- **THEN** the resolver records the canonical `io` function target and requires
  a declaration for each literal producer call instead of treating the alias or
  re-export as a different writer

#### Scenario: A changed producer is not already in the baseline

- **WHEN** Task 2.3 is implemented and a changed or untracked production
  Python file contains a call that resolves to a canonical warehouse writer
- **THEN** its bounded prefilter prevents a legacy-only plan, baseline
  validation runs, and an undeclared producer fails closed until the inventory
  declaration is added

#### Scenario: A producer signal is excluded by a path filter

- **WHEN** Task 2.3 is implemented and `--path` excludes a modified, added,
  deleted, rename-source, rename-target, or untracked producer-discovery signal
- **THEN** planning reports `architecture.partial_scope` and the excluded
  repository-relative paths before any target or baseline acceptance result

#### Scenario: A declared producer source is renamed or deleted

- **WHEN** Task 2.3 is implemented and a child renames or deletes a source
  named by a warehouse producer declaration
- **THEN** the baseline contributor runs from canonical rename/delete metadata
  and fails until the declaration is updated or removed with the corresponding
  producer inventory evidence

#### Scenario: Test-only calls do not create artifact declarations

- **WHEN** a test fixture calls either canonical warehouse writer
- **THEN** the production-universe filter excludes that source and validation
  does not require an artifact declaration for the fixture

#### Scenario: A producer source is unsafe or changes while read

- **WHEN** an included production-Python path is a symlink, non-regular file,
  escapes the repository, cannot be opened without following links, or has a
  changed identity between the verified descriptor read and post-read path
  check
- **THEN** discovery stops with
  `architecture.producer_discovery_unsafe_source` and emits no partial
  inventory

#### Scenario: Discovery exceeds a path or source budget

- **WHEN** the streamed Git enumeration, bounded delta selection, one source,
  or aggregate source bytes exceeds its declared path/file/byte limit
- **THEN** discovery stops with
  `architecture.producer_discovery_path_budget_exceeded` or
  `architecture.producer_discovery_budget_exceeded`, as applicable, before
  parsing further sources or reporting a pass

#### Scenario: A declared pointer or receipt source is changed

- **WHEN** Task 2.3 is implemented and a child renames, deletes, or rewrites
  the code declaration of a recorded artifact, pointer, receipt, or
  Capture-risk fact
- **THEN** the architecture contributor runs and fails until the baseline
  declaration and the owning child's migration evidence are updated together

#### Scenario: Baseline validation runs in a no-I/O fixture

- **WHEN** the baseline validator runs in its focused negative-I/O fixture
- **THEN** the fixture permits reads only of the baseline, declared
  repository source-evidence files, and verified regular descriptors in the
  bounded production-Python discovery universe, and patched `sqlite3.connect`,
  `duckdb.connect`, `pandas.read_parquet`, generic `open`/`Path.read_*` for
  in-repository `data/**`, `warehouse/**`, `market/**`, SQLite, Parquet,
  manifest, pointer, and receipt sentinels, and every out-of-repository path
  fail the test if the validator attempts to use them

#### Scenario: A source-only Capture fact changes

- **WHEN** a child changes a provider-time fallback, date-only precision rule,
  provider timezone/precision overwrite, catalog/environment override,
  replay/WAL behavior, or semantic quarantine source literal or changes the
  required current-behavior or migration-proof semantics
- **THEN** baseline validation fails until the record's risk kind, current
  behavior, required child, and required migration proof are updated in the
  same reviewed change

### Requirement: Compatibility and native baseline facts SHALL remain explicit

The baseline SHALL record the current root `trade` command facade, canonical
and retained hidden/deprecated CLI domains, FastAPI application/router source
roots, generated OpenAPI source, SSE route/media-type sources, and existing
CLI/HTTP/OpenAPI/SSE contract-test sources as source facts. Each interface
record SHALL name the source, exact literal, surface kind, current behavior,
compatibility owner, and the later `cli-http-sdk-compatibility` child required
to create snapshot parity and retire the old path. This child is a bounded
source-only interface baseline inventory: it SHALL NOT delegate routes, alter
payloads, generate a runtime snapshot, or implement compatibility adapters.

The baseline SHALL also record the current BTC compatibility pointer and C++
Python binding target as source facts. A later Dataset, package-layout, or
native child SHALL retain the legacy fact until it passes its own compatibility,
native-boundary, and rollback criteria.

#### Scenario: An interface evidence source changes

- **WHEN** Task 2.3 is implemented and a child renames, deletes, or changes
  the source declaration of a canonical CLI domain, FastAPI route/router,
  OpenAPI creation path, SSE media type, or its existing contract-test source
- **THEN** the architecture baseline contributor runs and fails until the
  source-only inventory and the owning interface child evidence are updated

#### Scenario: A later interface child delegates a route

- **WHEN** `cli-http-sdk-compatibility` moves a CLI, HTTP, OpenAPI, or SSE
  surface behind a compatibility adapter
- **THEN** it consumes this inventory, creates the required snapshot and
  behavior evidence, and preserves the current public contract before the
  source-only record can be retired

#### Scenario: A package transition proposes a native rename

- **WHEN** a package-layout child replaces the `trade_py` native binding target
- **THEN** it updates the baseline only after source/editable/wheel and
  C++/Python differential evidence proves the `_trade_native` boundary and
  retains a compatible rollback path

#### Scenario: A Dataset release replaces the BTC pointer

- **WHEN** a Dataset migration proposes a replacement for the recorded BTC
  compatibility pointer
- **THEN** it preserves the current pointer as a compatibility reader or
  rollback source until dual-read comparison or a readiness-gated switch has
  passed
