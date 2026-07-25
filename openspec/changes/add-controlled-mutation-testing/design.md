# Design: Controlled Mutation Testing

## Current-state audit

The repository is a mixed system, but the initial mutation target is deliberately
Python-only:

| Concern | Observed state |
|---|---|
| Primary application language | Python 3.10+ under `trade_py/`; C++ and React remain separate |
| Build/package | `uv`/`pyproject.toml`, CMake/CTest for `engine/`, npm/Vitest/Playwright for Web |
| Unit tests | pytest under `tests/`; 86 top-level/Observatory test files |
| Coverage | no committed pytest-cov/coverage.py configuration |
| CI | GitHub remote, but no committed `.github/workflows/` |
| Mutation | no existing tool or configuration |
| Core candidates | `trade_py/belief`, `decision`, selected `factors`, `signals`, `evaluation` |
| Excluded real paths | tests, `engine/vendor`, frontend node_modules/dist/test-results, DB migrations, fixtures/golden, devtools, CLI/Web/bootstrap surfaces |
| Full pytest baseline | 1161 passed, 1 failed, 12 warnings in 136.00 seconds |
| Existing failure | `test_verify_reports_current_then_stale` fails in full and two isolated reruns |
| Closed mapped cohort | `uv run pytest -q tests/test_decision_action.py tests/test_world_state.py tests/test_explanation.py tests/test_trust_layer.py tests/test_factor_groups.py tests/test_window_scorer.py`: 92 passed; exact core rows collect 90 |

The current failure is a stable pre-existing Observatory catalog behavior mismatch,
not a flaky mutation-test candidate. The mutation workflows remain non-blocking and
exclude Observatory from the initial core scope. The complete pytest result remains
visible in final validation rather than being hidden.

## Problems and root causes

1. Full-suite-per-mutant execution would cost roughly 136 seconds before accounting
   for setup or mutations and is not viable within PR budgets.
2. Popular default mutation sets include low-value and hazardous operators such as
   numeric literal replacement and `break`/`continue` swapping.
3. Cosmic Ray's built-in test runner classifies timeout as killed and does not promise
   process-tree termination, so it cannot own result truth or lifecycle here.
4. Source-to-test relationships are implicit. Guessing tests or falling back to all
   tests would either miss defects or violate the runtime budget.
5. A score-only report can be gamed by exclusions and can silently count no-coverage,
   timeout, or infrastructure failures as success.
6. No mutation history exists, so a high absolute score gate would block unrelated
   work before the signal is calibrated.

## Design Quality Brief

### Alternatives and trade-offs

Mutmut is active and integrates with pytest, but its current default operators include
integer literal changes and `break`/`continue` replacement. Its supported
configuration does not provide the precise closed operator whitelist required here.
Wrapping its generated cache after enumeration would retain unbounded/noisy planning
and make line-priority selection brittle. It is not selected.

#### Cosmic Ray full session/distributor

Cosmic Ray 8.4.6 exposes mature operator implementations, source positions, and
first-order work items. However, its default `_operators()` enumerates every
non-parameterized operator, and its runner classifies timeouts as killed. Its local
distributor also does not supply the repository's process-tree, partial-report, or
global-budget semantics. The design reuses only reviewed operator implementations and
AST mutation support.

#### Custom AST mutator

A fully custom engine could exactly match the policy but would duplicate parsing,
source-preserving mutation, operator correctness, and Python-version maintenance.
That is unnecessary. The chosen controller owns policy and lifecycle while Cosmic Ray
owns only single-location source transformation.

#### Chosen design

Use Cosmic Ray 8.4.6 as an exact optional development dependency and build a narrow
`trade_py.devtools.mutation_testing` controller. The controller imports named
operator classes with `cosmic_ray.plugins.get_operator`, parses with
`cosmic_ray.ast.get_ast`, and enumerates the exact preorder yielded by
`cosmic_ray.ast.ast_nodes` plus each operator's ordered `mutation_positions`. That is
the same node/position counter order used by `MutationVisitor.walk`. It transforms
source text only with `cosmic_ray.mutating.mutate_code`, materializes one mutation in
one worker-owned copy, and runs pytest through a mutation-owned bounded executor. A
compatibility test compares adapter occurrence/span output with `MutationVisitor` for
every allowlisted operator and nested syntax before any real target is admitted. The
controller never calls Cosmic Ray's in-place or test-running helpers. This separates
mutation semantics from execution truth.

### Requirements and acceptance

Users are repository developers and CI. Acceptance requires:

- one stable `scripts/mutation-test changed|core|full` surface;
- exact changed/core/full target and test selection without full-suite fallback;
- 150/1000/5000 mutant limits and 600/1800/5400 second wall budgets;
- first-order allowlisted mutations only;
- maximum worker count `min(4, max(1, effective CPU//2))`;
- one native math thread per pytest process;
- 10-second initial per-mutant timeout from the measured sub-3-second mapped baseline,
  dynamically recomputed per affected-test set and capped at 60 seconds for changed;
- supervisor-owned descendant containment and truthful status/count algebra;
- baseline line coverage and private-overlay provenance before mutant execution;
- generation-atomic JSON/Markdown/HTML output with surviving/no-coverage details;
- precise expiring equivalent-mutant exceptions and a non-established initial
  baseline;
- non-blocking GitHub PR/nightly/weekly/manual routing;
- focused unit, CLI, config, timeout, report, and workflow tests;
- exact definition-level eligibility inside the seven-source matrix;
- no production behavior, network access, credentials, or real-data reads/writes.

### Ownership and boundaries

```text
scripts/mutation-test                         thin executable facade
trade dev mutation                           canonical root dispatch to the facade
config/mutation-testing.toml                 policy and source/test ownership
config/mutation-baseline.json                reviewed score history seed
config/mutation-exceptions.toml              precise reviewed exceptions
config/mutation-capacity.json                identity-bound schedule qualification
trade_py/devtools/mutation_testing/
  bootstrap_contract.py                      Python 3.7 stdlib mode/receipt/safe-root contract
  supervisor.py                              Python 3.7 stdlib receipt/containment/fallback owner
  protocol.py                                Python 3.7-compatible closed control protocol
  bootstrap.py                               import-light Git/config/zero-work planner
  cli.py                                     argument parsing/rendering only
  application.py                             top-level use-case orchestration
  models.py                                  immutable plans/outcomes/status algebra
  config.py                                  closed configuration parser
  git_scope.py                               bounded base/diff/changed-line discovery
  selection.py                               deterministic targets/tests/mutants
  engine.py                                  Cosmic Ray operator adapter
  coverage.py                                coverage.py 7.10.7 line-data adapter
  process_supervisor.py                      authenticated supervisor spawn/lifecycle client
  isolation.py                               private trees, environment and guard evidence
  executor.py                                bounded mutation work scheduling only
  cache.py                                   staged outcomes and committed run markers
  trend.py                                   sequence transaction, digest chain and projection
  render.py                                  JSON-derived Markdown/HTML rendering
  report_store.py                            leases, atomic report publish and raw retention
  bundle.py                                  immutable CI bundle build/byte validation
  baseline.py                                comparable baseline evaluation
  exceptions.py                              exact exception identity and temporal validation
  capacity.py                                runner qualification and phase-budget admission
.github/workflows/mutation-testing.yml       CI routing only
docs/mutation-testing.md                     developer operations
```

The root `trade` script handles `dev mutation` before its ordinary `uv` dispatch and
`exec`s `scripts/mutation-test`, so both public paths enter the same supervisor-first
process tree and preserve argv/exit parity. The facade uses Bash built-ins to read
`/proc/uptime`, resolves an absolute supervisor interpreter from the explicit
`MUTATION_SUPERVISOR_PYTHON` override or `command -v python3`, rejects non-absolute
resolution and Python below 3.7, then `exec`s `python -I supervisor.py` with the
command-entry monotonic value. `supervisor.py` and `bootstrap_contract.py` are
syntax-tested on Python 3.7 and use no project imports, dataclasses, third-party
modules, or Python 3.8+ syntax. They own the closed mode ceilings, lowering-only
preparse, safe-root rules, receipt/fallback schemas, and child protocol. `config.py`
imports and validates those constants instead of redefining them. No `uv` process
exists before the supervisor, and the measured budget includes shell-to-supervisor
startup.

The supervisor creates the run ID, receipt, `CLOCK_BOOTTIME` deadlines, lease and
watchdog, enables Linux child-subreaper semantics, and is the sole process that calls
`fork`, `clone`, `posix_spawn`, or `Popen` for `uv`, coverage, pytest, and helper
children. Bootstrap and the controller cannot create an OS process. They submit
bounded authenticated requests over a private `SOCK_SEQPACKET` control channel:
`spawn`, `signal`, `wait`, `fallback_request`, and `shutdown`. Every request binds the
run ID, boot ID, monotonically increasing message sequence, deadline, closed command
kind, normalized argv/environment digest, working-directory identity, cgroup memory
reservation, and an exact `SCM_RIGHTS` descriptor role table. The supervisor validates
all fields against its own mode policy, creates the stopped child, completes cgroup/
pidfd/session registration, and returns an opaque random `worker_handle` plus verified
PID/start-token metadata required by the seccomp broker. Controller lifecycle actions
name only that handle; raw PIDs are diagnostic and cannot authorize signal, wait, or
cleanup.

The protocol is `trade.mutation.supervisor-protocol.v1`. The supervisor creates a
256-bit per-run channel capability before the controller starts, passes it only
through an inherited close-on-exec descriptor, and authenticates every request and response with
HMAC-SHA-256 over direction, run/boot identity, request ID, sequence, deadline,
payload digest, and descriptor-role digest. It also verifies the inherited client's
`SO_PEERCRED` PID/start token. The capability descriptor closes after the controller
establishes the one allowed session and is never inherited by workers. This channel
MAC prevents peer/replay corruption but is not supervisor-exclusive attestation.
The supervisor separately retains the only writable descriptor for a bounded,
hash-chained `trade.mutation.supervisor-attestation.v1` journal. Before controller
imports, it applies a Landlock controller policy that denies all controller writes,
renames, links, and removals below invocation/attestation/fallback roots while allowing
report-root writes. Reports reference journal entry digests and CI validates the full
chain. Unsupported write isolation fails execution preflight, so possession of the
shared channel key cannot forge supervisor-authored guard/wait/OOM/cleanup/forced-
close evidence. Reconnect and a second client are forbidden. Packets are at most 64 KiB, JSON payloads at most
48 KiB, and each packet carries at most eight role-labelled descriptors. One
controller multiplexer owns the socket, correlates unique request IDs, allows at most
`workers` in-flight lifecycle requests, one in-flight spawn per logical worker,
`workers` active handles, and `2 * workers + 4` queued requests. Completed handles are
removed immediately after verified wait/cleanup. A full queue blocks only until its
`CLOCK_BOOTTIME` send deadline, then returns `protocol_capacity_exceeded`; it never
spins or grows. MAC, peer, sequence, correlation, frame, FD, channel loss, or replay
failure closes admission, requests cleanup of registered handles, and becomes
phase-appropriate infrastructure failure.

The CLI parses and renders only; `application.py` coordinates owned use cases and
contains no adapter implementation. Cosmic Ray is
an adapter dependency and never imports application runtime modules. `models.py`
imports no controller adapter; planning and execution depend on models, while
selection/engine never import process, storage, cache, trend, or rendering modules.
`process_supervisor.py` is only the typed client for the supervisor protocol and owns
no OS spawning primitive. Every direct first-party import is represented in this
complete import DAG:

```text
bootstrap_contract <- protocol <- supervisor
bootstrap_contract, protocol <- bootstrap
models <- config, git_scope, selection, engine, coverage, process_supervisor,
          isolation, executor, render, report_store, cache, trend, baseline,
          exceptions, capacity, bundle, application, cli
config <- selection, engine, coverage, isolation, executor, baseline, exceptions,
          capacity, application
git_scope <- selection <- engine
protocol <- process_supervisor <- isolation, executor, application
engine, coverage, isolation <- executor <- application
models <- render <- report_store
models, report_store <- cache, trend, baseline, exceptions, bundle
report_store, cache, trend, baseline, exceptions, capacity, bundle, executor,
process_supervisor, git_scope, selection, config <- application <- cli
```

`supervisor.py` depends only on Python-3.7-compatible
`bootstrap_contract.py`/`protocol.py`; it cannot import controller modules.
`executor.py` cannot import report storage, cache, trend, baseline, exceptions,
capacity, bundle, or rendering. `report_store.py` owns report-generation bytes and
filesystem transitions only; it does not own outcomes, bundles, score policy, cache,
or trend. Cache/trend/baseline/exception/bundle modules consume immutable report DTOs
and cannot call executor or adapters. `bundle.py` validates copied bytes and never
selects global current state. `capacity.py` owns qualification math but cannot mutate
the reviewed capacity record. `bootstrap.py` depends only on the two Python
3.7-compatible modules and hands one validated bootstrap DTO to `application.py`; it
cannot import controller adapters. Static import-architecture tests enumerate every
module, enforce every edge, and forbid reverse/cyclic imports. Tests use synthetic
repositories and source files.

The controller is a devtool, not a domain service. It does not belong in the future
business Context graph and is excluded as a mutation target.

### Data and state invariants

The durable artifacts are developer reports and cache entries, never business state.
Identifiers and invariants:

- `run_id` is a UUID generated at supervisor entry. Before dependency preparation, the
  supervisor atomically writes `invocations/<run_id>.json` with schema
  `trade.mutation.invocation.v1`, mode, lifecycle state, shell command-entry
  `/proc/uptime` anchor, supervisor `CLOCK_BOOTTIME` value and clock ID, kernel boot
  ID, controller stop deadline, outer cleanup deadline, supervisor identity, and exact
  expected staging/run/fallback/bundle paths. All elapsed time, event ordering,
  deadlines, leases, and timeout decisions use that one `CLOCK_BOOTTIME` domain and
  boot ID. UTC is display/retention metadata only; a boot change invalidates leases
  rather than comparing monotonic values across boots.
  Bootstrap and the full CLI send bounded authenticated transition records over one
  inherited supervisor pipe; they never own or rewrite the receipt. The supervisor
  validates run ID, child PID/start token, sequence, schema and forward transition
  before atomically rewriting it. The receipt is also a bounded phase journal with
  plan digest, selected count, last durable phase, expected report root and projection
  state. CI receives its path before the first child starts.
- `CanonicalMutationPositionV1` is derived from the original UTF-8 source and the
  allowlisted operator before selection. It stores `kind=replace|insert_before`,
  zero-based start/end UTF-8 byte offsets, one-based line, zero-based UTF-8 byte
  column, and exact original token bytes. Replacement operators identify the unique
  `tokenize` token or contiguous compound token sequence inside the Cosmic Ray AST
  node that corresponds to that operator (`is not` remains one compound position).
  `AddNot` uses `insert_before` at the unique first lexical token of the target
  expression, with equal start/end byte offsets. Offsets use a line-start byte table,
  never Unicode code-point counting. A missing, duplicated, overlapping, or
  operator-inconsistent token is not selected and increments the
  `canonicalization_rejected` scanned reason. Immediately before mutation the same
  normalizer must reproduce the position and observed Cosmic Ray span; otherwise the
  already-selected record becomes `invalid_mutation`.
- `execution_id` (and compatibility field `mutant_id`) is SHA-256 over schema
  version, relative path, original source digest, `CanonicalMutationPositionV1`,
  observed AST span, operator name/arguments, `execution_occurrence`, and config
  digest. `execution_occurrence` is Cosmic Ray's zero-based source-wide counter for
  one `(source_digest, operator_name, operator_arguments)` traversal and is the only
  occurrence passed to `mutate_code`.
- `comparison_key` is a separate SHA-256 over schema version, relative path,
  exact qualified eligible definition, definition digest, definition-relative
  canonical mutation-token position, normalized original node text, operator
  name/arguments, and `definition_occurrence`. `definition_occurrence` is a separate
  zero-based counter inside the unique nearest named definition for the same
  operator/arguments and is never passed to Cosmic Ray. Candidates inside a nested
  named function or class belong to that nested qualified name, not its lexical
  parent; they are excluded unless that exact nested name is configured. Lambdas
  remain owned by their nearest named definition. Duplicate qualified ownership,
  missing parents, or ambiguous token positions fail validation. This permits a
  conservative cross-revision match after unrelated line/file edits without weakening
  exact execution/cache identity. Any changed node text, ambiguous key, added/deleted
  candidate, partial cohort, or changed policy is non-comparable.
- one `MutantPlan` contains exactly one `MutationSpec`.
- source paths are repository-relative normalized POSIX paths and regular files.
- changed line ranges are parsed from NUL-safe Git records and zero-context hunks for
  merge-base-to-HEAD plus staged and unstaged tracked state. Tracked rename
  destinations are eligible; deletion-only and rename-only-without-content-change are
  not. Untracked Python production paths are recorded as `deferred_untracked` and are
  never mutated or silently treated as ordinary zero work. Planning seals one
  `GitScopeSnapshot` containing explicit base OID, merge-base OID, HEAD OID, index
  entry digest, staged patch digest, tracked worktree content/mode digest, and the
  exact diff/hunk digest. It is part of run identity and is revalidated before private
  source copy and again before generation publication. Drift stops admission, makes
  uncommitted cache unreadable, and produces a typed source-drift result; it is never
  silently published for a different checkout. Changed mode admits only a candidate
  whose canonical Cosmic Ray mutation-token position is on an exact changed new-side
  line. It does not use a multiline AST span intersection and never widens to the
  enclosing definition.
- ordered selection is stable and does not depend on filesystem enumeration order.
- a cached result is reusable only when the bounded import-closure digest, mapped
  tests/fixtures, reviewed dynamic-import manifest, lock and pytest configuration,
  operator/config/exception policy, normalized command/environment including
  native-thread and kernel-isolation controls, platform, declared per-mapping clock
  and entropy policies, exact mutant cohort, and tool versions match. The complete
  tracked-Python tree digest remains report audit metadata but does not invalidate an
  unchanged closure. V1 accepts only `explicit_input|non_cacheable` clocks and
  `deterministic|entropy_non_cacheable` entropy; unsupported `fixed` values fail
  configuration until an injection contract exists. Clock/entropy checking is
  conservative over the entire copied repository-local closure executable by one
  mapped test tuple, not only the selected definition. An alias-aware AST/import
  binding pass resolves direct imports, `from` imports, aliases, and statically
  followed repository re-exports for datetime/date/time, UUID, random, secrets and OS
  entropy APIs. Unresolved star/dynamic imports, native-extension calls, reflective
  lookup, or ambiguous re-export cannot prove determinism and make the tuple
  non-cacheable unless the exact reviewed dynamic-import manifest declares an already
  non-cacheable policy. Unselected definitions in a copied executable module are
  included. The trust tuple is `entropy_non_cacheable` because its default path can
  call `uuid.uuid4()`.
- one run directory is staged and fsynced; one `current.json` pointer atomically
  publishes JSON, Markdown, HTML, and their digest manifest as one generation.
- `candidate_positions_scanned` counts deterministic eligible positions examined up
  to the scan ceiling and always equals `selected + not_selected_scanned`.
  `not_selected_scanned` is partitioned into closed reasons
  `outside_changed_line|lower_priority_after_mutant_cap|canonicalization_rejected|
  exception_filtered`; scan/deadline/source/visit ceilings stop before examining more
  positions and therefore affect only `unscanned_candidate_remainder="unknown"`.
  `generated_mutants` is exactly the number materialized as selected first-order
  mutant records and equals `selected`; it is bounded by 150/1000/5000.
- the closed terminal enum is `killed_fresh`, `survived_fresh`, `killed_cache`,
  `survived_cache`, `timeout_test_started`, `infrastructure_error_test_started`,
  `cancelled_test_started`, `no_coverage_line`, `baseline_unavailable`,
  `invalid_mutation`, `infrastructure_error_pre_test`, `equivalent_exception`,
  `not_run_plan`, `not_run_budget`, and `not_run_cancelled`. Every selected mutant
  has exactly one. The v1 matrix requires a non-empty test tuple for every eligible
  definition; a missing mapping fails configuration and matrix-external paths are
  `deferred_unmapped`, so the former `no_coverage_mapping` terminal is unreachable
  and removed.
- every record also carries booleans `mutation_applied_current`,
  `worker_spawned_current`, `worker_registered_current`, and
  `test_started_current`; aggregate phase counts are sums of those facts, not guesses
  from logs. Successful OS child creation immediately makes spawned true, before
  cgroup/pidfd/session registration can fail. `test_started_current` implies
  `worker_registered_current`, which implies `worker_spawned_current`, which implies
  `mutation_applied_current`. Cache terminals set all four false. Test-started
  terminals set all four true.
  `infrastructure_error_pre_test`, `not_run_budget`, or `not_run_cancelled` may set
  mutation-applied true only when their typed stop event wins after the
  position-verified write but before pytest spawn; in this contract `not_run` means
  the test did not start, not necessarily that transformation did not occur.
- only `killed_*` and `survived_*` enter the factual numerator/denominator. The report
  always preserves those counts, but `run_score_eligible=false` makes displayed score
  null under closed reason precedence `report_invalid`, `integrity_override`,
  `cleanup_unconfirmed`, `degraded_infrastructure`, then `zero_denominator`. Only an
  eligible run may compare/append a trend score or commit verified fresh
  killed/survived cache outcomes.
- every worker also has orthogonal cleanup status
  `not_required|confirmed|unconfirmed`. `cleanup_unconfirmed` is not a seventeenth
  terminal. If cleanup cannot be confirmed, the provisional terminal is replaced by
  `infrastructure_error_test_started` when `test_started_current=true`, otherwise
  `infrastructure_error_pre_test`; the run becomes `degraded_infrastructure`, score
  and cache eligibility are disabled for the affected record and run, and the receipt
  retains process/cgroup evidence. No provisional killed/survived/timeout/cancelled
  count survives that override.

The phase matrix below is the sole normative table. A tuple is
`(mutation_applied_current, worker_spawned_current, worker_registered_current,
test_started_current, cleanup)`.
No tuple outside it is serializable:

| Terminal | Allowed tuples |
|---|---|
| `killed_fresh`, `survived_fresh`, `timeout_test_started`, `cancelled_test_started` | `(true,true,true,true,confirmed)` |
| `infrastructure_error_test_started` | `(true,true,true,true,confirmed)`, `(true,true,true,true,unconfirmed)` |
| `killed_cache`, `survived_cache` | `(false,false,false,false,not_required)` |
| `no_coverage_line`, `baseline_unavailable`, `invalid_mutation`, `equivalent_exception`, `not_run_plan` | `(false,false,false,false,not_required)` |
| `infrastructure_error_pre_test` | `(false,false,false,false,not_required)`, `(true,false,false,false,not_required)`, `(true,true,false,false,confirmed)`, `(true,true,false,false,unconfirmed)`, `(true,true,true,false,confirmed)`, `(true,true,true,false,unconfirmed)` |
| `not_run_budget`, `not_run_cancelled` | `(false,false,false,false,not_required)`, `(true,false,false,false,not_required)`, `(true,true,false,false,confirmed)`, `(true,true,true,false,confirmed)` |

Any provisional non-infrastructure terminal whose spawned worker cleanup is
`unconfirmed` is replaced before aggregation by the phase-appropriate infrastructure
terminal with the corresponding allowed `unconfirmed` tuple. Such a record and its
run are score-ineligible, cache-ineligible, and trend-incomparable. No terminal
permits a false-to-true phase implication, and no cache terminal can claim current
mutation, worker, or test activity. Any created child requires confirmed or
unconfirmed cleanup; `not_required` means the OS child never existed.

Unavailable states are explicit: missing base, unsupported tool, invalid config,
baseline test failure, no eligible source, no test mapping, budget exhaustion, and
report write failure never become an empty successful mutation score.

### Persistent-write safety

Mutation state has explicit single writers: the supervisor alone owns and atomically
writes the invocation receipt and every fallback, including ordinary controller
publication failures. The controller may only send one typed bounded
`fallback_request`; it cannot open or replace a fallback path. The
controller/report-store owns one run staging/final generation; cache and trend owners
publish post-report projections under the shared output lock.
Output defaults to repository `.mutation-testing/`. An override must resolve inside that
directory, have no symlink ancestor, and either be absent or contain the exact
`.trade-mutation-output-v1` ownership marker; filesystem roots, repository `data/`,
DB/model/Parquet/generated-artifact paths, and unrelated existing directories are
rejected before locks or files are created. Each invocation owns one UUID `run_id`
staging directory and holds a per-run lease for its lifetime. The lease records PID,
process start token, and boot identity. An exclusive bounded output-root publication
lock has numeric acquisition and phase sub-deadlines and serializes current-pointer,
retention, cache projection, trend sequence reservation/source append, and trend
reconciliation. Compact retention indexes bound under-lock work; reaching a
sub-deadline leaves a typed projection gap rather than scanning past the reserve.
Cache records are content-addressed by complete cache identity
and execution ID but first stage under `cache/staging/<run_id>`. The immutable report
records only staged cache digests and eligibility. After `current.json` publishes, a
digest-bound `cache/commits/<run_id>.json` projection marker may expose eligible
entries; readers require the marker and ignore entries left by partial, cancelled,
degraded, report-failed, or cache-commit-failed runs. A later trend projection failure
does not revoke an already valid cache marker. Same-key writers use exclusive
creation or a per-key lock, and a losing writer accepts the predecessor only after
schema, identity, size, digest, and commit-marker verification.

Before publication, one safe-error layer redacts configured credential values,
credential-looking environment values, URL userinfo, and controlled home/temp roots
from console output, JSON, Markdown, HTML, fallback diagnostics, and retained staging
evidence. The controller validates the report schema and closed count equations, one
terminal state for every selected mutant, all configured size ceilings, and the
SHA-256 and byte size of JSON, Markdown, and HTML. `manifest.json` hashes every
generation member except itself. After read-back verification, the controller hashes
the manifest. Under the output lock, `report_store.py` reserves and fsyncs the next
monotonic `report_sequence` in `report-sequence.json`; this orders every published
changed/core/full/plan/zero-work generation and has no trend meaning. The reservation
binds run ID and expected manifest digest. The
controller then flushes and fsyncs each file and the directory, renames the directory
to its immutable final run path, and atomically replaces and directory-syncs
`current.json` containing run ID, manifest SHA-256, manifest byte size and reserved
report sequence. The supervisor updates the invocation receipt with the same root
hash and report sequence. Readers resolve the
receipt or pointer once, validate that root, then validate member digests and read only
that generation. They never compose output from separate runs. Cache records and the
bounded trend source/projection use the same write, fsync, atomic-replace,
directory-fsync discipline.

A crash before pointer replacement leaves the previous complete generation current;
a crash after replacement exposes only the already verified new generation.
On the next locked operation, every unresolved report reservation is reconciled before
a new reservation: it is committed only when `current.json` exactly binds the same
run ID, report sequence, manifest digest, manifest byte size, and valid generation
root. Otherwise a report tombstone is committed. A stale pointer, same run with a
different root/size, or a valid root not named by that exact pointer never commits a
source. Report sequence values are never reused.
Unreferenced staging is not successful evidence. A scavenger may remove it only after
non-blocking acquisition of its lease, validation of owner identity, a minimum
24-hour age, and proof that it is not current; an active concurrent invocation cannot
be removed. Malformed, oversized, symlinked,
path-escaping, identity-mismatched, or hash-mismatched report state fails closed and
is preserved for diagnosis. A missing pointer on first run is normal. Under the
publication lock, a malformed/symlinked pointer, missing referenced generation, or
manifest/digest mismatch is moved to a UUID quarantine record before a valid new
generation may replace it; the next manifest records the exact recovery code and
quarantine path. If quarantine cannot be completed, publication fails and emits exact
`trade dev mutation inspect --run-id <run_id>` and
`trade dev mutation repair --run-id <run_id> --expected-manifest-sha256 <digest>`
commands. `inspect` is read-only; `repair` takes the same ownership/publication lock,
defaults to dry-run, requires `--apply` to quarantine or replace owned state, and never
follows a symlink or touches an active lease. Corrupt or uncommitted cache is a miss
and is recomputed.

After report publication, the projection transaction exposes the eligible cache
marker. Only core/full then reserves the independent next `trend_sequence` in
`trend-sequence.json`, commits one immutable compact content-addressed
`trend-sources/<trend_sequence>-<run_id>-<digest>.json` record, advances the trend
high-water with this digest, then reconciles `trend.jsonl`. Changed/plan/zero-work
report publication cannot consume a trend sequence. Each source or tombstone includes
the previous retained digest, producing a validated hash chain anchored by the trend
high-water record. A retention checkpoint records the digest immediately before the
first retained record; stale restore below the high-water or a chain break fails
closed. The trend-source record contains the report root hash and report sequence,
run/mode/source identity, numerator/denominator key-set digests, score/counts,
comparability and trend sequence, but no mutant diagnostics. It is retained for 365
records, 32 MiB, and 400 days independently of 30-day report generations.
`trend.jsonl` is an idempotently rebuildable projection of that ledger. A
post-publication cache/ledger/
trend failure is recorded in the invocation receipt as `projection_degraded`; it
leaves the immutable factual report valid, leaves cache unreadable when its commit
failed, and makes trend expose an explicit gap until reconciliation. It never
retroactively changes the report to `report_failed`.

Deadline, cancellation, worker, or baseline degradation still produces a marked
partial generation when the report store is available. Deadline remainder is
`not_run_budget`; signal-stopped active work is `cancelled_test_started` and queued
remainder is `not_run_cancelled`. Render,
staging-validation, final-directory rename, and current-pointer publication failures
are distinct. A valid completed generation is published only if every validation
passes. Otherwise exit 2 preserves the failed run staging path and sends one bounded
authenticated `fallback_request` to the supervisor. Only the supervisor atomically
writes `fallback-<run_id>.json` with schema `trade.mutation.fallback.v1`, redacted
stable error code, failing stage, retryability, message, remediation command, receipt
path, and evidence paths outside `current.json`; it never labels fallback files as an
authoritative generation. A malformed, replayed, or unauthenticated request produces
a supervisor-owned protocol-error fallback. The invocation receipt identifies this
exact fallback, so CI never falls back to stale `current.json` or newest-file
globbing. The manifest audits run
ID, UTC time, mode, source/base
identity, complete config/tool/environment/cohort digests, budgets, stop reason,
status, member hashes, staged cache eligibility, and trend-source inputs without
credentials or raw environment values. Projection outcomes live in the receipt and
ledger, not the immutable manifest. If bootstrap/controller exits unexpectedly, is
SIGKILLed, or is proven OOM-killed by cgroup `memory.events`, the surviving supervisor
terminates descendants, verifies any run-ID final generation/current root already
published, and either anchors that root in the receipt or atomically writes a
`controller_lost` fallback. OOM attribution requires a unique invocation/worker
cgroup: a pre-spawn `memory.events` snapshot, a post-wait increase in `oom_kill`, the
signalled PID/start token being a member of that cgroup, `SIGKILL`, and no competing
member kill in the same interval. Shared-root counter deltas, absent membership, or
an ambiguous concurrent delta are `unknown_sigkill`, never OOM. `proven_oom` and
`unknown_sigkill` are closed infrastructure reasons and map to
`infrastructure_error_test_started` when `test_started_current=true`, otherwise
`infrastructure_error_pre_test`, using the sole phase/cleanup table above. Controller
OOM or SIGKILL is not a mutant record; the supervisor emits `controller_lost`
fallback. Only
host/runner/supervisor loss remains best effort.

Immutable generations are retained for at most 30 entries, 2 GiB, and 30 days;
compact trend-source records for 365 entries, 32 MiB, and 400 days; invocation
receipts/tombstones for 400 entries, 64 MiB, and 400 days; fallback/quarantine/
failed-staging evidence for at most 50 entries, 512 MiB, and 14 days. Deterministic
oldest-creation/run-ID eviction under the publication lock protects `current`, active
leases, and current failure evidence. Retention is a DAG, not one synchronous unit:
each receipt declares `bundle_expectation=required_ci|not_applicable_local` plus its
expected-member manifest. The raw-evidence unit is `receipt + supervisor-attestation
journal + report-or-fallback + expected bundles`; CI expects one named bundle and
local runs expect none. Required-missing and local-not-applicable are distinct.
Members are evicted together or the receipt becomes bounded
`trade.mutation.evidence-tombstone.v1` with former roots/sequences, reason, time, and
`detail_evidence=expired`. A compact trend source/aggregate is a separately retained
digest-bound derived record and may outlive raw evidence. It retains score/counts,
key-set digests, report root and sequence but never claims mutant detail remains.
Expired-detail trend records remain continuity projections only: they cannot establish
or reconstruct a baseline, satisfy a detailed comparison, or make `inspect` healthy.
`inspect` returns `evidence_expired`; every eviction is recorded. No backup is
required for ignored disposable developer state. CI uploads a validated immutable
invocation bundle, not mutable staging. Rollback removes tracked tooling
without rewriting or deleting these audit generations, which remain subject to
explicit manual cleanup and never enter production state.

### Contracts and compatibility

#### CLI

```text
trade dev mutation COMMAND [COMMAND_OPTIONS] [OUTPUT_OPTIONS]
scripts/mutation-test COMMAND [COMMAND_OPTIONS] [OUTPUT_OPTIONS]

COMMAND := changed | core | full | inspect | repair | qualify | reconcile

changed:
  [--base REF]

changed|core|full:
  [--output-dir PATH]
  [--max-mutants N]
  [--max-seconds N]
  [--workers N]
  [--plan-only]
  [--no-cache]

trade dev mutation inspect
  (--run-id UUID [--output-dir PATH] | --bundle PATH)
  [--format text|json]
trade dev mutation repair --run-id UUID
  [--output-dir PATH]
  --expected-manifest-sha256 SHA256
  [--format text|json]
  [--apply]

trade dev mutation qualify --mode core|full --runner-profile PROFILE
trade dev mutation reconcile --trend-epoch EPOCH
  --carrier PATH
  [--output-dir PATH]
  [--apply]

OUTPUT_OPTIONS := [--format text|json] [--quiet | --verbose]
```

`trade dev mutation` is canonical and root-dispatched before the ordinary `trade dev`
`uv` path; `scripts/mutation-test` supports all seven commands and is an argv/exit-
compatible facade. `COMMAND` is required and closed to the seven values above.
Execution-only options are rejected on administrative commands; `--base` is accepted
only by `changed`; `--mode`/`--runner-profile` only by `qualify`;
`--trend-epoch`/`--carrier`/`--apply` only by `reconcile`; `--bundle` only by
`inspect`; and `--run-id`, `--expected-manifest-sha256`, and `--apply` only as shown
for inspect/repair. `--output-dir` is valid on execution and run-root administrative
commands.
Changed mode defaults
to `${MUTATION_BASE_REF:-origin/master}` locally; CI always passes the pull-request
base SHA. Failure to resolve the explicit base is an infrastructure/configuration
error, never a full-scope fallback. `--output-dir` is subject to the owned safe-root
rules above.

`--plan-only` validates config and Cosmic Ray compatibility, discovers tracked scope,
parses eligible definitions, enumerates and selects deterministic first-order plans,
and publishes a report. It does not build private trees, run coverage/pytest, or read
cached outcomes. Its run status is `plan_only`, every selected mutant is
`not_run_plan`, score is `null`, and it exits 0 unless preflight or report publication
fails.

`inspect` is read-only and returns `trade.mutation.inspect.v1`; it resolves either one
explicit run under one owned output root or one explicit immutable bundle, never a
recursive scan or newest-file inference. Bundle inspection needs no original runner
or mutation dependency. `repair` returns
`trade.mutation.repair.v1`. Both support text/JSON, closed
`healthy|first_run|repairable|active_lease|stale_expected_digest|evidence_expired|
unsafe|internal_error` states, stable error/remediation fields and deterministic
evidence paths. Inspect exits 0 for `healthy|first_run`, 1 for
`repairable|active_lease|stale_expected_digest|evidence_expired`, and 2 for
`unsafe|internal_error`.
Repair dry-run exits 0 for no-op or a valid plan, 1 for active lease/stale expected
digest/evidence expired, and 2 for unsafe/internal errors; `--apply` exits 0 only
after read-back verification. Neither command invokes mutation dependencies, follows
symlinks, or touches active leases.

`qualify` measures one reviewed CI runner profile and emits an unsigned proposal; it
never edits the checked-in capacity record. Its response schema is
`trade.mutation.qualification.v1` with closed status
`qualified_proposal|unqualified_proposal|preflight_failed|measurement_failed|
internal_error`; a completed qualified or unqualified proposal exits 0 and the latter
three exit 2. `reconcile` returns `trade.mutation.reconcile.v1` with status
`clean|gap_plan|applied|active_lease|stale_epoch|unsafe|internal_error`, validates a
named trend epoch and explicit downloaded carrier,
shows missing/tombstoned sequences and projection repairs in dry-run form, and requires
`--apply` plus the publication lock to commit a gap tombstone or rebuilt projection
and emit a repaired-carrier proposal. Local apply never uploads. The manual CI
recovery route must bind artifact identity/digest, run dry-run first, require an
environment approval, revalidate, and publish only under a strictly newer workflow
tuple. It never invents a predecessor or score. `clean|gap_plan|applied` exits 0,
`active_lease|stale_epoch` exits 1, and `unsafe|internal_error` exits 2.

All commands support `--format text|json` and mutually exclusive `--quiet|--verbose`.
JSON mode writes exactly one UTF-8 JSON document plus newline to stdout, with no ANSI,
progress, warnings, or human prefixes; all diagnostics/progress go to stderr. Text
mode writes the final result to stdout and diagnostics/progress to stderr. `--quiet`
suppresses progress and non-error diagnostics but not the final stdout document;
`--verbose` emits bounded phase transitions at most once per phase and once per five
seconds, with no credentials. Non-TTY output never uses ANSI. Parse errors write only
diagnostic text to stderr and exit 2. Root/facade argv, output bytes/channels,
receipt export, supervisor-first execution, dependency-failure text, and exits are
golden parity-tested for every command, state, format, and verbosity.

Exit codes:

- `0`: `complete`, `zero_work`, `deferred_only`, `plan_only`, or `budget_partial`
  reports produced without infrastructure failures, including ordinary survivors,
  timeout, or no coverage;
- `1`: mutation policy regression only when an explicitly established baseline is
  configured for enforcement (not enabled in this change);
- `2`: `preflight_failed`, `degraded_infrastructure`, `signal_cancelled`, or
  `report_failed`, including invalid invocation/config/tool, baseline failure, source
  drift, provenance failure, launch failure, or report failure. SIGINT/SIGTERM are
  normalized to exit 2 after descendant cleanup and partial report publication.

An eligible-empty run with one or more unlisted/untracked production paths is
`deferred_only`, exits 0, has score `null`, and lists those paths without running
pytest. Run-status precedence is `report_failed` then `degraded_infrastructure`,
`signal_cancelled`, `preflight_failed`, `budget_partial`, `plan_only`,
`deferred_only`, `zero_work`, and `complete`; future policy regression affects exit 1
but does not overwrite factual run status. Any integrity or cleanup override forces
`degraded_infrastructure`; an earlier signal remains `stop_reason`/causation only.
Infrastructure failure in one mutant does not turn another mutant into killed. The
controller stops only the affected test tuple unless integrity, process ownership,
global deadline, signal, or report publication requires run-wide cancellation.

The report schema is `trade.mutation.report.v1`. CI consumes `summary.md` and uploads
all report files but does not infer truth from logs.

#### Error registry

`trade.mutation.error-code.v1` is the closed owner of machine error semantics. Every
fallback, administrative response, restore diagnostic, bundle validation, and CI
trusted summary carries one code below or `none`; ad hoc strings cannot become codes.
Command templates substitute only validated IDs/digests/paths:

| Code | Owner/stage | Retryable | Summary severity | Remediation template |
|---|---|---:|---|---|
| `invalid_invocation` | CLI/parse | no | error | `trade dev mutation COMMAND --help` |
| `missing_dependency` | bootstrap/dependency | yes | error | `uv sync --extra mutation --frozen` |
| `unsupported_execution_host` | preflight/kernel | no | error | `trade dev mutation COMMAND --plan-only` |
| `unsafe_output_root` | preflight/storage | no | error | `trade dev mutation inspect --run-id RUN_ID` |
| `source_drift` | plan/publication | yes | error | `trade dev mutation COMMAND --base REF` |
| `protocol_auth_failed` | supervisor/protocol | no | error | `trade dev mutation inspect --run-id RUN_ID` |
| `protocol_capacity_exceeded` | supervisor/protocol | yes | warning | `trade dev mutation COMMAND --workers N` |
| `spawn_failed` | supervisor/spawn | yes | error | `trade dev mutation inspect --run-id RUN_ID` |
| `isolation_failed` | worker/isolation | no | error | `trade dev mutation COMMAND --plan-only` |
| `incomplete_evidence` | finalization/integrity | yes | error | `trade dev mutation inspect --run-id RUN_ID` |
| `proven_oom` | supervisor/wait | yes | error | `trade dev mutation COMMAND --workers N` |
| `unknown_sigkill` | supervisor/wait | yes | error | `trade dev mutation inspect --run-id RUN_ID` |
| `cleanup_unconfirmed` | supervisor/cleanup | yes | error | `trade dev mutation inspect --run-id RUN_ID` |
| `baseline_unavailable` | coverage/baseline | yes | warning | `trade dev mutation COMMAND --plan-only` |
| `report_publish_failed` | report/publication | yes | error | `trade dev mutation repair --run-id RUN_ID --expected-manifest-sha256 DIGEST` |
| `projection_degraded` | cache/trend/projection | yes | warning | `trade dev mutation reconcile --trend-epoch EPOCH --carrier PATH` |
| `cache_restore_miss` | cache/restore | yes | notice | `trade dev mutation COMMAND --no-cache` |
| `trend_predecessor_unavailable` | trend/restore | yes | warning | `trade dev mutation reconcile --trend-epoch EPOCH --carrier PATH` |
| `trend_predecessor_invalid` | trend/restore | no | error | `trade dev mutation reconcile --trend-epoch EPOCH --carrier PATH` |
| `carrier_not_monotonic` | trend/publication | no | warning | `trade dev mutation reconcile --trend-epoch EPOCH --carrier PATH` |
| `evidence_expired` | retention/inspect | no | warning | `trade dev mutation COMMAND` |
| `active_lease` | storage/admin | yes | warning | `trade dev mutation inspect --run-id RUN_ID` |
| `stale_expected_digest` | storage/admin | no | error | `trade dev mutation inspect --run-id RUN_ID` |
| `bundle_missing` | CI/validation | yes | error | `gh run rerun RUN_ID` |
| `bundle_invalid` | CI/validation | no | error | `gh run download RUN_ID -n ARTIFACT -D DIR && trade dev mutation inspect --bundle DIR` |
| `artifact_upload_skipped_quota` | CI/upload | yes | warning | `gh api repos/OWNER/REPO/actions/artifacts` |
| `hard_runner_loss` | CI/runner | yes | error | `gh run rerun RUN_ID` |

The registry's implementation is shared immutable data in `models.py`; owner modules
may select but not redefine code, retryability, severity, or template. Unknown codes
make bundle validation fail.

#### Dependencies

`[project.optional-dependencies].mutation` pins `cosmic-ray==8.4.6`,
`coverage==7.10.7`, and `pytest>=8,<9`. The root dispatch/facade resolves the
repository, records command-entry `/proc/uptime`, and `exec`s the absolute selected
Python 3.7+ interpreter with
`-I trade_py/devtools/mutation_testing/supervisor.py`; it does not invoke `uv`. The
standard-library-only supervisor creates the run ID, receipt,
subreaper/watchdog, and absolute monotonic deadline before every `uv` child. It first
starts the import-light bootstrap in an owned session with
`uv run --frozen --no-sync python -m trade_py.devtools.mutation_testing.bootstrap`
and passes the same receipt/deadline to every child.
For changed mode the bootstrap resolves Git/config and writes a zero-work or
deferred-only report without importing or resolving Cosmic Ray. Eligible work, and
all core/full/plan-only requests, are handed off exactly through
`uv run --frozen --no-sync --extra mutation python -m
trade_py.devtools.mutation_testing.cli`; CI prepares the frozen mutation extra before
starting the bounded command. Local missing dependencies fail preflight with the
exact preparation command rather than synchronizing outside the deadline. Coverage
line data is mandatory before
mutant execution; the definition-to-test matrix remains authoritative and prevents a
full-suite fallback. Execution requires Linux with the configured Landlock ABI and
`no_new_privs`; unsupported systems may use deterministic `--plan-only` but fail
execution preflight rather than weakening isolation. Runtime dependency resolution is
unchanged.

#### Backward compatibility

All existing commands remain unchanged. The optional group and script are additive.
No current CI check is replaced. Removing the workflow and optional group restores
the previous developer surface.

### Mutation model

#### Operator allowlist

The configuration stores exact Cosmic Ray 8.4.6 plugin names, not broad families.
The v1 list is:

- `core/AddNot`;
- `core/ReplaceAndWithOr`, `core/ReplaceOrWithAnd`;
- `core/ReplaceTrueWithFalse`, `core/ReplaceFalseWithTrue`;
- `core/ReplaceComparisonOperator_Eq_NotEq`,
  `core/ReplaceComparisonOperator_NotEq_Eq`;
- `core/ReplaceComparisonOperator_Lt_LtE`,
  `core/ReplaceComparisonOperator_LtE_Lt`;
- `core/ReplaceComparisonOperator_Gt_GtE`,
  `core/ReplaceComparisonOperator_GtE_Gt`;
- `core/ReplaceComparisonOperator_Is_IsNot`,
  `core/ReplaceComparisonOperator_IsNot_Is`;
- `core/ReplaceBinaryOperator_Add_Sub`, `core/ReplaceBinaryOperator_Sub_Add`;
- `core/ReplaceBinaryOperator_Mul_Div`, `core/ReplaceBinaryOperator_Div_Mul`;
- `core/ReplaceBinaryOperator_FloorDiv_Div`,
  `core/ReplaceBinaryOperator_Mod_Mul`;
- no NumberReplacer, break/continue, zero-loop, variable, exception, decorator, or
  unary deletion operators.

The `True`/`False` plugins replace boolean literals, not generic return values. Generic
return-value mutation remains deferred because 8.4.6 has no such plugin. The
controller rejects any version other than 8.4.6, checks every name, and verifies the
required `get_operator` and `mutate_code` signatures. An installed operator never
becomes enabled automatically.

#### First-order proof

Enumeration calls `cosmic_ray.ast.get_ast(source)`, iterates
`cosmic_ray.ast.ast_nodes(tree)` in preorder, and for each node iterates the selected
operator's `mutation_positions(node)` in returned order while maintaining one
independent occurrence counter per `(source_digest, operator_name,
operator_arguments)`. This matches Cosmic Ray's new `MutationVisitor` per operator;
there is no cross-operator occurrence counter. Selection serializes one candidate per
plan.
Immediately before mutation, the adapter re-hashes/re-enumerates using that exact
contract, requires occurrence-to-span equality, calls `mutate_code`, parses the
output, and requires the diff to intersect only that planned location. Source drift
or a mismatch with `MutationVisitor` is infrastructure failure; unsupported output is
invalid. The worker also rejects `len(mutations) != 1`.

#### Deterministic priority

1. exact changed line;
2. configured core path priority;
3. configured defect-history priority;
4. relative path, line, column, operator priority, occurrence.

Changed mode admits only category 1 inside the closed matrix and never widens to an
unchanged line or file. Core/full use categories 2-4. Truncation records the last admitted key, the exact eligible
candidate count seen, `scan_truncated=true`, and
`unscanned_candidate_remainder="unknown"`; it never scans past a ceiling to invent an
exact omitted total.

### Target and test selection

The exact v1 matrix is definition-level:

| Scope | Source | Eligible qualified definitions | Tests | Clock policy | Entropy policy |
|---|---|---|---|---|---|
| core | `trade_py/decision/action.py` | `_confidence_label`, `derive_action_decision` | `tests/test_decision_action.py`, `tests/test_explanation.py` | explicit_input | deterministic |
| core | `trade_py/decision/world_state.py` | `infer_market_regime`, `infer_event_regime`, `infer_sentiment_regime`, `infer_technical_regime`, `infer_liquidity_regime`, `infer_uncertainty`, `_build_state_summary`, `build_world_state` | `tests/test_world_state.py`, `tests/test_decision_action.py`, `tests/test_explanation.py` | explicit_input | deterministic |
| core | `trade_py/decision/scenario.py` | `_count_bullish`, `_count_bearish`, `build_scenario_summary` | `tests/test_explanation.py` | explicit_input | deterministic |
| core | `trade_py/decision/explanation.py` | `build_explanation` | `tests/test_explanation.py` | explicit_input | deterministic |
| core | `trade_py/trust/compute.py` | `_freshness_score`, `_trust_level`, `compute_prediction_trust`, `compute_portfolio_trust` | `tests/test_trust_layer.py` | explicit_input | entropy_non_cacheable |
| full | `trade_py/factors/groups/event_features.py` | `build_event_group` | `tests/test_factor_groups.py` | explicit_input | deterministic |
| full | `trade_py/signals/window_scorer.py` | `_score_large_order` | `tests/test_window_scorer.py` | explicit_input | deterministic |

Configuration stores each definition's exact qualified top-level name; the run report
binds it to current source and definition digests. AST validation rejects a missing,
duplicated, nested-ambiguously, or moved definition until the matrix is reviewed.
Changed uses changed lines only inside eligible
definitions; core uses core rows; full uses all rows. Thus full is the complete
configured v1 scope, not the whole repository. Dataclass/DTO serializers and
`window_scorer` DB/cache/batch/provider entrypoints are explicitly absent, as are
event jobs, Web/CLI/devtools, migrations, generated/vendor, logging/startup, C++,
React, and unlisted Python modules. Unlisted tracked production changes and untracked
production paths are reported as `deferred_unmapped` or `deferred_untracked`; they
generate no mutant and do not affect score. Observatory's current catalog failure
does not create a blanket exclusion; independently stable Observatory/PIT definitions
require a later reviewed matrix addition with an explicit clock policy.

Static imports only validate mapping shape. A passing private-tree baseline must
collect coverage.py line data. A candidate whose exact line is absent becomes
`no_coverage_line` and does not start mutant pytest. A missing/empty mapping is an
invalid closed-matrix configuration; a matrix-external path is `deferred_unmapped`.
Coverage never adds an unconfigured test or falls back to all tests.

### Failure and recovery

The facade-provided `/proc/uptime` anchor is converted once at supervisor entry to
the supervisor's `CLOCK_BOOTTIME` domain; that clock and the kernel boot ID bound
planning, baselines, execution, cancellation, publication, and projection. A
controller stop deadline reserves cleanup/publication time before the outer script
deadline. The supervisor enables `PR_SET_CHILD_SUBREAPER`, launches every child in an
owned session from authenticated protocol requests, forwards signals, and maintains
PID/start-token/boot-identity descendants. Every supervisor child starts through a
Python 3.7-compatible fork-stop trampoline: before exec or application code the child
raises `SIGSTOP`. The supervisor creates one invocation cgroup and one unique child
cgroup. Immediately after successful OS child creation it records
`worker_spawned_current=true`, PID/start token, and a provisional registry entry in
the supervisor attestation journal. It then places the stopped child, sets resource
controls, reads back membership, and installs pidfd/session/watchdog ownership before
recording `worker_registered_current=true` and sending `SIGCONT`; placement or
read-back failure kills/reaps the stopped child and records confirmed or unconfirmed
cleanup instead of pretending cleanup was not required. When `clone3`
`CLONE_INTO_CGROUP` is available an equivalent tested supervisor helper may replace
this handshake. The supervisor stays outside the invocation cgroup, sets
`memory.oom.group=1`, and verifies every worker cgroup and the invocation
`cgroup.procs` are empty before exit. Execution requires a writable delegated cgroup
v2 subtree with finite `memory.max`; cgroup v1, a read-only/shared cgroup, or no
delegation permits `--plan-only` only and otherwise fails preflight. Subreaper and
bounded `/proc` closure checks remain defense in depth, not an execution fallback.
The trusted launcher activates Landlock/seccomp before application import and then
denies fork/vfork/exec and process-form clone; reviewed thread-form clone remains in
the same worker process/session/cgroup. Any ownership or policy-activation failure
kills the group and fails closed.
The supervisor terminates and reaps the remaining hierarchy if bootstrap/controller
hangs, crashes, or receives SIGKILL. Supervisor or host SIGKILL and runner loss remain
explicit best effort, not a no-orphan guarantee.

A controller-owned thread pool performs logical scheduling but submits every
coverage/pytest launch to the sole supervisor spawner and receives only an opaque
worker handle. Supervisor child creation, stopped-child registration, controller-visible
admission acknowledgement, typed stop-event enqueue, and wait observation share one
protocol/state gate. Every mutant follows
`PLANNED -> MUTATED? -> STARTED? -> EXIT_SEEN|STOP_SEEN -> FINALIZING -> TERMINAL`.
All events carry a `CLOCK_BOOTTIME` timestamp and supervisor-assigned sequence.
Natural exit, per-mutant timeout, global signal, and execution budget form one
first-observed control class. Integrity causes
`guard_violation|syscall_violation|listener_failure|provenance_failure|
protocol_failure|incomplete_evidence` are sticky and dominant regardless of sequence.
A natural/control event records `EXIT_SEEN` or `STOP_SEEN` but cannot terminalize
until the seccomp notification queue is drained, listener closure is resolved, and
both independent guard/audit streams reach authenticated completion or an allowed
supervisor forced-close record. Any integrity
cause observed during that bounded finalization replaces a provisional natural,
timeout, signal, or budget result with the phase-appropriate infrastructure terminal.
After process termination, cleanup status is evaluated last; `unconfirmed` performs
the same infrastructure override and disables run score/cache. Only when integrity
and cleanup are clean does the first control-class event own classification.
Budget or signal after mutation but before spawn yields the corresponding
`not_run_*` terminal with `mutation_applied_current=true`. Golden tests cover every
pairwise race, late integrity notification, both stream completion orders, cleanup
override, and both phase facts. Worker exceptions, lost protocol channels, repeated
signals, and parent cancellation use one supervisor TERM, bounded wait, KILL,
cgroup/group-existence check, and reap path.

A hard termination cannot require a killed child to emit `guard_completed`. For an
already provisional `timeout_test_started|cancelled_test_started`, forced close uses
one bounded ownership handoff: the controller stops broker admission, resolves or
denies already-received notifications, finalizes its syscall-audit digest, and passes
the listener plus digest in a typed request. The supervisor validates tracee/listener
identity, takes exclusive listener ownership, proves a nonblocking receive has no
pending notification, appends an ACK to its source-exclusive journal, and only then
allows the controller endpoint to close. It terminates the exact worker, observes
guard EOF, and confirms the unique cgroup empty plus wait/reap before appending
`trade.mutation.guard-forced-close.v1`. The attestation binds the last valid guard
sequence/digest, final audit digest, handoff/ACK sequences, provisional control
event/sequence, TERM/KILL facts, notification-drain result, cleanup result, and reason
`forced_timeout|forced_cancel`.
It substitutes only for `guard_completed`; any sequence gap, pre-existing truncation,
wrong identity, missing/late ACK, listener loss/deadline, nonempty notification queue,
unconfirmed cleanup, natural exit, or budget stop remains `incomplete_evidence`
infrastructure failure. Thus an infinite
loop can truthfully remain timeout after supervisor-proven forced closure without
turning arbitrary EOF into success.

Each thread creates one private tree from a sorted bounded import closure rooted at
the eligible source, mapped tests, required package `__init__.py` files, tracked
`tests/conftest.py` when present, and the injected provenance/isolation plugin. Static
imports are resolved only to repository-local regular Python files within the module,
edge, depth, file, byte, and deadline ceilings. Configuration must list an exact
reviewed dependency manifest for unresolved dynamic imports; missing or extra
requirements fail preflight. The complete tracked-Python digest remains audit metadata
but is neither the cache reuse identity nor the copy manifest. Symlinks, non-regular
files, bytecode, caches, data, and unrelated tests are rejected or absent. A 10x
unrelated-module fixture must leave the same bounded closure executable and cache-
reusable. The tree is reused across mutants. Before every work
item it restores and hashes the target, removes all
bytecode/cache paths, creates a fresh per-mutant `PYTHONPYCACHEPREFIX`, sets
`PYTHONDONTWRITEBYTECODE=1`, revalidates planned source identity/span, applies only
`get_operator` plus `mutate_code`, and restores/hash-checks afterward. Consecutive
same-size mutations with fixed mtimes are a required regression test.

Pytest runs from the private root with absolute mapped tests, importlib mode, checkout
removed from import paths, and a provenance plugin verifying module `__file__` and
digest. Before application/test import, an isolation launcher installs a Linux
Landlock ruleset whose read allowlist contains only the private closure, selected
tests, Python/runtime shared libraries and immutable system metadata, and whose write
allowlist contains only per-worker temp/output roots. Unsupported Landlock ABI,
unavailable `no_new_privs`, or incomplete rules fail preflight; there is no
non-isolated fallback. Worker spawn uses `close_fds` with an exact pass-FD list so no
pre-opened production descriptor bypasses policy.

Before application code, the launcher creates a seccomp user-notification filter and
transfers its listener exactly once over a pre-created `SOCK_SEQPACKET` channel using
`SCM_RIGHTS`; the worker never receives the controller audit writer. The controller
validates listener/tracee IDs before every decision, closes transfer endpoints at
defined handshake points, bounds receive/drain, and kills the process group if the
listener closes or the controller dies. The reviewed pass-FD table is exactly the
child's guard-pipe write end, listener-transfer socket until ACK, stdout and stderr;
the supervisor alone retains the guard-pipe read end, the child never receives it,
stdin is `/dev/null`, and every other descriptor is closed/read-back audited.

The broker records and validates `openat/openat2`, socket/connect, fork/vfork,
execve/execveat, clone and clone3 attempts before returning a decision. Allowed file
opens use controller-side `openat2` with `RESOLVE_BENEATH|RESOLVE_NO_MAGICLINKS|
RESOLVE_NO_SYMLINKS`, validate the resolved descriptor against the allowlist, then
inject it with `SECCOMP_IOCTL_NOTIF_ADDFD`; raw tracee pathname opens are never
continued. Network and process creation are recorded and denied. Thread-form clone is
allowed only for reviewed flags; process-form clone is denied. Landlock remains the
kernel backstop for any non-brokered path and for descriptor use after injection. This
controller-owned broker is the pathname authorization point and catches native
SQLite/pyarrow/DuckDB access even when application code catches EPERM, while permitting
required native library threads.

The environment sets temporary `TRADE_DATA_ROOT`, HOME/XDG/cache directories,
UTC/locale/hash seed, native thread variables `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`BLIS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to `1`, scrubs credentials/proxies,
and adds Python audit/guard hooks for higher-fidelity diagnostics. The supervisor
creates a per-worker 256-bit HMAC key, passes one sealed copy and the guard pipe's
write end to the launcher, retains the verifier copy plus guard read end itself, and
passes neither key nor endpoint to the controller. The child emits a gap-free
HMAC-SHA-256 sequence of `guard_started`,
`kernel_policy_active`, zero or more `guard_violation`, and `guard_completed`. The
supervisor verifies identity, MAC, sequence, and digest chain online, destroys the key
after finalization, appends the result or forced-close entry to its source-exclusive
attestation journal, then gives the controller only that entry digest over the
ordinary authenticated control channel.
The controller alone owns the syscall audit records and private append state; it writes
`syscall_audit_started`, zero or more `syscall_violation`, and
`syscall_audit_completed` before durable outcome serialization. The child cannot forge
the audit stream; the controller cannot forge the supervisor journal because its
Landlock boundary denies every write to that root and it owns no writable descriptor.
Both bind run ID, worker PID/start token, target digest and
policy digest. Classification drains pending notifications and requires both verified
sequences. Missing, empty, truncated, duplicate, wrong-key or incomplete evidence is
`infrastructure_error`. Any child or syscall violation
overrides pytest exit 0/1 even if application or native code catches an exception.
These controls and versions are cache/report identity. Baselines use the same controls
plus line coverage and a finite timeout bounded by remaining execution time.

The coverage adapter is exactly coverage.py 7.10.7 in statement/line mode, not branch
mode. It creates and hashes a controller-owned minimal rcfile disabling branch,
plugins, patches, parallel mode and relative files; clears every `COVERAGE_*`
environment variable; and passes `--rcfile` to both commands. It invokes `python -m
coverage run --rcfile <private-rc> --data-file <private-temp> --source
<canonical-private-target> -m pytest --import-mode=importlib
<absolute-mapped-tests>`, then `python -m coverage json --rcfile <private-rc>
--data-file <private-temp> --include <canonical-private-target> -o <private-json>`.
Both commands use the same supervisor and environment. The target-filtered JSON must
contain exactly one canonical private target. The adapter
maps it back to the repository-relative source, extracts bounded `executed_lines`,
rejects a missing/duplicate/unknown target or malformed/oversized data, and removes
raw data/JSON afterward. Before removal it persists one bounded immutable
`CoverageTupleEvidenceV1` containing tuple ID, repository source and source digest,
sorted collected node IDs, exact argv digest, rcfile digest, redacted environment and
provenance digests, coverage version, sorted unique executed lines plus digest,
process exits, duration, and closed outcome reason. Every selected record references
its tuple ID. Report and bundle validators reconstruct line membership and
`no_coverage_line` from this tuple rather than trusting stored aggregate flags.
Coverage exit/data failure is `baseline_unavailable`, never no-coverage or killed.

Pytest mutant exits map `0=survived`, `1=killed`, and `2/3/4/5` or unexpected signal
to infrastructure error only after both independent evidence streams finalize and
cleanup is confirmed. Typed precedence above resolves timeout/signal/budget/integrity
races. A baseline exit, coverage error, or baseline timeout produces
`baseline_unavailable` with the closed tuple-level reason
`test_failed|timeout|coverage_failed|provenance_failed|isolation_failed|
infrastructure_failed`; a baseline timeout is never counted as a mutant timeout. A
failed tuple does not execute its mutants and cannot create kills.

Cosmic Ray's `apply_mutation`, `use_mutation`, `mutate_and_test`, and `run_tests` are
forbidden because they write in place or collapse lifecycle/result truth.

### Performance and capacity

Mode constants:

| Limit | changed | core | full |
|---|---:|---:|---:|
| selected mutants | 150 | 1000 | 5000 |
| candidate scan | 3000 | 20000 | 100000 |
| wall clock | 600s | 1800s | 5400s |
| source files | 32 | 128 | 512 |
| aggregate source | 8 MiB | 32 MiB | 128 MiB |
| one source | 1 MiB / 100000 nodes / depth 256 | same | same |
| one private tree | 512 files / 8 MiB / 10s copy | same | same |
| all private trees | 2048 files / 32 MiB | same | same |
| free space after reservation | 512 MiB | 512 MiB | 512 MiB |
| import closure | 512 modules / 4096 edges / depth 32 | same | same |
| parse/operator visits | 3M | 20M | 100M |
| workers | computed max 4 | computed max 4 | computed max 4 |
| output per mutant | 64 KiB | 64 KiB | 64 KiB |
| retained diagnostic + diff | 8 KiB + 8 KiB | 8 KiB + 8 KiB | 8 KiB + 8 KiB |
| aggregate retained detail | 1 MiB | 8 MiB | 32 MiB |
| total JSON report | 32 MiB | 64 MiB | 128 MiB |
| Markdown / HTML / manifest | 4 / 8 / 1 MiB | 8 / 16 / 1 MiB | 16 / 32 / 1 MiB |
| complete generation | 48 MiB | 96 MiB | 192 MiB |
| peak report renderer memory | 64 MiB | 128 MiB | 256 MiB |
| shutdown/report grace | 30s | 60s | 120s |

The exact six-file mapped cohort collected 92 passing tests and each observed review
run remained below three seconds; exact core rows collected 90. This yields the
10-second initial floor without claiming a frozen duration. Each distinct affected-
test tuple is timed independently and records collected node IDs, revision, Python and
elapsed time. The changed cap remains 60 seconds even when a selected test baseline
would imply more; such a tuple is reported as incompatible with PR mode rather than
silently increasing the job.

The 30/60/120-second reserves are inside the 600/1800/5400-second controller budgets.
They split into TERM, KILL/observable-reap, render/hash/fsync, and
pointer-or-fallback sub-budgets of `5/5/15/5`, `10/10/30/10`, and `15/15/70/20`
seconds. Admission is rejected unless the computed mutant timeout plus the normal
TERM/KILL/reap allowance fits before the execution deadline. At that deadline the
controller closes admission and requests cancellation of every active worker cgroup
and process group; publication uses only the reserved remainder.

These are enforceable userspace budgets on qualified hosts, not a false promise that
Linux can synchronously reap a task stuck forever in uninterruptible kernel sleep or
that a failed filesystem must complete `fsync`. The supervisor uses pidfds where
available, `cgroup.kill`, process-group TERM/KILL, bounded observable-reap polling, and
a supervisor-owned fallback writer. A task still present after the cleanup sub-budget
receives cleanup status `unconfirmed`; its provisional terminal is replaced by the
phase-appropriate infrastructure error, the run is degraded, and no score/cache is
claimed. The receipt records its pidfd/cgroup evidence and the supervisor keeps
attempting cleanup until the outer CI timeout. An `fsync` or fallback write that does not return leaves no claimed
generation. The 15/45/110-minute CI timeouts are the final runner-level containment
for kernel/filesystem stalls and intentionally exceed controller budgets. Capacity
qualification fails if ordinary cancellation, reap, render, hash, fsync, and fallback
operations cannot complete inside their sub-budgets; documentation and reports call
out the remaining host-loss/kernel-stall limitation.

Source file/byte/tree/dependency and parse/operator-visit limits plus deadline
checkpoints bound operator-sparse trees where candidate count alone would not stop
planning. Effective CPU is the minimum of affinity, cgroup quota, and host count.
The CPU candidate is `min(4, max(1, floor(effective_cpu / 2)))`. Effective workers are
also limited by a finite enforceable delegated cgroup v2 memory ceiling. Qualification
cannot depend on an unmeasured peak: reviewed config sets finite
`controller_bootstrap_memory_max=512 MiB` and
`baseline_bootstrap_memory_max=1024 MiB`, both identity-bound and lowering-only.
Qualification caps/read-backs the controller at 512 MiB, then measures each selected
test tuple at `min(1024 MiB, effective_memory - controller/safety/renderer reserves)`.
The measurement must retain at least 256 MiB. Bootstrap-cap OOM, a peak reaching
`memory.high`, or inability to admit that finite cgroup fails the tuple/qualification;
there is no unbounded bootstrap. Later qualified execution uses measured controller
peak and each tuple's descendant `memory.peak`; the scheduler sets each worker cgroup
`memory.max = max(256 MiB, ceil(1.5 * measured_peak))`,
`memory.high = floor(0.9 * memory.max)`, `memory.swap.max = 0`, and `pids.max` to the
reviewed thread-only allowance. The invocation cgroup's finite `memory.max` is
`effective_memory`. Fixed reservations are the 512 MiB controller cap during
qualification and
`controller_reserve = max(256 MiB, ceil(measured_controller_peak * 1.25))` afterward,
`safety_reserve = max(256 MiB, floor(0.10 * effective_memory))`, and the mode's
64/128/256 MiB renderer reserve. Admission requires
`controller_reserve + safety_reserve + renderer_reserve +
sum(active worker memory.max) <= effective_memory`; values and read-back verification
are reported. A limit that cannot admit one worker fails execution preflight. Without
a writable delegated cgroup v2 subtree and finite enforceable memory limit, every
platform is plan-only; there is no one-worker execution fallback. The report records
every CPU/memory input, per-worker setting, reservation, read-back, and limiting
dimension.
Before copying, the complete deterministic manifest and aggregate worker reservation
must fit file, byte, copy-time, and remaining-space limits. Structural report capacity
and worst-case escaped detail are reserved for every selected mutant. JSON renders
incrementally; Markdown/HTML use the same bounded detail records rather than embedding
JSON. Detail truncates deterministically before any per-file, generation, or renderer
limit. CI disk admission additionally reserves the report generation, one full
uncompressed bundle copy, bounded 64 MiB upload/compression scratch, cache/aggregate
restore maxima, and both 20% and 512 MiB final free-space margins. Reflink or streaming
may reduce observed use but cannot reduce this conservative admission equation.

At 10x eligible source size, source and AST ceilings stop work before candidate
truncation; at 10x unrelated repository modules, the root-bounded import closure
remains within the same copy manifest rather than copying all `trade_py`, and retains
cache reuse because the whole-tree digest is audit-only.
At 10x test cost, baseline timing excludes incompatible tuples before scheduling or
the wall deadline terminates admission. At 10x output, bounded ring capture truncates
diagnostics without blocking pipes.

Qualification separates immutable runner capacity identity from measurement
throttling. `runner_capacity_identity` binds provider, image digest, architecture,
kernel, cgroup-v2 mount/delegation path and controller set, finite memory/quota,
Landlock ABI, seccomp notification/addfd support, filesystem type, Python/uv/tool/lock
digests, and isolation policy. `measurement_profile` records a requested
one-effective-CPU throttle, worker count, cache state, and selected cohort but is not
the runner's physical capacity identity. A core/full qualification passes only when:

- preflight and all isolation/cleanup probes pass under the same runner identity;
- every planned mode phase completes with no infrastructure, OOM, timeout, cleanup,
  report, or projection failure. The deterministic qualification cohort contains every
  configured source row and every distinct mapped-test tuple for that mode, with at
  least `max(30, 5 * tuple_count)` completed mutants and at least five completed
  mutants per tuple; insufficient candidates make qualification fail rather than
  substituting faster tuples;
- a capacity run observes
  `min(4, max(1, floor(effective_cpu/2)))` workers, while a separate serial diagnostic
  records one worker;
- each tuple records p50/p95 baseline, supervisor spawn/admission round-trip, execution,
  finalization and cleanup. Qualification deterministically list-schedules the
  configured mode's exact selected-limit distribution over `qualified_workers` using
  tuple-specific p95 end-to-end costs; unfilled tail slots use the slowest tuple p95.
  `projected_list_schedule_makespan + p95_baseline_total + p95_plan_copy +
  p95_render_fsync_cleanup <= execution_deadline * 0.80`;
- peak invocation memory is at most `effective_memory * 0.80`, each worker remains
  below `memory.high`, and disk use leaves both 20% and 512 MiB free;
- throughput `completed_mutants / active_execution_seconds` and projected worst-mode
  runtime both retain 20% time headroom.

One-effective-CPU evidence diagnoses serial rate; it cannot by itself qualify a
multi-worker schedule. The checked-in capacity record binds both serial and capacity-
run bundles.

### Observability and operations

Human output, Markdown and HTML all print factual run status, controller exit code,
base/head, eligible/deferred files, selected tests, budgets, progress, stop reason,
complete outcome counts, score and denominator, baseline comparability/reason, cache
freshness, projection state, cleanup/orphan result, remediation command, and exact
evidence paths. JSON is authoritative. Markdown contains the exact PR summary fields
requested by the user. HTML is static and self-contained.

Survivors and no-coverage entries display execution/mutant ID, comparison key,
outcome origin, original/mutated digest,
planned/observed span, source, definition, operator, bounded diff, tests, coverage,
duration, and cache identity. Timeout and infrastructure error remain separate.
Run status is one of `complete`, `zero_work`, `deferred_only`, `plan_only`,
`budget_partial`, `signal_cancelled`, `degraded_infrastructure`,
`preflight_failed`, or `report_failed`. Budget exhaustion records `mutant_limit`,
`candidate_scan_limit`, `parse_visit_limit`, `source_limit`, `execution_deadline`,
`signal`, or `report_limit`. A cancellation receipt records signal, admission-stop
UTC/monotonic offset, TERM/KILL/group-check/reap counts, cleanup duration, and
orphan-check result.

Core/full commit one compact trend-source record keyed by mode, commit, exact
comparison-key cohort, score-eligible key-set digest, exception/no-coverage membership
digests, scope/config/environment/tool digests, report-root hash, and run ID. The
immutable source ledger retains at most 365 records, 32 MiB, and 400 days;
`trend.jsonl` is a bounded 8 MiB projection. Its durable state is the reserved
trend-sequence high-water record, retention checkpoint, and content-addressed
source/tombstone hash chain described above. UTC is display and retention metadata,
never the sole ordering truth.

CI treats mutation result cache and trend storage differently. Result cache is a
disposable performance optimization capped at 20000 entries and 512 MiB. Remote
restore lists at most 20 candidate artifacts, downloads at most two candidates and
512 MiB total, and has a 180-second scheduled core/full deadline. Changed remote cache
restore is disabled so its 15-minute outer budget contains no hidden restore phase. It accepts
only an exact schema/tool/config prefix and the newest digest-valid candidate by
numeric `(workflow_run_id, run_attempt)` identity, validates every record and
committed-run marker, and saves one bounded archive under a new run-specific immutable
key with seven-day retention. Cache absence, eviction, restore ambiguity, validation
failure, or restore deadline is a visible miss and cannot remove factual report/trend
evidence. Under the output lock,
deterministic eviction removes the oldest `created_at,key` records first and records
evicted count, bytes, and key range in the receipt.

Each restore emits a bounded diagnostic object. `cache_restore` records at most 20
attempted carrier tuples/names/digests, per-carrier closed rejection code, selected
carrier or `null`, listed/downloaded count and bytes, elapsed monotonic milliseconds,
deadline/byte truncation, final hit/miss code, and retry command. `trend_restore`
records the same plus expected/observed epoch, predecessor/high-water tuple, affected
sequence gap, `predecessor_invalid|predecessor_unavailable|none`, and exact reconcile
command. JSON, human reports, and the trusted CI summary render these objects.

Only scheduled core/full runs own shared trend restore/sequence/high-water and publish
remote cache or aggregate carriers. Manual core/full may restore one carrier for
bounded diagnostics but always upload only their factual invocation bundle and skip
shared sequence reservation, source append, high-water advance, cache and aggregate
publication. Before upload, a bounded GitHub API inventory reads at most 100 metadata
records/20 MiB in 60 seconds. Cache carriers are capped per repository at 14 artifacts
and 7 GiB across seven days; aggregate carriers at 90 artifacts and 5760 MiB across
90 days. The job deletes nothing. If adding an artifact would exceed either count or
bytes, publication is skipped with `artifact_upload_skipped_quota`; report/trend local
truth remains valid and the summary names platform-owner cleanup.

For scheduled core and full, CI first retrieves prior
`mutation-aggregate-v1-<workflow_run_id>-<run_attempt>` artifacts by monotonically
decreasing tuple `(workflow_run_id, run_attempt)`. Restore lists at most 20 carriers,
reads metadata for at most 20 MiB, downloads at most the first two same-epoch
candidates and 64 MiB total, and stops after 180 seconds. The highest tuple is
authoritative. It is accepted only when its aggregate manifest, high-water,
checkpoint, complete retained hash chain, workflow/repository identity, configuration
epoch, and source-bundle digest validate. A second artifact with the same high-water
and identical manifest digest is a duplicate carrier and may be used; otherwise a
corrupt highest carrier starts a new `predecessor_invalid` epoch and lower tuples are
not silently accepted. Each reviewed configuration epoch contains one immutable,
initially unused genesis marker binding epoch/config digest,
`allow_no_predecessor=true`, and `predecessor_tuple=null`. Only that marker permits
the first checkpoint/high-water and first sequence. Any later missing carrier caused
by pause, expiry, deletion, or epoch mismatch starts `predecessor_unavailable`; genesis
never establishes a historical baseline or cross-epoch comparability. The restored aggregate seeds only
the local trend sequence, never the report sequence. Before reserving a new trend
sequence, the current `(workflow_run_id, run_attempt)` must be strictly greater than
the validated predecessor carrier tuple. An older workflow rerun still publishes its
factual report/bundle but skips trend append and aggregate upload with
`carrier_not_monotonic`; it cannot place a higher trend sequence in a lower carrier.
After validation, the current
run uploads a new immutable rolling aggregate containing the bounded ledger,
projection, high-water/checkpoint, its source bundle digest, and predecessor tuple;
retention is 90 days and aggregate size is capped at 64 MiB. Nightly cadence keeps a
recent carrier for records retained inside the aggregate. Every unavailable/invalid
epoch requires an explicit artifact-aware `trade dev mutation reconcile --trend-epoch
... --carrier ...` operation to close its visible gap; later runs cannot silently heal
it. The manual recovery workflow downloads and verifies one named carrier, dry-runs,
requires environment approval, reapplies under the publication lock, and publishes a
repair carrier only with a strictly newer workflow tuple. Local apply never uploads,
and ordinary manual core/full never writes shared trend. Local 400-day retention
is a cap, not a claim that GitHub guarantees storage for 400 days. Core/full no-overlap
prevents concurrent aggregate writers; PR changed runs do not write this aggregate.

The repository baseline changes only by explicit review. Comparisons require
identical complete comparison cohorts, identical killed/survived denominator key sets,
and identical exception/no-coverage membership; timeout, infrastructure,
cancellation, invalid, baseline-unavailable, partial status, changed node, policy
change, or trend-epoch boundary makes the pair non-comparable. Numerator and
denominator key-set digests are persisted. Execution IDs remain exact per revision.
Cache hit/miss/eviction reasons and identity versions are visible.

### Baseline and exceptions

`config/mutation-baseline.json` starts with `established: false`; schema
`trade.mutation.baseline.v1` still reserves the exact fields required for a future
reviewed baseline: verified bundle manifest digest, report manifest digest, run ID,
source commit and `GitScopeSnapshot` digest, mode, tool/lock/config/matrix/cohort/
environment/isolation identities, complete comparison-key cohort, killed-key set,
survived-key set, exception-key set, no-coverage-key set, each set digest, counts,
numerator, denominator, and score. The evaluator reads the referenced immutable bundle
from a reviewed fixture/artifact, validates every binding, reconstructs each key set
from per-mutant records, and recomputes counts and score; a stored-only aggregate is
never trusted. A future command can propose this artifact, but committing it requires
review. V1 never claims comparability merely because a source path matches.

`config/mutation-exceptions.toml` starts empty. Each future record must match exact
mutant ID, path/source digest, `CanonicalMutationPositionV1`, operator
name/arguments, `execution_occurrence`, owner, reason, and expiry. Validation rejects
every wildcard, ambiguity, and stale
source. Excepted mutants have `equivalent_exception` status and remain visible; they
do not enter score numerator or denominator.

Qualification and exception temporal validation uses RFC 3339 UTC plus a recorded
trusted validation time. At supervisor start and end, code samples
`CLOCK_REALTIME` immediately before and after `CLOCK_BOOTTIME`; expected realtime
advance is the boottime delta and a backward residual over five minutes is rollback.
Across runs, `last_trusted_utc` and its source digest come from the maximum validated
anchor in the reviewed capacity/exception record, validated local receipt high-water,
or validated predecessor aggregate. Current realtime more than five minutes behind
that anchor fails eligibility. A reboot permits a new boottime domain but does not
discard the UTC high-water. Capacity or a non-empty exception registry without a
required validated anchor fails closed; empty exceptions need no temporal eligibility.
Every successful validation advances only an ignored local receipt/aggregate anchor,
never edits reviewed config. `expires_at` must be strictly later than `qualified_at` or
`reviewed_at`, at most 30 days later for capacity and at most 180 days later for an
exception, and in the future at validation. A current clock earlier than the recorded
start by more than five minutes, invalid/leap-second text, a future start more than
five minutes ahead, a missing required anchor, or either rollback equation fails closed.
The report records the five-minute skew allowance and comparison result; UTC never
orders execution events.

### CI design

Runner infrastructure is an explicit external prerequisite, not an assumed repository
asset. The accountable owner is GitHub repository owner `huanwei1208`; the
prerequisite change must also name the human/team platform operator before approval.
A prerequisite child change
`provision-trade-mutation-v1-runner` owns the immutable image/build manifest, image
digest, ARC/VM scale-set and labels, one-job destruction policy, delegated cgroup
setup, finite memory, Landlock/seccomp kernel configuration, mount/credential policy,
capacity/availability SLO, and a runner-readiness workflow. Its accepted evidence is
a reviewed image digest plus three consecutive clean readiness jobs that prove label
pickup within five minutes, unique ephemeral identity/destruction, cgroup ownership,
kernel probes, no production mounts/secrets, and cleanup. Until that child change and
core/full mode capacity qualification pass, scheduled execution is disabled. Changed
execution requires the same runner readiness plus one exact cold-cache changed smoke
within its 150/600 limits; it does not require a nonexistent changed entry in the
core/full capacity schema. Repository CI may run plan-only on ordinary GitHub-hosted
runners and must not queue indefinitely on a nonexistent label.

One GitHub Actions workflow then has three execution routes plus one
evidence-validation job:

- every execution route uses the repository's dedicated ephemeral ARC/VM scale-set
  profile `runs-on: [self-hosted, linux, x64, ephemeral, trade-mutation-v1]`; hosted
  plan-only routing before readiness uses the ordinary hosted label and never claims
  this execution profile. One clean execution VM is
  destroyed after one job, has no production/data mounts or long-lived credentials,
  and delegates a fresh writable cgroup v2 subtree with `memory`, `pids`, and `cpu`
  controllers to the unprivileged runner user. Its immutable image enables Landlock
  ABI and seccomp user notification/addfd. The workflow's first preflight verifies
  exact runner image digest, ephemeral marker, mount namespace, delegation ownership,
  finite `memory.max`, controller list, writable child cgroup, Landlock/seccomp probes,
  and empty data/provider mounts. A mismatch publishes a preflight bundle and executes
  no mutant. Fork pull requests receive no secrets and run only on this disposable
  profile; ordinary persistent self-hosted and current GitHub-hosted runners are not
  assumed to satisfy the contract.
- `pull_request`: checkout full enough history, set up Python/uv, run
  `trade dev mutation changed --base $BASE_SHA`; remote cache restore is disabled and
  the job-level timeout is 15 minutes
  around the 10-minute total controller budget. Its concurrency key is
  `mutation-pr-${{ github.repository }}-${{ github.event.pull_request.number }}` with
  `cancel-in-progress: true`; cancellation upload is best effort.
- nightly/manual core: `core`, cron `17 18 * * 1-6` UTC, 45-minute timeout,
  compatible result cache and validated rolling trend aggregate.
- weekly/manual full: explicit `workflow_dispatch` mode or cron `17 18 * * 0` UTC,
  with a 110-minute job timeout.

Outer timeouts obey one static phase equation. Changed admits 2 minutes
checkout/setup/preflight + 10 controller + 2 bundle/upload + 1 cancellation margin =
15 minutes. Core runs cache and aggregate restore in parallel under one 3-minute
restore cap, then admits 4 minutes checkout/setup/preflight + 3 restore + 30 controller
+ 5 bundle/fsync/upload + 3 cancellation margin = 45 minutes. Full uses 4 + 3 + 90 +
10 + 3 = 110 minutes. Restore work is outside but explicitly represented in the outer
equation; bundle/upload reserves cannot be consumed by the controller. Static workflow
tests recompute these values. Hard outer timeout remains a last resort rather than a
normal conforming path.

All v1 mutation outcomes are report-only. The execution step captures controller exit
0/1/2, receipt path and run ID through `$GITHUB_OUTPUT`, then returns success so score,
survivor, timeout, baseline, or tool-status outcomes do not block the route. The same
execution job has `if: always()` steps that validate the receipt, build and independently
validate the immutable bundle locally, append its summary, and upload exactly that
bundle artifact. It exports artifact name plus bundle-manifest digest as job outputs.
The receipt preserves the real exit and summary prints it. The workflow is additive
and is not made a required branch-protection check by this change. Evidence integrity
is different: a dependent `validate-evidence` job runs `if: always()`, downloads that
named artifact into a fresh directory, rejects extra/path-escaping members, recomputes
all member/manifest digests and receipt/report/count/cleanup bindings without using an
execution-workspace path, and fails on missing or invalid evidence. No local path is
assumed to cross jobs. When GitHub schedules it, the validation job appends the sole
authoritative operator-facing `trade.mutation.ci-summary.v1`; whole-workflow
cancellation before scheduling remains best effort. Valid downloaded bytes dominate,
invalid downloaded bytes are `invalid`, absent bytes with exported artifact identity
are `missing`, absent identity plus failed/cancelled/timed-out execution is
`hard_loss`, and absent identity after success/skipped is `missing`. The summary carries
`trusted_validation_status=valid|missing|invalid|hard_loss`, expected and actual
artifact/manifest digest, artifact/run/workflow/attempt identity,
`controller_exit=0|1|2|null`, closed unavailable reason, verified outcome counts/
score/budget only when valid, closed error code, and exact remediation. The earlier
execution summary is visibly labelled
`UNTRUSTED UNTIL validate-evidence`; it cannot be mistaken for the final trust
decision. Mutation outcomes remain report-only. Evidence validation may be red on
integrity failure, but this change does not configure it as a required branch check.

Scheduled and manual core/full plus the approved reconcile recovery route share concurrency key
`mutation-long-${{ github.repository }}` with `cancel-in-progress: false`. Native GitHub
concurrency guarantees at most one running and one pending member; a newer dispatch
may supersede an older pending member even though it does not cancel the running job.
The workflow relies only on the no-overlap guarantee and does not claim a durable or
lossless queue. A dispatch cancelled before command entry has no run ID and creates no
trend evidence. Manual input is exactly `core|full`. Trend records use the
independent trend sequence allocated after report publication, not the report
sequence, workflow dispatch, or adjustable wall-clock order.

`config/mutation-capacity.json` has schema `trade.mutation.capacity.v1`. A core/full
execution qualification binds mode, config/matrix/cohort/import-closure/tool/lock/
isolation-policy digests, Python and dependency lock identities, the exact
`runner_capacity_identity` and separate serial/capacity `measurement_profile`s, and
the measured planning/copy/baseline/per-mutant p50/p95/render/fsync/cleanup values and
pass equations above. It records both qualification run/bundle digests, qualified UTC,
the finite 512 MiB controller and 1024 MiB baseline bootstrap caps, bootstrap OOM and
measured-controller evidence, expiry at no more than 30 days,
worker/throughput/time/memory/disk headroom and
`qualified: true|false`; review owns the checked-in file.
Before mutation dependency setup, scheduled/manual core/full recompute every identity
and freshness field. Missing, expired, false, mismatched, non-finite-memory, or
insufficient-headroom qualification produces a receipt-bound preflight report and
does not execute mutants. There is no permissive fallback. `trade dev mutation
qualify --mode core|full --runner-profile PROFILE` is an explicit report-only
operator command; it runs cold-cache serial and capacity profiles, validates the same
isolation/cleanup and pass equations, writes a proposal artifact, and never edits the
reviewed qualification file.

Every execution-job summary, bundle-construction and upload step uses `if: always()` and the exact
`trade.mutation.invocation.v1` receipt path exported through `$GITHUB_OUTPUT`; report
bundles have 14-day retention. The standard-library bundle validator validates the
receipt run ID, phase journal, lifecycle and cleanup facts, and exactly one referenced
immutable generation or typed fallback. It never reads global `current.json` or
chooses a newest path. It then creates a new immutable directory with schema
`trade.mutation.bundle.v1` containing:

- the exact receipt bytes;
- the complete supervisor-attestation journal;
- the complete referenced report generation and manifest, or the exact fallback;
- `validation.json` with schema `trade.mutation.bundle-validation.v1`, validator
  version, workflow/run identity, controller exit, receipt digest, evidence kind/root
  digest, lifecycle/count/cleanup validation results and closed failure codes;
- `bundle-manifest.json` with sizes and SHA-256 for every preceding member.

The validator read-backs the full bundle, fsyncs it, and exports its path and manifest
digest. The execution job uploads only that exact path under a run/attempt-specific
artifact name. The validation job downloads that artifact, requires the expected
bundle-manifest digest job output, and reruns validation from bytes; raw staging,
mutable output roots and unvalidated receipt-only artifacts are never uploaded or
trusted as successful evidence. Explicit missing-file behavior emits a GitHub failure
summary without manufacturing a bundle. This preserves evidence for controller exits,
internal deadlines and gracefully delivered signals while the job remains alive.
GitHub hard cancellation, runner loss, host loss, or an unreturning kernel/filesystem
operation cannot guarantee later steps and remain explicit best effort. Outer runner
timeouts are the final containment and exceed internal budgets.

The script always plans changed scope before installing/running the heavy mutation
engine. If no eligible files exist, it produces a zero-work report and exits. GitHub
path filters are an optimization only; script selection remains authoritative.

### Validation strategy

Tests cover:

- closed config and operator validation;
- root `trade dev mutation` and compatibility-facade argv/exit parity, pre-`uv`
  Python 3.7 selection, minimum-version rejection, and command-entry budget inclusion;
- changed Git rename/add/delete/tracked line hunks plus deferred untracked inventory;
- sealed Git-scope snapshot and drift detection before copy and publication;
- no eligible source zero-work;
- canonical mutation-token changed-line/core deterministic priority, unique nested
  definition ownership, execution/definition occurrence semantics, and limits;
- first-order rejection;
- exclusions and no broad exception patterns;
- test mapping and no full-suite fallback;
- exact-line baseline coverage and mapped-but-uncovered behavior;
- private-overlay import/digest provenance and original source hashes;
- Landlock/seccomp admission, real SQLite/parquet/os.open/Path.open/mmap and symlink
  denial, credential/proxy scrubbing, and fail-closed unsupported-kernel behavior;
- API/version/operator/get-ast/ast-nodes traversal compatibility,
  operator-local occurrence/span handshake, and forbidden in-place Cosmic Ray APIs;
- baseline timing/failed-baseline behavior and exact pytest exit mapping;
- killed/survived/timeout/no-coverage/baseline-unavailable/invalid/infrastructure
  /cancelled-active/not-run-cancelled classification and count equations;
- parent-owned descendant process termination on timeout, worker failure, signal, and
  global deadline, including cancellation between spawn and registry insertion;
- worker/native-thread bound, execution-deadline admission stop, and private-tree
  file/byte/copy/free-space limits;
- bytecode isolation across same-size fixed-mtime mutations;
- authenticated gap-free guard lifecycle sidecars, kernel-audit outcomes and missing/
  truncated/wrong-token evidence that override caught/uncaught pytest exits;
- supervisor tests for hung `uv`, bootstrap/controller crash, controller SIGKILL/OOM,
  unknown SIGKILL, wrapper SIGTERM, descendant/session escape attempts, and
  fail-closed missing delegated cgroup;
- sole-supervisor authenticated spawn/signal/wait/fallback protocol, opaque handles,
  replay/FD-role rejection, and proof that controller modules contain no spawn primitive;
- fork-before-cgroup-move prevention, pidfd identity, listener-FD handoff/closure,
  brokered `openat2` descriptor injection, pathname/symlink races, controller-private
  audit integrity, and supervisor fallback after controller loss;
- per-mutant natural-exit versus cancellation linearization in both race orders;
- sticky integrity precedence after both stream/drain orders, orthogonal cleanup
  override, and the complete terminal x mutation-applied x worker-spawned x
  test-started x cleanup matrix;
- run-generation atomic partial JSON/Markdown/HTML publication, independent fallback,
  invocation receipt binding, corrupt-pointer recovery, output-root safety, per-run
  lease scavenging, redaction, and generation/aggregate retention bounds;
- closed terminal/phase/cache count algebra, stale/corrupt cache invalidation,
  post-publication committed-run markers, manifest-root anchoring, compact immutable
  trend-source reservation/tombstone/hash-chain recovery, numeric eviction,
  aggregate-carrier validation, epoch gaps and projection-gap reporting;
- receipt/fallback/report/trend-anchor retention coupling, inspect/repair closed
  schemas including evidence-expired, explicit reconcile, and immutable
  bundle/validator digest binding;
- tuple-specific per-worker cgroup memory controls, no-delegation plan-only behavior,
  unique-cgroup OOM proof, cleanup-unconfirmed override, and outer-timeout containment;
- unestablished/comparison-key baseline policy and detail-derived baseline binding;
- capacity identity versus measurement throttling, freshness/headroom pass equations,
  and schedule enforcement;
- dedicated coverage rcfile/environment independence and missing dependency diagnostics;
- workflow literal cron/mode/timeout/shared-long no-overlap/latest-pending semantics,
  report-only outcomes, execution-job receipt-bound bundle upload, validation-job
  artifact download/revalidation, bounded aggregate/cache restore ordering, invalid
  predecessor epochs, and hard-runner-loss best-effort documentation.

Validation commands include focused pytest, a real small changed run, TOML/JSON/YAML/
static workflow checks, `bash -n`, ShellCheck when available, Ruff, BasedPyright,
compileall, `./trade dev check`, the exact mapped tests, full pytest with the
pre-existing stable failure reported, and `git diff --check`. Before schedules execute
mutants, cold-cache core and full qualification run both serial and capacity profiles
on the exact selected runner. They record planning/copy/baseline/per-mutant
p50/p95/render/fsync/cleanup time and only pass when the reviewed runner identity,
equations, temporal rules, 30-day freshness, and time/memory/disk headroom validate.

### Runtime concurrency evidence

Ownership is one Python 3.7 standard-library supervisor/subreaper/sole OS spawner plus
one non-spawning controller process and bounded logical thread pool. The authenticated
control protocol returns opaque handles into the supervisor's synchronized
pidfd/PGID/cgroup registry; one receipt-bound `CLOCK_BOOTTIME` deadline and boot ID are
created before every `uv` child with an earlier execution deadline. The supervisor
creates a stopped child and required delegated worker cgroup; the child cannot enter
the trusted launcher until cgroup placement, resource read-back, pidfd/session/
watchdog ownership, and the one-shot start gate are verified. The launcher must
activate kernel policy before application import. Ordering is fixed before admission;
completion order never changes report ordering. Each logical thread owns one bounded
import-closure tree and restores the target between work items.

Atomicity is one root-hash-published run generation. Report sequence reservation
precedes pointer publication; a separate core/full trend sequence drives
source/tombstone reconciliation as a post-publication idempotent projection journaled
in the supervisor receipt. A child write-only/supervisor-read-only guard channel,
supervisor-only write-protected attestation journal, and controller-private syscall-
audit state are independently authenticated and jointly required. Timeout and
cancellation name opaque handles and target full worker
cgroups/process groups; the supervisor owns adopted descendants, every fallback and
continued cleanup. Backpressure is the CPU-and-memory admission rule, finite queue,
source/AST/dependency ceilings, detail budget, and cleanup reserve. Partial failure is
aggregated by exact terminal, phase, sticky integrity, and cleanup facts without
converting infrastructure, cancellation, unconfirmed cleanup, or tool errors to kills.

### Rollout and rollback

1. Land the controller, configuration, tests, and docs with all modes report-only.
2. Enable PR changed mode non-blocking.
3. Collect nightly core trends and review no-coverage/survivors.
4. Establish a baseline only after repeated stable comparable runs.
5. In a later change, consider blocking only a >5-point comparable regression.
6. Consider raising selected mature core targets from 70% to 80%, never an unconditional
   repository-wide threshold.

Rollback removes the workflow first, then the script/controller and optional
dependencies. Before removal, operators disable new dispatch, let the shared long-run
group drain or cancel one job, wait for its bounded cleanup/fallback, and verify no
active PGID or run lease remains. Cache/report directories are ignored and retained
for manual audit; stale lease/staging cleanup uses the same owned scavenger and is
never an automatic rollback side effect. No production data or behavior rollback
exists.
