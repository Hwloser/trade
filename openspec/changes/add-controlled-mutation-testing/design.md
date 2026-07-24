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
  bootstrap.py                               import-light Git/config/zero-work planner
  cli.py                                     argument parsing and orchestration only
  models.py                                  immutable plans/outcomes/status algebra
  config.py                                  closed configuration parser
  git_scope.py                               bounded base/diff/changed-line discovery
  selection.py                               deterministic targets/tests/mutants
  engine.py                                  Cosmic Ray operator adapter
  coverage.py                                coverage.py 7.10.7 line-data adapter
  process_supervisor.py                      spawn gate, PGIDs, deadlines, cancellation
  isolation.py                               private trees, environment and guard evidence
  executor.py                                bounded mutation work scheduling only
  cache.py                                   staged outcomes and committed run markers
  trend.py                                   sequence transaction, digest chain and projection
  render.py                                  JSON-derived Markdown/HTML rendering
  report_store.py                            leases, atomic publish, bundles and retention
  baseline.py                                comparable baseline evaluation
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

The supervisor creates the run ID, receipt, deadline, lease and watchdog, enables
Linux child-subreaper semantics, and parents every `uv` handoff. The CLI coordinates
owned modules but does not contain mutation logic. Cosmic Ray is an adapter dependency
and never imports application runtime modules. `models.py` imports no controller
adapter; planning and execution depend on models, while selection/engine never import
process, storage, cache, trend, or rendering modules. `render.py` consumes persisted
report models and `report_store.py` owns bytes and filesystem transitions, not outcome
semantics. Tests use synthetic repositories and source files.

The controller is a devtool, not a domain service. It does not belong in the future
business Context graph and is excluded as a mutation target.

### Data and state invariants

The durable artifacts are developer reports and cache entries, never business state.
Identifiers and invariants:

- `run_id` is a UUID generated at supervisor entry. Before dependency preparation, the
  supervisor atomically writes `invocations/<run_id>.json` with schema
  `trade.mutation.invocation.v1`, mode, lifecycle state, shell command-entry monotonic
  value, controller stop deadline, outer cleanup deadline, supervisor identity, and
  exact expected staging/run/fallback/bundle paths.
  Bootstrap and the full CLI send bounded authenticated transition records over one
  inherited supervisor pipe; they never own or rewrite the receipt. The supervisor
  validates run ID, child PID/start token, sequence, schema and forward transition
  before atomically rewriting it. The receipt is also a bounded phase journal with
  plan digest, selected count, last durable phase, expected report root and projection
  state. CI receives its path before the first child starts.
- `execution_id` (and compatibility field `mutant_id`) is SHA-256 over schema
  version, relative path, original source digest, start/end position, operator
  name/arguments, occurrence, and config digest.
- `comparison_key` is a separate SHA-256 over schema version, relative path,
  qualified eligible definition, definition-relative span, normalized original node
  text, operator name/arguments, and occurrence within that definition. It permits a
  conservative cross-revision match after unrelated line/file edits without weakening
  exact execution/cache identity. Any changed node text, ambiguous key, added/deleted
  candidate, partial cohort, or changed policy is non-comparable.
- one `MutantPlan` contains exactly one `MutationSpec`.
- source paths are repository-relative normalized POSIX paths and regular files.
- changed line ranges are parsed from NUL-safe Git records and zero-context hunks for
  merge-base-to-HEAD plus staged and unstaged tracked state. Tracked rename
  destinations are eligible; deletion-only and rename-only-without-content-change are
  not. Untracked Python production paths are recorded as `deferred_untracked` and are
  never mutated or silently treated as ordinary zero work. The dirty-tree digest is
  part of run identity. Changed mode admits only candidates whose planned start line
  intersects that exact changed-line set; it never widens to the enclosing definition.
- ordered selection is stable and does not depend on filesystem enumeration order.
- a cached result is reusable only when the bounded import-closure digest, mapped
  tests/fixtures, reviewed dynamic-import manifest, lock and pytest configuration,
  operator/config/exception policy, normalized command/environment including
  native-thread and kernel-isolation controls, platform, declared per-mapping clock
  and entropy policies, exact mutant cohort, and tool versions match. The complete
  tracked-Python tree digest remains report audit metadata but does not invalidate an
  unchanged closure. V1 accepts only `explicit_input|non_cacheable` clocks and
  `deterministic|entropy_non_cacheable` entropy; unsupported `fixed` values fail
  configuration until an injection contract exists. A tuple whose bounded import
  closure references an undeclared clock or entropy API is non-cacheable. The trust
  tuple is `entropy_non_cacheable` because its default path can call `uuid.uuid4()`.
- one run directory is staged and fsynced; one `current.json` pointer atomically
  publishes JSON, Markdown, HTML, and their digest manifest as one generation.
- `generated` counts eligible candidates seen up to the candidate-scan ceiling and
  never claims the unknown unscanned remainder; `generated = selected +
  not_selected_scanned`.
- the closed terminal enum is `killed_fresh`, `survived_fresh`, `killed_cache`,
  `survived_cache`, `timeout_test_started`, `infrastructure_error_test_started`,
  `cancelled_test_started`, `no_coverage_mapping`, `no_coverage_line`,
  `baseline_unavailable`, `invalid_mutation`, `infrastructure_error_pre_test`,
  `equivalent_exception`, `not_run_plan`, `not_run_budget`, and
  `not_run_cancelled`. Every selected mutant has exactly one.
- every record also carries booleans `mutation_applied_current` and
  `test_started_current`; aggregate phase counts are sums of those facts, not guesses
  from logs. Cache terminals set both false. Test-started terminals set both true.
  `infrastructure_error_pre_test`, `not_run_budget`, or `not_run_cancelled` may set
  mutation-applied true only when their typed stop event wins after the
  position-verified write but before pytest spawn; in this contract `not_run` means
  the test did not start, not necessarily that transformation did not occur.
- only `killed_*` and `survived_*` enter score; score is `null` when their sum is zero.
  Only verified fresh killed/survived outcomes are cache candidates.

Unavailable states are explicit: missing base, unsupported tool, invalid config,
baseline test failure, no eligible source, no test mapping, budget exhaustion, and
report write failure never become an empty successful mutation score.

### Persistent-write safety

Mutation state has explicit single writers: the supervisor owns the invocation
receipt and controller-loss fallback; the controller/report store owns one run
staging/final generation; cache and trend owners publish post-report projections
under the shared output lock.
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
the manifest. Before pointer publication, `trend.py` reserves and fsyncs the next
monotonic publication sequence in `publication-sequence.json`. The reservation binds
run ID, previous committed sequence/digest and expected manifest digest. The
controller then flushes and fsyncs each file and the directory, renames the directory
to its immutable final run path, and atomically replaces and directory-syncs
`current.json` containing run ID, manifest SHA-256, manifest byte size and reserved
sequence. The supervisor updates the invocation receipt with the same root hash and
sequence. Readers resolve the
receipt or pointer once, validate that root, then validate member digests and read only
that generation. They never compose output from separate runs. Cache records and the
bounded trend source/projection use the same write, fsync, atomic-replace,
directory-fsync discipline.

A crash before pointer replacement leaves the previous complete generation current;
a crash after replacement exposes only the already verified new generation.
On the next locked operation, every unresolved sequence reservation is reconciled
before a new reservation: a matching root publishes its trend source, while an
absent/invalid root publishes a content-addressed tombstone. Sequence values are never
reused.
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
marker, commits one immutable compact content-addressed
`trend-sources/<publication_sequence>-<run_id>-<digest>.json` record, advances the
sequence high-water record with this digest, then reconciles `trend.jsonl`. Each
source or tombstone includes the previous retained digest, producing a validated hash
chain anchored by the high-water record. A retention checkpoint records the digest
immediately before the first retained record; stale restore below the high-water or a
chain break fails closed. The trend-source record contains the report root hash,
run/mode/source identity, numerator/denominator key-set digests, score/counts,
comparability and publication sequence, but no mutant diagnostics. It is retained for
365 records, 32 MiB, and 400 days independently of 30-day report generations.
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
passes. Otherwise exit 2 preserves the failed run staging path and writes one
independent bounded `fallback-<run_id>.json` with schema
`trade.mutation.fallback.v1`, redacted stable error code, failing stage, retryability,
message, remediation command, receipt path, and evidence paths outside
`current.json`; it never labels fallback files as an authoritative generation. The
invocation receipt identifies this exact fallback, so CI never falls back to stale
`current.json` or newest-file globbing. The manifest audits run
ID, UTC time, mode, source/base
identity, complete config/tool/environment/cohort digests, budgets, stop reason,
status, member hashes, staged cache eligibility, and trend-source inputs without
credentials or raw environment values. Projection outcomes live in the receipt and
ledger, not the immutable manifest. If bootstrap/controller exits unexpectedly, is
SIGKILLed, or is proven OOM-killed by cgroup `memory.events`, the surviving supervisor
terminates descendants, verifies any run-ID final generation/current root already
published, and either anchors that root in the receipt or atomically writes a
`controller_lost` fallback. Unknown SIGKILL is never labelled OOM. Only
host/runner/supervisor loss remains best effort.

Immutable generations are retained for at most 30 entries, 2 GiB, and 30 days;
compact trend-source records for 365 entries, 32 MiB, and 400 days; invocation
receipts for 400 entries, 64 MiB, and 400 days; fallback/quarantine/failed-staging
evidence for at most 50 entries, 512 MiB, and 14 days. Deterministic
oldest-creation/run-ID eviction under the
publication lock protects `current`, all active leases, and current failure evidence
and prevents a receipt from being evicted before its retained report, fallback or
trend anchor. It records every eviction. No backup is required for ignored disposable
developer state. CI uploads a validated immutable invocation bundle, not mutable
staging. Rollback removes tracked tooling
without rewriting or deleting these audit generations, which remain subject to
explicit manual cleanup and never enter production state.

### Contracts and compatibility

#### CLI

```text
trade dev mutation MODE [OPTIONS]
scripts/mutation-test MODE
  [--base REF]
  [--output-dir PATH]
  [--max-mutants N]
  [--max-seconds N]
  [--workers N]
  [--plan-only]
  [--no-cache]

trade dev mutation inspect --run-id UUID [--format text|json]
trade dev mutation repair --run-id UUID
  --expected-manifest-sha256 SHA256
  [--format text|json]
  [--apply]
```

`trade dev mutation` is canonical and root-dispatched before the ordinary `trade dev`
`uv` path; `scripts/mutation-test` is an argv/exit-compatible facade. `MODE` is
required and closed to `changed`, `core`, or `full`. Changed mode defaults
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

`inspect` is read-only and returns `trade.mutation.inspect.v1`; `repair` returns
`trade.mutation.repair.v1`. Both support text/JSON, closed
`healthy|first_run|repairable|active_lease|stale_expected_digest|unsafe|internal_error`
states, stable error/remediation fields and deterministic evidence paths. Inspect
exits 0 for healthy/first-run, 1 for repairable, and 2 for unsafe/internal errors.
Repair dry-run exits 0 for no-op or a valid plan, 1 for active lease/stale expected
digest, and 2 for unsafe/internal errors; `--apply` exits 0 only after read-back
verification. Neither command invokes mutation dependencies, follows symlinks, or
touches active leases.

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
pytest. Run-status precedence is `report_failed` then `signal_cancelled`,
`preflight_failed`, `degraded_infrastructure`, `budget_partial`, `plan_only`,
`deferred_only`, `zero_work`, and `complete`; future policy regression affects exit 1
but does not overwrite factual run status.
Infrastructure failure in one mutant does not turn another mutant into killed. The
controller stops only the affected test tuple unless integrity, process ownership,
global deadline, signal, or report publication requires run-wide cancellation.

The report schema is `trade.mutation.report.v1`. CI consumes `summary.md` and uploads
all report files but does not infer truth from logs.

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
`no_coverage_line`; a missing mapping becomes `no_coverage_mapping`. Neither starts
mutant pytest. Coverage never adds an unconfigured test or falls back to all tests.

### Failure and recovery

The facade-provided command-entry monotonic timestamp bounds planning, baselines,
execution, cancellation, publication, and projection. A controller stop deadline
reserves cleanup/publication time before the outer script deadline. The supervisor enables
`PR_SET_CHILD_SUBREAPER`, launches every `uv` handoff in an owned session, forwards
signals, and maintains PID/start-token/boot-identity descendants. Every supervisor
child starts through a Python 3.7-compatible fork-stop trampoline: before exec or
application code the child raises `SIGSTOP`. With delegated cgroup v2, the supervisor
places it in the invocation/worker cgroup, reads back membership and installs
pidfd/session/watchdog ownership before sending `SIGCONT`; placement or read-back
failure kills the stopped child and fails preflight. When `clone3`
`CLONE_INTO_CGROUP` is available an equivalent tested helper may replace this
handshake. When delegated cgroup v2 is writable, the supervisor stays outside, sets
`memory.oom.group=1`, and verifies `cgroup.procs` is empty before exit. Without
delegated cgroup containment, the supervisor still registers the stopped child's
pidfd/session/watchdog before release, forces the single-worker/no-finite-memory
fallback, and uses subreaper reparenting plus bounded `/proc` descendant closure
checks. That fallback is accepted only because the trusted launcher activates
Landlock/seccomp before application import and then denies fork/vfork/exec and
process-form clone; reviewed thread-form clone remains inside the same worker process
and session. Any ownership or policy-activation failure kills the group and fails
closed.
The supervisor terminates and reaps the remaining hierarchy if bootstrap/controller
hangs, crashes, or receives SIGKILL. Supervisor or host SIGKILL and runner loss remain
explicit best effort, not a no-orphan guarantee.

A controller-owned thread pool is used so the controller directly creates every
pytest `Popen`. Spawn, PGID registration, typed stop-event enqueue, and wait
observation share one gate. Every mutant follows
`PLANNED -> MUTATED? -> STARTED? -> EXIT_SEEN|STOP_SEEN -> FINALIZING -> TERMINAL`.
Stop events carry a monotonic sequence and closed cause
`guard_violation > syscall_violation > listener_failure > launch_or_provenance_failure
> per_mutant_timeout > global_signal > execution_budget`. A natural exit records
`EXIT_SEEN` but cannot terminalize until both independent guard/audit streams complete
and pending notifications drain. Any integrity event overrides exit 0/1; otherwise
the earlier sequenced natural-exit or timeout/signal/budget event owns classification.
Budget or signal after mutation but before spawn yields the corresponding
`not_run_*` terminal with `mutation_applied_current=true`. Golden tests cover every
pairwise race and both phase facts. Worker exceptions, lost pipes, repeated signals,
and parent cancellation use one TERM, bounded wait, KILL, group-existence check, and
reap path.

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
read-only child-guard channel, listener-transfer socket until ACK, stdout and stderr;
stdin is `/dev/null` and every other descriptor is closed/read-back audited.

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
and adds Python audit/guard hooks for higher-fidelity diagnostics. The child receives
one write-only guard pipe and child-only MAC key, and emits a gap-free authenticated
sequence of `guard_started`, `kernel_policy_active`, zero or more `guard_violation`,
and `guard_completed`. The controller alone owns the syscall audit records and key;
it writes `syscall_audit_started`, zero or more `syscall_violation`, and
`syscall_audit_completed` to controller-private memory before durable outcome
serialization. Both streams bind run ID, worker PID/start token, target digest and
policy digest, but cannot forge or truncate each other. Classification drains pending
notifications and requires both complete sequences. Missing, empty, truncated,
duplicate, wrong-token or incomplete evidence is `infrastructure_error`. Any child or syscall violation
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
data/JSON afterward. Coverage exit/data failure is `baseline_unavailable`, never
no-coverage or killed.

Pytest mutant exits map `0=survived`, `1=killed`, and `2/3/4/5` or unexpected signal
to infrastructure error only after both independent evidence streams finalize. Typed
event precedence above resolves timeout/signal/budget/integrity races. Nonzero
baseline means `baseline_unavailable`; timeout is always timeout. A failed tuple does
not execute its mutants and cannot create kills.

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
a controller-independent fallback writer. A task still present after the cleanup
sub-budget is `cleanup_unconfirmed`, never killed or successful; the receipt records
its pidfd/cgroup evidence and the supervisor keeps attempting cleanup until the outer
CI timeout. An `fsync` or fallback write that does not return leaves no claimed
generation. The 15/35/100-minute CI timeouts are the final runner-level containment
for kernel/filesystem stalls and intentionally exceed controller budgets. Capacity
qualification fails if ordinary cancellation, reap, render, hash, fsync, and fallback
operations cannot complete inside their sub-budgets; documentation and reports call
out the remaining host-loss/kernel-stall limitation.

Source file/byte/tree/dependency and parse/operator-visit limits plus deadline
checkpoints bound operator-sparse trees where candidate count alone would not stop
planning. Effective CPU is the minimum of affinity, cgroup quota, and host count.
The CPU candidate is `min(4, max(1, floor(effective_cpu / 2)))`. Effective workers are
also limited by a finite enforceable cgroup v1/v2 memory ceiling. Baseline execution
measures each selected test tuple's peak descendant RSS and the scheduler reserves
`max(256 MiB, ceil(1.5 * measured_peak))` for that active tuple, not one optimistic
global average. Admission requires
`controller + safety + renderer + sum(active tuple reservations) <= effective_memory`.
A finite limit that cannot admit one worker fails preflight. When no finite enforceable
memory ceiling exists, local execution is forced to exactly one worker and records
`memory_bound_unenforced`; a caller may instead choose `--plan-only`. CI execution,
capacity qualification, and any request for more than one worker fail preflight in
that state. The report records every input, tuple reservation, and limiting dimension.
Before copying, the complete deterministic manifest and aggregate worker reservation
must fit file, byte, copy-time, and remaining-space limits. Structural report capacity
and worst-case escaped detail are reserved for every selected mutant. JSON renders
incrementally; Markdown/HTML use the same bounded detail records rather than embedding
JSON. Detail truncates deterministically before any per-file, generation, or renderer
limit.

At 10x eligible source size, source and AST ceilings stop work before candidate
truncation; at 10x unrelated repository modules, the root-bounded import closure
remains within the same copy manifest rather than copying all `trade_py`, and retains
cache reuse because the whole-tree digest is audit-only.
At 10x test cost, baseline timing excludes incompatible tuples before scheduling or
the wall deadline terminates admission. At 10x output, bounded ring capture truncates
diagnostics without blocking pipes.

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
publication-sequence high-water record, retention checkpoint, and content-addressed
source/tombstone hash chain described above. UTC is display and retention metadata,
never the sole ordering truth.

CI treats mutation result cache and trend storage differently. Result cache is a
disposable performance optimization capped at 20000 entries and 512 MiB. A run restores
only a schema/tool/config-compatible exact or prefix archive, validates every record
and committed-run marker, and saves one bounded archive under a run-specific immutable
key. Cache absence, eviction, restore ambiguity, or validation failure is a visible
miss and cannot remove factual report/trend evidence. Under the output lock,
deterministic eviction removes the oldest `created_at,key` records first and records
evicted count, bytes, and key range in the receipt.

For scheduled/manual core and full, CI first retrieves prior
`mutation-aggregate-v1-<workflow_run_id>` artifacts by monotonically increasing GitHub
workflow run ID, newest first, accepting only a bundle whose aggregate manifest,
high-water, checkpoint, complete retained hash chain, workflow/repository identity,
and configuration epoch validate. It never chooses by artifact timestamp or filename
glob alone. The restored aggregate seeds the local sequence reservation before the
controller starts. After validation, the current run uploads a new immutable rolling
aggregate containing the bounded ledger, projection, high-water/checkpoint, and its
own source bundle digest; retention is 90 days. Nightly cadence keeps a recent carrier
for records retained inside the rolling aggregate. If no valid carrier exists after a
schedule pause or artifact expiry, the run creates a new explicitly identified trend
epoch with `predecessor_unavailable`, so reports show a continuity gap rather than
inventing a predecessor. Local 400-day retention is a cap, not a claim that GitHub
guarantees storage for 400 days. Core/full no-overlap prevents concurrent aggregate
writers; PR changed runs do not write this aggregate.

The repository baseline changes only by explicit review. Comparisons require
identical complete comparison cohorts, identical killed/survived denominator key sets,
and identical exception/no-coverage membership; timeout, infrastructure,
cancellation, invalid, baseline-unavailable, partial status, changed node, policy
change, or trend-epoch boundary makes the pair non-comparable. Numerator and
denominator key-set digests are persisted. Execution IDs remain exact per revision.
Cache hit/miss/eviction reasons and identity versions are visible.

### Baseline and exceptions

`config/mutation-baseline.json` starts with `established: false`; it defines schema,
scope identity, comparison-key cohort, and the future five-point regression rule
without fabricating a score. A future command can propose a baseline artifact, but
committing it requires review. V1 never claims comparability merely because a source
path matches.

`config/mutation-exceptions.toml` starts empty. Each future record must match exact
mutant ID, path, source digest, start/end line and column, operator, occurrence,
owner, reason, and expiry. Validation rejects every wildcard, ambiguity, and stale
source. Excepted mutants have `equivalent_exception` status and remain visible; they
do not enter score numerator or denominator.

### CI design

One GitHub Actions workflow has three execution routes plus one evidence-validation
job:

- `pull_request`: checkout full enough history, set up Python/uv, run
  `trade dev mutation changed --base $BASE_SHA`; job-level timeout is 15 minutes
  around the 10-minute total controller budget. Its concurrency key is
  `mutation-pr-${repository}-${pull_request.number}` with `cancel-in-progress: true`;
  cancellation upload is best effort.
- nightly/manual core: `core`, cron `17 18 * * 1-6` UTC, 35-minute timeout,
  compatible result cache and validated rolling trend aggregate.
- weekly/manual full: explicit `workflow_dispatch` mode or cron `17 18 * * 0` UTC,
  with a 100-minute job timeout.

All v1 mutation outcomes are report-only. The execution step captures controller exit
0/1/2, receipt path and run ID through `$GITHUB_OUTPUT`, then returns success so score,
survivor, timeout, baseline, or tool-status outcomes do not block the route. The
receipt preserves the real exit and summary prints it. The workflow is additive and
is not made a required branch-protection check by this change. Evidence integrity is
different: a separate `validate-evidence` job runs `if: always()`, and a missing,
malformed, unsafe, digest-mismatched or receipt-unbound bundle is a real workflow
failure rather than an invented mutation result.

Scheduled and manual core/full share concurrency key
`mutation-long-${repository}` with `cancel-in-progress: false`. Native GitHub
concurrency guarantees at most one running and one pending member; a newer dispatch
may supersede an older pending member even though it does not cancel the running job.
The workflow relies only on the no-overlap guarantee and does not claim a durable or
lossless queue. A dispatch cancelled before command entry has no run ID and creates no
trend evidence. Manual input is exactly `core|full`. Trend records use the
publication sequence allocated under the report lock, not workflow dispatch or
adjustable wall-clock order.

`config/mutation-capacity.json` has schema `trade.mutation.capacity.v1`. A core/full
execution qualification binds mode, config/matrix/cohort/import-closure/tool/lock/
isolation-policy digests, Python and dependency lock identities, runner image digest,
effective CPU/quota, finite enforceable memory/cgroup identity, filesystem type, and
the measured planning/copy/baseline/per-mutant p50/p95/render/fsync/cleanup values.
It records qualification run/bundle digests, qualified UTC, expiry at no more than 30
days, budget headroom and `qualified: true|false`; review owns the checked-in file.
Before mutation dependency setup, scheduled/manual core/full recompute every identity
and freshness field. Missing, expired, false, mismatched, non-finite-memory, or
insufficient-headroom qualification produces a receipt-bound preflight report and
does not execute mutants. There is no permissive fallback. `trade dev mutation
qualify --mode core|full --runner-image-digest SHA256` is an explicit report-only
operator command; it runs cold-cache with one effective CPU, validates the same
isolation and cleanup contract, writes a proposal artifact, and never edits the
reviewed qualification file.

Every summary, bundle-construction and upload step uses `if: always()` and the exact
`trade.mutation.invocation.v1` receipt path exported through `$GITHUB_OUTPUT`; report
bundles have 14-day retention. The standard-library bundle validator validates the
receipt run ID, phase journal, lifecycle and cleanup facts, and exactly one referenced
immutable generation or typed fallback. It never reads global `current.json` or
chooses a newest path. It then creates a new immutable directory with schema
`trade.mutation.bundle.v1` containing:

- the exact receipt bytes;
- the complete referenced report generation and manifest, or the exact fallback;
- `validation.json` with schema `trade.mutation.bundle-validation.v1`, validator
  version, workflow/run identity, controller exit, receipt digest, evidence kind/root
  digest, lifecycle/count/cleanup validation results and closed failure codes;
- `bundle-manifest.json` with sizes and SHA-256 for every preceding member.

The validator read-backs the full bundle, fsyncs it, and exports its path and manifest
digest. Upload and the validation job accept only that exact path/digest; raw staging,
mutable output roots and unvalidated receipt-only artifacts are never uploaded as
successful evidence. Explicit missing-file behavior emits a GitHub failure summary
without manufacturing a bundle. This preserves evidence for controller exits,
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
- no eligible source zero-work;
- exact changed-line/core deterministic priority, definition allowlist, and limits;
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
  wrapper SIGTERM, descendant/session escape attempts, cgroup and subreaper fallback;
- fork-before-cgroup-move prevention, pidfd identity, listener-FD handoff/closure,
  brokered `openat2` descriptor injection, pathname/symlink races, controller-private
  audit integrity, and supervisor fallback after controller loss;
- per-mutant natural-exit versus cancellation linearization in both race orders;
- stop-event precedence at mutation-applied/pre-test/test-started boundaries and
  truthful phase facts for budget, signal, integrity, timeout and natural exit;
- run-generation atomic partial JSON/Markdown/HTML publication, independent fallback,
  invocation receipt binding, corrupt-pointer recovery, output-root safety, per-run
  lease scavenging, redaction, and generation/aggregate retention bounds;
- closed terminal/phase/cache count algebra, stale/corrupt cache invalidation,
  post-publication committed-run markers, manifest-root anchoring, compact immutable
  trend-source reservation/tombstone/hash-chain recovery, numeric eviction,
  aggregate-carrier validation, epoch gaps and projection-gap reporting;
- receipt/fallback/report/trend-anchor retention coupling, inspect/repair closed
  schemas and exit codes, and immutable bundle/validator digest binding;
- tuple-specific peak-memory admission, no-finite-bound single-worker fallback,
  cgroup/OOM evidence, cleanup-unconfirmed handling, and outer-timeout containment;
- unestablished/comparison-key baseline policy;
- capacity schema identity/freshness/headroom and schedule enforcement;
- dedicated coverage rcfile/environment independence and missing dependency diagnostics;
- workflow literal cron/mode/timeout/shared-long no-overlap/latest-pending semantics,
  report-only outcomes, strict evidence-validation job, receipt-bound bundle upload,
  aggregate restore ordering, and hard-runner-loss best-effort documentation.

Validation commands include focused pytest, a real small changed run, TOML/JSON/YAML/
static workflow checks, `bash -n`, ShellCheck when available, Ruff, BasedPyright,
compileall, `./trade dev check`, the exact mapped tests, full pytest with the
pre-existing stable failure reported, and `git diff --check`. Before schedules execute
mutants, cold-cache core and full qualification runs execute on one effective CPU and
the exact standard runner image. They record planning/copy/baseline/per-mutant
p50/p95/render/fsync/cleanup time and only pass when the reviewed schema identity,
30-day freshness and configured reserve/headroom validate in the workflow.

### Runtime concurrency evidence

Ownership is one Python 3.7 standard-library supervisor/subreaper plus one controller
process, a bounded thread pool, one synchronized pidfd/PGID/cgroup and per-mutant
typed-transition registry, and one receipt-bound monotonic deadline created before
every `uv` child with an earlier execution deadline. The supervisor creates a stopped
child and, when delegated cgroups exist, its worker cgroup; the child cannot enter the
trusted launcher until cgroup placement when applicable and pidfd/session/watchdog
ownership are read-back verified and the one-shot start gate is released. The launcher
must activate kernel policy before application import. Ordering is fixed before
admission; completion order never changes report ordering. Each thread owns one
bounded import-closure tree and restores the target between work items.

Atomicity is one root-hash-published run generation. Sequence reservation precedes
pointer publication; cache commit and trend-source/tombstone reconciliation are
post-publication idempotent projections whose state is journaled in the supervisor
receipt. A child write-only guard channel and controller-private syscall-audit state
are independently authenticated and jointly required. Timeout and cancellation target
full worker cgroups/process groups; the supervisor owns adopted descendants,
controller-loss fallback and continued cleanup. Backpressure is the CPU-and-memory
admission rule, finite queue, source/AST/dependency ceilings, detail budget, and
cleanup reserve. Partial failure is aggregated by exact terminal and phase facts
without converting infrastructure, cancellation, unconfirmed cleanup, or tool errors
to kills.

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
