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
| Stable core baseline | 97 passed in 1.77 seconds |

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
- 10-second initial per-mutant timeout from the measured 1.77-second core baseline,
  dynamically recomputed per affected-test set and capped at 60 seconds for changed;
- parent-owned process-group termination and truthful status/count algebra;
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
config/mutation-testing.toml                 policy and source/test ownership
config/mutation-baseline.json                reviewed score history seed
config/mutation-exceptions.toml              precise reviewed exceptions
trade_py/devtools/mutation_testing/
  bootstrap.py                               import-light scope/deadline/receipt owner
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
  trend.py                                   projection from committed generations
  render.py                                  JSON-derived Markdown/HTML rendering
  report_store.py                            receipt, leases, atomic publish, retention
  baseline.py                                comparable baseline evaluation
.github/workflows/mutation-testing.yml       CI routing only
docs/mutation-testing.md                     developer operations
```

The shell facade performs no parsing beyond locating the repository and invoking the
locked optional environment. The CLI coordinates owned modules but does not contain
mutation logic. Cosmic Ray is an adapter dependency and never imports application
runtime modules. `models.py` imports no controller adapter; planning and execution
depend on models, while selection/engine never import process, storage, cache, trend,
or rendering modules. `render.py` consumes persisted report models and
`report_store.py` owns bytes and filesystem transitions, not outcome semantics. Tests
use synthetic repositories and source files.

The controller is a devtool, not a domain service. It does not belong in the future
business Context graph and is excluded as a mutation target.

### Data and state invariants

The durable artifacts are developer reports and cache entries, never business state.
Identifiers and invariants:

- `run_id` is a UUID generated at wrapper entry. Before dependency preparation, the
  wrapper atomically writes `invocations/<run_id>.json` with schema
  `trade.mutation.invocation.v1`, mode, lifecycle state, one absolute monotonic
  deadline, and exact expected staging/run/fallback paths. Bootstrap and full CLI
  update only that receipt through validated transitions and export its path to CI.
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
  part of run identity.
- ordered selection is stable and does not depend on filesystem enumeration order.
- a cached result is reusable only when the complete tracked Python tree, mapped
  tests/fixtures, lock and pytest configuration, operator/config/exception policy,
  normalized command/environment including native-thread controls, platform, declared
  per-mapping clock and entropy policies, complete mutant cohort, and tool versions
  match. V1 mappings declare `explicit_input` clock policy and an explicit entropy
  policy; a tuple whose bounded import closure references an undeclared clock or
  entropy API is non-cacheable. The trust tuple is `entropy_non_cacheable` because its
  default path can call `uuid.uuid4()`.
- one run directory is staged and fsynced; one `current.json` pointer atomically
  publishes JSON, Markdown, HTML, and their digest manifest as one generation.
- mutation score excludes timeout, no-coverage, baseline-unavailable, invalid,
  infrastructure, exception, and budget-not-run states.
- `generated` counts eligible candidates seen up to the candidate-scan ceiling and
  never claims the unknown unscanned remainder; `selected` is bounded by mode;
  `mutation_applied_current` requires a verified current-invocation transformation;
  `test_started_current`/`executed` requires a current-invocation mutant pytest.
- every killed/survived outcome has `outcome_origin=fresh|cache`. Only verified
  killed/survived outcomes are cacheable. Cache hits increase `cache_resolved`, not
  `test_started_current`.
- every selected mutant has exactly one terminal state, including
  `cancelled_test_started` or `not_run_cancelled` when a signal wins, and score is
  `null` when no killed or survived mutant exists.

Unavailable states are explicit: missing base, unsupported tool, invalid config,
baseline test failure, no eligible source, no test mapping, budget exhaustion, and
report write failure never become an empty successful mutation score.

### Persistent-write safety

The controller is the only writer of mutation-testing developer state. Output
defaults to repository `.mutation-testing/`. An override must resolve inside that
directory, have no symlink ancestor, and either be absent or contain the exact
`.trade-mutation-output-v1` ownership marker; filesystem roots, repository `data/`,
DB/model/Parquet/generated-artifact paths, and unrelated existing directories are
rejected before locks or files are created. Each invocation owns one UUID `run_id`
staging directory and holds a per-run lease for its lifetime. The lease records PID,
process start token, and boot identity. An exclusive bounded output-root publication
lock serializes current-pointer, retention, and core/full trend updates. Cache records
are content-addressed by complete cache identity and execution ID but first stage
under `cache/staging/<run_id>`. A digest-bound `cache/commits/<run_id>.json` marker is
published only after the run is complete and cache-eligible; readers require the
marker and ignore entries left by partial, cancelled, degraded, or report-failed
runs. Same-key writers use exclusive creation or a per-key lock, and a losing writer
accepts the predecessor only after schema, identity, size, digest, and commit-marker
verification.

Before publication, one safe-error layer redacts configured credential values,
credential-looking environment values, URL userinfo, and controlled home/temp roots
from console output, JSON, Markdown, HTML, fallback diagnostics, and retained staging
evidence. The controller validates the report schema and count equations, one terminal
state for every selected mutant, all configured size ceilings, and the SHA-256 and
byte size of JSON, Markdown, and HTML. It writes those files and a
manifest below the private staging directory, flushes and fsyncs each file and the
directory, renames the directory to its immutable final run path, then atomically
replaces and directory-syncs `current.json`. Readers resolve the pointer once,
validate its run ID, manifest, and member digests, and read only that generation.
They never compose output from separate runs. Cache records and the bounded trend
file use the same write, fsync, atomic-replace, directory-fsync discipline.

A crash before pointer replacement leaves the previous complete generation current;
a crash after replacement exposes only the already verified new generation.
Unreferenced staging is not successful evidence. A scavenger may remove it only after
non-blocking acquisition of its lease, validation of owner identity, a minimum
24-hour age, and proof that it is not current; an active concurrent invocation cannot
be removed. Malformed, oversized, symlinked,
path-escaping, identity-mismatched, or hash-mismatched report state fails closed and
is preserved for diagnosis. A missing pointer on first run is normal. Under the
publication lock, a malformed/symlinked pointer, missing referenced generation, or
manifest/digest mismatch is moved to a UUID quarantine record before a valid new
generation may replace it; the next manifest records the exact recovery code and
quarantine path. If quarantine cannot be completed, publication fails with a stable
copy-pasteable repair command and leaves `current.json` unchanged. Corrupt or
uncommitted cache is a miss and is recomputed. Trend is a rebuildable projection of
validated immutable committed generations: startup idempotently reconciles missing
run IDs, quarantines corrupt projection state, and rebuilds within its bound instead
of treating an independent write as source truth.

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
status, member hashes, and cache/trend decisions without credentials or raw
environment values. Immutable generations are retained for at most 30 entries,
2 GiB, and 30 days; fallback/quarantine/failed-staging evidence is retained for at
most 50 entries, 512 MiB, and 14 days. Deterministic oldest-creation/run-ID eviction
under the publication lock protects `current`, all active leases, and the current
failure evidence and records every eviction. No backup is required for ignored disposable developer state;
CI uploads the immutable verified run directory. Rollback removes tracked tooling
without rewriting or deleting these audit generations, which remain subject to
explicit manual cleanup and never enter production state.

### Contracts and compatibility

#### CLI

```text
scripts/mutation-test MODE
  [--base REF]
  [--output-dir PATH]
  [--max-mutants N]
  [--max-seconds N]
  [--workers N]
  [--plan-only]
  [--no-cache]
```

`MODE` is required and closed to `changed`, `core`, or `full`. Changed mode defaults
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
`coverage==7.10.7`, and `pytest>=8,<9`. `scripts/mutation-test` creates the run ID,
receipt, parent watchdog, and absolute monotonic deadline before any `uv` child. It
first executes the import-light bootstrap with
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
full-suite fallback. Runtime dependency resolution is unchanged.

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
2. enclosing definition containing a changed line;
3. configured core path priority;
4. configured defect-history priority;
5. remaining eligible changed file;
6. relative path, line, column, operator priority, occurrence.

Changed mode never reaches priority categories in an unchanged file. Core/full use
categories 3-6. Truncation records the last admitted key, the exact eligible
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

The command-entry monotonic hard deadline owns planning, baselines, execution,
cancellation, and publication. An earlier execution deadline reserves publication
time inside that hard limit. A controller-owned thread pool is used so the controller
process directly creates every `Popen`. Spawn, PGID registration, and cancellation
share one gate lock: admission checks cancellation under the lock, starts the process
group, inserts the PGID before releasing the lock, then immediately terminates it if
an asynchronous signal flag won during spawn. The cancellation coordinator takes the
same lock before snapshotting active groups. Worker exceptions, lost pipes, repeated
signals, and parent cancellation all use the same TERM, bounded wait, KILL,
group-existence check, and reap path.

Each thread creates one private tree from a sorted bounded import closure rooted at
the eligible source, mapped tests, required package `__init__.py` files, tracked
`tests/conftest.py` when present, and the injected provenance/isolation plugin. Static
imports are resolved only to repository-local regular Python files within the module,
edge, depth, file, byte, and deadline ceilings. Configuration must list an exact
reviewed dependency manifest for unresolved dynamic imports; missing or extra
requirements fail preflight. The complete tracked-Python digest remains cache identity
but is not the copy manifest. Symlinks, non-regular files, bytecode, caches, data, and
unrelated tests are rejected or absent. A 10x unrelated-module fixture must leave the
same bounded closure executable. The tree is reused across mutants. Before every work
item it restores and hashes the target, removes all
bytecode/cache paths, creates a fresh per-mutant `PYTHONPYCACHEPREFIX`, sets
`PYTHONDONTWRITEBYTECODE=1`, revalidates planned source identity/span, applies only
`get_operator` plus `mutate_code`, and restores/hash-checks afterward. Consecutive
same-size mutations with fixed mtimes are a required regression test.

Pytest runs from the private root with absolute mapped tests, importlib mode, checkout
removed from import paths, and a provenance plugin verifying module `__file__` and
digest. The environment sets temporary `TRADE_DATA_ROOT`, HOME/XDG/cache directories,
UTC/locale/hash seed, native thread variables `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`BLIS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to `1`, scrubs credentials/proxies,
denies sockets and external process launch through the test isolation plugin, and
guards canonical/symlink-resolved opens under real data/model/DB/parquet/artifact
roots. The plugin writes every provenance/isolation violation to a controller-created
0600 sidecar authenticated by a per-process random token and append sequence outside
application exception handling. The controller validates the sidecar before exit-code
classification; any violation is `infrastructure_error` even if production code
catches the raised exception or pytest otherwise exits 0/1. These controls and
versions are cache/report identity. Baselines use the same controls plus line coverage
and a finite timeout bounded by remaining execution time.

The coverage adapter is exactly coverage.py 7.10.7 in statement/line mode, not branch
mode. It invokes `python -m coverage run --data-file <private-temp> --source
<private-package> -m pytest --import-mode=importlib <absolute-mapped-tests>`, then
`python -m coverage json --data-file <private-temp> -o <private-json>`. Both commands
use the same supervisor and environment. The adapter accepts only the target
private-tree path, canonicalizes it back to the repository-relative source path,
extracts `executed_lines` from bounded JSON, rejects unknown/multiple paths or
malformed/oversized data, and removes data/JSON after use. Coverage exit/data failure
is `baseline_unavailable`, never no-coverage or killed.

Pytest mutant exits map `0=survived`, `1=killed`, and `2/3/4/5` or signal to
infrastructure error only after the authenticated violation sidecar is clean. A
controller signal produces `cancelled_test_started` for a started mutant, not
infrastructure error; unstarted selected work becomes `not_run_cancelled`. Nonzero
baseline means `baseline_unavailable`. Timeout is always timeout. A failed tuple does
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

The audited stable core set took 1.77 seconds, yielding a 10-second initial timeout.
Each distinct affected-test tuple is timed independently. The changed cap remains 60
seconds even when a selected test baseline would imply more; such a tuple is reported
as incompatible with PR mode rather than silently increasing the job.

The 30/60/120-second reserves are inside, never after, the 600/1800/5400-second hard
limits. They split into TERM, KILL/reap, render/hash/fsync, and pointer-or-fallback
sub-budgets of `5/5/15/5`, `10/10/30/10`, and `15/15/70/20` seconds. Admission is
rejected unless the computed mutant timeout plus TERM/KILL/reap allowance fits before
the execution deadline. At that deadline all active process groups are cancelled;
publication then uses the remaining reserve and never extends the hard deadline.

Source file/byte/tree/dependency and parse/operator-visit limits plus deadline
checkpoints bound operator-sparse trees where candidate count alone would not stop
planning. Effective CPU is the minimum of affinity, cgroup quota, and host count.
Before copying, the complete deterministic manifest and aggregate worker reservation
must fit file, byte, copy-time, and remaining-space limits. Structural report capacity
and worst-case escaped detail are reserved for every selected mutant. JSON renders
incrementally; Markdown/HTML use the same bounded detail records rather than embedding
JSON. Detail truncates deterministically before any per-file, generation, or renderer
limit.

At 10x eligible source size, source and AST ceilings stop work before candidate
truncation; at 10x unrelated repository modules, the root-bounded import closure
remains within the same copy manifest rather than copying all `trade_py`.
At 10x test cost, baseline timing excludes incompatible tuples before scheduling or
the wall deadline terminates admission. At 10x output, bounded ring capture truncates
diagnostics without blocking pipes.

### Observability and operations

Human output prints mode, base/head, eligible files, selected tests, budgets, progress,
outcome counts, score, and report paths. JSON is authoritative. Markdown contains the
exact PR summary fields requested by the user. HTML is static and self-contained.

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

Core/full append one trend record keyed by mode, commit, exact comparison-key cohort,
scope/config/environment/tool digests, and run ID. Trend retention is at most 365
records, 8 MiB, and 400 days. Result cache retention is at most 20000 entries and
512 MiB. Under the output publication lock, deterministic eviction removes the oldest
`created_at, key` records first and records evicted count/bytes/range in the current
report. Trend is derived/reconciled idempotently from validated committed generation
manifests and uses a monotonic publication sequence allocated under the lock; UTC is
display/retention metadata, not sole ordering truth. CI restores only a
schema/tool/config-compatible cache prefix, validates every entry and its committed-run
marker before use, and saves under a run-specific immutable key. The repository
baseline changes only by explicit review. Comparisons require identical complete
comparison-key cohorts with no ambiguity, partial status, changed node, or policy
change; execution IDs remain exact per revision. Cache hit/miss/eviction reasons and
identity versions are visible.

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

One GitHub Actions workflow has three jobs/routes:

- `pull_request`: checkout full enough history, set up Python/uv, run
  `scripts/mutation-test changed --base $BASE_SHA`; job-level timeout is 15 minutes
  around the 10-minute total controller budget. Its concurrency key is
  `mutation-pr-${repository}-${pull_request.number}` with `cancel-in-progress: true`;
  cancellation upload is best effort.
- nightly/manual core: `core`, cron `17 18 * * 1-6` UTC, 35-minute timeout,
  compatible trend/result cache, and `continue-on-error: true`.
- weekly/manual full: explicit `workflow_dispatch` mode or cron `17 18 * * 0` UTC,
  with a 100-minute job timeout.

Scheduled and manual core/full share concurrency key
`mutation-long-${repository}` with `cancel-in-progress: false`. Native GitHub
concurrency guarantees at most one running and one pending member; a newer dispatch
may supersede an older pending member even though it does not cancel the running job.
The workflow relies only on the no-overlap guarantee and does not claim a durable or
lossless queue. A dispatch cancelled before command entry has no run ID and creates no
trend evidence. Manual input is exactly `core|full`. Trend records use the
publication sequence allocated under the report lock, not workflow dispatch or
adjustable wall-clock order.

Every summary and artifact step uses `if: always()`, 14-day retention, and the exact
`trade.mutation.invocation.v1` receipt path exported through `$GITHUB_OUTPUT`.
CI validates receipt run ID/lifecycle and referenced manifest or typed fallback; it
never reads global `current.json` or chooses a newest path. Explicit missing-file
behavior writes a fallback failure summary. This guarantees
evidence for controller exits, internal deadlines, and gracefully delivered signals
while the job remains alive. GitHub hard job cancellation, runner loss, or host loss
cannot guarantee later steps and are explicitly best effort. Outer timeouts remain
strictly beyond internal hard deadlines and cleanup reserves.

The script always plans changed scope before installing/running the heavy mutation
engine. If no eligible files exist, it produces a zero-work report and exits. GitHub
path filters are an optimization only; script selection remains authoritative.

### Validation strategy

Tests cover:

- closed config and operator validation;
- changed Git rename/add/delete/tracked line hunks plus deferred untracked inventory;
- no eligible source zero-work;
- exact-line/definition/core deterministic priority, definition allowlist, and limits;
- first-order rejection;
- exclusions and no broad exception patterns;
- test mapping and no full-suite fallback;
- exact-line baseline coverage and mapped-but-uncovered behavior;
- private-overlay import/digest provenance and original source hashes;
- temporary data roots, credential/proxy scrubbing, socket denial, and real-data
  filesystem guards;
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
- authenticated isolation-violation sidecars that override caught/uncaught pytest exits;
- run-generation atomic partial JSON/Markdown/HTML publication, independent fallback,
  invocation receipt binding, corrupt-pointer recovery, output-root safety, per-run
  lease scavenging, redaction, and generation/aggregate retention bounds;
- complete cache/cohort identity, fresh/cache count algebra, stale/corrupt
  invalidation, committed-run markers, numeric eviction, and derived trend recovery;
- unestablished/comparison-key baseline policy;
- script forwarding and missing dependency diagnostics;
- workflow literal cron/mode/timeout/shared-long no-overlap/latest-pending semantics,
  non-blocking receipt-bound always-upload assertions, and hard-runner-loss best-effort
  documentation.

Validation commands include focused pytest, a real small changed run, core plan or
bounded small run, TOML/JSON/YAML/static workflow checks, `bash -n`, ShellCheck when
available, Ruff, BasedPyright, compileall, `./trade dev check`, original stable core
tests, full pytest with the pre-existing failure reported, and `git diff --check`.

### Runtime concurrency evidence

Ownership is one wrapper watchdog plus one controller process, a bounded thread pool,
one synchronized active PGID registry with a shared spawn/cancel gate, and one
receipt-bound monotonic deadline created before every `uv` child with an earlier
execution deadline. Ordering is determined before admission; completion order never
changes report ordering. Each thread owns one bounded import-closure tree and restores
the target between work items. Atomicity is one published run generation plus staged
cache entries made visible only by a completed-run marker; trend is derived from
committed generations. Timeout and
cancellation own full process groups. Backpressure is the bounded pool, finite queue,
source/AST/dependency ceilings, detail budget, and cleanup reserve. Partial failure is
aggregated by exact terminal status without converting infrastructure errors to
kills.

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
