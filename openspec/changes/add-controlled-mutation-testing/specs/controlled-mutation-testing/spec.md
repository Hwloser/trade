# Controlled Mutation Testing Specification

## ADDED Requirements

### Requirement: Mutation execution SHALL be bounded by mode

The repository SHALL expose `scripts/mutation-test changed|core|full`. Each mode SHALL
use an explicit source selection, mutant limit, worker limit, per-mutant timeout, and
wall-clock budget. No mode SHALL silently expand to a broader source or test scope.

The normative budgets SHALL be:

| Mode | Source scope | Mutant limit | Wall clock | CI use |
|---|---|---:|---:|---|
| `changed` | eligible changed production lines in configured definitions relative to an explicit target branch | 150 | 600 seconds | pull request |
| `core` | configured Python core business definitions | 1000 | 1800 seconds | nightly/manual |
| `full` | all configured eligible Python production definitions | 5000 | 5400 seconds | weekly/manual only |

The worker count SHALL be `min(4, max(1, floor(available_cpu/2)))` unless an explicit
lower value is supplied. An explicit value above the computed maximum SHALL fail
before mutation enumeration. Available CPU SHALL use the minimum of process affinity,
cgroup quota, and host CPU count when those values are available. Every baseline and
mutant process SHALL set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` to `1`; those controls SHALL enter environment/cache
identity.

The wall clock SHALL start at command entry and include Git discovery, source parsing,
coverage baselines, mutation execution, cancellation, and report publication. The
controller SHALL reserve 30/60/120 seconds of the changed/core/full wall clock for
cleanup and report publication inside, not after, the hard limit. The reserves SHALL
split into TERM, KILL/reap, render/hash/fsync, and pointer-or-fallback sub-budgets of
`5/5/15/5`, `10/10/30/10`, and `15/15/70/20` seconds. The execution deadline SHALL
be command entry plus wall budget minus reserve. The controller SHALL reject a
subprocess unless its timeout plus termination/reap allowance fits before that
deadline, and SHALL cancel all active groups when that deadline arrives. Candidate
enumeration SHALL stop at 3000/20000/100000 candidates.
Changed/core/full SHALL additionally cap source files at 32/128/512, one source file
at 1 MiB, aggregate source bytes at 8/32/128 MiB, one parsed tree at 100000 nodes and
depth 256, dependency traversal at 512 modules, 4096 edges, and depth 32, and
cumulative parse/operator visits at 3/20/100 million. Each worker private tree SHALL
be at most 512 files, 8 MiB, and 10 seconds to materialize; aggregate private trees
SHALL be at most 2048 files and 32 MiB, with at least 512 MiB free after reservation.
Every discovery, copy, and parse loop SHALL check the same monotonic deadline.

CLI `--max-mutants`, `--max-seconds`, and `--workers` options SHALL only lower the
selected mode's limits. Values above a mode or computed limit SHALL fail before
source parsing. No test, configuration, or retry path SHALL broaden a mode.

#### Scenario: Pull request changes no eligible production source

- **WHEN** `changed` compares the working revision with its target base and finds only
  documentation, configuration prose, tests, fixtures, generated, vendor, migration,
  logging, DTO, startup, or otherwise excluded paths
- **THEN** it exits successfully
- **AND** writes reports containing zero generated and executed mutants
- **AND** does not invoke pytest or mutate any source.

#### Scenario: Changed scope exceeds a budget

- **WHEN** eligible work exceeds the mutant or 600-second budget
- **THEN** execution stops scheduling new mutants
- **AND** active worker process groups are cancelled at the execution deadline within
  the in-budget TERM/KILL/reap allowance
- **AND** the partial report is retained with `budget_exhausted = true` and a precise
  stop reason.

### Requirement: Only deterministic first-order high-value mutations SHALL run

Every work item SHALL contain exactly one mutation specification affecting one source
location. Higher-order and multiple-location mutation SHALL be rejected both during
planning and immediately before execution.

The initial dependency SHALL be exactly Cosmic Ray 8.4.6. At startup the controller
SHALL verify the version, `cosmic_ray.plugins.get_operator`, and
`cosmic_ray.mutating.mutate_code`, `cosmic_ray.ast.get_ast`, and
`cosmic_ray.ast.ast_nodes`; other versions or missing APIs SHALL fail closed.
The exact initial operator allowlist SHALL be:

- `core/AddNot`;
- `core/ReplaceAndWithOr` and `core/ReplaceOrWithAnd`;
- `core/ReplaceTrueWithFalse` and `core/ReplaceFalseWithTrue`;
- `core/ReplaceComparisonOperator_Eq_NotEq` and
  `core/ReplaceComparisonOperator_NotEq_Eq`;
- `core/ReplaceComparisonOperator_Lt_LtE` and
  `core/ReplaceComparisonOperator_LtE_Lt`;
- `core/ReplaceComparisonOperator_Gt_GtE` and
  `core/ReplaceComparisonOperator_GtE_Gt`;
- `core/ReplaceComparisonOperator_Is_IsNot` and
  `core/ReplaceComparisonOperator_IsNot_Is`;
- `core/ReplaceBinaryOperator_Add_Sub` and
  `core/ReplaceBinaryOperator_Sub_Add`;
- `core/ReplaceBinaryOperator_Mul_Div` and
  `core/ReplaceBinaryOperator_Div_Mul`;
- `core/ReplaceBinaryOperator_FloorDiv_Div` and
  `core/ReplaceBinaryOperator_Mod_Mul`.

The boolean operators above replace boolean literals wherever Cosmic Ray supports
them; they are not generic return-value mutators. Generic return-value mutation is
deferred because Cosmic Ray 8.4.6 does not provide that operator.

The controller SHALL NOT enable numeric literal replacement, string replacement,
decorator removal, exception replacement, variable insertion/replacement, zero-loop
replacement, `break`/`continue` replacement, logging/metrics call removal,
void-call removal, constructor mutation, generated-method mutation, or any
multiple-location/higher-order operator.

Mutants SHALL be ordered deterministically by priority, repository-relative source
path, source line/column, operator priority/name, and occurrence. No unseeded random
sampling is permitted. If future sampling is introduced, its fixed seed and selection
algorithm SHALL be recorded in the report.

Enumeration SHALL call `get_ast(source)`, iterate `ast_nodes(tree)` in preorder, and
iterate each operator's `mutation_positions(node)` in returned order using one
module-wide occurrence counter. Startup compatibility tests SHALL compare that
occurrence/span sequence with Cosmic Ray `MutationVisitor` for every allowlisted
operator and nested syntax. A mismatch SHALL fail preflight.

#### Scenario: A loop condition is mutated

- **WHEN** a while-loop condition contains an eligible boolean, relational, or
  arithmetic expression
- **THEN** that single expression MAY be mutated
- **AND** the test runs in an independently terminable process group
- **BUT** loop-step deletion, `break`/`continue` replacement, or forced zero-iteration
  mutation SHALL NOT be generated.

### Requirement: Mutation target and test selection SHALL remain narrow

Production targets SHALL be regular tracked Python files and exact qualified
top-level definitions in the closed matrix below. The closed exclusions SHALL cover
actual repository test, build, generated, vendor, dependency, migration,
fixture/golden/test-data, DTO/data-only/serialization, logging and monitoring
wrapper, startup/dependency-injection, DB/cache/batch entrypoint, config-mapping,
devtool, and third-party client code.

`changed` SHALL prefer exact changed lines. If no candidate exists on those lines, it
MAY consider the enclosing changed definition in the same file. It SHALL NOT mutate
unchanged files. `core` SHALL use only configured core domain modules. `full` SHALL use
only the configured eligible Python production roots and SHALL still apply every
exclusion.

The v1 eligible definition-to-test matrix SHALL be closed:

| Scope | Source | Eligible qualified definitions | Affected unit tests | Clock policy |
|---|---|---|---|---|
| core | `trade_py/decision/action.py` | `_confidence_label`, `derive_action_decision` | `tests/test_decision_action.py`, `tests/test_explanation.py` | explicit_input |
| core | `trade_py/decision/world_state.py` | `infer_market_regime`, `infer_event_regime`, `infer_sentiment_regime`, `infer_technical_regime`, `infer_liquidity_regime`, `infer_uncertainty`, `_build_state_summary`, `build_world_state` | `tests/test_world_state.py`, `tests/test_decision_action.py`, `tests/test_explanation.py` | explicit_input |
| core | `trade_py/decision/scenario.py` | `_count_bullish`, `_count_bearish`, `build_scenario_summary` | `tests/test_explanation.py` | explicit_input |
| core | `trade_py/decision/explanation.py` | `build_explanation` | `tests/test_explanation.py` | explicit_input |
| core | `trade_py/trust/compute.py` | `_freshness_score`, `_trust_level`, `compute_prediction_trust`, `compute_portfolio_trust` | `tests/test_trust_layer.py` | explicit_input |
| full | `trade_py/factors/groups/event_features.py` | `build_event_group` | `tests/test_factor_groups.py` | explicit_input |
| full | `trade_py/signals/window_scorer.py` | `_score_large_order` | `tests/test_window_scorer.py` | explicit_input |

The configuration SHALL identify exact qualified definitions and the run report SHALL
bind their source/definition digests. Missing, duplicate, nested-ambiguous, or moved
definitions SHALL fail validation. `changed` SHALL consider changed lines only in this
complete matrix. `core` SHALL use the rows marked core. `full` SHALL use every row;
full means the complete configured eligible matrix, not all Python files. Adding a
source, definition, test, or clock policy SHALL require an explicit configuration and
test change.

Unlisted tracked production changes SHALL be reported as `deferred_unmapped`.
Untracked production Python paths SHALL be reported as `deferred_untracked`; neither
category SHALL generate mutants or affect score. There SHALL be no broad Observatory
exclusion: Observatory and all other unlisted definitions are deferred because they
are not yet in the closed v1 matrix.

The controller SHALL run only the mapped tests and SHALL NOT fall back to the full
pytest suite. Before mutation, one baseline per distinct test tuple SHALL collect
coverage.py line data from the verified private source tree. A candidate whose exact
planned start line is not executed SHALL become `no_coverage_line`; a source with no
mapping SHALL become `no_coverage_mapping`. Both roll up to `no_coverage` and neither
starts mutant pytest.

#### Scenario: Changed file has no trusted unit-test mapping

- **WHEN** a changed eligible source file has mutation candidates but no configured
  affected unit-test mapping
- **THEN** candidates are reported as `no_coverage`
- **AND** they are not counted as killed
- **AND** the report names the source, line, operator, and missing mapping.

#### Scenario: A production path is outside the reviewed matrix

- **WHEN** changed scope contains an unlisted tracked production Python definition or
  an untracked production Python path
- **THEN** it is reported as `deferred_unmapped` or `deferred_untracked`
- **AND** it generates and executes no mutant
- **AND** an otherwise deferred-only run is distinct from an ordinary
  documentation-only `zero_work` run.

### Requirement: Test execution SHALL be isolated and truthfully classified

The controller SHALL never mutate the developer's source file in place. It SHALL use a
thread pool so the controller process directly creates every `Popen`, owns every
process group ID in one synchronized active registry, and can terminate it if a
worker thread fails. Spawn/registration and cancellation SHALL share one gate lock:
cancellation is checked before spawn, PGID insertion completes before release, and a
group SHALL be terminated immediately if cancellation wins during spawn. Each thread
SHALL own one bounded private source tree, restore and hash-check the target file
between mutants, and remove the tree at shutdown.

A private tree SHALL be built from a sorted tracked manifest containing only
`trade_py/**/*.py`, mapped tests, tracked `tests/conftest.py` when present, and the
isolation/provenance plugin. It SHALL contain no symlink, non-regular file, data,
unmapped test, cache, `__pycache__`, or `.pyc`. Before every mutant the worker SHALL
purge bytecode/cache paths, create a fresh empty `PYTHONPYCACHEPREFIX`, and set
`PYTHONDONTWRITEBYTECODE=1`. File/byte/copy/free-space limits from the bounded-mode
requirement SHALL be checked before admission.

The adapter SHALL instantiate only `get_operator(name)()` and call
`mutate_code(original_text, operator, occurrence)`. It SHALL NOT call
`apply_mutation`, `use_mutation`, `mutate_and_test`, or Cosmic Ray `run_tests`.
Immediately before mutation it SHALL re-hash and re-enumerate the source using the
exact `get_ast`/`ast_nodes`/`mutation_positions` order, verify occurrence maps to the
planned start/end span, require
parseable non-identical output, and verify the diff intersects only that planned
location. Drift or API mismatch SHALL be `infrastructure_error`; unsupported output
SHALL be `invalid`. Results SHALL retain original/mutated digests and
planned/observed spans.

Pytest SHALL run with the private tree as working directory, absolute mapped test
paths, `--import-mode=importlib`, the private tree first and the checkout absent from
`PYTHONPATH`, and an injected provenance guard that fails unless the target module's
`__file__` and digest belong to the private tree. The process SHALL use a temporary
`TRADE_DATA_ROOT`, HOME, XDG and cache directories; scrub provider/API credentials and
proxy variables; fix `TZ=UTC`, locale, and `PYTHONHASHSEED`; install a Python socket
deny guard; deny external process launch; and reject canonical/symlink-resolved
filesystem opens beneath the real repository `data/`, model, database, parquet, and
generated-artifact roots. Tests SHALL prove provider, scheduler, event, real-data,
network, and external-client access fail closed. V1 SHALL refuse cache reuse for a
tuple that observes an undeclared wall clock; every current mapping declares
`explicit_input`.

The baseline time `T` for a selected affected-test set SHALL be measured before mutant
execution under the same provenance, coverage, environment, and process-group
controls. The baseline SHALL have its own finite timeout limited by remaining global
time. The per-mutant timeout SHALL be `max(10 seconds, 2.5 * T)`. In `changed` mode it
SHALL be capped at 60 seconds. A tuple requiring more SHALL be unavailable in changed
mode. Insufficient remaining time SHALL produce `not_run_budget`, not a late launch.

On timeout, cancellation, or global deadline, the controller SHALL terminate the
entire process group, wait a bounded grace period, escalate to kill, and reap the
process before returning. A timeout SHALL be `timeout`, not `killed`. Failure to
launch, copy, mutate, parse, collect, or execute infrastructure SHALL be
`infrastructure_error`, not `killed`.

Pytest exits SHALL map exactly: `0=survived`, `1=killed`, and `2/3/4/5` or a signal
SHALL be `infrastructure_error` in mutant phase. Baseline exit `0` is usable; any
other exit is `baseline_unavailable`. The classification SHALL be:

- baseline failure: mark that test tuple's selected mutants `baseline_unavailable`;
- pytest exits zero under the mutant: `survived`;
- pytest reports a test failure under a valid mutant: `killed`;
- deadline expires: `timeout`;
- no mapping or baseline line coverage: `no_coverage_mapping` or `no_coverage_line`;
- the operator cannot produce valid changed source: `invalid`;
- controller/tool/process failure: `infrastructure_error`.

#### Scenario: Mutant causes an infinite loop with descendants

- **WHEN** the selected test exceeds its computed timeout after creating child
  processes
- **THEN** the controller terminates and reaps the full process group
- **AND** records one timeout mutant
- **AND** no child remains alive after the worker completes.

### Requirement: Reports SHALL be complete, machine-readable, and recoverable

The default output root SHALL be repository `.mutation-testing/`. `--output-dir`
SHALL resolve inside that root, have no symlink ancestor, and either be absent or
contain the exact `.trade-mutation-output-v1` ownership marker. Filesystem roots,
repository data, DB, model, Parquet, generated-artifact roots, path escapes, and
unrelated existing directories SHALL fail before a lock or write.

Every invocation SHALL stage one run-ID directory containing:

- `report.json` using schema `trade.mutation.report.v1`;
- `summary.md` suitable for GitHub job summaries;
- `index.html` with summary counts and surviving/no-coverage details;
- `manifest.json` with the run ID and digest of every output.

JSON SHALL be authoritative and Markdown/HTML SHALL render from the persisted JSON.
After fsync of files and the staged directory, the controller SHALL atomically replace
one `current.json` pointer to publish the generation. A crash SHALL expose either the
previous complete generation or the new complete generation, never mixed files.
Atomic per-mutant cache/journal entries SHALL permit a retry to reuse only verified
matching outcomes; v1 does not expose a `--resume-from` report contract.

Each invocation SHALL hold a per-run staging lease containing PID, process-start
token, and boot identity. Staging cleanup SHALL require non-blocking lease acquisition,
valid stale owner identity, age at least 24 hours, and proof the run is not current.
Render, staging-validation, final-directory rename, and pointer-publication failures
SHALL remain distinct. If a complete generation cannot be validated and published,
the controller SHALL preserve owned staging, return exit 2, and atomically write only
one bounded `fallback-<run_id>.json` diagnostic outside `current.json`; fallback is
not a report generation.

One safe-error layer SHALL redact configured credential values,
credential-looking environment values, URL userinfo, and controlled home/temp roots
from console, JSON, Markdown, HTML, fallback, cache/trend recovery diagnostics, and
retained staging before persistence.

The report SHALL include mode, source/base/head identities, tool versions, config
digest, deterministic selection strategy, fixed seed or `null`, budgets, worker
limit, baseline timings, generated/selected/mutation-applied/test-started counts,
killed, survived, timeout, no-coverage, baseline-unavailable, invalid,
infrastructure-error, equivalent-exception and not-run counts, mutation score,
budget-exhausted state/reason, elapsed time, and every selected mutant's path,
line/column, definition, operator, related tests, status, duration, and bounded
diagnostic, original/mutated digest, planned/observed span, and cache identity.

Every selected mutant SHALL have exactly one terminal status. Each killed/survived
outcome SHALL carry `outcome_origin=fresh|cache`. Only killed/survived outcomes from a
complete identity-valid run SHALL be cacheable; timeout, infrastructure, invalid,
no-coverage, baseline-unavailable, exception, cancellation, and budget outcomes SHALL
never be reused as successful outcomes. `mutation_applied_current` SHALL count valid
source transformations performed in this invocation. `test_started_current` and the
compatibility alias `executed` SHALL count only current-invocation mutant pytest
starts. Therefore:

- `selected` equals the sum of every terminal status;
- `resolved = fresh_resolved + cache_resolved`;
- `fresh_resolved = killed_fresh + survived_fresh + timeout +
  infrastructure_error_test_started`;
- `cache_resolved = killed_cache + survived_cache`;
- `killed = killed_fresh + killed_cache` and
  `survived = survived_fresh + survived_cache`;
- `test_started_current = executed = fresh_resolved`;
- `no_coverage = no_coverage_mapping + no_coverage_line`;
- `not_run = not_run_budget + not_run_plan + baseline_unavailable +
  equivalent_exception`;
- mutation score SHALL be `null` when `killed + survived == 0`, otherwise
  `killed / (killed + survived)`.

Timeout, no-coverage, baseline-unavailable, invalid, infrastructure-error, exception,
and not-run outcomes SHALL NOT improve the score.

`--plan-only` SHALL validate config/tool APIs, enumerate and select candidates, read no
outcome cache, run no coverage/pytest, assign every selected mutant `not_run_plan`,
publish run status `plan_only`, return score `null`, and exit 0 unless preflight or
publication fails. Run-status precedence SHALL be `report_failed`,
`signal_cancelled`, `preflight_failed`, `degraded_infrastructure`,
`budget_partial`, `plan_only`, `deferred_only`, `zero_work`, then `complete`. A run
with no eligible candidate but one or more deferred production paths SHALL be
`deferred_only`, return score `null`, run no pytest, and exit 0. A future policy
regression changes exit code, not factual run status.

Core and full modes SHALL persist a content-addressed result cache. Its identity SHALL
include execution/mutant ID, the digest of every tracked Python source file, mapped tests,
fixtures/conftest, `pyproject.toml`, `setup.cfg`, `uv.lock`, mutation configuration
and exceptions, normalized pytest arguments, Python/Cosmic Ray/coverage/pytest/plugin
versions, platform/architecture, controlled environment including native-thread
controls, declared clock policy, selection algorithm, and complete mutant cohort.
Corrupt, partial, unbounded, undeclared-clock, or
identity-mismatched entries SHALL be ignored. Changed mode MAY read a matching entry
but SHALL remain correct without it.

Core/full SHALL append one `trend.jsonl` record keyed by mode, commit, exact
comparison-key cohort, scope/config/environment digests, and report generation.
Trend retention SHALL be at most 365 records, 8 MiB, and 400 days. Result cache
retention SHALL be at most 20000 entries and 512 MiB. Under the output publication
lock, deterministic eviction SHALL remove oldest `created_at,key` records first and
record evicted counts, bytes, and range. CI SHALL restore only a
schema/tool/config-compatible cache prefix, validate entries, append the current
record, and save it under a new run-specific immutable key. Score regression is
comparable only over identical complete, unambiguous comparison-key cohorts with
unchanged normalized node text and policy; execution IDs remain source-exact.
Different exclusions, budgets, partial runs, ambiguous/added/deleted/changed
candidates, or cohorts SHALL be non-comparable.

Raw subprocess output SHALL be capped at 64 KiB per mutant, but JSON SHALL retain at
most 8 KiB diagnostic and 8 KiB diff excerpts per detailed mutant. Mode-level detail
budgets SHALL be 1/8/32 MiB for changed/core/full. Space for one minimal structural
record per selected mutant SHALL be reserved before admission. Detail pressure SHALL
truncate marked excerpts deterministically or stop admission; it SHALL never omit a
selected mutant or prevent the 32/64/128 MiB JSON report from publishing.
Markdown SHALL be capped at 4/8/16 MiB, HTML at 8/16/32 MiB, manifest at 1 MiB,
complete staged generation at 48/96/192 MiB, and peak report-renderer memory at
64/128/256 MiB. Renderers SHALL stream structural records and reuse the same bounded
detail rather than embed an unrestricted JSON copy. Worst-case escaped output SHALL be
reserved before admission.

#### Scenario: Global time expires after partial execution

- **WHEN** some mutants completed before the wall-clock budget
- **THEN** their outcomes remain in all reports
- **AND** unstarted selected mutants are marked `not_run_budget`
- **AND** report generation completes within the in-budget 30/60/120-second reserve
- **AND** the absolute 600/1800/5400-second hard deadline is never extended.

### Requirement: Mutation quality policy SHALL be gradual and auditable

This change SHALL keep all CI mutation jobs non-blocking. The controller SHALL still
evaluate a repository baseline when it is established and comparable:

- a changed-scope mutation-score decrease greater than five percentage points SHALL be
  reported as a policy regression only when base and head have one identical complete
  comparison-key cohort;
- modified core business code SHALL display a 70% target;
- 80% SHALL be documented only as a future core target, not a current repository-wide
  gate;
- no-coverage mutants SHALL be reported separately.

`execution_id`/`mutant_id` SHALL remain source-digest exact for execution, cache, and
exceptions. A separate `comparison_key` SHALL include source path, qualified eligible
definition, definition-relative span, normalized original node text, operator
name/arguments, and occurrence within that definition. Base and head SHALL be
enumerated separately. Added, deleted, changed-node, ambiguous, policy-different, or
partial candidates SHALL make the run non-comparable; line shifts or unrelated edits
MAY remain comparable. V1 SHALL report `baseline_comparable=false` rather than invent
a mapping.

Equivalent or meaningless mutants MAY be registered only with exact `mutant_id`,
source path, source digest, start/end line and column, operator, occurrence, owner,
reason, and review expiry. Directory, filename, definition, location, or operator
wildcards SHALL be rejected. Expired, stale-digest, missing, or ambiguous exceptions
SHALL fail configuration validation. Exceptions SHALL remain visible in reports and
SHALL not be counted as killed.

#### Scenario: Baseline is not yet established

- **WHEN** the saved baseline is marked unestablished or no comparable scope exists
- **THEN** the invocation remains report-only
- **AND** states that no regression decision was possible
- **AND** does not manufacture a baseline from the current result.

### Requirement: CI SHALL route bounded modes without making them universal PR gates

GitHub Actions SHALL:

- run a cheap changed-mode plan on pull requests and execute mutation only when at
  least one eligible production Python file is present;
- pass the pull request base SHA explicitly and never infer a missing base by
  broadening scope;
- cap the changed job at 15 minutes around the 600-second total script budget;
- use PR concurrency key
  `mutation-pr-${repository}-${pull_request.number}` with
  `cancel-in-progress: true`;
- use `if: always()` to append a success or fallback failure summary and upload the
  run/staging/fallback path with 14-day retention for controller exits, internal
  deadlines, and gracefully delivered signals while the job remains alive;
- document upload after hard GitHub job cancellation, runner loss, or host loss as
  best effort rather than guaranteed;
- run core on cron `17 18 * * 1-6` UTC and on manual request, with 1800-second script
  budget, non-blocking job semantics, and a 35-minute outer timeout;
- run full only on cron `17 18 * * 0` UTC or explicit manual request, with
  5400-second script budget, non-blocking job semantics, and a 100-minute outer
  timeout;
- give scheduled/manual core and full one shared concurrency key
  `mutation-long-${repository}` with `cancel-in-progress: false`, so they queue rather
  than overlap or destroy evidence;
- order trend records by report UTC completion time then run ID, not workflow dispatch
  order.

The ordinary pull-request workflow SHALL never invoke `core` or `full`. Documentation,
test-only, fixture, generated, and excluded changes SHALL not execute mutant tests.

#### Scenario: A documentation-only pull request runs CI

- **WHEN** the changed-mode plan sees no eligible source
- **THEN** the mutation job exits successfully after producing a zero-work report
- **AND** no mutation dependency or test worker is required beyond planning.

#### Scenario: Long-running routes overlap

- **WHEN** nightly core, weekly full, or manual core/full are requested concurrently
- **THEN** the shared long-run concurrency group runs at most one
- **AND** queued runs are not cancelled by newer runs
- **AND** no two long modes compete for mutation workers or trend publication.
