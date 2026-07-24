# Controlled Mutation Testing Specification

## ADDED Requirements

### Requirement: Mutation execution SHALL be bounded by mode

The repository SHALL expose `scripts/mutation-test changed|core|full`. Each mode SHALL
use an explicit source selection, mutant limit, worker limit, per-mutant timeout, and
wall-clock budget. No mode SHALL silently expand to a broader source or test scope.

The normative budgets SHALL be:

| Mode | Source scope | Mutant limit | Wall clock | CI use |
|---|---|---:|---:|---|
| `changed` | eligible changed production lines/files relative to an explicit target branch | 150 | 600 seconds | pull request |
| `core` | configured Python core business modules | 1000 | 1800 seconds | nightly/manual |
| `full` | all configured eligible Python production modules | 5000 | 5400 seconds | weekly/manual only |

The worker count SHALL be `min(4, max(1, floor(available_cpu/2)))` unless an explicit
lower value is supplied. An explicit value above the computed maximum SHALL fail
before mutation enumeration. Available CPU SHALL use the minimum of process affinity,
cgroup quota, and host CPU count when those values are available.

The wall clock SHALL start at command entry and include Git discovery, source parsing,
coverage baselines, mutation execution, cancellation, and report publication. The
controller SHALL reserve 30/60/120 seconds of the changed/core/full wall clock for
cleanup and report publication and SHALL stop new subprocess admission before that
reserve begins. Candidate enumeration SHALL stop at 3000/20000/100000 candidates.
Changed/core/full SHALL additionally cap source files at 32/128/512, one source file
at 1 MiB, aggregate source bytes at 8/32/128 MiB, one parsed tree at 100000 nodes and
depth 256, and dependency traversal at 512 modules, 4096 edges, and depth 32. Every
discovery and parse loop SHALL check the same monotonic deadline.

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
- **AND** running worker process groups are allowed only a bounded shutdown grace
- **AND** the partial report is retained with `budget_exhausted = true` and a precise
  stop reason.

### Requirement: Only deterministic first-order high-value mutations SHALL run

Every work item SHALL contain exactly one mutation specification affecting one source
location. Higher-order and multiple-location mutation SHALL be rejected both during
planning and immediately before execution.

The initial dependency SHALL be exactly Cosmic Ray 8.4.6. At startup the controller
SHALL verify the version, `cosmic_ray.plugins.get_operator`, and
`cosmic_ray.mutating.mutate_code`; other versions or missing APIs SHALL fail closed.
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

#### Scenario: A loop condition is mutated

- **WHEN** a while-loop condition contains an eligible boolean, relational, or
  arithmetic expression
- **THEN** that single expression MAY be mutated
- **AND** the test runs in an independently terminable process group
- **BUT** loop-step deletion, `break`/`continue` replacement, or forced zero-iteration
  mutation SHALL NOT be generated.

### Requirement: Mutation target and test selection SHALL remain narrow

Production targets SHALL be regular tracked Python files beneath configured eligible
roots. The closed exclusions SHALL cover actual repository test, build, generated,
vendor, dependency, migration, fixture/golden/test-data, DTO/data-only, logging and
monitoring wrapper, startup/dependency-injection, config-mapping, devtool, and
third-party client paths.

`changed` SHALL prefer exact changed lines. If no candidate exists on those lines, it
MAY consider the enclosing changed definition in the same file. It SHALL NOT mutate
unchanged files. `core` SHALL use only configured core domain modules. `full` SHALL use
only the configured eligible Python production roots and SHALL still apply every
exclusion.

The v1 eligible source-to-test matrix SHALL be closed:

| Scope | Source | Affected unit tests |
|---|---|---|
| core | `trade_py/decision/action.py` | `tests/test_decision_action.py` |
| core | `trade_py/decision/world_state.py` | `tests/test_world_state.py`, `tests/test_decision_action.py`, `tests/test_explanation.py` |
| core | `trade_py/decision/scenario.py` | `tests/test_explanation.py` |
| core | `trade_py/decision/explanation.py` | `tests/test_explanation.py` |
| core | `trade_py/trust/compute.py` | `tests/test_trust_layer.py` |
| full | `trade_py/factors/groups/event_features.py` | `tests/test_factor_groups.py` |
| full | `trade_py/signals/window_scorer.py` | `tests/test_window_scorer.py` |

`changed` SHALL consider changed lines only in this complete matrix. `core` SHALL use
the rows marked core. `full` SHALL use every row; full means the complete configured
eligible matrix, not all Python files. Adding one source or test SHALL require an
explicit configuration and test change. There SHALL be no broad Observatory
exclusion: Observatory and all other unlisted modules are deferred because they are
not yet in the closed v1 matrix.

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

### Requirement: Test execution SHALL be isolated and truthfully classified

The controller SHALL never mutate the developer's source file in place. It SHALL use a
thread pool so the controller process directly creates every `Popen`, owns every
process group ID in one synchronized active registry, and can terminate it if a
worker thread fails. Each thread SHALL own one bounded private source tree, restore
and hash-check the target file between mutants, and remove the tree at shutdown.
Private-tree file count, bytes, temporary free-space use, and copy time SHALL be
bounded before worker admission.

The adapter SHALL instantiate only `get_operator(name)()` and call
`mutate_code(original_text, operator, occurrence)`. It SHALL NOT call
`apply_mutation`, `use_mutation`, `mutate_and_test`, or Cosmic Ray `run_tests`.
Immediately before mutation it SHALL re-hash and re-enumerate the source using the
same visitor order, verify occurrence maps to the planned start/end span, require
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
deny guard; and reject filesystem opens beneath the real repository `data/`, model,
database, parquet, and generated-artifact roots. Tests SHALL prove provider,
scheduler, event, real-data, and network access fail closed.

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

The report SHALL include mode, source/base/head identities, tool versions, config
digest, deterministic selection strategy, fixed seed or `null`, budgets, worker
limit, baseline timings, generated/selected/mutation-applied/test-started counts,
killed, survived, timeout, no-coverage, baseline-unavailable, invalid,
infrastructure-error, equivalent-exception and not-run counts, mutation score,
budget-exhausted state/reason, elapsed time, and every selected mutant's path,
line/column, definition, operator, related tests, status, duration, and bounded
diagnostic, original/mutated digest, planned/observed span, and cache identity.

Every selected mutant SHALL have exactly one terminal status. `mutation_applied`
SHALL count valid source transformations. `test_started` and the compatibility alias
`executed` SHALL count only mutants for which pytest actually started. Therefore:

- `selected` equals the sum of every terminal status;
- `test_started = killed + survived + timeout + infrastructure_error_test_started`;
- `no_coverage = no_coverage_mapping + no_coverage_line`;
- `not_run = not_run_budget + baseline_unavailable + equivalent_exception`;
- mutation score SHALL be `null` when `killed + survived == 0`, otherwise
  `killed / (killed + survived)`.

Timeout, no-coverage, baseline-unavailable, invalid, infrastructure-error, exception,
and not-run outcomes SHALL NOT improve the score.

Core and full modes SHALL persist a content-addressed result cache. Its identity SHALL
include mutant ID, the digest of every tracked Python source file, mapped tests,
fixtures/conftest, `pyproject.toml`, `setup.cfg`, `uv.lock`, mutation configuration
and exceptions, normalized pytest arguments, Python/Cosmic Ray/coverage/pytest/plugin
versions, platform/architecture, controlled environment, UTC run date, selection
algorithm, and complete mutant cohort. Corrupt, partial, unbounded, unfrozen-clock, or
identity-mismatched entries SHALL be ignored. Changed mode MAY read a matching entry
but SHALL remain correct without it.

Core/full SHALL append one bounded `trend.jsonl` record keyed by mode, commit, cohort,
scope/config/environment digests, and report generation. CI SHALL restore the latest
compatible trend cache, reject malformed/incompatible records, append the current
record, and save it under a new run-specific cache key. Score regression is comparable
only over identical matched mutant IDs; different exclusions, budgets, partial runs,
or cohorts SHALL be non-comparable.

Raw subprocess output SHALL be capped at 64 KiB per mutant, but JSON SHALL retain at
most 8 KiB diagnostic and 8 KiB diff excerpts per detailed mutant. Mode-level detail
budgets SHALL be 1/8/32 MiB for changed/core/full. Space for one minimal structural
record per selected mutant SHALL be reserved before admission. Detail pressure SHALL
truncate marked excerpts deterministically or stop admission; it SHALL never omit a
selected mutant or prevent the 32/64/128 MiB JSON report from publishing.

#### Scenario: Global time expires after partial execution

- **WHEN** some mutants completed before the wall-clock budget
- **THEN** their outcomes remain in all reports
- **AND** unstarted selected mutants are marked `not_run_budget`
- **AND** report generation completes within a separate bounded shutdown allowance.

### Requirement: Mutation quality policy SHALL be gradual and auditable

This change SHALL keep all CI mutation jobs non-blocking. The controller SHALL still
evaluate a repository baseline when it is established and comparable:

- a changed-scope mutation-score decrease greater than five percentage points SHALL be
  reported as a policy regression;
- modified core business code SHALL display a 70% target;
- 80% SHALL be documented only as a future core target, not a current repository-wide
  gate;
- no-coverage mutants SHALL be reported separately.

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
- use `if: always()` to append a success or fallback failure summary and upload the
  run directory with 14-day retention even when mutation exits 2, is cancelled, or
  reaches its deadline;
- run core nightly and on manual request, with 1800-second script budget and
  non-blocking job semantics, a 35-minute outer timeout, an exact UTC cron, and a
  non-overlapping core concurrency group;
- run full only on weekly schedule or explicit manual request, with 5400-second
  script budget, non-blocking job semantics, a 100-minute outer timeout, an exact UTC
  cron, and a non-overlapping full concurrency group.

The ordinary pull-request workflow SHALL never invoke `core` or `full`. Documentation,
test-only, fixture, generated, and excluded changes SHALL not execute mutant tests.

#### Scenario: A documentation-only pull request runs CI

- **WHEN** the changed-mode plan sees no eligible source
- **THEN** the mutation job exits successfully after producing a zero-work report
- **AND** no mutation dependency or test worker is required beyond planning.
