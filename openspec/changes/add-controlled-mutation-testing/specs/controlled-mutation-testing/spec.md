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
before mutation enumeration. Candidate enumeration SHALL have its own finite scan
ceiling so a mutant execution limit cannot hide unbounded planning work.

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

The initial operator allowlist SHALL contain only supported Cosmic Ray operators that
implement:

- conditional negation;
- boolean `and`/`or` replacement;
- boolean return replacement (`True`/`False`);
- relational equality, inequality, and boundary replacement;
- arithmetic replacement among `+`, `-`, `*`, `/`, `//`, and `%`;
- null-related identity replacement between `is` and `is not`.

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

Every target module SHALL resolve to a finite explicit list of affected unit-test
files. The controller SHALL run only those tests for each mutant. It SHALL NOT fall
back to the full pytest suite. An unmapped or empty test selection SHALL produce a
`no_coverage` mutant without running pytest.

#### Scenario: Changed file has no trusted unit-test mapping

- **WHEN** a changed eligible source file has mutation candidates but no configured
  affected unit-test mapping
- **THEN** candidates are reported as `no_coverage`
- **AND** they are not counted as killed
- **AND** the report names the source, line, operator, and missing mapping.

### Requirement: Test execution SHALL be isolated and truthfully classified

The controller SHALL never mutate the developer's source file in place. Each worker
SHALL operate on its own temporary source tree or file overlay and SHALL execute
pytest in a new process session/group. The process environment SHALL disable bytecode
writes and SHALL not point at real production data roots.

The baseline time `T` for a selected affected-test set SHALL be measured before mutant
execution. The per-mutant timeout SHALL be `max(10 seconds, 2.5 * T)`. In `changed`
mode it SHALL be capped at 60 seconds. The configured mode wall-clock remains a
separate outer deadline.

On timeout, cancellation, or global deadline, the controller SHALL terminate the
entire process group, wait a bounded grace period, escalate to kill, and reap the
process before returning. A timeout SHALL be `timeout`, not `killed`. Failure to
launch, copy, mutate, parse, collect, or execute infrastructure SHALL be
`infrastructure_error`, not `killed`.

The classification SHALL be:

- baseline failure: abort that target test set and report infrastructure/baseline
  failure;
- pytest exits zero under the mutant: `survived`;
- pytest reports a test failure under a valid mutant: `killed`;
- deadline expires: `timeout`;
- no trusted tests cover the target: `no_coverage`;
- the operator cannot produce valid changed source: `invalid`;
- controller/tool/process failure: `infrastructure_error`.

#### Scenario: Mutant causes an infinite loop with descendants

- **WHEN** the selected test exceeds its computed timeout after creating child
  processes
- **THEN** the controller terminates and reaps the full process group
- **AND** records one timeout mutant
- **AND** no child remains alive after the worker completes.

### Requirement: Reports SHALL be complete, machine-readable, and resumable

Every invocation SHALL atomically write:

- `report.json` using schema `trade.mutation.report.v1`;
- `summary.md` suitable for GitHub job summaries;
- `index.html` with summary counts and surviving/no-coverage details.

The report SHALL include mode, source/base/head identities, tool versions, config
digest, deterministic selection strategy, fixed seed or `null`, budgets, worker
limit, baseline timings, generated/selected/executed counts, killed, survived,
timeout, no-coverage, invalid, infrastructure-error counts, mutation score,
budget-exhausted state/reason, elapsed time, and every selected mutant's path,
line/column, definition, operator, related tests, status, duration, and bounded
diagnostic.

Mutation score SHALL be `killed / (killed + survived)`. Timeout, no-coverage, invalid,
and infrastructure-error outcomes SHALL be reported separately and SHALL NOT improve
the score.

Core and full modes SHALL persist a content-addressed result cache keyed by source
content, operator/location, selected tests, test content, tool/config version, and
Python version. Reuse SHALL be explicit in the report. Core reports SHALL include a
trend record and surviving-mutant detail.

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

Equivalent or meaningless mutants MAY be registered only with exact source path,
line, operator, source digest, owner, reason, and review expiry. Directory, filename,
definition, or operator wildcards SHALL be rejected. Expired, stale-digest, missing,
or ambiguous exceptions SHALL fail configuration validation. Exceptions SHALL remain
visible in reports and SHALL not be counted as killed.

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
- cap the changed job at a duration compatible with the 600-second script budget;
- upload JSON, Markdown, and HTML reports and append the Markdown summary;
- run core nightly and on manual request, with 1800-second script budget and
  non-blocking job semantics;
- run full only on weekly schedule or explicit manual request, with 5400-second
  script budget and non-blocking job semantics.

The ordinary pull-request workflow SHALL never invoke `core` or `full`. Documentation,
test-only, fixture, generated, and excluded changes SHALL not execute mutant tests.

#### Scenario: A documentation-only pull request runs CI

- **WHEN** the changed-mode plan sees no eligible source
- **THEN** the mutation job exits successfully after producing a zero-work report
- **AND** no mutation dependency or test worker is required beyond planning.
