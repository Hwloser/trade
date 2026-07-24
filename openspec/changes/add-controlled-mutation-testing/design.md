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
operator classes with `cosmic_ray.plugins.get_operator`, enumerates only the closed
allowlist through each operator's `mutation_positions`, transforms source text only
with `cosmic_ray.mutating.mutate_code`, materializes one mutation in one worker-owned
copy, and runs pytest through a mutation-owned bounded executor. It never calls
Cosmic Ray's in-place or test-running helpers. This separates mutation semantics from
execution truth.

### Requirements and acceptance

Users are repository developers and CI. Acceptance requires:

- one stable `scripts/mutation-test changed|core|full` surface;
- exact changed/core/full target and test selection without full-suite fallback;
- 150/1000/5000 mutant limits and 600/1800/5400 second wall budgets;
- first-order allowlisted mutations only;
- maximum worker count `min(4, max(1, effective CPU//2))`;
- 10-second initial per-mutant timeout from the measured 1.77-second core baseline,
  dynamically recomputed per affected-test set and capped at 60 seconds for changed;
- parent-owned process-group termination and truthful status/count algebra;
- baseline line coverage and private-overlay provenance before mutant execution;
- generation-atomic JSON/Markdown/HTML output with surviving/no-coverage details;
- precise expiring equivalent-mutant exceptions and a non-established initial
  baseline;
- non-blocking GitHub PR/nightly/weekly/manual routing;
- focused unit, CLI, config, timeout, report, and workflow tests;
- no production behavior, network access, credentials, or real-data reads/writes.

### Ownership and boundaries

```text
scripts/mutation-test                         thin executable facade
config/mutation-testing.toml                 policy and source/test ownership
config/mutation-baseline.json                reviewed score history seed
config/mutation-exceptions.toml              precise reviewed exceptions
trade_py/devtools/mutation_testing/
  cli.py                                     argument parsing and exit contract
  config.py                                  closed configuration parser
  git_scope.py                               bounded base/diff/changed-line discovery
  selection.py                               deterministic targets/tests/mutants
  engine.py                                  Cosmic Ray operator adapter
  executor.py                                process groups, deadlines, cancellation
  report.py                                  typed results and generation publication
  baseline.py                                comparable baseline evaluation
.github/workflows/mutation-testing.yml       CI routing only
docs/mutation-testing.md                     developer operations
```

The shell facade performs no parsing beyond locating the repository and invoking the
locked optional environment. The CLI coordinates owned modules but does not contain
mutation logic. Cosmic Ray is an adapter dependency and never imports application
runtime modules. Tests use synthetic repositories and source files.

The controller is a devtool, not a domain service. It does not belong in the future
business Context graph and is excluded as a mutation target.

### Data and state invariants

The durable artifacts are developer reports and cache entries, never business state.
Identifiers and invariants:

- `run_id` is a UUID generated per invocation.
- `mutant_id` is SHA-256 over schema version, relative path, original source digest,
  start/end position, operator name/arguments, occurrence, and config digest.
- one `MutantPlan` contains exactly one `MutationSpec`.
- source paths are repository-relative normalized POSIX paths and regular files.
- changed line ranges are parsed from NUL-safe Git records and zero-context hunks for
  merge-base-to-HEAD plus staged, unstaged, and untracked state. Rename destinations
  are eligible, deletion-only and rename-only-without-content-change are not, and the
  dirty-tree digest is part of the run identity.
- ordered selection is stable and does not depend on filesystem enumeration order.
- a cached result is reusable only when the complete tracked Python tree, mapped
  tests/fixtures, lock and pytest configuration, operator/config/exception policy,
  normalized command/environment, platform, UTC run date, complete mutant cohort,
  and tool versions match.
- one run directory is staged and fsynced; one `current.json` pointer atomically
  publishes JSON, Markdown, HTML, and their digest manifest as one generation.
- mutation score excludes timeout, no-coverage, baseline-unavailable, invalid,
  infrastructure, exception, and budget-not-run states.
- `generated` counts eligible candidates seen up to the candidate-scan ceiling;
  `selected` is bounded by mode; `mutation_applied` requires a verified source
  transformation; `test_started`/`executed` requires a started mutant pytest.
- every selected mutant has exactly one terminal state and score is `null` when no
  killed or survived mutant exists.

Unavailable states are explicit: missing base, unsupported tool, invalid config,
baseline test failure, no eligible source, no test mapping, budget exhaustion, and
report write failure never become an empty successful mutation score.

### Persistent-write safety

The controller is the only writer of mutation-testing developer state. Each
invocation owns one UUID `run_id` staging directory; an exclusive, bounded
output-root publication lock serializes only current-pointer and core/full trend
updates. Cache records are content-addressed by the complete comparability identity
and mutant ID. Same-key writers use exclusive creation or a per-key lock, and a
losing writer accepts the predecessor only after schema, identity, size, and digest
verification. These writes never target runtime databases, repository `data/`,
Parquet, provider state, model artifacts, or production configuration.

Before publication, the controller validates the report schema and count equations,
one terminal state for every selected mutant, all configured size ceilings, and the
SHA-256 and byte size of JSON, Markdown, and HTML. It writes those files and a
manifest below the private staging directory, flushes and fsyncs each file and the
directory, renames the directory to its immutable final run path, then atomically
replaces and directory-syncs `current.json`. Readers resolve the pointer once,
validate its run ID, manifest, and member digests, and read only that generation.
They never compose output from separate runs. Cache records and the bounded trend
file use the same write, fsync, atomic-replace, directory-fsync discipline.

A crash before pointer replacement leaves the previous complete generation current;
a crash after replacement exposes only the already verified new generation.
Unreferenced staging is not successful evidence and is removed on a later invocation
only after proving it is not current. Malformed, oversized, symlinked,
path-escaping, identity-mismatched, or hash-mismatched report state fails closed and
is preserved for diagnosis. Corrupt cache is a miss and is recomputed. Corrupt trend
state is quarantined and replaced by a fresh bounded trend carrying an explicit
recovery record; it is never silently interpreted as historical truth.

Deadline, cancellation, worker, or baseline degradation still produces a marked
partial generation when the report store is available, with every admitted mutant
classified and the remaining selected mutants marked `budget_not_run`. Publication
failure returns infrastructure exit 2 and leaves staged evidence rather than
exposing a mixed generation. The manifest audits run ID, UTC time, mode, source/base
identity, complete config/tool/environment/cohort digests, budgets, stop reason,
status, member hashes, and cache/trend decisions without credentials or raw
environment values. No backup is required for ignored disposable developer state;
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
error, never a full-scope fallback.

Exit codes:

- `0`: `complete`, `zero_work`, or `budget_partial` reports produced without
  infrastructure failures, including ordinary survivors, timeout, or no coverage;
- `1`: mutation policy regression only when an explicitly established baseline is
  configured for enforcement (not enabled in this change);
- `2`: `preflight_failed`, `degraded_infrastructure`, `signal_cancelled`, or
  `report_failed`, including invalid invocation/config/tool, baseline failure, source
  drift, provenance failure, launch failure, or report failure. SIGINT/SIGTERM are
  normalized to exit 2 after descendant cleanup and partial report publication.

Infrastructure failure in one mutant does not turn another mutant into killed. The
controller stops only the affected test tuple unless integrity, process ownership,
global deadline, signal, or report publication requires run-wide cancellation.

The report schema is `trade.mutation.report.v1`. CI consumes `summary.md` and uploads
all report files but does not infer truth from logs.

#### Dependencies

`[project.optional-dependencies].mutation` pins `cosmic-ray==8.4.6` plus a bounded
coverage.py version. Coverage line data is mandatory before mutant execution; the
explicit module-to-test matrix remains authoritative and prevents a full-suite
fallback. Runtime dependency resolution is unchanged.

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

Enumeration walks the same Parso tree order used by Cosmic Ray and records one
internal candidate for each `mutation_positions()` result. Selection serializes one
candidate per plan. Immediately before mutation, the adapter re-hashes/re-enumerates,
requires occurrence-to-span equality, calls `mutate_code`, parses the output, and
requires the diff to intersect only that planned location. Source drift is
infrastructure failure; unsupported output is invalid. The worker also rejects
`len(mutations) != 1`.

#### Deterministic priority

1. exact changed line;
2. enclosing definition containing a changed line;
3. configured core path priority;
4. configured defect-history priority;
5. remaining eligible changed file;
6. relative path, line, column, operator priority, occurrence.

Changed mode never reaches priority categories in an unchanged file. Core/full use
categories 3-6. Truncation records the last admitted key and omitted candidate count.

### Target and test selection

The exact v1 matrix is:

| Scope | Source | Tests |
|---|---|---|
| core | `trade_py/decision/action.py` | `tests/test_decision_action.py` |
| core | `trade_py/decision/world_state.py` | `tests/test_world_state.py`, `tests/test_decision_action.py`, `tests/test_explanation.py` |
| core | `trade_py/decision/scenario.py` | `tests/test_explanation.py` |
| core | `trade_py/decision/explanation.py` | `tests/test_explanation.py` |
| core | `trade_py/trust/compute.py` | `tests/test_trust_layer.py` |
| full | `trade_py/factors/groups/event_features.py` | `tests/test_factor_groups.py` |
| full | `trade_py/signals/window_scorer.py` | `tests/test_window_scorer.py` |

Changed uses changed lines only inside this matrix; core uses core rows; full uses all
rows. Thus full is the complete configured v1 scope, not the whole repository. DTOs,
DB/data access, event/jobs/providers, Web/CLI/devtools, migrations, generated/vendor,
logging/startup, C++, React, and unlisted Python modules are out of scope by absence
from the closed matrix, not by score-oriented broad wildcards. Observatory's current
catalog failure does not create a blanket exclusion; independently stable
Observatory/PIT files require a later reviewed matrix addition.

Static imports only validate mapping shape. A passing private-tree baseline must
collect coverage.py line data. A candidate whose exact line is absent becomes
`no_coverage_line`; a missing mapping becomes `no_coverage_mapping`. Neither starts
mutant pytest. Coverage never adds an unconfigured test or falls back to all tests.

### Failure and recovery

The command-entry monotonic deadline owns planning, baselines, execution, cancellation,
and publication. A controller-owned thread pool is used so the parent process directly
creates every `Popen` and synchronously registers its PGID before monitoring. Worker
exceptions, lost pipes, repeated signals, and parent cancellation all use the same
TERM, bounded wait, KILL, group-existence check, and reap path.

Each thread creates one size/free-space-bounded private tree and reuses it across
mutants. Before every work item it restores and hashes the target, revalidates the
planned source identity/span, calls only `get_operator` plus `mutate_code`, writes the
result to the private tree, and restores/hash-checks afterward. This avoids
O(mutants x source-tree) copy I/O while preserving isolation.

Pytest runs from the private root with absolute mapped tests, importlib mode, checkout
removed from import paths, and a provenance plugin verifying module `__file__` and
digest. The environment sets temporary `TRADE_DATA_ROOT`, HOME/XDG/cache directories,
UTC/locale/hash seed, scrubs credentials/proxies, denies sockets, and guards reads
under real data/model/DB/parquet/artifact roots. Baselines use the same controls plus
line coverage and a finite timeout bounded by remaining global time.

Pytest mutant exits map `0=survived`, `1=killed`, and `2/3/4/5` or signal to
infrastructure error. Nonzero baseline means `baseline_unavailable`. Timeout is always
timeout. A failed tuple does not execute its mutants and cannot create kills.

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
| one source/tree | 1 MiB / 100000 nodes / depth 256 | same | same |
| dependency graph | 512 modules / 4096 edges / depth 32 | same | same |
| workers | computed max 4 | computed max 4 | computed max 4 |
| output per mutant | 64 KiB | 64 KiB | 64 KiB |
| retained diagnostic + diff | 8 KiB + 8 KiB | 8 KiB + 8 KiB | 8 KiB + 8 KiB |
| aggregate retained detail | 1 MiB | 8 MiB | 32 MiB |
| total JSON report | 32 MiB | 64 MiB | 128 MiB |
| shutdown/report grace | 30s | 60s | 120s |

The audited stable core set took 1.77 seconds, yielding a 10-second initial timeout.
Each distinct affected-test tuple is timed independently. The changed cap remains 60
seconds even when a selected test baseline would imply more; such a tuple is reported
as incompatible with PR mode rather than silently increasing the job.

Source file/byte/tree/dependency limits plus deadline checkpoints bound operator-sparse
trees where candidate count alone would not stop parsing. Effective CPU is the minimum
of affinity, cgroup quota, and host count. One private tree per worker bounds copy I/O.
Structural report capacity is reserved for every selected mutant; detail is truncated
deterministically before aggregate report limits.

At 10x source size, source and AST ceilings stop work before candidate truncation.
At 10x test cost, baseline timing excludes incompatible tuples before scheduling or
the wall deadline terminates admission. At 10x output, bounded ring capture truncates
diagnostics without blocking pipes.

### Observability and operations

Human output prints mode, base/head, eligible files, selected tests, budgets, progress,
outcome counts, score, and report paths. JSON is authoritative. Markdown contains the
exact PR summary fields requested by the user. HTML is static and self-contained.

Survivors and no-coverage entries display mutant ID, original/mutated digest,
planned/observed span, source, definition, operator, bounded diff, tests, coverage,
duration, and cache identity. Timeout and infrastructure error remain separate.
Run status is one of `complete`, `zero_work`, `budget_partial`, `signal_cancelled`,
`degraded_infrastructure`, `preflight_failed`, or `report_failed`. Budget exhaustion
records `mutant_limit`, `candidate_scan_limit`, `source_limit`, `wall_clock`,
`signal`, or `report_limit`.

Core/full append one bounded trend record keyed by mode, commit, exact mutant cohort,
scope/config/environment/tool digests, and run ID. CI restores only a compatible
cache, rejects corrupt records, and saves a new run-specific cache. The repository
baseline changes only by explicit review. Comparisons use identical matched mutant
IDs; partial or different cohorts are non-comparable. Cache hit/miss reasons and
identity versions are visible.

### Baseline and exceptions

`config/mutation-baseline.json` starts with `established: false`; it defines schema,
scope identity, and the future five-point regression rule without fabricating a
score. A future command can propose a baseline artifact, but committing it requires
review.

`config/mutation-exceptions.toml` starts empty. Each future record must match exact
mutant ID, path, source digest, start/end line and column, operator, occurrence,
owner, reason, and expiry. Validation rejects every wildcard, ambiguity, and stale
source. Excepted mutants have `equivalent_exception` status and remain visible; they
do not enter score numerator or denominator.

### CI design

One GitHub Actions workflow has three jobs/routes:

- `pull_request`: checkout full enough history, set up Python/uv, run
  `scripts/mutation-test changed --base $BASE_SHA`; job-level timeout is 15 minutes
  around the 10-minute total controller budget.
- nightly/manual core: `core`, exact UTC cron, non-overlapping concurrency group,
  35-minute timeout, compatible trend/result cache, `continue-on-error: true`.
- weekly/manual full: explicit `workflow_dispatch` mode or weekly cron, 100-minute
  job timeout, exact UTC cron, and a non-overlapping full concurrency group.

Every summary and artifact step uses `if: always()`, 14-day retention, stable run
paths, explicit missing-file behavior, and a fallback failure summary so exit 2,
cancellation, or timeout cannot silently erase evidence.

The script always plans changed scope before installing/running the heavy mutation
engine. If no eligible files exist, it produces a zero-work report and exits. GitHub
path filters are an optimization only; script selection remains authoritative.

### Validation strategy

Tests cover:

- closed config and operator validation;
- changed Git rename/add/delete/untracked and line hunks;
- no eligible source zero-work;
- exact-line/definition/core deterministic priority and limits;
- first-order rejection;
- exclusions and no broad exception patterns;
- test mapping and no full-suite fallback;
- exact-line baseline coverage and mapped-but-uncovered behavior;
- private-overlay import/digest provenance and original source hashes;
- temporary data roots, credential/proxy scrubbing, socket denial, and real-data
  filesystem guards;
- API/version/operator compatibility, occurrence/span handshake, and forbidden
  in-place Cosmic Ray APIs;
- baseline timing/failed-baseline behavior and exact pytest exit mapping;
- killed/survived/timeout/no-coverage/baseline-unavailable/invalid/infrastructure
  classification and count equations;
- parent-owned descendant process termination on timeout, worker failure, signal, and
  global deadline;
- worker bound and admission stop;
- run-generation atomic partial JSON/Markdown/HTML publication and output bounds;
- complete cache/cohort identity, stale/corrupt invalidation, and trend recovery;
- unestablished/comparable baseline policy;
- script forwarding and missing dependency diagnostics;
- workflow event/mode/timeout/concurrency/non-blocking/always-upload assertions.

Validation commands include focused pytest, a real small changed run, core plan or
bounded small run, TOML/JSON/YAML/static workflow checks, `bash -n`, ShellCheck when
available, Ruff, BasedPyright, compileall, `./trade dev check`, original stable core
tests, full pytest with the pre-existing failure reported, and `git diff --check`.

### Runtime concurrency evidence

Ownership is one controller process, a bounded thread pool, one synchronized active
PGID registry, and one invocation-wide deadline. Ordering is determined before
admission; completion order never changes report ordering. Each thread owns one
private tree and restores the target between work items. Atomicity is one published
run generation plus individual verified cache/journal records. Timeout and
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
dependencies. Cache/report directories are ignored and disposable. No production data
or behavior rollback exists.
