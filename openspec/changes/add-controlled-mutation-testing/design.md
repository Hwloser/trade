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

Use Cosmic Ray 8.x as an optional development library and build a narrow
`trade_py.devtools.mutation_testing` controller. The controller imports named
operator classes through Cosmic Ray's public plugin interface, enumerates only the
closed allowlist, materializes one mutation in one worker-owned copy, and runs pytest
through the repository's bounded executor. This separates mutation semantics from
execution truth.

### Requirements and acceptance

Users are repository developers and CI. Acceptance requires:

- one stable `scripts/mutation-test changed|core|full` surface;
- exact changed/core/full target and test selection without full-suite fallback;
- 150/1000/5000 mutant limits and 600/1800/5400 second wall budgets;
- first-order allowlisted mutations only;
- maximum worker count `min(4, max(1, CPU//2))`;
- 10-second initial per-mutant timeout from the measured 1.77-second core baseline,
  dynamically recomputed per affected-test set and capped at 60 seconds for changed;
- full process-group termination and truthful status classification;
- atomic JSON/Markdown/HTML output with surviving/no-coverage details;
- precise expiring equivalent-mutant exceptions and a non-established initial
  baseline;
- non-blocking GitHub PR/nightly/weekly/manual routing;
- focused unit, CLI, config, timeout, report, and workflow tests;
- no production behavior or real-data mutation.

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
  report.py                                  typed results and atomic renderers
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
- `mutant_id` is SHA-256 over schema version, relative path, source digest, start/end
  position, operator name, occurrence, and config digest.
- one `MutantPlan` contains exactly one `MutationSpec`.
- source paths are repository-relative normalized POSIX paths and regular files.
- changed line ranges are parsed from NUL-safe Git path records and zero-context
  unified hunks against an explicit base SHA.
- ordered selection is stable and does not depend on filesystem enumeration order.
- a cached result is reusable only when source, test files, config, Python, Cosmic Ray,
  pytest, and runner versions match.
- reports are written to a temporary sibling and atomically replaced.
- mutation score excludes timeout, no-coverage, invalid, infrastructure, exception,
  and budget-not-run states.
- `generated` counts eligible candidates seen up to the candidate-scan ceiling;
  `selected` is bounded by mode; `executed` excludes no-coverage and not-run work.

Unavailable states are explicit: missing base, unsupported tool, invalid config,
baseline test failure, no eligible source, no test mapping, budget exhaustion, and
report write failure never become an empty successful mutation score.

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

- `0`: reports produced, including zero eligible work, ordinary survivors, policy
  advisory, timeout, or no-coverage outcomes;
- `1`: mutation policy regression only when an explicitly established baseline is
  configured for enforcement (not enabled in this change);
- `2`: invalid invocation/config/tool, baseline-test failure, source mutation failure,
  report failure, or other infrastructure error.

The report schema is `trade.mutation.report.v1`. CI consumes `summary.md` and uploads
all report files but does not infer truth from logs.

#### Dependencies

`[project.optional-dependencies].mutation` adds compatible bounded versions of
Cosmic Ray and coverage.py. Coverage is used to map tests to changed source when
available; the explicit module-to-test mapping remains authoritative and prevents a
full-suite fallback. Runtime dependency resolution is unchanged.

#### Backward compatibility

All existing commands remain unchanged. The optional group and script are additive.
No current CI check is replaced. Removing the workflow and optional group restores
the previous developer surface.

### Mutation model

#### Operator allowlist

The configuration stores exact Cosmic Ray plugin names, not broad families. The first
version includes:

- `core/AddNot`;
- `core/ReplaceAndWithOr`, `core/ReplaceOrWithAnd`;
- `core/ReplaceTrueWithFalse`, `core/ReplaceFalseWithTrue`;
- selected `ReplaceComparisonOperator_*` edges for `Eq/NotEq`, `Lt/LtE`,
  `Gt/GtE`, and `Is/IsNot`;
- selected arithmetic edges among Add/Sub, Mul/Div/FloorDiv, and Mod;
- no NumberReplacer, break/continue, zero-loop, variable, exception, decorator, or
  unary deletion operators.

The controller checks that every configured name exists and that the plugin version is
inside the supported range. An operator appearing in the installed package does not
become enabled automatically.

#### First-order proof

Enumeration creates one internal candidate for each operator position. Selection
serializes a single candidate per plan. The worker rejects `len(mutations) != 1`
immediately before mutation. Tests inspect plans and force a multi-mutation input to
prove fail-closed behavior.

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

Core mutation paths are individual files or tightly scoped globs in belief, decision,
factor groups/definitions, signals, and evaluation. Data access, DB, event runtime,
jobs, providers, Web, CLI, devtools, migrations, and Observatory are not initial core
targets.

The configuration maps source globs to explicit pytest files. The initial mapping is
grounded in imports and current tests:

- belief -> `tests/test_belief_engine.py`;
- decision causal/world-state/action/explanation -> corresponding decision tests;
- factor groups/definitions -> `tests/test_factor_groups.py` and relevant feature
  tests;
- signals window scorer -> `tests/test_window_scorer.py`;
- evaluation trust/gate -> `tests/test_trust_layer.py` and recommendations tests.

During implementation, a static import collector validates that mapped tests import
or transitively exercise the module. Dynamic coverage can narrow tests after a passing
baseline, but cannot add unconfigured integration suites. No mapping yields
no-coverage.

The `full` mode means all *eligible configured Python production roots*, not the
repository, C++ engine, frontend, or every `trade_py` file. Its larger source map may
be extended only through reviewed config and tests.

### Failure and recovery

Each mutation worker:

1. creates a private temporary directory under the configured ignored state root;
2. copies the target package file set needed for imports, using regular-file,
   no-follow checks;
3. applies one Cosmic Ray mutation to the copy;
4. launches `python -m pytest --maxfail=1 -q <mapped tests>` with the copy first on
   `PYTHONPATH`, new session/process group, bytecode disabled, bounded stdout/stderr,
   and a temporary data/home/cache root;
5. monitors the per-mutant and global monotonic deadlines;
6. sends TERM to the process group, waits a short grace, sends KILL, and reaps;
7. removes the private source copy after retaining bounded diagnostics and diff.

Worker pool ownership remains in one parent process. It stops admission on global
deadline or interrupt, cancels queued futures, kills active groups, writes the partial
report, then exits. Signal handling is idempotent.

Baseline tests run once per distinct test tuple before mutants. A failed baseline
marks that tuple unavailable and no mutants using it execute. This prevents existing
test failures from being counted as kills.

Cosmic Ray's own `run_tests()` is not used because it maps timeout to killed. The
controller calls only plugin lookup and source mutation APIs.

### Performance and capacity

Mode constants:

| Limit | changed | core | full |
|---|---:|---:|---:|
| selected mutants | 150 | 1000 | 5000 |
| candidate scan | 3000 | 20000 | 100000 |
| wall clock | 600s | 1800s | 5400s |
| workers | computed max 4 | computed max 4 | computed max 4 |
| output per mutant | 64 KiB | 64 KiB | 64 KiB |
| total JSON report | 32 MiB | 64 MiB | 128 MiB |
| shutdown/report grace | 30s | 60s | 120s |

The 97-test initial core set takes 1.77 seconds, yielding a 10-second initial timeout.
Each distinct affected-test tuple is timed independently. The changed cap remains 60
seconds even when a selected test baseline would imply more; such a tuple is reported
as incompatible with PR mode rather than silently increasing the job.

At 10x source size, candidate-scan ceilings stop AST work and report truncation.
At 10x test cost, baseline timing excludes incompatible tuples before scheduling or
the wall deadline terminates admission. At 10x output, bounded ring capture truncates
diagnostics without blocking pipes.

### Observability and operations

Human output prints mode, base/head, eligible files, selected tests, budgets, progress,
outcome counts, score, and report paths. JSON is authoritative. Markdown contains the
exact PR summary fields requested by the user. HTML is static and self-contained.

Survivors and no-coverage entries display source, line, definition, operator,
mutation diff, related tests, and duration. Timeout and infrastructure error remain
separate. Budget exhaustion records `mutant_limit`, `candidate_scan_limit`,
`wall_clock`, `signal`, or `report_limit`.

Core/full trend files are CI artifacts, not committed moving state. The repository
baseline is changed only by explicit review. Cache hit counts and cache key versions
are visible.

### Baseline and exceptions

`config/mutation-baseline.json` starts with `established: false`; it defines schema,
scope identity, and the future five-point regression rule without fabricating a
score. A future command can propose a baseline artifact, but committing it requires
review.

`config/mutation-exceptions.toml` starts empty. Each future record must match exact
path, line, operator, source digest, owner, reason, and expiry. Validation rejects
wildcards and stale source. Excepted mutants have `equivalent_exception` status and
remain visible; they do not enter score numerator or denominator.

### CI design

One GitHub Actions workflow has three jobs/routes:

- `pull_request`: checkout full enough history, set up Python/uv, run
  `scripts/mutation-test changed --base $BASE_SHA`, append summary, upload artifact;
  job-level timeout is 12 minutes to allow bounded setup/report grace and
  `continue-on-error: true`.
- nightly/manual core: `core`, 32-minute job timeout, cache keyed by lock/config/source
  hashes, artifact/trend upload, `continue-on-error: true`.
- weekly/manual full: explicit `workflow_dispatch` mode or weekly cron, 92-minute job
  timeout, artifact/cache upload, `continue-on-error: true`.

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
- baseline timing/failed-baseline behavior;
- killed/survived/timeout/no-coverage/invalid/infrastructure classification;
- descendant process termination on timeout and global deadline;
- worker bound and admission stop;
- atomic partial JSON/Markdown/HTML reports and output bounds;
- cache identity and stale invalidation;
- unestablished/comparable baseline policy;
- script forwarding and missing dependency diagnostics;
- workflow event/mode/timeout/non-blocking/report assertions.

Validation commands include focused pytest, a real small changed run, core plan or
bounded small run, TOML/JSON/YAML/static workflow checks, `bash -n`, ShellCheck when
available, Ruff, BasedPyright, compileall, `./trade dev check`, original stable core
tests, full pytest with the pre-existing failure reported, and `git diff --check`.

### Runtime concurrency evidence

Ownership is one controller process and a bounded worker pool. Ordering is determined
before admission; completion order never changes report ordering. Workers share no
mutable source tree. Atomicity is per report replace and per cache entry. Timeout and
cancellation own full process groups. Backpressure is the bounded pool plus finite
queue. Partial failure is aggregated by status without converting infrastructure
errors to kills. Capacity tests exercise worker, candidate, time, output, and report
ceilings.

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
