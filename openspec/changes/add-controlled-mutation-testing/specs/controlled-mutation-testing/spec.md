# Controlled Mutation Testing Specification

## ADDED Requirements

### Requirement: Mutation execution SHALL be bounded by mode

The repository SHALL expose canonical `trade dev mutation changed|core|full`
commands. `scripts/mutation-test changed|core|full` SHALL be an argv/exit-compatible
facade into the same supervisor-first process tree. Each mode SHALL use an explicit
source selection, mutant limit, worker limit, per-mutant timeout, and wall-clock
budget. No mode SHALL silently expand to a broader source or test scope.

The normative budgets SHALL be:

| Mode | Source scope | Generated/executed mutant limit | Candidate-position scan limit | Wall clock | CI use |
|---|---|---:|---:|---:|---|
| `changed` | eligible changed production lines in configured definitions relative to an explicit target branch | 150 | 3000 | 600 seconds | pull request |
| `core` | configured Python core business definitions | 1000 | 20000 | 1800 seconds | nightly/manual |
| `full` | all configured eligible Python production definitions | 5000 | 100000 | 5400 seconds | weekly/manual only |

`candidate_positions_scanned` SHALL count eligible positions examined before
selection. `generated_mutants` SHALL count materialized selected first-order mutant
records, SHALL equal `selected`, and SHALL NOT exceed 150/1000/5000. Scanned but
unselected positions and the unknown unscanned remainder SHALL be reported separately
and SHALL NOT be labelled generated mutants.

The worker count SHALL be `min(4, max(1, floor(available_cpu/2)))` unless an explicit
lower value is supplied. An explicit value above the computed maximum SHALL fail
before mutation enumeration. Available CPU SHALL use the minimum of process affinity,
cgroup quota, and host CPU count when those values are available. Every baseline and
mutant process SHALL set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` to `1`; those controls SHALL enter environment/cache
identity.

Worker admission SHALL require a writable delegated cgroup v2 subtree with finite
`memory.max` and the `memory`, `pids`, and `cpu` controllers. Cgroup v1, shared or
read-only cgroups, missing delegation, and unbounded memory SHALL permit
`--plan-only` only and SHALL fail execution preflight. Baseline execution SHALL
measure each selected test tuple's descendant `memory.peak`. Each worker cgroup SHALL
set and read back `memory.max = max(256 MiB, ceil(1.5 * measured_peak))`,
`memory.high = floor(0.9 * memory.max)`, `memory.swap.max = 0`, and a reviewed
thread-only `pids.max`. Fixed reservations SHALL be
`controller_reserve = max(256 MiB, 1.25 * measured_controller_peak)`,
`safety_reserve = max(256 MiB, 0.10 * effective_memory)`, and the mode's
64/128/256 MiB renderer reserve. Admission SHALL require
`controller_reserve + safety_reserve + renderer_reserve +
sum(active worker memory.max) <= effective_memory`; the lower CPU/memory bound
SHALL win and every setting, read-back, input, and reservation SHALL be reported.
A finite limit that cannot admit one worker SHALL fail preflight.

The wall clock SHALL start in a Python 3.7-compatible standard-library-only supervisor
before bootstrap, dependency verification, or any `uv` child. The root `trade`
dispatch and facade SHALL use Bash built-ins to capture command-entry `/proc/uptime`,
resolve an absolute interpreter from an explicit `MUTATION_SUPERVISOR_PYTHON` or
`command -v python3`, reject a non-absolute path or Python below 3.7, and `exec`
`python -I` without starting `uv`. The supervisor and its shared bootstrap contract
SHALL use no project imports, third-party modules, dataclasses, or Python 3.8+ syntax.
At entry the supervisor SHALL convert that anchor once to `CLOCK_BOOTTIME`, record the
clock ID and kernel boot ID, and use only that clock domain for event order, elapsed
time, deadlines, leases, and timeout decisions. UTC SHALL be display/retention
metadata only; a boot-ID change SHALL invalidate a lease rather than compare monotonic
values across boots. The supervisor SHALL create one run ID, atomic
`trade.mutation.invocation.v1` receipt, child-subreaper/watchdog, and absolute
`CLOCK_BOOTTIME` deadline passed unchanged through bootstrap and the full controller. It
SHALL parent every `uv` handoff in an owned session and include
dependency preparation/verification, Git discovery, source parsing, coverage
baselines, mutation execution, cancellation, and report publication. CI SHALL prepare
the frozen mutation environment before starting this bounded invocation; local
missing dependencies SHALL fail with a preparation command rather than sync outside
the budget. The controller SHALL reserve 30/60/120 seconds of the changed/core/full wall clock for
cleanup and report publication inside the controller budget. The reserves SHALL split
into TERM, KILL/observable-reap, render/hash/fsync, and pointer-or-fallback sub-budgets of
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

The supervisor SHALL use pidfds where available, cgroup kill, process-group
TERM/KILL, bounded observable-reap polling, and its sole fallback writer.
A task still present after its cleanup sub-budget SHALL receive orthogonal cleanup
status `unconfirmed`. Its provisional terminal SHALL be replaced by
`infrastructure_error_test_started` when a test started or
`infrastructure_error_pre_test` otherwise; the run SHALL be
`degraded_infrastructure`, score/cache SHALL be disabled for the affected record and
run, and no provisional killed/survived/timeout/cancelled count SHALL remain. The
receipt SHALL retain pidfd/cgroup evidence while the supervisor continues cleanup
attempts until the outer CI timeout. An unreturning filesystem operation SHALL create no claimed
generation. CI timeouts of 15/35/100 minutes SHALL be the final containment for
uninterruptible kernel/filesystem stalls and SHALL exceed the 600/1800/5400-second
controller budgets. Capacity qualification SHALL fail if ordinary cancellation,
reap, render, hash, fsync, or fallback cannot complete within its sub-budget.

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
independent occurrence counter per `(source digest, operator name, operator
arguments)`. This SHALL match Cosmic Ray's new `MutationVisitor` and zero-based
counter for each operator; no occurrence counter SHALL be shared across operators.
That source-wide value SHALL be named `execution_occurrence` and SHALL be the only
occurrence passed to `mutate_code`. A separate `definition_occurrence` SHALL count
the same operator/arguments inside the unique nearest named definition and SHALL be
used only for comparison identity. Nested named functions/classes SHALL belong to
their exact nested qualified name, not a configured lexical parent; lambdas SHALL
belong to their nearest named definition. Duplicate qualified ownership, a missing
parent, or an ambiguous canonical token position SHALL fail validation.
Startup compatibility tests SHALL compare that occurrence/span sequence with Cosmic
Ray `MutationVisitor` for every allowlisted operator and nested multi-operator syntax.
A mismatch SHALL fail preflight.

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

`changed` SHALL admit only candidates whose canonical Cosmic Ray mutation-token
position lies on an exact new-side changed line. A multiline AST span intersecting a
changed line SHALL NOT admit a token on an unchanged line. It SHALL NOT widen to an
enclosing definition or mutate unchanged lines/files. `core` SHALL use only configured
core domain modules. `full` SHALL use only the configured eligible Python production
roots and SHALL still apply every exclusion.

Planning SHALL seal `GitScopeSnapshot` with explicit base OID, merge-base OID, HEAD
OID, index-entry digest, staged-patch digest, tracked worktree content/mode digest,
and exact diff/hunk digest. That snapshot SHALL enter run identity and SHALL be
revalidated before private-tree copy and before report generation publication. Drift
SHALL stop admission, keep staged cache unreadable, and produce typed source-drift
infrastructure evidence rather than publishing for a different checkout.

The v1 eligible definition-to-test matrix SHALL be closed:

| Scope | Source | Eligible qualified definitions | Affected unit tests | Clock policy | Entropy policy |
|---|---|---|---|---|---|
| core | `trade_py/decision/action.py` | `_confidence_label`, `derive_action_decision` | `tests/test_decision_action.py`, `tests/test_explanation.py` | explicit_input | deterministic |
| core | `trade_py/decision/world_state.py` | `infer_market_regime`, `infer_event_regime`, `infer_sentiment_regime`, `infer_technical_regime`, `infer_liquidity_regime`, `infer_uncertainty`, `_build_state_summary`, `build_world_state` | `tests/test_world_state.py`, `tests/test_decision_action.py`, `tests/test_explanation.py` | explicit_input | deterministic |
| core | `trade_py/decision/scenario.py` | `_count_bullish`, `_count_bearish`, `build_scenario_summary` | `tests/test_explanation.py` | explicit_input | deterministic |
| core | `trade_py/decision/explanation.py` | `build_explanation` | `tests/test_explanation.py` | explicit_input | deterministic |
| core | `trade_py/trust/compute.py` | `_freshness_score`, `_trust_level`, `compute_prediction_trust`, `compute_portfolio_trust` | `tests/test_trust_layer.py` | explicit_input | entropy_non_cacheable |
| full | `trade_py/factors/groups/event_features.py` | `build_event_group` | `tests/test_factor_groups.py` | explicit_input | deterministic |
| full | `trade_py/signals/window_scorer.py` | `_score_large_order` | `tests/test_window_scorer.py` | explicit_input | deterministic |

The configuration SHALL identify exact qualified definitions and the run report SHALL
bind their source/definition digests. Missing, duplicate, nested-ambiguous, or moved
definitions SHALL fail validation. Each row SHALL use a closed clock policy
`explicit_input|non_cacheable` and entropy policy
`deterministic|entropy_non_cacheable`. V1 SHALL reject `fixed` because no clock or
entropy injection contract is implemented. The bounded import closure SHALL be
checked for undeclared `datetime/date/time`, UUID, random, secrets, or OS entropy
access, and an undeclared access SHALL disable cache reuse. `changed` SHALL consider
changed lines only in this complete matrix. `core` SHALL use the rows marked core. `full` SHALL use every row;
full means the complete configured eligible matrix, not all Python files. Adding a
row or changing its scope, source, definition, tests, clock policy, entropy policy,
or dependency manifest SHALL require an explicit configuration, snapshot-test, and
review change.

Unlisted tracked production changes SHALL be reported as `deferred_unmapped`.
Untracked production Python paths SHALL be reported as `deferred_untracked`; neither
category SHALL generate mutants or affect score. There SHALL be no broad Observatory
exclusion: Observatory and all other unlisted definitions are deferred because they
are not yet in the closed v1 matrix.

The controller SHALL run only the mapped tests and SHALL NOT fall back to the full
pytest suite. Before mutation, one baseline per distinct test tuple SHALL collect
coverage.py line data from the verified private source tree. A candidate whose exact
canonical mutation-token line is not executed SHALL become `no_coverage_line`; a source with no
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

The controller SHALL never mutate the developer's source file in place. The
standard-library supervisor SHALL enable Linux child-subreaper semantics before any
child and SHALL be the sole process that calls `fork`, `clone`, `posix_spawn`, or
`Popen` for every `uv`, coverage, pytest, and helper child. Bootstrap/controller SHALL
have no OS spawning primitive. They SHALL submit closed authenticated bounded
`spawn|signal|wait|fallback_request|shutdown` messages over a private
`SOCK_SEQPACKET` channel. Every message SHALL bind run ID, boot ID, increasing
sequence, `CLOCK_BOOTTIME` deadline, closed command kind, normalized argv/environment
digest, working-directory identity, memory reservation, and exact `SCM_RIGHTS`
descriptor roles. The supervisor SHALL reject replay, unknown fields/commands, broad
argv/environment, stale deadlines, and descriptor-role mismatch.

For each accepted spawn the supervisor SHALL create a stopped child and its unique
delegated worker cgroup. The child SHALL NOT enter the trusted launcher until the
supervisor moves it into that cgroup, read-back verifies membership plus
pidfd/process-group/watchdog ownership, and releases a one-shot start gate. The
cgroup path SHALL set `memory.oom.group=1`, record `memory.events`, and verify
`cgroup.procs` during cleanup. The supervisor SHALL return an opaque random
`worker_handle` plus diagnostic verified PID/start token; controller lifecycle
requests SHALL authorize by handle, not raw PID. When delegated cgroup containment is
unavailable, execution SHALL fail preflight and only `--plan-only` MAY proceed.
Subreaper and bounded `/proc` closure checks SHALL remain defense in depth. The trusted
launcher SHALL activate Landlock/seccomp before application import and then deny
fork/vfork/exec and process-form clone; any activation failure SHALL kill the group
and fail closed. Reviewed thread-form clone SHALL remain inside the worker
process/session/cgroup. The
supervisor SHALL contain the remaining hierarchy and write controller-loss fallback
after a hung/crashed/SIGKILLed/OOM controller. Host, runner, supervisor SIGKILL, or
indefinite uninterruptible kernel sleep remains explicitly best effort and SHALL
never be reported as confirmed cleanup.

The controller SHALL use a thread pool only for logical scheduling and SHALL retain
opaque supervisor handles, while the supervisor owns every pidfd, process-group ID,
cgroup, spawn, signal, wait, and reap in one synchronized registry. Spawn admission,
registration, cancellation, mutation application, test-start observation, integrity
failure, timeout and natural-exit observation SHALL share one typed event gate. Every
event SHALL carry a supervisor-assigned sequence and `CLOCK_BOOTTIME` timestamp.
Every mutant SHALL follow a locked phase machine that records
`mutation_applied_current` immediately after position-verified write and
`test_started_current` only after acknowledged pytest spawn. Natural exit,
per-mutant timeout, global signal, and execution budget SHALL form one control class
whose first observed event is provisional. Integrity causes
`guard_violation|syscall_violation|listener_failure|provenance_failure|
protocol_failure|incomplete_evidence` SHALL be sticky and dominant regardless of
event sequence. No provisional control result SHALL terminalize until the seccomp
notification queue drains, listener state closes, and both authenticated evidence
streams complete. Any integrity cause observed during finalization SHALL replace the
provisional result with phase-appropriate infrastructure error. Cleanup status SHALL
be evaluated last and `unconfirmed` SHALL perform the same override and disable run
score/cache. Only a clean integrity and cleanup finalization SHALL allow the first
control event to own classification.
Each thread SHALL own one bounded private source tree, restore and hash-check the
target file between mutants, and remove the tree at shutdown.

A private tree SHALL be built from a sorted bounded repository-local import closure
rooted at the eligible source, mapped tests, required package `__init__.py` files,
tracked `tests/conftest.py` when present, and the isolation/provenance plugin. Static
resolution SHALL obey the configured module/edge/depth/file/byte/deadline limits.
Unresolved dynamic imports SHALL require an exact reviewed dependency manifest or
fail preflight. The complete tracked-Python digest SHALL remain audit metadata but
SHALL NOT enter cache reuse identity or force the entire package into the copy
manifest. A 10x fixture of unrelated modules SHALL not change the closure, invalidate
cache reuse, or prevent eligible v1 work. The tree SHALL
contain no symlink, non-regular file, data, unmapped test, cache, `__pycache__`, or
`.pyc`. Before every mutant the worker SHALL purge bytecode/cache paths, create a
fresh empty `PYTHONPYCACHEPREFIX`, and set `PYTHONDONTWRITEBYTECODE=1`.
File/byte/copy/free-space limits from the bounded-mode requirement SHALL be checked
before admission.

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
`__file__` and digest belong to the private tree. Before application/test import, a
launcher SHALL set `no_new_privs` and install a Linux Landlock ruleset whose read
allowlist contains only the private closure, selected tests, Python/runtime shared
libraries and immutable system metadata, and whose write allowlist contains only
per-worker temp/output roots. It SHALL install seccomp denial of socket/connect and
process creation as specified below. Unsupported Landlock ABI, incomplete rules, or
unavailable `no_new_privs` SHALL fail preflight; there is no unisolated execution
fallback. Worker spawn SHALL close every descriptor except an exact reviewed pass-FD
list so a pre-opened production descriptor cannot bypass policy.
This boundary SHALL cover `openat/openat2`, `os.open`, `Path.open`, SQLite,
pandas/pyarrow parquet, DuckDB, mmap and symlink races.
Mutation execution SHALL require Linux with the configured Landlock ABI. An
unsupported platform MAY run deterministic `--plan-only` but SHALL fail execution
preflight rather than weaken isolation.

Before application code, the launcher SHALL install a seccomp user-notification
filter and transfer its listener exactly once to the controller over a pre-created
`SOCK_SEQPACKET` channel with `SCM_RIGHTS`. Controller and worker SHALL validate
tracee/listener identity, use an ACK to close transfer endpoints at fixed handshake
points, and kill the worker group if transfer or listener liveness fails. The
listener SHALL record and validate `openat/openat2`, socket/connect, fork/vfork,
execve/execveat, clone and clone3 attempts before deciding. For an allowed open, the
controller SHALL open the object itself with `openat2` and
`RESOLVE_BENEATH|RESOLVE_NO_MAGICLINKS|RESOLVE_NO_SYMLINKS`, validate the resulting
descriptor, and inject it using `SECCOMP_IOCTL_NOTIF_ADDFD`; it SHALL NOT continue a
raw tracee pathname open. Landlock SHALL remain the kernel backstop for any
non-brokered path and descriptor use after injection.
Out-of-allowlist opens, network and process creation SHALL be recorded and denied.
Thread-form clone SHALL be allowed only for reviewed flags; process-form clone SHALL
be denied. This SHALL make native I/O violations observable without pathname TOCTOU,
including when application/native code catches EPERM.

The process SHALL use a temporary `TRADE_DATA_ROOT`, HOME, XDG and cache directories;
scrub provider/API credentials and proxy variables; fix `TZ=UTC`, locale, and
`PYTHONHASHSEED`; and install Python audit/guard hooks for diagnostic attribution.
Tests SHALL prove provider, scheduler, event, real-data, network, external-process and
external-client access fail closed. V1 SHALL refuse cache reuse for a
tuple whose bounded import closure accesses an undeclared clock or entropy API. The
trust tuple SHALL remain entropy-non-cacheable while its default path can call
`uuid.uuid4()`.

The child SHALL receive only the guard pipe's write end and a child-only MAC key; the
controller SHALL retain only its read end, and neither endpoint SHALL be inherited in
the opposite process. The
launcher/plugin SHALL emit a gap-free authenticated child sequence of
`guard_started`, `kernel_policy_active`, zero or more `guard_violation`, and
`guard_completed`. The controller SHALL keep its syscall audit and key in
controller-private memory and emit its own
`syscall_audit_started`, zero or more `syscall_violation`, and
`syscall_audit_completed` sequence. The child SHALL NOT possess the controller audit
writer/key, and the controller SHALL NOT possess the child MAC key. Both SHALL bind
run ID, worker PID/start token, target digest and policy digest. Before interpreting
pytest exit, the controller SHALL drain pending notifications and require both
complete independent handshakes. Missing, empty, truncated, duplicated, wrong-token,
wrong-process or sequence-gap evidence SHALL be `infrastructure_error`. Any child or
syscall violation SHALL also be
`infrastructure_error`, including when application or native code catches an
exception or pytest otherwise exits 0 or 1.

Line coverage SHALL use exactly coverage.py 7.10.7 in statement/line mode. The
controller SHALL create and hash a minimal private rcfile that disables branch,
plugins, patches, parallel mode and relative files; it SHALL clear every
`COVERAGE_*` environment variable. The adapter SHALL run `python -m coverage run
--rcfile <private-rc> --data-file <private-data> --source
<private-package> -m pytest --import-mode=importlib <absolute-mapped-tests>` followed
by `python -m coverage json --rcfile <private-rc> --data-file <private-data> --include
<canonical-private-target> -o <private-json>`. Other closure modules MAY exist in the
raw coverage database, but the filtered JSON SHALL contain exactly one canonical
target. It SHALL map that target back to the repository-relative source, read bounded
`executed_lines`, reject malformed/oversized/missing/duplicate/unknown targets, and
remove data and JSON afterward. Coverage process or data failure SHALL be
`baseline_unavailable`, never `no_coverage` or `killed`.

The baseline time `T` for a selected affected-test set SHALL be measured before mutant
execution under the same provenance, coverage, environment, and process-group
controls. The baseline SHALL have its own finite timeout limited by remaining global
time. The per-mutant timeout SHALL be `max(10 seconds, 2.5 * T)`. In `changed` mode it
SHALL be capped at 60 seconds. A tuple requiring more SHALL be unavailable in changed
mode. Insufficient remaining time SHALL produce `not_run_budget`, not a late launch.

On timeout, cancellation, or global deadline, the controller SHALL request termination
by opaque worker handle. The supervisor SHALL terminate that worker cgroup/process
group, wait a bounded observable grace period, escalate to kill, and reap before
returning. A timeout SHALL be `timeout_test_started`, not `killed`. Failure to confirm
cleanup SHALL set receipt cleanup status `unconfirmed` and replace the provisional
terminal with phase-appropriate infrastructure error, never a kill. Failure to launch,
copy, mutate, parse, collect, or execute infrastructure SHALL be
`infrastructure_error`, not `killed`.

After a complete clean guard lifecycle, pytest exits SHALL map exactly:
`0=survived`, `1=killed`, and `2/3/4/5` SHALL be `infrastructure_error` in mutant
phase. A controller SIGINT/SIGTERM that is the first control-class event SHALL
provisionally classify a started mutant `cancelled_test_started`; an earlier natural
exit SHALL provisionally keep its result, subject to later integrity and cleanup
override. Unstarted selected work SHALL be `not_run_cancelled`; neither SHALL be
conflated with infrastructure or budget.
Baseline exit `0` is usable. Baseline nonzero exit, timeout, coverage, provenance,
isolation, or infrastructure failure SHALL be `baseline_unavailable` with exactly one
tuple-level reason `test_failed|timeout|coverage_failed|provenance_failed|
isolation_failed|infrastructure_failed`. A baseline timeout SHALL NOT increment the
mutant-timeout count. The
classification SHALL be:

- baseline failure: mark that test tuple's selected mutants `baseline_unavailable`;
- pytest exits zero under the mutant: `survived`;
- pytest reports a test failure under a valid mutant: `killed`;
- deadline expires: `timeout`;
- no mapping or baseline line coverage: `no_coverage_mapping` or `no_coverage_line`;
- the operator cannot produce valid changed source: `invalid`;
- controller/tool/process or isolation-guard failure: `infrastructure_error`;
- controller signal after mutant pytest start: `cancelled_test_started`;
- selected work not admitted after controller signal: `not_run_cancelled`.

OOM attribution SHALL require a unique invocation/worker cgroup, pre-spawn and
post-wait `memory.events` snapshots showing an `oom_kill` increase, the SIGKILLed
PID/start token's membership, and no competing member kill in that interval. Shared
counters, absent membership, or an ambiguous delta SHALL be `unknown_sigkill`, never
OOM. Tests SHALL cover both exit/cancel race orders, late integrity after natural
exit, both evidence-stream completion orders, cleanup override, hung `uv`,
bootstrap/controller crash, proven OOM/unknown SIGKILL, wrapper SIGTERM, cgroup/
subreaper cleanup, and attempted session or process-creation escape. The no-child guarantee applies after supervisor-
managed child failure and graceful supervisor signals; host/runner/supervisor SIGKILL
is best effort.

#### Scenario: Mutant causes an infinite loop with descendants

- **WHEN** the selected test exceeds its computed timeout after creating child
  processes
- **THEN** the controller requests termination and the supervisor terminates and reaps
  the full worker cgroup/process group
- **AND** records one timeout mutant
- **AND** no child remains alive after the worker completes.

### Requirement: Reports SHALL be complete, machine-readable, and recoverable

The default output root SHALL be repository `.mutation-testing/`. `--output-dir`
SHALL resolve inside that root, have no symlink ancestor, and either be absent or
contain the exact `.trade-mutation-output-v1` ownership marker. Filesystem roots,
repository data, DB, model, Parquet, generated-artifact roots, path escapes, and
unrelated existing directories SHALL fail before a lock or write.

At supervisor entry, each invocation SHALL atomically create
`invocations/<run_id>.json` using schema `trade.mutation.invocation.v1` with run ID,
mode, `/proc/uptime` anchor, `CLOCK_BOOTTIME` value/clock ID, kernel boot ID, absolute
deadlines, supervisor identity, lifecycle/projection states, and exact expected
staging, final, fallback, and bundle paths. Bootstrap/controller SHALL send bounded
authenticated transitions over one inherited supervisor channel; the supervisor SHALL
validate run ID, child PID/start token, sequence, schema and forward state before it
alone atomically rewrites the receipt. The receipt SHALL be a bounded phase journal
that records plan digest, selected count, mutation/test phase facts, controller exit,
cleanup evidence, last durable phase, expected report root/report sequence and
projection state. The supervisor SHALL be the sole writer of the receipt and every
fallback. A controller SHALL only submit one authenticated bounded
`fallback_request`; it SHALL NOT open or replace a fallback path. If the controller
exits unexpectedly, is SIGKILLed, or is proven OOM-killed by the unique-cgroup rule,
the surviving supervisor SHALL clean descendants, validate
any already-published run-ID root, and bind that root or atomically write a
`controller_lost` fallback. Unknown SIGKILL SHALL NOT be labelled OOM. CI SHALL consume the receipt path
exported through `$GITHUB_OUTPUT`,
validate the run ID and referenced evidence, and SHALL NOT infer this invocation from
global `current.json`, timestamps, or newest-file globbing.

Every invocation SHALL stage one run-ID directory containing:

- `report.json` using schema `trade.mutation.report.v1`;
- `summary.md` suitable for GitHub job summaries;
- `index.html` with summary counts and surviving/no-coverage details;
- `manifest.json` with the run ID and digest/size of every other generation member.

JSON SHALL be authoritative and Markdown/HTML SHALL render from the persisted JSON.
`manifest.json` SHALL NOT hash itself. After read-back verification and fsync of files
and the staged directory, the controller SHALL hash the manifest. Under the output
lock, it SHALL reserve and fsync the next `report_sequence` in
`report-sequence.json`, binding run ID and expected manifest digest, before atomically
replacing one `current.json` pointer containing run ID, manifest SHA-256, manifest
byte size and reserved report sequence. The supervisor SHALL write the same
root/report sequence into the invocation receipt. Readers and
CI SHALL validate the root before member hashes. A crash SHALL expose either the
previous complete generation or the new complete generation, never mixed files. On
the next locked operation an unresolved report reservation SHALL be reconciled before
any new report sequence. It SHALL commit only if `current.json` exactly binds the same
run ID, report sequence, manifest digest, manifest size, and valid generation root;
otherwise it SHALL commit a report tombstone. A stale pointer, same run with a
different root/size, or an unreferenced valid root SHALL NOT commit a source. Report
sequence values SHALL never be reused and SHALL be independent from trend sequence.
Per-mutant cache outcomes SHALL stage under their run ID. A digest-bound completed-run
commit marker SHALL be published only after `current.json` succeeds and the generation
is complete/cache-eligible; cache reads SHALL require that marker. Partial, cancelled,
degraded, report-failed, or cache-commit-failed entries SHALL remain uncommitted and
unreadable. A later trend-source/projection failure SHALL NOT revoke an already valid
cache marker. The immutable report SHALL contain staged cache digests/eligibility, not
post-publication projection outcomes. V1 does not expose a `--resume-from` contract.

Each invocation SHALL hold a per-run staging lease containing PID, process-start
token, and boot identity. Staging cleanup SHALL require non-blocking lease acquisition,
valid stale owner identity, age at least 24 hours, and proof the run is not current.
Render, staging-validation, final-directory rename, and pointer-publication failures
SHALL remain distinct. If a complete generation cannot be validated and published,
the controller SHALL preserve owned staging, return exit 2, and submit one bounded
authenticated fallback request. The supervisor alone SHALL atomically write one
`fallback-<run_id>.json` diagnostic using schema
`trade.mutation.fallback.v1` with redacted stable `error_code`, failed stage,
retryability, message, remediation command, receipt path, and exact evidence paths
outside `current.json`; fallback is not a report generation. A malformed, replayed,
or unauthenticated request SHALL produce a supervisor-owned protocol-error fallback.

A missing `current.json` before the first successful generation SHALL be normal.
Under the publication lock, a malformed or symlinked pointer, missing referenced
generation, or manifest/member digest mismatch SHALL be moved to a UUID quarantine
record before a valid successor can publish. The successor manifest SHALL record the
recovery code and quarantine path. If safe quarantine cannot complete, publication
SHALL fail, leave the predecessor untouched, and emit exact read-only
`trade dev mutation inspect --run-id <run_id>` plus idempotent
`trade dev mutation repair --run-id <run_id> --expected-manifest-sha256 <digest>`
commands. `repair` SHALL default to dry-run, require `--apply` for mutation, take the
same lock, verify ownership/root hash again, reject symlinks, and never touch an active
lease. `inspect` SHALL return schema `trade.mutation.inspect.v1` and closed states
`healthy|first_run|repairable|active_lease|stale_expected_digest|evidence_expired|
unsafe|internal_error`
with exits 0 for healthy/first-run, 1 for repairable and 2 for unsafe/internal error.
`repair` SHALL return `trade.mutation.repair.v1`; dry-run SHALL exit 0 for no-op/valid
plan, 1 for active lease/stale expected digest and 2 for unsafe/internal error, while
`--apply` SHALL exit 0 only after read-back verification.
`trade dev mutation reconcile --trend-epoch <epoch>` SHALL be read-only by default,
validate the named trend chain and gaps, and require `--apply` plus the publication
lock to commit a gap tombstone or rebuild `trend.jsonl`. It SHALL NOT invent a
predecessor, sequence, or score.

One safe-error layer SHALL redact configured credential values,
credential-looking environment values, URL userinfo, and controlled home/temp roots
from console, JSON, Markdown, HTML, fallback, cache/trend recovery diagnostics, and
retained staging before persistence.

The report SHALL include mode, source/base/head identities, tool versions, config
digest, deterministic selection strategy, fixed seed or `null`, budgets, worker
limit including CPU/memory inputs, baseline command/collected node IDs/timings,
candidate-positions-scanned/generated-mutants/selected/mutation-applied/test-started
counts,
killed, survived, timeout, no-coverage, baseline-unavailable, invalid,
infrastructure-error, equivalent-exception and not-run counts, mutation score,
budget-exhausted state/reason, elapsed time, and every selected mutant's path,
line/column, definition, operator, related tests, status, duration, and bounded
diagnostic, original/mutated digest, planned/observed span, and cache identity.

The terminal-status enum SHALL be closed to `killed_fresh`, `survived_fresh`,
`killed_cache`, `survived_cache`, `timeout_test_started`,
`infrastructure_error_test_started`, `cancelled_test_started`,
`no_coverage_mapping`, `no_coverage_line`, `baseline_unavailable`,
`invalid_mutation`, `infrastructure_error_pre_test`, `equivalent_exception`,
`not_run_plan`, `not_run_budget`, and `not_run_cancelled`. Every selected mutant SHALL
have exactly one. Each record SHALL also have booleans `mutation_applied_current` and
`test_started_current`; cached and pre-admission outcomes set both false, all
test-started terminals set both true, and
`infrastructure_error_pre_test` MAY set mutation-applied true only when the verified
transformation completed before later infrastructure failed.
Each record SHALL also carry orthogonal cleanup status
`not_required|confirmed|unconfirmed`. `cleanup_unconfirmed` SHALL NOT be a seventeenth
terminal and SHALL force the phase-appropriate infrastructure terminal as specified
above.

The terminal/phase matrix SHALL be closed:

| Terminal group | Allowed `(mutation_applied_current, test_started_current)` | Cleanup |
|---|---|---|
| fresh killed/survived, timeout/infrastructure/cancelled test-started | `(true,true)` | `confirmed` |
| cache killed/survived | `(false,false)` | `not_required` |
| no-coverage, baseline-unavailable, invalid, equivalent, plan-only | `(false,false)` | `not_required` |
| infrastructure pre-test | `(false,false)` or `(true,false)` | `not_required` before spawn, otherwise `confirmed` |
| not-run budget/cancelled | `(false,false)` or `(true,false)` | `not_required` |

An unconfirmed started cleanup SHALL serialize as
`infrastructure_error_test_started/(true,true)`; unconfirmed pre-test cleanup SHALL
serialize as `infrastructure_error_pre_test` while preserving its durable mutation
fact. No terminal SHALL permit `(false,true)`.

Only killed/survived outcomes from a complete identity-valid run SHALL be cacheable;
all other terminals SHALL never be reused as successful outcomes. Every aggregate
SHALL be recomputable as follows:

- `generated_mutants = selected`; `candidate_positions_scanned`,
  `not_selected_scanned`, and an unscanned remainder `"unknown"` stay separate;
- `selected` equals the sum of all sixteen closed terminal counts;
- `mutation_applied_current` equals the sum of the per-record fact and equals
  `test_started_current + applied_without_test`; `applied_without_test` is the sum of
  pre-test records whose phase fact is true, including infrastructure, budget or
  cancellation that wins after a verified transformation but before pytest spawn;
- `test_started_current = executed = killed_fresh + survived_fresh +
  timeout_test_started + infrastructure_error_test_started +
  cancelled_test_started`;
- `killed = killed_fresh + killed_cache` and
  `survived = survived_fresh + survived_cache`;
- `timeout = timeout_test_started`;
- `infrastructure_error = infrastructure_error_test_started +
  infrastructure_error_pre_test`;
- `cancelled = cancelled_test_started + not_run_cancelled`;
- `no_coverage = no_coverage_mapping + no_coverage_line`;
- `not_run = no_coverage_mapping + no_coverage_line + baseline_unavailable +
  invalid_mutation + infrastructure_error_pre_test + equivalent_exception +
  not_run_plan + not_run_budget + not_run_cancelled`;
- mutation score SHALL be `null` when `killed + survived == 0`, otherwise
  `killed / (killed + survived)`.

Timeout, cancellation, no-coverage, baseline-unavailable, invalid,
infrastructure-error, exception, and not-run outcomes SHALL NOT improve the score or
be cacheable. Cancellation before admission, between spawn/registration, during
pytest, and while queued SHALL each have a golden count-algebra test.

Console, Markdown, and HTML SHALL all expose factual run status, controller exit code,
stop reason, deferred paths, every aggregate above, numerator/denominator, baseline
comparability and reason, cache freshness, projection state/gaps, cleanup/orphan
result, stable remediation command, and exact receipt/report/fallback paths. Exit zero
SHALL NOT hide `budget_partial`, timeout, no-coverage, or deferred work.

`--plan-only` SHALL validate config/tool APIs, enumerate and select candidates, read no
outcome cache, run no coverage/pytest, assign every selected mutant `not_run_plan`,
publish run status `plan_only`, return score `null`, and exit 0 unless preflight or
publication fails. Run-status precedence SHALL be `report_failed`,
`signal_cancelled`, `preflight_failed`, `degraded_infrastructure`,
`budget_partial`, `plan_only`, `deferred_only`, `zero_work`, then `complete`. A run
with no eligible candidate but one or more deferred production paths SHALL be
`deferred_only`, return score `null`, run no pytest, and exit 0. A future policy
regression changes exit code, not factual run status.

Core and full modes SHALL persist a content-addressed result cache. Its reuse identity
SHALL include execution/mutant ID, the bounded import-closure digest, mapped tests,
fixtures/conftest, reviewed dynamic-import manifest, `pyproject.toml`, `setup.cfg`,
`uv.lock`, mutation configuration and exceptions, normalized pytest arguments,
Python/Cosmic Ray/coverage/pytest/plugin/kernel-isolation versions,
platform/architecture, controlled environment including native-thread controls,
declared clock and entropy policies, selection algorithm, and exact mutant cohort.
The complete tracked-Python tree digest SHALL remain audit metadata only and SHALL NOT
invalidate an unchanged closure. Corrupt, partial, unbounded, undeclared-clock/entropy,
identity-mismatched, or entries without a valid completed-run commit marker SHALL be
ignored. Changed mode MAY read a matching entry but SHALL remain correct without it.

After report publication, core/full SHALL expose eligible cache markers, commit one
content-addressed immutable compact
`trend-sources/<trend_sequence>-<run_id>-<digest>.json` after reserving an independent
core/full `trend_sequence`, advance the trend high-water with its digest, then
reconcile `trend.jsonl`. Changed, plan-only, deferred, and zero-work report
publication SHALL consume no trend sequence. Each source or tombstone
SHALL bind the previous retained digest, forming a validated hash chain anchored by
the high-water and a retention checkpoint. Each source record SHALL bind report
manifest root/report sequence, mode, commit, exact comparison cohort,
killed/survived numerator and denominator key-set digests, exception/no-coverage
membership digests, score/counts, scope/config/environment digests, run ID, and its
pre-reserved trend sequence.
UTC is display/age metadata, not sole ordering truth. A stale restore below high-water
or chain break SHALL fail closed.
The trend-source ledger SHALL retain at most 365 records, 32 MiB, and 400 days;
`trend.jsonl` SHALL be an idempotently rebuildable 8 MiB projection of that ledger.
Result cache
retention SHALL be at most 20000 entries and 512 MiB. Under the output publication
lock, deterministic eviction SHALL remove oldest `created_at,key` records first and
record evicted counts, bytes, and range. Immutable generations SHALL be at most 30
entries, 2 GiB, and 30 days; fallback, quarantine, and failed-staging evidence SHALL
be at most 50 entries, 512 MiB, and 14 days. Invocation receipts SHALL be at most 400
entries, 64 MiB, and 400 days. Eviction SHALL protect `current`, active leases,
current failure evidence, and each retained receipt/report/fallback/trend carrier as
one retention unit. They SHALL be evicted synchronously, or the retained receipt SHALL
be replaced by `trade.mutation.evidence-tombstone.v1` binding former roots/sequences,
reason and time; `inspect` SHALL return `evidence_expired`, not healthy. CI result
cache restore SHALL list at most 20 candidates, download at most two and 512 MiB total,
stop after 60 seconds for changed or 180 seconds for core/full, accept only exact
schema/tool/config prefixes and newest valid numeric `(workflow_run_id, run_attempt)`,
validate entries/commit markers, and save one bounded run-specific immutable archive
with seven-day retention.
Cache SHALL remain disposable and SHALL NOT be treated as trend evidence.

Scheduled/manual core and full SHALL restore trend by decreasing numeric tuple
`(workflow_run_id, run_attempt)` from immutable
`mutation-aggregate-v1-<workflow_run_id>-<run_attempt>` artifacts, never timestamp or
a filename glob. Restore SHALL list at most 20 carriers, read at most 20 MiB metadata,
download at most the first two same-epoch candidates and 64 MiB total, and stop after
180 seconds. The highest tuple is authoritative and SHALL validate bounded ledger,
projection, high-water/checkpoint, complete retained hash chain,
workflow/repository/configuration-epoch identity, and prior source-bundle digest. Only
an identical-manifest duplicate at the same high-water MAY replace it. A corrupt
highest carrier SHALL start `predecessor_invalid`; lower tuples SHALL NOT be silently
accepted. No carrier due to expiry/pause SHALL start `predecessor_unavailable`.
After the run CI SHALL upload one new aggregate capped at 64 MiB with 90-day
retention. Both epochs SHALL expose a gap until explicit `reconcile`; comparisons
across them SHALL be non-comparable. PR changed mode SHALL NOT write the aggregate.
Post-publication projection failure SHALL be
recorded as `projection_degraded` in the invocation receipt, leave the factual report
valid, leave uncommitted cache unreadable, and expose a trend gap until reconciliation.
Score regression is comparable only over identical complete unambiguous comparison
cohorts, identical killed/survived denominator key sets, and identical exception/no-
coverage memberships with unchanged normalized node text and policy; numerator and
denominator key-set digests SHALL persist. Any timeout, infrastructure, cancellation,
invalid, baseline-unavailable, differing exception/no-coverage membership, partial
run, ambiguous/added/deleted/changed candidate, or cohort difference SHALL be
non-comparable. Execution IDs remain source-exact.

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

For CI, a standard-library validator SHALL create exactly one immutable
`trade.mutation.bundle.v1` directory from the exported receipt and its one referenced
generation or fallback. It SHALL contain the exact receipt bytes, complete referenced
evidence, `validation.json` using schema
`trade.mutation.bundle-validation.v1`, and `bundle-manifest.json` with size/SHA-256
for every preceding member. Validation SHALL bind workflow/run identity, controller
exit, receipt digest, evidence kind/root digest, lifecycle/count/cleanup checks and
closed failure codes. CI SHALL read-back and fsync the bundle and upload only the
exported exact path/digest from the execution job under a run/attempt-specific
artifact name. A separate job SHALL download that named artifact into a fresh
directory, reject extra/path-escaping members, require the expected bundle-manifest
digest job output, and independently recompute all member and receipt/report/count/
cleanup bindings from bytes. No execution-workspace path SHALL cross jobs. A missing,
malformed, unsafe, digest-mismatched or receipt-unbound bundle SHALL fail the
evidence-validation job and SHALL NOT be relabelled as a mutation outcome.

#### Scenario: Global time expires after partial execution

- **WHEN** some mutants completed before the wall-clock budget
- **THEN** their outcomes remain in all reports
- **AND** unstarted selected mutants are marked `not_run_budget`
- **AND** report generation completes within the in-budget 30/60/120-second reserve
- **AND** ordinary qualified-host execution does not admit work beyond the
  600/1800/5400-second controller budget
- **AND** any uninterruptible kernel/filesystem stall remains unconfirmed and is
  contained by the larger outer runner timeout rather than reported as success.

### Requirement: Mutation quality policy SHALL be gradual and auditable

This change SHALL keep all CI mutation jobs non-blocking. The controller SHALL still
evaluate a repository baseline when it is established and comparable:

- a changed-scope mutation-score decrease greater than five percentage points SHALL be
  reported as a policy regression only when base and head have one identical complete
  comparison-key cohort, identical killed/survived denominator key-set digest, and
  identical exception/no-coverage membership digests;
- modified core business code SHALL display a 70% target;
- 80% SHALL be documented only as a future core target, not a current repository-wide
  gate;
- no-coverage mutants SHALL be reported separately.

`execution_id`/`mutant_id` SHALL remain source-digest exact for execution, cache, and
exceptions. A separate `comparison_key` SHALL include source path, qualified eligible
definition/digest, definition-relative canonical mutation-token position, normalized
original node text, operator name/arguments, and `definition_occurrence`. Base and
head SHALL be enumerated separately. Added, deleted, changed-node, ambiguous,
policy-different, or partial candidates, differing score-eligible sets, timeout,
infrastructure, cancellation, invalid, baseline-unavailable, or differing
exception/no-coverage membership SHALL make the run non-comparable; line shifts or
unrelated edits MAY remain comparable. V1 SHALL persist numerator/denominator key-set
digests and report `baseline_comparable=false` rather than invent a mapping.

Schema `trade.mutation.baseline.v1` SHALL start `established: false`. An established
record SHALL bind a verified bundle-manifest digest, report-manifest digest, run ID,
source commit and `GitScopeSnapshot` digest, mode, tool/lock/config/matrix/cohort/
environment/isolation identities, complete comparison cohort, killed/survived/
exception/no-coverage key sets and their digests, counts, numerator, denominator, and
score. Evaluation SHALL open the referenced immutable bundle, validate every binding,
reconstruct all key sets from per-mutant records, and recompute counts/score. A
stored-only aggregate SHALL NOT establish a baseline.

Equivalent or meaningless mutants MAY be registered only with exact `mutant_id`,
source path, source digest, start/end line and column, operator, occurrence, owner,
reason, and review expiry. Directory, filename, definition, location, or operator
wildcards SHALL be rejected. Expired, stale-digest, missing, or ambiguous exceptions
SHALL fail configuration validation. Exceptions SHALL remain visible in reports and
SHALL not be counted as killed.

Capacity and exception times SHALL be RFC 3339 UTC intervals. `expires_at` SHALL be
strictly after `qualified_at`/`reviewed_at`, at most 30 days later for capacity and
180 days for an exception, and in the future at validation. A current clock earlier
than the start by more than five minutes, a start over five minutes in the future,
invalid/leap-second text, or a detected rollback across the interval SHALL fail
closed. UTC SHALL NOT order execution events.

#### Scenario: Baseline is not yet established

- **WHEN** the saved baseline is marked unestablished or no comparable scope exists
- **THEN** the invocation remains report-only
- **AND** states that no regression decision was possible
- **AND** does not manufacture a baseline from the current result.

### Requirement: CI SHALL route bounded modes without making them universal PR gates

GitHub Actions SHALL:

- use only `runs-on: [self-hosted, linux, x64, ephemeral, trade-mutation-v1]`, a
  reviewed disposable one-job VM/ARC profile with no production/data mounts or
  long-lived credentials, a fresh writable delegated cgroup v2 subtree with finite
  memory and `memory,pids,cpu` controllers, and supported Landlock/seccomp
  notification/addfd. Preflight SHALL verify runner image digest, ephemeral marker,
  mount namespace, cgroup ownership/controllers/finite `memory.max`, writable child
  cgroup, kernel probes, and absence of data/provider mounts before mutation; mismatch
  SHALL produce preflight evidence and execute no mutant. Fork PRs SHALL receive no
  secrets;
- run a cheap changed-mode plan on pull requests and execute mutation only when at
  least one eligible production Python file is present;
- pass the pull request base SHA explicitly and never infer a missing base by
  broadening scope;
- cap the changed job at 15 minutes around the 600-second total script budget;
- make changed/core/full mutation outcomes report-only by capturing exit 0/1/2,
  receipt path and run ID through `$GITHUB_OUTPUT`, preserving the original exit in
  the receipt/summary, then returning success from the execution step; score,
  survivor, timeout, baseline or tool-status outcomes SHALL NOT become a merge gate
  in this change;
- in the execution job use `if: always()` to validate the receipt, construct and
  locally validate `trade.mutation.bundle.v1`, upload exactly that bundle under a
  run/attempt-specific artifact name, and export the name plus manifest digest;
- run a separate dependent `if: always()` evidence-validation job that SHALL download
  that named artifact into a fresh directory, independently validate bytes, and fail
  on missing, extra, malformed, unsafe, digest-mismatched or receipt-unbound evidence;
- use PR concurrency key
  `mutation-pr-${{ github.repository }}-${{ github.event.pull_request.number }}` with
  `cancel-in-progress: true`;
- use `if: always()` to validate the exact invocation receipt exported through
  `$GITHUB_OUTPUT`, append its success or typed fallback summary, construct the
  immutable bundle above, and upload only the exported exact bundle path/digest with
  14-day retention for controller exits, internal deadlines and gracefully delivered
  signals while the job remains alive; global `current.json`, raw staging and
  newest-file globbing SHALL NOT select CI evidence;
- document upload after hard GitHub job cancellation, runner loss, or host loss as
  best effort rather than guaranteed;
- run core on cron `17 18 * * 1-6` UTC and on manual request, with 1800-second script
  budget and a 35-minute outer timeout;
- run full only on cron `17 18 * * 0` UTC or explicit manual request, with
  5400-second script budget and a 100-minute outer timeout;
- give scheduled/manual core and full one shared concurrency key
  `mutation-long-${{ github.repository }}` with `cancel-in-progress: false`, which guarantees at
  most one running and one pending member but permits a newer dispatch to supersede an
  older pending member; the workflow SHALL rely only on no overlap and SHALL NOT claim
  a durable/lossless queue;
- create no run ID or trend record for a pending dispatch superseded before command
  entry, and order committed trend records by their independent trend sequence, not report sequence, workflow
  dispatch or adjustable UTC order;
- require reviewed `config/mutation-capacity.json` schema
  `trade.mutation.capacity.v1` before core/full execution. Qualification SHALL bind
  mode, config/matrix/cohort/import-closure/tool/lock/isolation-policy digests, Python
  and dependency lock identities, immutable `runner_capacity_identity`, separate
  serial/capacity `measurement_profile`s, qualification bundle/run digests, and
  planning/copy/baseline/per-mutant p50/p95/render/fsync/cleanup measurements. It
  SHALL bind at least 30 completed measured mutants, zero infrastructure/OOM/timeout/
  cleanup/report/projection failure, computed worker count, serial throughput,
  `p95_mutant * selected_limit / workers + baseline + plan/copy + render/fsync/cleanup
  <= execution_deadline * 0.80`, peak memory/disk at most 80%, and each worker below
  `memory.high`. It SHALL expire within 30 days, record headroom and
  `qualified: true|false`, and fail execution preflight on any missing, expired,
  false, mismatched, non-finite-memory or insufficient-headroom field;
- expose `trade dev mutation qualify --mode core|full
  --runner-profile PROFILE` as an explicit report-only serial-plus-capacity
  qualification proposal command that never edits the reviewed capacity file;
- restore result cache and rolling aggregate only within the candidate/byte/deadline
  limits above; identify aggregate carriers by `(workflow_run_id, run_attempt)`, open
  `predecessor_invalid` rather than falling back below a corrupt highest carrier, and
  require explicit reconcile for epoch gaps.

The ordinary pull-request workflow SHALL never invoke `core` or `full`. Documentation,
test-only, fixture, generated, and excluded changes SHALL not execute mutant tests.
Before scheduled routes execute mutants, cold-cache core and full qualification SHALL
run both serial and capacity profiles on the exact runner, record planning/copy/
baseline/per-mutant p50/p95/render/fsync/cleanup time, and satisfy the reviewed
identity, pass equations, temporal rules, 30-day freshness, and configured time/
memory/disk headroom contract.

#### Scenario: A documentation-only pull request runs CI

- **WHEN** the changed-mode plan sees no eligible source
- **THEN** the mutation job exits successfully after producing a zero-work report
- **AND** no mutation dependency or test worker is required beyond planning.

#### Scenario: Long-running routes overlap

- **WHEN** nightly core, weekly full, or manual core/full are requested concurrently
- **THEN** the shared long-run concurrency group runs at most one
- **AND** at most the latest pending request is retained under native GitHub semantics
- **AND** a superseded pre-start request creates no false report or trend record
- **AND** no two long modes compete for mutation workers or trend publication.
