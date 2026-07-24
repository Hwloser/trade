# Proposal: Controlled Mutation Testing

## Why

Trade has broad pytest coverage, but line execution alone does not prove that tests
detect incorrect decision, belief, factor, signal, or evaluation behavior. The
repository currently has no mutation-testing tool, no coverage tool, and no committed
CI workflow. A naive mutation rollout would be unsafe: the full pytest baseline takes
about 136 seconds and currently has one reproducible unrelated failure, so running the
entire suite for every unbounded mutant would consume excessive CI time and conflate
tool failures, timeouts, and real kills.

The project needs a source-only, bounded mutation-quality signal that is useful on
pull requests without becoming a repository-wide mandatory gate. It must prefer
changed core business lines, use only reviewed first-order operators, run selected
unit tests in isolated process groups, preserve partial reports at every budget
boundary, and keep mutation score advisory until a trustworthy history exists.

## What Changes

- Add canonical `trade dev mutation changed|core|full` developer commands, with
  `scripts/mutation-test` retained as an argv/exit-compatible facade.
- Add a typed Python controller under `trade_py/devtools/mutation_testing/` that:
  - pins Cosmic Ray 8.4.6 and uses only `get_operator` plus `mutate_code` for
    deterministic first-order
    Python mutation generation;
  - enumerates occurrences independently per operator, applies exactly one
    position-verified mutation to a worker-owned bounded import-closure tree, and
    proves pytest imported that tree;
  - selects only the closed v1 definition-to-test matrix and requires baseline line
    coverage from pinned coverage.py 7.10.7 before mutant execution;
  - admits only exact changed-line candidates in PR mode and uses configured core
    definitions for scheduled/manual modes without widening to an enclosing
    definition;
  - admits a candidate only when its canonical Cosmic Ray mutation-token position is
    on a changed line, seals/revalidates the complete Git scope, and distinguishes the
    150/1000/5000 generated-mutant ceilings from broader bounded candidate scanning;
  - enforces mutant, candidate-scan, worker, per-mutant, output, and wall-clock
    budgets across bootstrap, dependency preparation, execution, and publication;
  - selects an absolute Python 3.7+ supervisor interpreter and starts one
    standard-library supervisor before every `uv` child, makes it the invocation
    subreaper/watchdog, sole OS-level child spawner, and sole receipt/fallback writer;
    the non-spawning controller uses an authenticated bounded protocol with opaque
    worker handles, and the supervisor contains and observes cleanup even when the
    controller is killed;
  - denies network, process launch, and real-data paths with Linux Landlock/seccomp
    containment, transfers a seccomp listener over `SCM_RIGHTS`, brokers allowed
    opens with `openat2` plus descriptor injection, and requires independent
    controller-owned syscall audit and child guard evidence before a mutant can be
    killed or survived;
  - distinguishes killed, survived, timeout, mapping/line no-coverage,
    baseline-unavailable, cancellation, invalid, and infrastructure-error outcomes;
    integrity evidence is sticky and dominant after notification drain, while
    unconfirmed cleanup is an orthogonal status that replaces any provisional
    killed/survived result with a phase-appropriate infrastructure terminal;
  - publishes deterministic JSON, Markdown, and HTML as one bounded atomic run
    generation with closed status/count algebra, a manifest hash anchored by the
    invocation receipt/current pointer, and an independent typed fallback if
    publication itself fails;
  - commits cache outcomes only after report publication, keeps report publication
    sequence separate from core/full trend sequence, and derives bounded trend views
    from a hash-chained immutable trend-source ledger with explicit reconciliation;
  - validates and packages the exact receipt plus referenced generation/fallback as
    one digest-bound `trade.mutation.bundle.v1` CI artifact; the execution job uploads
    it and a separate job downloads and independently validates the bytes.
- Add `config/mutation-testing.toml`, an initial unestablished baseline, and a
  location-precise equivalent-mutant exception registry. Add an identity- and
  freshness-bound capacity qualification contract for scheduled modes.
- Add focused controller, selection, process-isolation, reporting, baseline, and CLI
  tests using temporary repositories and synthetic source.
- Add optional locked mutation dependencies without changing the existing pytest
  framework or runtime dependencies.
- Add GitHub Actions workflows:
  - pull requests plan and run `changed` only when eligible production code changed;
  - nightly and manual runs execute `core`;
  - weekly or explicit manual runs execute `full`;
  - all mutation outcomes are initially report-only, while an independent evidence
    validator fails on missing, malformed, unsafe, or receipt-unbound bundles;
  - scheduled modes require a reviewed, unexpired capacity qualification and carry
    trend history through validated immutable rolling aggregate artifacts rather than
    treating result cache as evidence;
  - every execution route uses a reviewed disposable `trade-mutation-v1` runner with
    writable delegated cgroup v2, finite memory, Landlock and seccomp notification;
    unsupported runners produce plan/preflight evidence and execute no mutant;
  - native GitHub concurrency prevents simultaneous long runs but may supersede an
    older pending request, which is reported honestly rather than described as a
    durable queue.
- Add `docs/mutation-testing.md` covering operation, interpretation, exceptions,
  budgets, troubleshooting, and anti-gaming rules.

## Compatibility

The stable runtime CLI, HTTP, Web, SDK, scheduler, event, database, data, model, and
C++ contracts do not change. The new command is a developer-only additive surface.
Production modules are read and copied but never edited in place by the controller.
Mutation dependencies live in an optional development group and are not installed in
runtime environments unless explicitly requested.

The initial quality policy is report-only. A saved baseline may report a regression
only when separately enumerated base/head revisions have one complete, unambiguous
comparison-key cohort; exact source-bound execution IDs remain distinct. Changed
nodes, partial cohorts, additions/deletions, or policy changes are explicitly
non-comparable. CI remains non-blocking in this change. A future separately reviewed
change may make a stable delta policy blocking after sufficient history and exception
review.

## Scope

In scope:

- Python core business unit-test mutation only.
- Deterministic changed/core/full planning and execution.
- Bounded concurrency, native thread pools, source/AST/dependency/private-tree/report
  limits, supervisor-owned process-tree containment and fallback, kernel I/O
  containment,
  cancellation algebra, partial reporting, committed cache/trend artifacts, bounded
  report retention, sealed Git scope, independently reconstructable baseline evidence,
  bounded remote restore, exact mutant exceptions, and explicit trend reconciliation.
- GitHub Actions because the authoritative remote is GitHub.

Out of scope:

- C++, TypeScript/React, generated clients, migrations, DTO-only modules, startup and
  dependency-injection code, logging/metrics wrappers, third-party code, or tests as
  mutation targets.
- Higher-order mutation, random sampling, full-repository mutation on pull requests,
  or running the complete pytest suite per mutant.
- Fixing the pre-existing Observatory catalog test failure.
- Changing production behavior merely to improve mutation score.

## Risk and Rollback

The main risks are runaway test processes, excessive enumeration, incorrect
test-to-source mapping, native I/O escaping Python guards, false kills from
infrastructure failures, and score gaming through broad exclusions. An
invocation-wide supervisor/deadline, subreaper/cgroup process containment,
Landlock plus brokered seccomp opens and independent syscall audit,
source/AST/dependency/private-tree limits, bytecode-clean private source trees,
target-filtered line coverage, closed
operator/definition maps, generation-atomic output plus safe fallback, exact
fresh/cache/plan/cancellation status algebra, authenticated guard lifecycle evidence,
committed cache markers, separate report/trend sequence reservations and tombstones,
coupled receipt retention/evidence tombstones, identity-bound capacity qualification,
bounded remote restore with invalid-predecessor epochs, immutable cross-job CI
bundles, corrupt-pointer recovery, and mutant-ID exception validation address those
risks. Linux cannot
guarantee userspace reaping of an indefinitely uninterruptible task or completion of
a stuck filesystem `fsync`; those cases remain `cleanup_unconfirmed` or lack a
claimed generation and are contained by the larger CI runner timeout, never counted
as kills.

Rollback removes the optional mutation dependency group, controller, wrapper,
configuration, workflows, documentation, and tests. Generated reports and caches are
ignored artifacts. No production state, schema, data, or runtime process requires
restoration.
