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

- Add the `scripts/mutation-test changed|core|full` developer entrypoint.
- Add a typed Python controller under `trade_py/devtools/mutation_testing/` that:
  - pins Cosmic Ray 8.4.6 and uses only `get_operator` plus `mutate_code` for
    deterministic first-order
    Python mutation generation;
  - applies exactly one position-verified mutation to a worker-owned temporary
    source tree and proves pytest imported that tree;
  - selects only the closed v1 definition-to-test matrix and requires baseline line
    coverage before mutant execution;
  - prioritizes changed lines, then changed definitions, then configured core paths;
  - enforces mutant, candidate-scan, worker, per-mutant, output, and wall-clock
    budgets;
  - lets the controller process directly own and terminate every mutant test process
    group and descendant on timeout, worker failure, signal, or global cancellation;
  - denies network, provider credentials, and real-data paths in mutation workers;
  - distinguishes killed, survived, timeout, mapping/line no-coverage,
    baseline-unavailable, invalid, and infrastructure-error outcomes;
  - publishes deterministic JSON, Markdown, and HTML as one bounded atomic run
    generation, with an independent safe fallback if publication itself fails.
- Add `config/mutation-testing.toml`, an initial unestablished baseline, and a
  location-precise equivalent-mutant exception registry.
- Add focused controller, selection, process-isolation, reporting, baseline, and CLI
  tests using temporary repositories and synthetic source.
- Add optional locked mutation dependencies without changing the existing pytest
  framework or runtime dependencies.
- Add GitHub Actions workflows:
  - pull requests plan and run `changed` only when eligible production code changed;
  - nightly and manual runs execute `core`;
  - weekly or explicit manual runs execute `full`;
  - all mutation jobs are initially non-blocking and upload reports.
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
  limits, process-tree termination, partial reporting, comparable trend artifacts,
  and exact mutant exceptions.
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
test-to-source mapping, false kills from infrastructure failures, and score gaming
through broad exclusions. Invocation-wide deadlines, parent-owned process groups,
source/AST/dependency/private-tree limits, a serialized spawn/cancel registry,
bytecode-clean private source trees, mandatory line coverage, closed
operator/definition maps, bounded generation-atomic output plus safe fallback, exact
fresh/cache/plan status algebra, and mutant-ID exception validation address those
risks.

Rollback removes the optional mutation dependency group, controller, wrapper,
configuration, workflows, documentation, and tests. Generated reports and caches are
ignored artifacts. No production state, schema, data, or runtime process requires
restoration.
