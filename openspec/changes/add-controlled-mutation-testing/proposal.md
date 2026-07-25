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

- Add canonical `trade dev mutation
  changed|core|full|inspect|repair|qualify|reconcile` developer commands, with
  `scripts/mutation-test` retained as an all-command argv/output/exit-compatible
  facade and a closed versioned error/remediation registry.
- Add a typed Python controller under `trade_py/devtools/mutation_testing/` that:
  - keeps CLI parsing/rendering in `cli.py`, top-level use-case coordination in
    `application.py`, adapters behind an explicit complete acyclic import graph, and
    every OS spawn primitive exclusively in the supervisor;
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
    the non-spawning controller uses a per-run-capability-authenticated, numerically
    bounded single-session protocol with opaque worker handles; the shared channel MAC
    protects peer/replay integrity but is not treated as source attestation. The
    supervisor alone owns the writable descriptor for a bounded hash-chained
    attestation journal, while controller Landlock policy denies writes to receipt,
    attestation, and fallback roots. It records a child immediately after OS creation,
    before registration, and contains and observes cleanup even when registration or
    the controller fails;
  - denies network, process launch, and real-data paths with Linux Landlock/seccomp
    containment, transfers a seccomp listener over `SCM_RIGHTS`, brokers allowed
    opens with `openat2` plus descriptor injection, and requires independent
    controller-owned syscall audit and supervisor-owned child guard verification,
    including an explicit listener-drain ownership handoff and tightly constrained
    forced-close attestation after timeout/cancellation, before a mutant can be
    killed, survived, or truthfully timed out;
  - distinguishes killed, survived, timeout, line no-coverage,
    baseline-unavailable, cancellation, invalid, and infrastructure-error outcomes;
    integrity evidence is sticky and dominant after notification drain, while
    unconfirmed cleanup is an orthogonal status that replaces any provisional
    killed/survived result with a phase-appropriate infrastructure terminal. Reports
    preserve factual numerator/denominator, but an integrity, cleanup, degraded-run,
    invalid-report, or zero-denominator condition makes the displayed score null and
    prohibits cache, trend, or baseline use;
  - publishes deterministic JSON, Markdown, and HTML as one bounded atomic run
    generation with closed status/count algebra, a manifest hash anchored by the
    invocation receipt/current pointer, and an independent typed fallback if
    publication itself fails;
  - commits cache outcomes only after report publication, keeps report publication
    sequence separate from core/full trend sequence, and derives bounded trend views
    from a hash-chained immutable trend-source ledger with explicit reconciliation;
  - validates and packages the exact receipt plus referenced generation/fallback as
    one digest-bound `trade.mutation.bundle.v1` CI artifact, including the complete
    supervisor attestation journal; the execution job uploads it and a separate job
    downloads and independently validates the bytes and publishes the sole trusted
    `trade.mutation.ci-summary.v1`. Operators can inspect downloaded bundles without
    the original runner and reconcile an explicit downloaded trend carrier through a
    dry-run-first, approval-bound route.
- Add `config/mutation-testing.toml`, an initial unestablished baseline, and a
  location-precise equivalent-mutant exception registry. Add finite reviewed
  controller/baseline memory bootstrap caps, one immutable trend genesis marker per
  configuration epoch, and an identity- and freshness-bound capacity qualification
  contract for scheduled modes.
- Add focused controller, selection, process-isolation, reporting, baseline, and CLI
  tests using temporary repositories and synthetic source.
- Add optional locked mutation dependencies without changing the existing pytest
  framework or runtime dependencies.
- Add GitHub Actions workflows:
  - pull requests plan `changed` only when eligible production code changed and run
    it only after runner readiness plus an exact cold-cache changed smoke; changed
    mode never restores remote result cache;
  - nightly runs may execute `core`, and weekly runs may execute `full`, only after
    representative capacity qualification;
  - capacity-qualified manual core/full runs remain factual diagnostics and never
    advance shared trend sequence/high-water or publish shared cache/aggregate
    carriers;
  - all mutation outcomes are initially report-only, while an independent evidence
    validator fails on missing, malformed, unsafe, or receipt-unbound bundles;
  - scheduled modes require a reviewed, unexpired capacity qualification and carry
    trend history through validated immutable rolling aggregate artifacts rather than
    treating result cache as evidence. Only one reviewed unused genesis marker may
    establish the first no-predecessor checkpoint for an epoch;
  - every execution route uses a reviewed disposable `trade-mutation-v1` runner with
    writable delegated cgroup v2, finite memory, Landlock and seccomp notification;
    GitHub repository owner `huanwei1208` is accountable for a prerequisite runner-
    provisioning child change, named platform operator, and three-run readiness proof;
    until it and capacity qualification pass, workflow execution remains disabled and
    hosted CI is plan-only;
  - native GitHub concurrency prevents simultaneous long runs but may supersede an
    older pending request, which is reported honestly rather than described as a
    durable queue;
  - when the validation job is scheduled, its explicit valid/invalid/missing/hard-loss
    decision table and nullable controller exit produce the trusted summary. Whole
    workflow cancellation before that job starts remains an explicitly best-effort
    observability boundary.
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
  cancellation algebra, finite first-measurement memory admission, partial reporting,
  committed cache/trend artifacts, scheduled-only shared trend authority, bounded
  raw/detail versus compact-trend retention, sealed Git scope, independently
  reconstructable coverage/baseline evidence, bounded diagnosed remote restore and
  producer quotas, exact mutant exceptions, reviewed trend genesis, downloaded-bundle
  inspection, and artifact-aware trend reconciliation.
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
fresh/cache/plan/cancellation status algebra, supervisor-verified guard and forced-
close attestation in a supervisor-only write-protected journal, spawn-before-
registration cleanup accounting, finite bootstrap memory caps, score-eligibility
precedence, committed cache markers, separate report/trend sequence reservations and
tombstones, invocation-kind-aware raw-evidence tombstones plus separately retained
compact trend anchors, representative identity-bound capacity qualification, no
changed restore phase, scheduled-only shared trend ownership, reviewed genesis,
bounded diagnosed remote restore and producer quotas with invalid-predecessor epochs
and monotonic carriers, immutable cross-job CI bundles, explicit hard-loss summaries,
artifact-aware inspection/reconciliation, corrupt-pointer recovery, and exact mutant-
ID exception validation address those
risks. Linux cannot
guarantee userspace reaping of an indefinitely uninterruptible task or completion of
a stuck filesystem `fsync`; those cases remain `cleanup_unconfirmed` or lack a
claimed generation and are contained by the larger CI runner timeout, never counted
as kills.

Rollback removes the optional mutation dependency group, controller, wrapper,
configuration, workflows, documentation, and tests. Generated reports and caches are
ignored artifacts. No production state, schema, data, or runtime process requires
restoration.
