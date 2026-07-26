## 1. Governed Architecture Design

- [x] 1.1 Audit current repository structure, `AGENTS.md`, OpenSpec workflow,
  CLI facade, Python packaging, Web routes/pages, EventBus, database/schema,
  artifact stores, Observatory/PIT, notebooks and C++ boundary without mutating
  real data. Objective: establish code facts rather than rely on historic
  documents. Inputs: current source tree and OpenSpec policy. Outputs:
  `current-state-inventory.md`, `file-ownership-map.md`,
  `table-and-artifact-ownership.md`, `interface-compatibility-matrix.md` and
  summarized evidence in `design.md`. Affected contracts: all retained public
  surfaces. Validation: source-only audit, route/CLI registry reconciliation
  and `git status -sb` preservation check. Rollback: none because no runtime
  state changes. Completion evidence: audited paths, 56-table/eight-artifact
  ledger, 13+8 CLI registry and 68/77 HTTP registry are cited from the governed
  design. [validation:test]

- [x] 1.2 Write proposal, Design Quality Brief, target architecture, dependency [validates:architecture.boundaries] [validation:test]
  graph, runtime diagrams, compatibility matrix, ownership map, risks, rollback
  and child-change plan. Objective: create an implementation-independent design
  baseline. Inputs: audited facts and `design-policy/v1.toml`. Outputs:
  governed OpenSpec proposal, design, tasks, four audit attachments and ten
  capability specs. Affected contracts: architecture and dependency rules.
  Validation: OpenSpec scenario, attachment reconciliation and Design Quality
  Brief completeness check. Rollback: edit only governed design artifacts
  before review. Completion evidence: required artifacts exist under the
  change directory and every attachment is referenced by `design.md`.
  [validates:architecture.boundaries]
  [validation:test]

- [x] 1.3 Run `./trade dev design-check restructure-trade-architecture-v1`, [validates:architecture.boundaries] [validates:migration.governance] [validation:test]
  resolve deterministic blockers and assign
  every warning to design, task or explicit future follow-up. Objective: make
  the initial design evidence machine-valid. Inputs: proposal, design, specs,
  tasks and quality declaration. Outputs: passing non-strict report. Affected
  contracts: migration governance and architecture boundaries. Validation:
  design-check output and `git diff --check`. Rollback: correct only the
  governed artifacts; no source/data rollback exists. Completion evidence:
  zero blocker report saved in the design review record. [validates:architecture.boundaries]
  [validates:migration.governance] [validation:test]

- [ ] 1.4 Run the required six-role design review from an isolated review
  worktree and synthesize architecture, reliability, performance, data-quality,
  observability and news/future findings. Objective: challenge the design
  before strict approval. Inputs: frozen design artifact generation and real
  code references. Outputs: consensus report and candidate `design-review.toml`
  evidence. Affected contracts: all architecture capabilities. Validation:
  review evidence contains file/line findings, consensus counts and P0/P1/P2
  disposition. Rollback: discard review-only worktree; no production state is
  changed. Completion evidence: six judge reports and a synthesized finding
  list. [validation:review]

- [ ] 1.5 Resolve every P0 and material P1 design finding, refresh the
  non-strict design check, record digest-bound review approval and run strict
  design-check. Objective: make the architecture implementable but still
  design-only. Inputs: review consensus and current artifact digest. Outputs:
  approved `design-review.toml` and strict result. Affected contracts: all
  architecture capabilities. Validation: `./trade dev design-check
  restructure-trade-architecture-v1 --strict`. Rollback: revise governed
  artifacts and repeat review when the digest changes. Completion evidence:
  strict exit code zero and zero unresolved P0. [validation:review]

## 2. Foundational Prerequisites for Child Changes

- [ ] 2.1 Prepare `architecture-guardrails-and-baselines` as an independent [validates:architecture.boundaries] [validates:dependency.guardrails] [validation:test]
  child OpenSpec change. Objective: define static import guard, contract type
  leakage guard, DB-owner guard and baseline inventories before any module
  extraction. Inputs: dependency graph, current imports and table inventory.
  Outputs: scoped child proposal/tasks and baseline test plan. Affected
  contracts: context imports and table ownership. Validation: proposed
  architecture test validates allowed/forbidden import samples and DB owner
  fixtures. Rollback: remove only guardrail additions if they prove invalid; do
  not alter existing source ownership. Completion evidence: child scope maps
  each rule to a current path and a test fixture. [validates:architecture.boundaries]
  [validates:dependency.guardrails] [validation:test]

- [ ] 2.2 Prepare `kernel-and-public-contracts` as an independent [validates:architecture.boundaries] [validates:processes.recovery] [validates:interfaces.compatibility] [validation:test]
  prerequisite child OpenSpec change. Objective: establish framework-free IDs,
  envelope/command/query DTOs, immutable refs/policy identities, trusted
  ActorContext, OperationReceipt, ProcessView, ErrorEnvelope and explicit
  status taxonomy before Platform or interface extraction consumes them.
  Inputs: current CLI/HTTP/scheduler/event mutation audit and immutable-ref
  inventory. Outputs: canonical serialization/versioning policy, compatibility
  import bridge and DTO fixture plan. Affected contracts: all public command,
  receipt, error, ref and process-view shapes. Validation: round-trip,
  forbidden-framework, trusted-actor provenance, unknown/not-observed/
  unavailable distinction and legacy snapshot tests. Rollback: stop new
  consumers and retain legacy DTO/import paths. Completion evidence: Platform
  can consume public DTOs without introducing a Context repository or framework
  type. [validates:architecture.boundaries] [validates:processes.recovery]
  [validates:interfaces.compatibility] [validation:test]

- [ ] 2.3 Prepare `platform-persistence-events-and-bootstrap-foundation` as [validates:platform.foundation] [validates:processes.recovery] [validates:migration.governance] [validation:test]
  an independent prerequisite child OpenSpec change. Objective: supply the
  transaction/outbox port, command ingress/OperationReceipt, inbox/lease/ack/
  DLQ and OrderingContract delivery, EventBus/LegacySchemaBootstrapAdapter,
  DatabaseRuntime/MigrationCoordinator, CapacityEnvelope, verified restore and
  Bootstrap composition before any Context relies on cross-context delivery.
  It consumes the prior Kernel/public contracts and does not require
  unextracted Context repositories. Inputs: EventBus, `TradeDB`, migrations,
  runtime resources, backup audit and public DTOs. Outputs: generic public
  APIs, compatibility bridge, crash/mixed-version/restore fixture plan,
  capacity result schema and one Bootstrap-owned shutdown lifecycle for CLI,
  Web, workers and schedulers. Affected contracts: event envelope, persistence
  transaction, operation receipt, migration capability, runtime composition
  and shutdown receipt.
  Validation:
  crash-after-commit, duplicate ingress, inbox dedup, lease recovery, DLQ,
  N+1-before-N ordering gap, mixed-binary fence, staged corrupted-backup
  rejection, activation/rebind rollback, repeated stop, stuck child-process
  TERM/KILL escalation, executor/heartbeat drain, database-close ordering and
  1x/10x backlog tests. Rollback:
  select the legacy EventBus/TradeDB construction bridge without deleting
  outbox, receipt or restore evidence. Completion evidence: no Context child
  has to invent an atomic outbox, command handoff, runtime container or
  non-comparable capacity report. [validates:platform.foundation]
  [validates:processes.recovery]
  [validates:migration.governance] [validation:test]

- [ ] 2.4 Prepare `formal-pit-and-revision-semantics` as an independent [validates:datasets.products] [validates:studies.reproducibility] [validation:test]
  prerequisite child change. Objective: make required clocks fail closed and
  implement real as-known/latest-restated mapping before a formal SnapshotRef
  or Study migration. Inputs: existing ArtifactRef, SnapshotContext, PIT
  resolver and research workflow audit. Outputs: PIT/revision contract and
  golden fixture plan. Affected contracts: DatasetSnapshotRef, policy refs,
  StudyResultRef and evidence-gap event. Validation: raw-input rejection,
  null-clock/revision/timezone goldens, insufficient-data and deterministic
  rerun tests. Rollback: retain legacy non-formal reader and block formal
  release/run rather than expose an unproven snapshot. Completion evidence:
  revision/retraction mapping and all formal ref policy identities are
  verifier-visible; no missing clock is visible. [validates:datasets.products]
  [validates:studies.reproducibility] [validation:test]

- [ ] 2.5 Prepare `cli-http-sdk-compatibility` as an independent child OpenSpec [validates:interfaces.compatibility] [validates:platform.foundation] [validation:test]
  change. Objective: freeze actual CLI help/parse/exit behavior, HTTP/OpenAPI/
  SSE route behavior, Web BFF payloads, Observatory capability semantics and
  notebook entry contracts before delegation. Inputs: root `trade`, CLI
  registries, FastAPI route inventory, React API consumers and current
  notebook. The current FastAPI application registers 72 routes in the
  default-off/error Observatory mode and 81 when its full data router is
  enabled, while schema generation fails on the unresolved local
  `PredictRequest` forward reference;
  therefore the child SHALL first freeze the registered route/method/signature
  table and golden payloads, then repair and add the OpenAPI snapshot without
  omitting `/predict`. Outputs: compatibility matrix, route-registry baseline,
  golden-response fixtures and OpenAPI repair/snapshot plan. Affected
  contracts: all retained interfaces. Validation: CLI, OpenAPI/SSE, BFF,
  ProcessView/ErrorEnvelope/status taxonomy, RetentionView/GC receipt and SDK
  contract snapshot tests against temporary roots. Rollback: keep legacy
  interface adapter selected until snapshot parity returns. Completion evidence:
  each legacy entrance and mutation has a named adapter, durable receipt/
  recovery path and retirement condition; BFF/SSE budgets cannot cause an
  unbounded client-specific poller or queue.
  [validates:interfaces.compatibility] [validates:platform.foundation]
  [validation:test]

## 3. Durable Product and Research Migration Preparation

- [ ] 3.1 Prepare `capture-boundary` implementation readiness for a pilot [validates:capture.receipts] [validates:platform.foundation] [validates:migration.governance] [validation:test]
  source after the Platform foundation and its child design are strictly
  approved. Objective: document context-owned capture tables/artifacts,
  SourceManifest rights/temporal/finality policy, provider ports,
  stage/digest/commit reconciliation, checkpoint/retry/quarantine/redrive
  policy and compatibility bridge without moving implementation in this parent
  change. Inputs: child contract, source rights audit and crypto run-store
  audit. Outputs: migration slice, additive schema plan, retention/tombstone
  plan, source/credential durable quota ledger, Retry-After/deadline and
  stream-buffer/checkpoint admission plan, capacity envelope and capture
  fixture matrix. Affected contracts: CaptureArtifactRef, QuarantineReceipt,
  rights-restriction propagation event and existing source commands.
  Validation: temporary-root replay, supersession, stream segment,
  no-provider replay, shared-credential concurrent-worker admission, rights
  revocation through retained lineage, absent publication time, quarantined
  access/revalidation, commit crash and 1x/10x admission tests defined in the
  child. Rollback: previous capture adapter and immutable prior artifacts.
  Completion evidence: child change has an owned migration/rollback design,
  policy digest, explicitly persisted quota/Retry-After evidence and code
  worktree plan. [validates:capture.receipts]
  [validates:platform.foundation] [validates:migration.governance]
  [validation:test]

- [ ] 3.2 Prepare `dataset-product-boundary` as an independent child OpenSpec [validates:datasets.products] [validates:migration.governance] [validation:test]
  change. Objective: define canonical build/version/snapshot/release, quality,
  lineage, canonicalization/quality/revision/clock/transform environment and
  physical-layout identities, QueryBudget, catalog rebuild and
  generation-stamped legacy pointer bridge for the same pilot source. Inputs:
  proven PIT/revision contract, Capture artifact contract, crypto run store and
  warehouse/catalog audit. Outputs: Dataset repository/migration/projection
  plan, reference verifier, SemanticSchemaPolicyRef and
  MigrationReconciliationManifest schema. Affected contracts:
  DatasetVersionRef, DatasetSnapshotRef, quality/PIT query and
  DerivationReceipt. Validation: lineage, source reconciliation, catalog
  rebuild, immutable build input, policy/clock/revision/reference tamper,
  manifest-verified formal and compatibility reads, deterministic and
  provider-backed derivation receipts, physical query-budget, pointer
  reconciliation and rollback fixtures.
  Rollback: restore verified prior release pointer and retain the newer
  immutable version for audit. Completion evidence: child proposal identifies
  the one Datasets transaction boundary per state transition and cannot release
  a SnapshotRef without formal PIT proof or expose an unverifiable artifact,
  derivation or schema-breaking semantic output as an existing DatasetVersion.
  [validates:datasets.products]
  [validates:migration.governance] [validation:test]

- [ ] 3.3 Prepare `study-boundary` implementation readiness after Dataset [validates:datasets.products] [validates:studies.reproducibility] [validation:test]
  contracts and the formal PIT/revision gate exist. Objective: specify one
  Study's preregistration, proven pinned snapshot input, feature
  classification, validation, promotion, stale result and evidence-gap flow.
  Inputs: Dataset snapshot/policy contract and current research workflow audit.
  Outputs: Study lifecycle migration plan and golden fixture matrix. Affected
  contracts: StudyResultRef and Decision Support read inputs. Validation: PIT
  proof rejection, raw-input rejection, deterministic rerun, revision
  staleness and insufficient-data tests. Rollback: preserve prior research
  query path and expose new outputs as unpublished/stale. Completion evidence:
  child proposal declares all metrics, horizon and unavailable semantics.
  [validates:datasets.products] [validates:studies.reproducibility]
  [validation:test]

- [ ] 3.4 Prepare `decision-support-boundary` implementation readiness after [validates:decision_support.audit] [validates:interfaces.compatibility] [validates:migration.governance] [validation:test]
  Study contracts exist and before Process/interface cutover. Objective:
  classify recommendation, causal decision, picks, actions, portfolio-intent,
  rationale, trust and override records file by file; establish DecisionCase,
  Review, Rationale, Override, PortfolioIntent, Expiry and AuditTrail ownership
  without adding trade execution. Inputs: DatasetSnapshotRef, StudyResultRef,
  existing recommendation/action/causal paths and compatibility snapshots.
  Outputs: owner-local repository/migration plan, evidence/staleness/expiry
  transition matrix, read-only query DTOs and legacy adapter plan. Affected
  contracts: DecisionCase views, reviews, overrides, non-executable intents and
  accepted/rejected/expired/stale compatibility states. Validation: immutable
  evidence rejection, stale/revision propagation, append-only override,
  expiry, GET read-only, audit correlation and unsupported-execution tests.
  Rollback: select the legacy recommendation/action read adapter, stop new case
  admission and retain append-only decision/audit facts. Completion evidence:
  no child treats Decision Support as Studies, a Web page, a global service or
  an execution context, and every migrated table/artifact has one writer.
  [validates:decision_support.audit] [validates:interfaces.compatibility]
  [validates:migration.governance] [validation:test]

- [ ] 3.5 Prepare `tests-and-legacy-cleanup` migration rehearsal criteria. [validates:migration.governance] [validation:test]
  Objective: define additive schema/version, old reader preservation,
  idempotent replay/shadow-copy, dual-read comparison, pointer switch and
  retirement checks for all later children. Inputs: table/artifact ownership map
  and platform backup behavior. Outputs: migration test harness and rollback
  checklist design. Affected contracts: SQLite/parquet readers, release
  pointers and legacy imports. Validation: migration rollback, old/new reader,
  reconciliation manifest, artifact digest, staged verified backup restore,
  writer-fence/activate/rebind/health-window rollback, protected-reference
  retention, GC dry-run/run receipts and projection rebuild tests. Rollback:
  restore previous generation or verified backup snapshot without deleting
  immutable records. Completion evidence: every durable child has a selected
  migration mode, mixed-version fence, RestoreOperation recovery state and
  rollback source.
  [validates:migration.governance] [validation:test]

## 4. Runtime and Interface Orchestration Preparation

- [ ] 4.1 Prepare `process-manager-boundary` after the Platform foundation as [validates:studies.reproducibility] [validates:processes.recovery] [validates:platform.foundation] [validation:test]
  an independent child OpenSpec change. Objective: define durable Process
  records and the normal refresh, evidence-gap, revision propagation,
  registered study, publication request, projection and daily workspace flows
  over the existing command/outbox substrate. Inputs: Platform foundation,
  EventBus/runtime/job/agenda audit and Context contracts. Outputs: process
  state schema, ActorContext/OperationReceipt/ProcessView, idempotency/
  deadline/compensation policy and temporary-root recovery fixtures. Affected
  contracts: commands, events, process receipts and schedule envelopes.
  Validation: duplicate delivery, crash-after-commit, inbox/lease recovery,
  partial fan-out, deadline, cancellation, DLQ redrive and replay tests.
  Rollback: disable new process command while retaining pending outbox/process
  facts for compatible recovery. Completion evidence: each process maps every
  step to a context command, `PublishDataset` remains a Datasets transaction,
  and there is no cross-context transaction. [validates:studies.reproducibility]
  [validates:processes.recovery] [validates:platform.foundation]
  [validation:test]

- [ ] 4.2 Prepare and validate `operational-sli-slo-alert-runbook-matrix` as an [validates:processes.recovery] [validates:platform.foundation] [validates:interfaces.compatibility] [validates:migration.governance] [validation:test]
  independent child OpenSpec change before any production cutover. Objective:
  bind every Platform, Process, Capture, Dataset and interface operational
  state to a versioned signal, SLI/SLO, threshold, owner, escalation and
  recovery runbook. Inputs: OperationReceipt, ProcessView, ErrorEnvelope,
  RetentionView, GC receipts, CapacityEnvelope and delivery/restore state
  contracts. Outputs: operational matrix, bounded status/query DTO inventory,
  synthetic-alert fixture plan and authorized recovery evidence requirements.
  Affected contracts: operation/process/retention views, status taxonomy,
  alert payloads and operator commands. Validation: synthetic alert,
  correlation drill-down, stale/unknown/blocked distinction, authorized
  recovery, retention forecast and error-envelope compatibility tests.
  Rollback: disable only the new alert/routing adapter while retaining durable
  receipts, views and recovery evidence. Completion evidence: each production
  cutover criterion names a measured signal, owner, escalation path and
  runbook rather than relying on an undocumented dashboard.
  [validates:processes.recovery] [validates:platform.foundation]
  [validates:migration.governance] [validation:test]

- [ ] 4.3 Prepare interface composition migration slices. Objective: select [validates:processes.recovery] [validates:interfaces.compatibility] [validation:test]
  low-risk CLI/HTTP/Web/SDK surfaces and route them through read-only query
  handles or command receipts, preserving existing compatibility adapters.
  Inputs: compatibility snapshots and Process/Platform contracts. Outputs:
  per-surface BFF/adapter sequence for Today, Observatory, Research, Data Ops
  and Operations before broader pages. Affected contracts: route payloads,
  SSE, page state and command receipts. Validation: BFF contract, GET
  read-only guard, bounded query, 1x/10x BFF/SSE shared-hub slow-client and
  compatibility snapshot tests. Rollback: route to the legacy adapter without
  removing URL/payload aliases. Completion evidence: no selected interface
  module directly queries an owner table, provider or lifecycle pointer, and
  unavailable/process errors map through the versioned compatible envelope.
  [validates:processes.recovery]
  [validates:interfaces.compatibility] [validation:test]

- [ ] 4.4 Prepare `btc-observation-analysis-ui-v1` as an independent child [validates:interfaces.compatibility] [validates:datasets.products] [validates:studies.reproducibility] [validation:test]
  after compatibility baselines and Dataset/Study query contracts. Objective:
  reorganize the existing BTC Observatory into Market, Quality, Research and
  Lineage work views without creating a business Context or changing old
  routes. Inputs: current capability gate, URL/local state, decimal-string DTOs,
  K-line/date/Trust/run/research components, BFF matrix and immutable query
  contracts. Outputs: `BtcWorkspaceView` contract or equivalent bounded batched
  BFF, responsive interaction design, route/payload compatibility adapter and
  page migration plan. Affected contracts: Observatory capability, `obsLens`
  deep links, granular BTC routes, snapshot identity, typed panel states and
  workspace cache/ETag metadata. Validation: disabled/error/ready capability,
  URL restore, old/new payload goldens, snapshot mismatch, partial/stale panel,
  decimal precision, bounded fan-out, keyboard/ARIA, desktop/mobile screenshot,
  no-overlap and chart nonblank pixel tests. Rollback: select the current
  four-lens page and granular endpoints while preserving URL state and immutable
  evidence. Completion evidence: every visible metric names source/ref and
  unavailable state; no UI/BFF query performs capture, repair, publication,
  Study execution or Decision transition.
  [validates:interfaces.compatibility] [validates:datasets.products]
  [validates:studies.reproducibility] [validation:test]

## 5. Final Design Approval and Handoff

- [ ] 5.1 Reconcile the approved architecture with all compatibility and [validates:interfaces.compatibility] [validates:dependency.guardrails] [validates:platform.foundation] [validation:test]
  dependency baselines. Objective: ensure the child-change order has no hidden
  import, table-owner or interface dependency. Inputs: completed design review,
  contract inventories and task graph. Outputs: final child-change ordering and
  named compatibility windows. Affected contracts: dependency guards and
  retained interfaces. Validation: architecture/import/contract snapshot
  review and `git diff --check`. Rollback: revise this architecture design and
  repeat review; no production changes are in scope. Completion evidence:
  every child can be implemented and rolled back independently, the Platform
  foundation consumes Kernel/public contracts before Context outbox use,
  formal PIT semantics precede formal SnapshotRef/Study migration, and the
  package-layout child is blocked on its approved distribution/import/native
  compatibility decision record. [validates:interfaces.compatibility]
  [validates:dependency.guardrails] [validates:platform.foundation]
  [validation:test]

- [ ] 5.2 Record digest-bound six-role design consensus and strict approval. [validates:migration.governance] [validates:platform.foundation] [validation:review]
  Objective: formally gate implementation until the current design is
  approved. Inputs: current artifact digest, non-strict report and six judges'
  file/line evidence. Outputs: `design-review.toml`, strict gate report and
  final status. Affected contracts: all migration governance obligations.
  Validation: consensus review resolves every P0, assigns material P1 items,
  then `./trade dev design-check restructure-trade-architecture-v1 --strict`
  exits zero. Rollback: alter governed artifacts only, regenerate the digest
  and repeat review if evidence changes. Completion evidence: six approved
  roles, zero P0 and a current strict approval record.
  [validates:migration.governance] [validates:platform.foundation]
  [validation:review]
