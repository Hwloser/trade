# BTC Observation and Analysis UI v2 Tasks

This architecture child is design and migration-preparation only. Tasks that
change production behavior are intentionally delegated to separately governed
child changes after this design receives strict approval and user confirmation.

## 1. Audit and design approval

- [x] 1.1 Audit and review actual BTC source evidence across Web, BFF, query, DTO, style, and test paths plus [validates:btc-ui.compatibility] [validates:btc-ui.task-flow] [validation:review]
  the architecture parent and existing BTC OpenSpec changes. Objective: establish
  the real implementation baseline and avoid duplicating the K-line, viewport,
  Catalog, or H1 designs. Inputs: `ObservatoryPage.tsx`, `MarketWorkspace.tsx`,
  Observatory components/helpers/tests, `router.py`, query facade/SDK, parent
  architecture artifacts, and three existing BTC changes. Outputs: current-state
  evidence and supersession/compatibility decisions in proposal/design. Affected
  contracts: Observatory URL, HTTP, selected-snapshot, K-line, assurance, H1,
  Data-page BTC consumer. Validation: source paths and line-count/route/contract
  inventories are cited and cross-checked against active changes. Rollback:
  documentation-only correction. Completion evidence: the Current-State Audit
  and Relationship sections identify existing owners and non-duplicated scope.
- [x] 1.2 Author and review governed design evidence: proposal, Design Quality Brief, three capability specs, target [validates:btc-ui.task-flow] [validates:btc-analysis.product-boundary] [validates:btc-ui.bff-boundary] [validates:btc-ui.temporal-coherence] [validation:review]
  dependency/runtime diagrams, budgets, rollout/rollback, child-change plan, and
  governance marker. Objective: define an implementable UI architecture without
  changing production behavior. Inputs: task 1.1 audit and repository design
  policy. Outputs: complete governed artifacts under this change. Affected
  contracts: WorkspaceContext, AnalysisSnapshotRef, MetricObservation, BFF
  slices, URL/API/SDK compatibility, responsive Web behavior. Validation:
  OpenSpec artifacts contain normative scenarios and every selected policy
  profile has concrete evidence. Rollback: edit or remove only this design
  change before approval. Completion evidence: proposal/design/specs/tasks and
  `design-quality.toml` exist and map every obligation.
- [x] 1.3 Run `openspec validate btc-observation-analysis-ui-v2 --strict`, [validates:btc-ui.compatibility] [validates:btc-analysis.product-boundary] [validates:btc-ui.bff-boundary] [validates:btc-ui.temporal-coherence] [validation:test]
  `./trade dev design-check btc-observation-analysis-ui-v2`, and
  `git diff --check`; resolve every deterministic blocker and own every warning.
  Objective: prove native schema and policy completeness before semantic review.
  Inputs: frozen artifacts from task 1.2. Outputs: zero-issue native validation,
  non-strict design report, artifact/policy digests. Affected contracts: all
  three capabilities. Validation: commands exit zero and report the intended
  `contract`, `forecast`, and `concurrency` profiles. Rollback: correct only
  governed design artifacts and rerun. Completion evidence: command output and
  task checkmark record the exact result.
- [ ] 1.4 Run the six-role design review in a separate review worktree for [validates:btc-ui.task-flow] [validates:btc-analysis.product-boundary] [validates:btc-ui.bff-boundary] [validates:btc-ui.temporal-coherence] [validation:review]
  reliability, performance, architecture, data quality, observability, and
  news/future integration. Objective: challenge UI semantics, temporal
  correctness, data ownership, workload bounds, compatibility, and future
  event-data isolation. Inputs: current artifact digest and cited source
  baseline. Outputs: six file/line reports and one consensus with P0/P1/P2
  disposition. Affected contracts: all three capabilities. Validation: every
  judge returns a substantive report; contradictions receive at most one
  reconciliation round; all findings are evidence-backed. Rollback: discard the
  review-only worktree and no review evidence is fabricated. Completion
  evidence: reports and consensus are retained outside governed artifact digest.
- [ ] 1.5 Resolve all P0 and material P1 findings, rerun the non-strict check, [validates:btc-ui.compatibility] [validates:btc-analysis.product-boundary] [validates:btc-ui.bff-boundary] [validates:btc-ui.temporal-coherence] [validation:review]
  record current digest-bound `design-review.toml`, and run
  `./trade dev design-check btc-observation-analysis-ui-v2 --strict`.
  Objective: obtain genuine design approval before any child implementation.
  Inputs: task 1.4 consensus. Outputs: resolved design/spec/task evidence,
  approved six-role record, strict exit zero. Affected contracts: all three
  capabilities. Validation: zero unresolved P0, every material P1 is resolved or
  assigned to a named child, and strict approval binds the current date, commit,
  policy, and artifact digest. Rollback: modify design, invalidate old approval,
  and repeat review. Completion evidence: strict PASS and approved digest.
## 2. Compatibility and test baselines

- [ ] 2.1 Prepare a frozen compatibility inventory for the later implementation [validates:btc-ui.compatibility] [validates:btc-ui.bff-boundary] [validation:test]
  changes. Objective: make all existing BTC consumers and public behaviors
  testable before delegation. Inputs: current URL parser/serializer, capability
  gate, OpenAPI, Observatory response fixtures, Data-page BTC adapter, SDK, K-line
  and viewport tests. Outputs: proposed URL/HTTP/SDK/BFF/OpenAPI golden fixture
  matrix and consumer/retirement map. Affected contracts: all existing
  `obs*` selectors, `/api/v1/observatory/*`, capability/ETag/error behavior,
  Data-page selected series, notebook/SDK. Validation: every route/method/input/
  status/payload/SSE-or-none/capability/error shape and frontend consumer has a
  named snapshot or explicit no-test reason. Rollback: retain existing routes
  and tests unchanged. Completion evidence: the query-contract child proposal
  includes the frozen matrix and parity exit criteria.
- [ ] 2.2 Prepare the UI interaction and responsive acceptance matrix. [validates:btc-ui.task-flow] [validates:btc-ui.compatibility] [validation:test]
  Objective: turn the Observe/Analyze/Assurance/Research information architecture
  into testable behavior without coding it in this change. Inputs: information
  architecture spec, existing desktop/mobile E2E/a11y suites, current CSS/layout
  and chart constraints. Outputs: view/state/evidence/focus matrix plus
  360x800, 768x1024, 1280x800, and 1600x900 screenshot/canvas checks. Affected
  contracts: task navigation, evidence rail, controls, chart/table layout,
  loading/partial/unavailable/failed states. Validation: every normative UI
  scenario maps to a component/contract test or Playwright visual/a11y check;
  overlap and nonblank canvas checks are explicit. Rollback: keep the legacy
  page selected. Completion evidence: the Web child proposal includes the
  matrix, budgets, and feature-flag rollback.
## 3. Datasets and Studies boundary preparation

- [ ] 3.1 Create `btc-descriptive-analysis-product-v1` as an independent [validates:btc-analysis.product-boundary] [validates:btc-ui.temporal-coherence] [validation:test]
  governed child after the prerequisite package/contracts and formal PIT
  changes are ready. Objective: implement the reusable BTC analysis product in
  Datasets, not React or the BFF. Inputs: DatasetSnapshotRef, formal PIT/revision
  policy, current BTC daily contract, metric-method table, immutable-ref rules.
  Outputs: child proposal/design/spec/tasks for method registry, build/version/
  snapshot/release, lineage, quality, query projection, repository/adapters, and
  rollback. Affected contracts: AnalysisSnapshotRef, MetricObservation,
  analysis series, Dataset lineage/revision. Validation: exact-decimal and
  temporal golden fixtures cover return, volatility, range, drawdown,
  percentile, coverage, missing/quarantine/revision/minimum samples, identical
  input determinism, tamper rejection, and no moving inputs. Rollback: select
  the prior verified analysis release or disable analysis capability while
  retaining immutable artifacts. Completion evidence: child strict design
  approval defines one Dataset owner and no UI/BFF business calculation.
- [ ] 3.2 Prepare the Studies relationship contract in the analysis product or [validates:btc-analysis.product-boundary] [validates:btc-ui.temporal-coherence] [validation:test]
  study-boundary child. Objective: prevent forward labels, inference, validation,
  and promotion from leaking into descriptive Analyze metrics. Inputs: H1
  receipt, Study lifecycle parent spec, Dataset/Analysis refs, revision flow.
  Outputs: StudyResultRef relationship DTO, stale/evidence-gap states, and
  forbidden-field/dependency test plan. Affected contracts: Research slice,
  Study result input refs, stale-after-revision mapping. Validation: contract
  tests reject moving/raw/provider inputs and reject forecast/recommendation
  fields in the descriptive analysis schema; deterministic Study fixtures retain
  their own method/sample/uncertainty/lifecycle. Rollback: retain legacy H1
  rendering and keep new relationship unavailable. Completion evidence: the
  named child owns every inferential field in Studies.
## 4. BFF and SDK boundary preparation

- [ ] 4.1 Create `btc-workspace-query-contracts-v1` as an independent governed [validates:btc-ui.bff-boundary] [validates:btc-ui.compatibility] [validates:btc-ui.temporal-coherence] [validation:test]
  child. Objective: introduce WorkspaceContext and active-view query slices
  through Interfaces while preserving legacy Observatory contracts. Inputs:
  approved public refs/DTOs, Dataset/Study query contracts, current router/facade,
  URL/OpenAPI matrix and query budgets. Outputs: versioned HTTP/SDK DTOs,
  context/observe/analyze/assurance/lineage/research/evidence endpoints,
  compatibility mapper, capability bits and ErrorEnvelope mapping. Affected
  contracts: additive workspace API, legacy Observatory HTTP/ETag/error behavior,
  SDK/notebook semantics. Validation: OpenAPI/DTO/legacy snapshots, context-first
  topology, same-identity ETag, mismatch rejection, partial-owner errors and
  Web/SDK parity. Rollback: disable additive route registration and select the
  legacy adapter. Completion evidence: child strict approval and dark-launch
  dual-read plan with no legacy removal.
- [ ] 4.2 Define and validate read-only, deadline, cancellation and capacity [validates:btc-ui.bff-boundary] [validates:btc-ui.temporal-coherence] [validation:test]
  guards in the query-contract child. Objective: ensure the BFF cannot become a
  workflow or unbounded aggregation owner. Inputs: owner query ports,
  QueryBudget, Platform status/error APIs, current resource cancellation,
  specified 1x/10x envelopes. Outputs: instrumented read-only integration
  fixtures and capacity-result schema. Affected contracts: every workspace GET,
  cache identity, timeout ErrorEnvelope, response budgets, telemetry. Validation:
  provider/write/repair/publish/Study-run spies fail the read; disconnect and
  deadline terminate owned work; four-request, 2 MiB, 2,000-point, 100-metric,
  pagination and 10x overload cases remain bounded and preserve partial reports.
  Rollback: disable the new BFF; no business state exists to migrate. Completion
  evidence: child tests distinguish timeout/tool error from unavailable product
  and show zero residual query owners.
## 5. Web and compatibility cutover preparation

- [ ] 5.1 Create `btc-observation-analysis-web-v2` as an independent governed [validates:btc-ui.task-flow] [validates:btc-ui.compatibility] [validates:btc-ui.bff-boundary] [validation:test]
  frontend child after query contracts are approved. Objective: implement the
  four-task UI with no business calculation or compatibility break. Inputs:
  approved BFF DTOs, current K-line/viewport components, UI acceptance matrix,
  existing URL/capability helpers. Outputs: focused page containers, typed
  resource clients, compact context header, Analyze views, evidence rail,
  responsive styles and feature gate. Affected contracts: Web navigation, URL,
  request topology, visual/a11y behavior, K-line compatibility. Validation:
  Vitest/typecheck/build/bundle plus Playwright functional/a11y/visual/canvas
  checks at all target viewports, rapid-switch cancellation, independent
  failures, and a static/contract guard proving Analyze does not consume raw
  OHLCV for metric calculation. Rollback: turn off V2 and restore the legacy
  page bundle without data cleanup. Completion evidence: child strict approval,
  focused green tests, and implemented-diff six-role review.
- [ ] 5.2 Create `btc-workspace-compatibility-cutover` only after analysis, [validates:btc-ui.compatibility] [validates:btc-ui.temporal-coherence] [validation:test]
  query, and Web children meet their independent exit criteria. Objective:
  switch Observe/Assurance/Research composition and V2 default safely while
  keeping old clients and the Data-page BTC adapter working. Inputs: dual-read
  reports, compatibility snapshots, feature/capability telemetry, rollback
  runbook. Outputs: staged percentage/local gate, monitored default, consumer
  inventory, and later retirement proposal. Affected contracts: legacy and V2
  HTTP, URL, SDK, Data page, page assets. Validation: parity, PIT identity,
  read-only, error/status, latency/payload, accessibility and rollback drills
  pass before each stage. Rollback: select legacy routes/page and previous
  verified Dataset release; retain immutable facts. Completion evidence: one
  reviewable cutover PR with no legacy deletion.
## 6. Final handoff audit

- [ ] 6.1 Review child-proposal evidence against this design before any [validates:btc-ui.task-flow] [validates:btc-analysis.product-boundary] [validates:btc-ui.bff-boundary] [validates:btc-ui.compatibility] [validates:btc-ui.temporal-coherence] [validation:review]
  implementation starts. Objective: prove ownership, dependency order,
  compatibility and rollback remain complete after child decomposition. Inputs:
  child governed artifacts and current architecture parent. Outputs:
  prompt-to-artifact checklist covering every capability requirement, DTO,
  route, budget, test, gate, risk and rollback. Affected contracts: all
  capabilities and child boundaries. Validation: uncertainty is treated as not
  ready; every requirement maps to concrete child owner/task/evidence and no
  child depends on an unapproved future big-bang move. Rollback: revise the
  child or this design, invalidate review digest and regain strict approval.
  Completion evidence: dependency audit records package/contracts and formal
  PIT prerequisites before Datasets/BFF/Web cutover.
- [ ] 6.2 Review final handoff evidence and confirm this design change modified no production code, real data, [validates:btc-ui.compatibility] [validation:review]
  schema, generated artifact or runtime behavior and record the approved handoff.
  Objective: close the design round without accidentally beginning
  implementation. Inputs: git diff/status, strict report, review consensus and
  child order. Outputs: final design status and user decisions/confirmation
  request. Affected contracts: none at runtime. Validation: only files under
  this OpenSpec change and an intentional architecture-parent task reference, if
  added, appear in the design diff; production tests are not claimed as changed.
  Rollback: revert documentation commits only. Completion evidence: clean
  worktree after commit, production-code-changed=no, and implementation-ready
  only if strict approval passes.
