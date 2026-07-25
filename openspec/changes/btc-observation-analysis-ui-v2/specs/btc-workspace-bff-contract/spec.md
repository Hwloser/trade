## ADDED Requirements

### Requirement: Workspace context is resolved before dependent slices

The BTC workspace BFF SHALL first resolve one read-only `WorkspaceContext`
containing asset contract identity, selected lifecycle channel, selected market
DatasetSnapshotRef, effective knowledge cut, knowledge mode, revision policy,
watermarks, purpose fitness, context version, and request/correlation identity.
Observe, Analyze, and selected-snapshot Assurance queries SHALL require that
context identity. Research and catalog-wide lineage SHALL expose their separate
scope explicitly.

#### Scenario: Workspace opens
- **WHEN** a client requests a supported BTC workspace context
- **THEN** the BFF resolves the context through Datasets contracts or the legacy Observatory compatibility query
- **AND THEN** it returns one immutable selected-market reference plus the effective temporal/revision selectors
- **AND THEN** no dependent owner query starts until that context is confirmed

#### Scenario: Dependent query supplies a stale context
- **WHEN** an Observe, Analyze, or selected-snapshot Assurance query supplies a context id or market reference different from the active selection
- **THEN** the BFF returns `workspace_context_mismatch`
- **AND THEN** it does not substitute the newest context or combine responses from both identities

### Requirement: BFF composes owner query contracts without owning business facts

The BFF SHALL call context-owned read/query contracts and map their DTOs into
versioned workspace payloads. It MAY perform authorization, bounded parallel
composition, identity verification, response shaping, cache metadata, and
status/error mapping. It SHALL NOT query owner tables directly, read parquet or
artifact paths, call providers, calculate analysis metrics, repair data, build
or publish Datasets, execute Studies, move lifecycle pointers, or write business
state.

#### Scenario: Analyze slice is composed
- **WHEN** the client requests Analyze for a confirmed context
- **THEN** the BFF resolves the compatible AnalysisSnapshotRef through Datasets query contracts
- **AND THEN** it verifies immutable-reference and lineage relationships before returning metrics and series descriptors
- **AND THEN** no metric is derived from the Observe OHLCV response inside the BFF

#### Scenario: Research slice is composed
- **WHEN** the client requests Research
- **THEN** the BFF queries Studies for registered hypotheses and StudyResultRef values
- **AND THEN** it maps the input snapshot relationship to the active workspace without changing the Study result
- **AND THEN** it does not start a StudyRun or close an EvidenceGap

#### Scenario: Read-only guard observes a write
- **WHEN** a workspace GET attempts a database write, artifact write, provider call, repair, publish, or lifecycle transition
- **THEN** the read-only guard fails the request and records a safe operational diagnostic
- **AND THEN** no partial business mutation is committed

### Requirement: Query slices are versioned and view-local

The BFF SHALL expose versioned, bounded query DTOs for `context`, `observe`,
`analyze`, `assurance`, `lineage`, `research`, and `evidence`. The client SHALL
request only the active view and explicit drill-down. Hidden views SHALL NOT be
prefetched by default. A composite all-page response SHALL NOT be required for
correctness.

#### Scenario: Observe is active
- **WHEN** the client opens Observe
- **THEN** it requests context followed by the Observe slice and only an explicitly selected date's evidence
- **AND THEN** it does not request Analyze, Assurance detail, Lineage, or Research payloads

#### Scenario: Analyze is active
- **WHEN** the client opens Analyze
- **THEN** it requests context followed by the bounded analysis summary and only the selected metric series or evidence detail
- **AND THEN** it does not load every rolling series, every window, or every evidence reference eagerly

#### Scenario: User switches views rapidly
- **WHEN** a client changes the active view before a slice response completes
- **THEN** the old request is cancelled where supported and ignored by complete request identity otherwise
- **AND THEN** only the active view's matching response can enter confirmed state

### Requirement: Complete request identity prevents stale truth

Each request identity SHALL include endpoint/contract version, asset id, selected
channel, WorkspaceContext id, immutable market or analysis reference, effective
knowledge cut, knowledge mode, revision policy, view, selector/range, evidence
target, and refresh generation as applicable. Cache reuse and conditional
requests SHALL be allowed only for the same semantic identity.

#### Scenario: Channel changes during an ETag revalidation
- **WHEN** the prior observed response is revalidating and the user switches to Formal
- **THEN** the observed response cannot satisfy, populate, or remain labelled as the Formal request
- **AND THEN** the new context resolves before Formal-dependent slices begin

#### Scenario: Server returns not modified
- **WHEN** a conditional request receives `304 Not Modified`
- **THEN** the client reuses payload bytes only when the cached semantic identity and schema version exactly match the active request
- **AND THEN** a missing or mismatched cache entry becomes unavailable or a full retry, never a cross-identity fallback

### Requirement: Failure and freshness states are structured

Workspace payloads and errors SHALL distinguish transport failure, capability
denial, context unavailable, product unavailable, partial product, stale product,
identity mismatch, PIT not proven, quality blocked, query-budget exceeded, and
unsupported selector. Every error SHALL use the versioned ErrorEnvelope or its
legacy-compatible mapping with safe message, stable code, retryability,
correlation id, and bounded evidence references.

#### Scenario: Analysis product has not been built
- **WHEN** the active market snapshot has no compatible analysis product
- **THEN** Analyze receives `analysis_product_unavailable` or `analysis_pending_for_snapshot`
- **AND THEN** Observe and selected-snapshot Assurance remain independently available
- **AND THEN** the BFF does not treat the condition as a killed request, success, or zero metrics

#### Scenario: One owner query times out
- **WHEN** one requested slice exceeds its bounded deadline
- **THEN** that slice returns a timeout ErrorEnvelope and releases its resources
- **AND THEN** already confirmed independent slices keep their own identities and states
- **AND THEN** the server does not continue an unbounded background query solely for the disconnected client

#### Scenario: Workspace process is shutting down
- **WHEN** the serving process receives shutdown while workspace queries or owned child processes are active
- **THEN** admission closes before cancellation propagates to every owned query and child process group
- **AND THEN** shutdown waits only through a finite documented grace period before escalating owned child process groups from graceful termination to forced termination
- **AND THEN** every owned process is reaped, late results are discarded, and a bounded diagnostic identifies any owner that missed the deadline
- **AND THEN** shutdown does not wait for or signal unrelated processes, shared external services, or another runtime owner's resources

### Requirement: Request, response, and rendering budgets are enforceable

The BFF and client SHALL enforce finite budgets per slice. Initial design
envelopes SHALL be no more than four concurrent same-workspace HTTP requests,
one in-flight request per slice identity, 7,300 Observe daily positions, 2,000
analysis series points per response, 100 metric observations per summary, 50
lineage rows per page, 100 evidence references per response, 2 MiB uncompressed
JSON per slice, and 15 seconds server/query deadline. Exceeding a budget SHALL be
explicit and SHALL NOT silently truncate evidence.

#### Scenario: Analysis series request is within budget
- **WHEN** a requested metric series contains at most 2,000 points and the encoded response fits the byte budget
- **THEN** the BFF returns the complete ordered slice with range and point counts
- **AND THEN** the client renders it with bounded Canvas/SVG work and bounded accessible summaries

#### Scenario: Evidence fan-out exceeds budget
- **WHEN** a query would return more evidence references, rows, points, or bytes than its contract allows
- **THEN** the BFF returns a budget error or an explicit paginated/summary response with omitted counts and continuation contract
- **AND THEN** it does not silently discard references or start unbounded parallel reads

#### Scenario: Ten-times workload is exercised
- **WHEN** capacity tests run 10x the expected concurrent workspace sessions and maximum supported slice sizes
- **THEN** admission, latency, memory, cancellation, cache, and error rates are measured against documented limits
- **AND THEN** overload produces bounded rejection/degradation rather than process exhaustion or cross-user identity reuse

### Requirement: Caching is immutable-reference aware and bounded

Server and browser caching SHALL be limited to read-only payloads keyed by
complete semantic identity and immutable reference. Browser evidence payloads
SHALL use bounded in-memory caching only unless a separately governed encrypted
offline mode is introduced. LocalStorage SHALL remain limited to non-evidence
presentation preferences such as the existing identity-bound K-line viewport.

#### Scenario: Same immutable analysis is revisited
- **WHEN** a user returns to the same confirmed AnalysisSnapshotRef and selectors
- **THEN** a bounded same-identity memory/ETag cache may serve or revalidate it
- **AND THEN** cache metadata retains source identity, schema version, stored time, and validation state

#### Scenario: Browser restarts
- **WHEN** a browser restarts after viewing analysis or evidence
- **THEN** market, analysis, assurance, and research payloads are fetched or revalidated rather than restored from LocalStorage
- **AND THEN** the existing viewport preference may restore only under its already defined identity policy

### Requirement: Legacy Observatory contracts remain compatible

The system MUST keep existing `/api/v1/observatory/*` routes, status codes, ETag
behavior, capability gate, error shape, and primary payload fields available
during the published compatibility window. New workspace routes SHALL be
additive and versioned. Compatibility adapters SHALL map legacy context, series,
trust, runs, and H1 responses conservatively and SHALL NOT invent immutable
references or analysis facts absent from legacy responses.

#### Scenario: Legacy frontend is served with the new backend
- **WHEN** a pre-V2 frontend calls the existing Observatory endpoints
- **THEN** it receives compatible methods, paths, selectors, status codes, ETags, and major payload fields
- **AND THEN** no V2 analysis field is required for existing Observe, Assurance, Lineage, or H1 behavior

#### Scenario: New frontend runs before analysis backend cutover
- **WHEN** the V2 frontend is enabled while only compatible legacy Observe/Assurance/Research queries are available
- **THEN** those views use the compatibility adapter
- **AND THEN** Analyze is explicitly unavailable behind its capability bit
- **AND THEN** it does not calculate browser display estimates as a substitute

#### Scenario: Compatibility adapter cannot prove an identity
- **WHEN** a legacy response lacks a required owner, immutable reference, digest, clock, or method identity
- **THEN** the adapter returns `unproven` or unavailable for the affected V2 fact
- **AND THEN** it does not synthesize an id from a path, mutable current pointer, or response time

### Requirement: Web, SDK, and notebook share query DTO semantics

The versioned workspace query contracts SHALL be usable by Web, SDK, and
notebook callers without exposing ORM models, repositories, database
connections, DataFrames, filesystem paths, or framework-specific response
objects. Notebook consumers SHALL NOT modify `sys.path`, scan the repository,
read internal parquet directly, or import adapters.

#### Scenario: Notebook requests BTC analysis
- **WHEN** a notebook requests a confirmed analysis snapshot and metric series through the SDK
- **THEN** it receives the same reference, method, metric, clocks, availability, and error semantics as the Web BFF
- **AND THEN** it does not need to locate repository files or call an internal repository

#### Scenario: SDK consumer receives an unavailable metric
- **WHEN** a metric is unavailable under the owning contract
- **THEN** the SDK exposes the structured unavailable state and reason codes
- **AND THEN** it does not coerce the result to `0`, `NaN`, an empty DataFrame, or a successful optional value without status

### Requirement: BFF observability is bounded and privacy-safe

The BFF SHALL emit correlation-aware operational telemetry for slice name,
contract version, result state, reason code, owner-query latency, response
bytes, cache outcome, cancellation, deadline, and identity-mismatch count. Logs
and metrics SHALL NOT include access tokens, raw payloads, full evidence content,
local paths, or unbounded reference arrays.

#### Scenario: User reports a failed Analyze panel
- **WHEN** an Analyze request returns a structured failure
- **THEN** the UI exposes a safe correlation id and reason code
- **AND THEN** operators can correlate it with bounded BFF/owner-query timing and result-state telemetry
- **AND THEN** telemetry does not reveal raw market rows, credentials, or filesystem layout

#### Scenario: Stale response is discarded
- **WHEN** a response arrives after its request identity was superseded
- **THEN** the client records a bounded stale-response-discard diagnostic
- **AND THEN** the payload is not rendered, persisted, or logged in full
