## ADDED Requirements

### Requirement: Reusable BTC descriptive analysis is a Datasets product

Reusable BTC descriptive analysis SHALL be published by Datasets as an
immutable, versioned derived Dataset product. Its formal build inputs SHALL be
immutable DatasetVersionRef or DatasetSnapshotRef values. The product SHALL
carry lineage, semantic schema, method policy, unit, timezone, event-time,
available-time, knowledge-cut, revision-policy, quality, and coverage
identities. The UI, BFF, provider adapter, and moving database state SHALL NOT
be formal build inputs.

#### Scenario: Analysis product is built
- **WHEN** Datasets builds a BTC descriptive-analysis version
- **THEN** every input is a verified immutable DatasetVersionRef or DatasetSnapshotRef
- **AND THEN** the output identifies the source BTC daily-bar snapshot, transform and method policy versions, schema, units, UTC calendar policy, available-time policy, revision policy, and lineage digest
- **AND THEN** an identical ordered input and policy set yields the same content identity

#### Scenario: Moving input is supplied
- **WHEN** a formal analysis build is asked to read `latest`, current directory contents, a provider response, an unpinned DataFrame, or current database rows
- **THEN** the build is rejected before publication
- **AND THEN** no DatasetVersionRef or analysis snapshot is issued

#### Scenario: Product cannot prove point-in-time input
- **WHEN** a required source row lacks a provable available-time or violates the requested knowledge/revision policy
- **THEN** the affected metric/series is excluded or marked unavailable under the method policy
- **AND THEN** a formal analysis snapshot is not published as complete
- **AND THEN** missing temporal proof is never replaced by collector-now, file mtime, or zero

### Requirement: Analysis references bind market, product, and method identity

Every analysis response SHALL carry an `AnalysisSnapshotRef` that identifies the
owner context, analysis DatasetVersionRef or DatasetSnapshotRef, source market
DatasetSnapshotRef, content digest, schema version, transform environment,
method-policy version, knowledge cut, revision policy, and creation clock. A
consumer SHALL be able to verify that the analysis input lineage contains the
selected market snapshot without resolving a filesystem path.

#### Scenario: Workspace binds analysis to selected market evidence
- **WHEN** the BFF composes a selected market context with an analysis response
- **THEN** it verifies the analysis reference and source market snapshot relationship
- **AND THEN** it returns both immutable references and a relationship value of `same_snapshot`, `derived_from_snapshot`, `different_snapshot`, or `unproven`
- **AND THEN** only `same_snapshot` or a policy-approved `derived_from_snapshot` may be shown as current Analyze evidence

#### Scenario: Reference is tampered or unverifiable
- **WHEN** owner, type, version, digest, schema, source reference, or method-policy identity does not verify
- **THEN** the response fails closed with `analysis_reference_invalid`
- **AND THEN** the consumer does not open a path, scan a directory, or fall back to the latest product

### Requirement: Metric values include method and availability semantics

Each descriptive metric SHALL be represented as a structured `MetricObservation`
rather than a bare number. It SHALL include `metric_id`, display label, value or
explicit absence, unit, direction-neutral interpretation class, window,
observation/event as-of, knowledge cut, available time, sample count, expected
sample count, coverage ratio, coverage state, quality state, method id/version,
input and output references, evidence references, and reason codes.

#### Scenario: Complete metric is returned
- **WHEN** a metric meets its method's temporal, sample, coverage, and quality requirements
- **THEN** the response sets `state=confirmed`, provides its decimal value and unit, and identifies its method, window, sample, coverage, clocks, and references
- **AND THEN** the value is serialized as an exact decimal string when decimal preservation is required

#### Scenario: Metric is unavailable
- **WHEN** the method cannot calculate a valid value because lookback, temporal proof, unit identity, or minimum sample requirements are unmet
- **THEN** the response sets `state=unavailable`, omits the numeric value, and provides stable reason codes and evidence references
- **AND THEN** zero, the prior value, a neutral percentile, or an imputed normal state is not substituted

#### Scenario: Metric is partial
- **WHEN** the method permits a result with incomplete but policy-sufficient coverage
- **THEN** the response sets `state=partial`, reports actual and expected samples, coverage, exclusions, and the numeric value if policy permits it
- **AND THEN** the UI must visibly retain the partial state beside that value

### Requirement: Initial descriptive metric set is bounded and versioned

The initial analysis product SHALL support a bounded registry of descriptive
historical metrics: close return over approved trailing windows, realized
volatility over approved windows, peak-to-current and maximum drawdown, true or
high-low range where source semantics permit it, volume change only when volume
identity is comparable, and empirical distribution percentile over a declared
historical window. Each metric SHALL have one versioned method definition and
minimum-sample/coverage policy. Arbitrary user expressions SHALL NOT execute in
the query or UI path.

#### Scenario: Supported metric is queried
- **WHEN** a consumer requests a registered metric and supported window
- **THEN** Datasets returns the prebuilt or boundedly projected observation under its registered method version
- **AND THEN** the response identifies whether the window is calendar-day, available-bar, rolling-observation, or drawdown-episode based

#### Scenario: Unsupported metric or window is queried
- **WHEN** a consumer supplies an unregistered metric id, arbitrary expression, or unsupported window
- **THEN** the query returns `analysis_selector_invalid`
- **AND THEN** it performs no dynamic code execution, provider access, dataset build, or unbounded scan

#### Scenario: Volume identity is absent
- **WHEN** source volume has no stable comparable unit or mixes incompatible venue semantics
- **THEN** volume-derived observations are unavailable with `volume_unit_unproven`
- **AND THEN** price-based observations remain independently queryable

### Requirement: Rolling series are immutable, bounded projections

An analysis series SHALL identify one metric method and one immutable analysis
snapshot. It SHALL return ordered UTC points with event date, available time,
value or explicit gap, quality/revision state, and evidence reference. Queries
SHALL require a supported range and SHALL enforce response-point and byte
budgets without silent truncation.

#### Scenario: Rolling volatility series is requested
- **WHEN** the user opens a supported realized-volatility series for a confirmed analysis snapshot and range
- **THEN** the query returns points in deterministic UTC event-date order with method/version, units, input reference, and per-point availability
- **AND THEN** excluded, revised, or unavailable points remain explicit gaps or marked points rather than interpolated normals

#### Scenario: Series exceeds its contract budget
- **WHEN** the requested range would exceed the maximum points or encoded bytes
- **THEN** the query returns `analysis_query_budget_exceeded` with supported coarser resolutions or bounded ranges
- **AND THEN** it does not silently drop early or late points

### Requirement: Revisions create new analysis versions and stale relationships

A source Dataset revision SHALL produce a new derived DatasetVersion rather than
mutating an existing analysis product. The relationship between old analysis
references, new source snapshots, and affected StudyResultRef values SHALL be
queryable. A superseded analysis result SHALL remain auditable.

#### Scenario: BTC source snapshot is superseded
- **WHEN** Datasets releases a revised BTC daily-bar DatasetVersion
- **THEN** the previous analysis DatasetVersion remains immutable
- **AND THEN** a new analysis build uses the revised immutable input and records the supersession/lineage relationship
- **AND THEN** queries against the old reference continue to return its historical facts with a stale/superseded relationship

#### Scenario: Current workspace sees old analysis after revision
- **WHEN** the active market snapshot is newer than the available analysis product
- **THEN** the BFF returns `analysis_pending_for_snapshot` or `analysis_stale_for_snapshot`
- **AND THEN** the UI does not label the old metrics current or recompute them locally

### Requirement: Studies own inferential and forward-looking results

The system MUST assign every hypothesis, forward outcome, label, effect
estimate, model fit, confidence interval, significance test, placebo,
walk-forward result, benchmark, multiple-testing correction, validation state,
promotion, or rejection to Studies, and each formal StudyRun SHALL read only
DatasetSnapshotRef inputs. Such results SHALL NOT be added to the descriptive
analysis Dataset or inferred by the UI.

#### Scenario: Historical regime description has no future label
- **WHEN** Datasets publishes a volatility percentile or drawdown state using only as-of historical data
- **THEN** Analyze may display it as a descriptive observation
- **AND THEN** it does not claim that price will rise, fall, persist, mean-revert, or reach a target

#### Scenario: H1 evaluates a future volatility outcome
- **WHEN** H1 compares later realized volatility with a registered historical condition
- **THEN** the computation and result are owned by a StudyRun bound to DatasetSnapshotRef
- **AND THEN** Research displays its validation, sample, uncertainty, evidence, and lifecycle state
- **AND THEN** Analyze does not copy its result into a market-status metric

#### Scenario: Study input is not immutable
- **WHEN** a StudyRun is given an AnalysisSnapshotRef or DatasetSnapshotRef that cannot be verified, or is asked to read moving current state
- **THEN** the StudyRun is rejected or concludes insufficient data
- **AND THEN** it does not query Capture, a provider, or a raw artifact

### Requirement: Analysis has no recommendation semantics

The analysis contract SHALL be direction-neutral and SHALL NOT expose buy/sell,
long/short, risk-on/risk-off command, target price, expected future return,
direction probability, position size, asset ranking, or recommendation fields.
Any future Decision Support use SHALL consume a separately governed
StudyResultRef and DatasetSnapshotRef through its own contract.

#### Scenario: Consumer requests a directional conclusion
- **WHEN** a consumer asks the descriptive-analysis query for a trade direction, target, probability, or recommendation
- **THEN** the query rejects the selector as unsupported
- **AND THEN** it returns no synthesized neutral or directional fallback

#### Scenario: High volatility is displayed
- **WHEN** a confirmed metric places current realized volatility in a high historical percentile
- **THEN** the response and UI describe the historical percentile, window, sample, and method
- **AND THEN** they do not map `high` to buy, sell, avoid, increase risk, or reduce risk
