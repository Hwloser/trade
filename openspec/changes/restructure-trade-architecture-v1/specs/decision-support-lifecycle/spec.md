## ADDED Requirements

### Requirement: Decision Support SHALL own human-assist decision state

Decision Support SHALL own `DecisionCase`, `Review`, `Rationale`, `Override`,
`PortfolioIntent`, `Expiry` and `AuditTrail`. A DecisionCase SHALL reference
immutable `DatasetSnapshotRef` and/or `StudyResultRef` evidence and SHALL NOT
mutate Capture, Dataset or Study state. Decision Support SHALL expose an
explicit lifecycle of `draft`, `ready_for_review`, `under_review`, `accepted`,
`rejected`, `expired`, `superseded` and `withdrawn`; every transition SHALL
record actor, reason, correlation, causation, policy identity and evidence
identity in the owner-local transaction.

#### Scenario: A reviewer accepts a supported decision case

- **WHEN** an authorized reviewer accepts a ready DecisionCase whose immutable
  evidence is available and whose review and expiry policy is satisfied
- **THEN** Decision Support records the Review, Rationale, accepted transition
  and audit entry atomically and emits `DecisionCaseAccepted` without changing
  any upstream DatasetSnapshot or StudyResult

#### Scenario: Evidence is unavailable during case creation

- **WHEN** a caller supplies a moving alias, raw provider payload, mutable
  DataFrame, unknown reference or unavailable immutable evidence
- **THEN** Decision Support rejects formal case readiness with a typed
  evidence-unavailable reason and does not synthesize a neutral score,
  recommendation or successful review state

### Requirement: Decision evidence SHALL remain immutable and revision-aware

An accepted Review, Rationale or Override SHALL be append-only evidence.
Correction, expiry, withdrawal and supersession SHALL create a new state or
record rather than rewriting prior rationale. When an upstream Dataset or Study
reference becomes stale, withdrawn, quarantined or superseded, the owning
Process Manager SHALL issue a command that records the affected DecisionCase
as `under_review`, `expired`, `superseded` or otherwise unavailable according
to its pinned policy. Decision Support SHALL preserve the original evidence
identity and the revision cause.

#### Scenario: A referenced StudyResult becomes stale

- **WHEN** a revision-propagation process reports that an accepted
  DecisionCase references a stale StudyResultRef
- **THEN** Decision Support records the revision evidence, prevents the prior
  case from appearing current, and requires the declared review or supersession
  transition without rewriting the accepted historical audit trail

#### Scenario: A decision case expires without replacement evidence

- **WHEN** the case expiry time or evidence-freshness policy is reached and no
  validated replacement reference exists
- **THEN** the current DecisionCase view becomes explicitly `expired` or
  unavailable and does not reuse the old rationale as a current recommendation

### Requirement: Overrides and portfolio intents SHALL be bounded and auditable

An Override SHALL name the superseded recommendation or review conclusion,
authorized actor, bounded scope, reason, issue time and expiry. A
`PortfolioIntent` SHALL be a human-assist planning artifact with evidence and
policy references; it SHALL NOT be an executable order, broker instruction or
claim of validated profitability. Query paths SHALL not create a Review,
Override or PortfolioIntent.

#### Scenario: An operator records an override

- **WHEN** an authorized operator overrides a reviewed case
- **THEN** Decision Support appends the override and audit evidence, preserves
  both the original and overriding rationale, applies the declared expiry, and
  emits a past-tense fact without invoking an exchange, broker or execution
  adapter

#### Scenario: A page reads an accepted case

- **WHEN** Today, Symbol Workspace, Candidates, Actions or Trust requests a
  DecisionCase view
- **THEN** the Decision Support query returns bounded evidence, review, expiry
  and stale-state fields and performs no case transition, upstream repair,
  Study execution or portfolio mutation

### Requirement: Decision Support SHALL remain outside trade execution

This architecture change SHALL NOT add order placement, exchange connectivity,
broker settlement, autonomous portfolio mutation or capital-risk execution to
Decision Support. Any future execution context SHALL require a separate
governed OpenSpec, risk model, public contracts, safety controls and approval;
`PortfolioIntent` and existing recommendation/action compatibility records
SHALL remain non-executable until then.

#### Scenario: A caller attempts to execute a PortfolioIntent

- **WHEN** a caller submits a PortfolioIntent to a Decision Support use case as
  an executable order
- **THEN** the use case rejects the request with a stable unsupported-capability
  reason and records no broker, exchange, order or position state
