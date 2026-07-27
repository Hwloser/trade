## ADDED Requirements

### Requirement: Every current path SHALL have a source-evidenced disposition before movement or retirement

Each candidate path SHALL be classified using current consumers, state transitions, table and
artifact access, transaction behavior, provider/framework/native dependencies,
commands/events/queries, import side effects, semantic owner, target cell,
compatibility obligation, validation and rollback.

A path SHALL have exactly one of: `owner_ready`, `compatibility`,
`split_required`, `tool`, `deployment`, `example`, `test`, `historical`,
`blocked` or `retireable`. A directory name alone SHALL not determine the
disposition.

#### Scenario: An Observatory file is classified
- **WHEN** the file composes HTTP/SDK queries rather than owning a business aggregate
- **THEN** it is classified as an Interface/BFF compatibility path and not as an Observatory bounded Context

#### Scenario: A root script mutates data
- **WHEN** a root script can relocate data paths or mutate durable business data or metadata
- **THEN** it remains blocked or moves only under its owning migration change with backup, dry-run, verification and rollback evidence

#### Scenario: Historical material contains a still-referenced decision
- **WHEN** a governed document or source comment links to an `_bmad-output` file
- **THEN** the valid decision is extracted with provenance and the historical file is retained until link and retention checks pass

### Requirement: Notebook and SDK consumers SHALL use installed public contracts

Target notebooks and examples SHALL import an installed SDK shared with CLI and
HTTP Interface adapters. They SHALL NOT modify `sys.path`, search repository
parents, scan source files, read formal parquet directly, import repositories
or adapters, or rely on current working directory.

Notebook validation SHALL execute from a clean working directory against
temporary immutable fixtures and compare explicit reference identity, status
and error semantics. Notebook movement SHALL not change a moving `latest` read
into a formal reproducible result.

#### Scenario: The installed SDK is not ready
- **WHEN** a notebook still requires a legacy internal resolver or repository path
- **THEN** the notebook remains at its current path and no source-path workaround is added to the target example

#### Scenario: The same immutable snapshot is opened from Web and notebook
- **WHEN** both consumers use the same reviewed snapshot reference and knowledge policy
- **THEN** they observe the same snapshot identity, lifecycle/quality state and owner query semantics through public contracts

### Requirement: Root scripts SHALL be classified by operational ownership

Each root script SHALL be inventoried as one of:

- public CLI adapter;
- owner-specific migration;
- deployment/operations tool;
- example;
- historical one-time utility;
- blocked mixed-responsibility code.

A script SHALL not remain an installable package merely because current
setuptools discovery includes `scripts*`. Migration scripts SHALL never be
imported as business libraries. Backup code SHALL separate Platform technical
capability from business repository ownership before movement.

#### Scenario: A migration script is present in a wheel
- **WHEN** a clean wheel contains a destructive one-time migration that was not explicitly approved as an installed command
- **THEN** the wheel-member guard fails and reports that script

#### Scenario: A backup script mixes driver and TradeDB operations
- **WHEN** classification finds technical remote storage plus business database access
- **THEN** it is split-required and remains at the compatibility path until both owners and contracts exist

### Requirement: Test layout migration SHALL preserve component ownership and collection evidence

Tests SHALL move gradually into `unit`, `integration`, `contract`,
`architecture`, `golden`, `e2e` and `fixtures` according to behavior, while
Python, frontend and C++ component runners retain appropriate ownership.
Production movement and broad test reclassification SHALL not occur in the
same migration slice.

Each test-layout slice SHALL record old/new node mapping, collection count,
marker/fixture resolution, data/network isolation and focused behavior result.
No test SHALL be dropped, duplicated or silently deselected because its path
changed.

#### Scenario: A test file moves to a category directory
- **WHEN** pytest collection runs before and after the slice
- **THEN** every reviewed old test maps to exactly one new test node and no unrelated node disappears or duplicates

#### Scenario: A frontend or C++ test is classified
- **WHEN** the test belongs to Vite/Vitest/Playwright or CMake/ctest
- **THEN** it remains with that component runner and is referenced by repository validation rather than copied into Python pytest

### Requirement: Migration validation SHALL be bounded, deterministic and process-clean

Package/layout inventories SHALL use sorted Git-tracked paths and fixed
selection rules. One authority slice SHALL contain no more than 50 production
modules and 500 consumer records. Validation SHALL not randomly sample or
silently truncate; exceeding a bound SHALL require a smaller explicit slice.

Build, install, import, ASGI, browser and native checks SHALL run in terminable
process groups with finite deadlines. Timeout, test failure, contract mismatch,
tool failure and unavailable prerequisite SHALL be reported as distinct
states. Reaching a deadline SHALL terminate the process tree and retain partial
reports without advancing authority.

#### Scenario: A package smoke process hangs
- **WHEN** an install, import, console, ASGI or native command exceeds its reviewed deadline
- **THEN** the whole owned process group is terminated, state is `timeout`, completed evidence is retained and no module authority advances

#### Scenario: The candidate slice exceeds the consumer budget
- **WHEN** more than 500 current consumers map to the slice
- **THEN** validation refuses the slice and requires a narrower deterministic grouping rather than taking the first 500 or sampling

### Requirement: Every migration slice SHALL be independently reversible

Each slice SHALL name its prior authority, target authority, activation
condition, rollback trigger, rollback selector, retained compatibility path and
post-rollback validation. Python package, ASGI/backend, frontend build and
native adapter generations SHALL be independently selectable.

Rollback SHALL not require database migration, artifact deletion, provider
access or reversal of previously completed semantic owner migrations.

#### Scenario: A post-cutover consumer is discovered
- **WHEN** an unsupported or missed consumer fails after target authority activation
- **THEN** the affected module returns to its prior authority or forwarding state, the consumer inventory is updated and only that slice is revalidated

#### Scenario: Rollback validation fails
- **WHEN** reinstalling or selecting the prior generation does not restore its recorded imports/contracts
- **THEN** rollout halts, the bridge remains installed and the failure is P0 until the prior authority is recoverable

### Requirement: Package and layout operations SHALL not access real business data

Source audit and package/layout validation SHALL not open real business data or provider state.
Package builds, wheel installs, Web builds, import tests and layout guards
SHALL not open real DB, parquet, manifest, pointer or provider
state. Tests SHALL use temporary roots and explicit fake or immutable fixtures.

#### Scenario: A compatibility smoke tries to initialize TradeDB
- **WHEN** package, import or route-registration validation opens a real or default data root
- **THEN** the guard fails the test and requires dependency injection or a temporary fixture before continuing

#### Scenario: A Web route requires data for its golden
- **WHEN** contract validation needs representative payload state
- **THEN** it uses the frozen temporary fixture generation and does not repair, backfill or refresh real data
