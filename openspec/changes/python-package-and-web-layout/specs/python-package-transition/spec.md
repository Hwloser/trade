## ADDED Requirements

### Requirement: The distribution SHALL expose legacy and target packages through one deterministic build

The distribution SHALL remain named `trade-py` during the compatibility
window and SHALL install both the admitted legacy compatibility packages and
the `trade` namespace from `src/trade`. Package discovery SHALL be explicit,
deterministic and restricted to reviewed source roots. It SHALL NOT install
tests, data, notebooks, frontend source, build output, vendor code or
`_bmad-output`.

The source tree, editable installation and clean wheel installation SHALL
produce the same declared public import set. Clean-wheel validation SHALL run
outside the repository with no checkout path on `sys.path` and SHALL record the
wheel digest, sorted member set and module origins.

#### Scenario: The target package exists only in the checkout
- **WHEN** `import trade` succeeds from the repository but the clean wheel omits the package or resolves it from the checkout
- **THEN** package transition fails before any logical module becomes target-authoritative

#### Scenario: The built wheel contains an unreviewed root
- **WHEN** the wheel contains a test, notebook, frontend, data, build, vendor or historical-planning path not present in the member allowlist
- **THEN** validation rejects the wheel and reports the exact unexpected members

#### Scenario: Optional native code is absent
- **WHEN** the pure Python wheel is installed without `_trade_native`
- **THEN** all non-native target and legacy imports remain usable and native capability is explicitly unavailable rather than causing package import failure

### Requirement: Each migrated logical module SHALL have exactly one implementation authority

Each migrated logical module SHALL have an immutable authority record naming
the legacy module, target module, semantic owner, contract generation,
`ConsumerInventoryRef`, implementation digest, compatibility direction,
`LayoutActivationPlanV1`, `MigrationEvidenceRef`, state, rollback target and
retirement condition. At most one module SHALL own business behavior,
registries, singletons, handlers, repositories, resources or import-time side
effects in a generation.

A compatibility module SHALL forward explicitly from the legacy path to the
target public module. Target modules SHALL NOT import legacy implementation,
extend package search paths into legacy roots, install broad `sys.modules`
aliases or use module-level fallback lookup to hide missing classifications.

#### Scenario: Legacy and target imports occur in either order
- **WHEN** a process imports the legacy then target path, target then legacy path, or both concurrently
- **THEN** the selected public symbols have the reviewed identity and behavior, and registries, handlers and resources are initialized no more than once

#### Scenario: A target module imports a legacy implementation
- **WHEN** the dependency guard finds a target-to-legacy implementation edge
- **THEN** the migration record cannot advance and the target module is not authoritative

#### Scenario: A private legacy consumer is not classified
- **WHEN** a current consumer imports a legacy private symbol not covered by the forwarding export map
- **THEN** the bridge fails explicitly and retirement is blocked rather than resolving through a broad fallback

### Requirement: Physical movement SHALL follow implemented semantic ownership

A production module SHALL move into `src/trade` only after its owning Context,
Platform, Process, Interface or Bootstrap child has passed strict design
approval and implemented the required public contracts, repositories, Ports or
query handles. Directory names and import compatibility SHALL NOT constitute
ownership evidence.

A mixed-owner file SHALL remain at its legacy path or be split by its owning
child. It SHALL NOT be copied whole to make the target tree appear complete.

#### Scenario: A file has dependencies on multiple business owners
- **WHEN** classification finds cross-owner state transitions, SQL, artifacts or provider calls in one file
- **THEN** this change records the file as blocked or split-required and does not move or duplicate it

#### Scenario: An owner-ready module is selected
- **WHEN** its semantic child is implemented, public contracts are stable and the consumer inventory is current
- **THEN** one bounded migration slice may add the target implementation and explicit legacy forwarder with focused rollback evidence

### Requirement: Root and installed CLI contracts SHALL remain compatible

The root `./trade` facade and installed `trade-py` console command SHALL remain
available. Canonical and hidden compatibility domains, arguments, defaults,
help, stdout/stderr routing, exit status and `./trade dev` frozen/no-sync
behavior SHALL remain compatible until `cli-http-sdk-compatibility` explicitly
versions a change.

The console target MAY change from a legacy module to a target Interface module
only after source, editable and clean-wheel parity proves that both entrypoints
select one authoritative CLI and do not import unrelated business modules for
help.

#### Scenario: A console target is changed
- **WHEN** the installed entry point begins delegating to `trade.interfaces.cli`
- **THEN** root and installed command snapshots match the approved baseline for every canonical and hidden domain

#### Scenario: Lazy import behavior broadens
- **WHEN** CLI help or one domain import loads an unrelated Context, Web server, provider, database or native module
- **THEN** compatibility validation fails even if command text and exit status match

### Requirement: Compatibility bridges SHALL have measured retirement criteria

Each legacy bridge SHALL name an owner, introduction generation, supported
consumer set, deprecation behavior, last observed supported use, minimum
compatibility window, deadline, rollback target and cleanup child. A bridge
SHALL remain installed for at least 30 days after target authority and SHALL
not be removed while usage is unknown or any supported consumer remains.

The owner Interface adapter SHALL emit only bounded bridge/generation,
supported-consumer-class and outcome observations at an already-owned facade.
Compatibility module import SHALL perform no network, file or telemetry I/O.
The deployment observability adapter SHALL produce `BridgeUseCoverageRef`
containing the supported-consumer and deployment-population digests, coverage
interval, collector version/health, last successful observation, coverage state,
last supported use and report digest. Unsupported or dynamically unobservable
consumers SHALL remain explicit inventory entries until their removal is
source- and deployment-proven.

Bridge retirement SHALL require a complete consumer scan, zero observed
supported use for the full window, `complete` population coverage without a
missing or stale interval, release/deprecation evidence, source, editable and
wheel validation without the bridge, and a successful rollback drill.

#### Scenario: Telemetry is unavailable
- **WHEN** supported-use evidence cannot be collected for part of the compatibility window
- **THEN** use state is `unknown` and retirement remains blocked

#### Scenario: A deployment is outside collector coverage
- **WHEN** the supported deployment-population digest contains an instance or consumer class absent from the coverage report
- **THEN** coverage is `partial`, zero use is not inferred and retirement remains blocked

#### Scenario: A legacy module is imported
- **WHEN** the compatibility module loads in a supported process
- **THEN** import itself performs no external or durable I/O and any usage observation occurs only through the owning facade adapter

#### Scenario: A bridge meets its age but still has a consumer
- **WHEN** the minimum window elapsed but a script, deployment command, SDK, notebook or test still imports the legacy path
- **THEN** cleanup reports the consumer and retains the bridge
