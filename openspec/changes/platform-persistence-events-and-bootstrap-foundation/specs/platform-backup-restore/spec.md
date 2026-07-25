## ADDED Requirements

### Requirement: Backup creation SHALL certify a bounded immutable generation

Platform Backup SHALL create a `BackupManifest` before a snapshot is eligible
for restore or remote publication. The manifest SHALL contain schema/version,
backup/snapshot identity, source generation, creation time, database schema
capability range, required Context/artifact set, archive SHA-256 and size, and a
deterministically ordered member list. Each member SHALL contain normalized
relative path, kind, size, SHA-256, required/optional state and owner namespace.

Backup creation SHALL use an explicit consistency strategy for each included
database or mutable artifact: SQLite online backup or an equivalent
read-transaction snapshot for the database, and an immutable committed
generation/ref for external artifacts. It SHALL not recursively tar a live data
root while writers can mutate included files. The manifest and certification
receipt SHALL be written atomically after all staged bytes are hashed. A failed
or partial snapshot SHALL remain `preparing`/`failed`, SHALL not be published as
restorable and SHALL preserve bounded diagnostics.

#### Scenario: SQLite receives writes during backup
- **WHEN** a backup starts while compatible writers continue
- **THEN** the archive contains one database snapshot generation produced by the reviewed SQLite snapshot mechanism rather than a mixture of copied database, WAL and SHM moments

#### Scenario: An artifact changes during staging
- **WHEN** a mutable path identity, size or digest changes between selection and staged verification
- **THEN** certification fails and the manifest is not marked restorable

#### Scenario: Remote publication is partial
- **WHEN** the archive uploads but the manifest upload or remote digest verification fails
- **THEN** the backup remains not-published with an explicit retryable outcome and is never selected as the latest verified remote generation

### Requirement: RestoreOperation SHALL verify before extraction or activation

Every restore SHALL create an append-only `RestoreOperation` owned by Platform
Backup with identity, trusted actor, source backup, prior and target generation,
correlation/causation, migration capability decision, state, current step,
deadline, manifest/archive digests, safe reason/error and audit timestamps.

Before extracting a member, restore SHALL parse and bound the manifest and archive
directory, reject absolute paths, traversal, links, devices, FIFOs, sockets,
duplicate normalized paths, undeclared members, missing required members,
case/Unicode-normalization collisions, per-member or aggregate size/count excess,
unsupported compression, digest/size mismatch and incompatible schema capability.
It SHALL extract only declared regular files/directories into a no-follow staged
root on the target filesystem, then re-hash and validate the staged database and
required artifacts. It SHALL never call an unrestricted archive `extractall`.

The core state path SHALL be:
`prepared -> staged_verified -> writers_fenced -> activated ->
health_verified -> committed`.
Failure/recovery states SHALL be `verification_failed`, `fence_failed`,
`activation_incomplete`, `health_failed`, `rollback_pending`, `rolled_back` and
`failed`. Only `committed`, `verification_failed`, `rolled_back` and `failed`
are terminal. Every transition SHALL be durable and idempotent.

#### Scenario: The archive contains traversal
- **WHEN** an archive member normalizes outside the staged root or is a link/device
- **THEN** restore records `verification_failed` before extraction and the active generation remains unchanged

#### Scenario: A declared member has another digest
- **WHEN** staged bytes do not match the manifest SHA-256 or size
- **THEN** restore rejects the generation, preserves mismatch evidence and exposes no restored reader

#### Scenario: Restore restarts after staged verification
- **WHEN** the process crashes with a durable `staged_verified` operation
- **THEN** reconciliation revalidates the staged identity and resumes fencing without re-downloading or activating unverified bytes

### Requirement: Activation SHALL be one fenced generation compare-and-swap

After staged verification, `MigrationCoordinator` SHALL close new writer
admission, drain or durably fence existing writers under one finite deadline and
record the current active generation. Platform Backup SHALL append one activation
journal entry that compare-and-swaps the expected prior generation to the verified
target. Runtime readers/writers SHALL rebind only to the journal-selected
generation and only after schema capability, repository probes and required
artifact/reference checks pass.

Activation SHALL never expose two writable generations. A crash before journal
commit leaves the prior generation active; a crash after journal commit is
reconciled from the journal and operation state. A bounded health window SHALL
either move to `health_verified`/`committed` or activate the preserved prior
generation and record `rolled_back`. Writer fences SHALL not reopen merely
because an HTTP/CLI observation timed out.

#### Scenario: Activation loses power before compare-and-swap
- **WHEN** writers are fenced but the activation journal does not contain the target commit
- **THEN** restart retains or restores the prior generation and resumes/rolls back the operation without selecting the staged root by directory presence

#### Scenario: Activation commits but runtime rebind fails
- **WHEN** the journal points to the target but required repository or artifact health fails
- **THEN** the operation enters rollback, compare-and-swaps back to the preserved prior generation and reopens admission only after prior-generation readiness

#### Scenario: A stale runtime writes after restore
- **WHEN** a runtime holding the pre-restore fence attempts a transaction after target activation
- **THEN** the transaction is rejected and cannot mutate either generation

### Requirement: Backup and restore evidence SHALL be immutable and operator-visible

Platform SHALL store immutable backup certification, remote publication,
restore transition, activation, health and rollback receipts. An authorized
operator query SHALL distinguish preparing, verified, published, corrupt,
incompatible, restoring, active, rolled-back and unavailable states without
reading archive bytes. It SHALL expose manifest/archive digests, generation,
capability range, bounded member/count/size summary, current state/step, last
safe failure, timestamps and permitted control actions.

Logs, metrics and public errors SHALL not expose credentials, service-account
paths, archive contents or arbitrary exception text. Remote drivers SHALL have
finite timeout/retry/cost bounds and SHALL use staging plus digest verification;
remote availability SHALL not be represented as local backup validity.

#### Scenario: A corrupt backup is listed
- **WHEN** certification or later integrity verification detects corruption
- **THEN** the query reports `corrupt` with safe digest/generation evidence and excludes that backup from automatic restore selection

#### Scenario: A restore action is queried
- **WHEN** an operator reads a non-terminal restore operation
- **THEN** the query has no side effect and returns the exact current step, retained generation/fence state and allowed recovery controls

#### Scenario: A remote credential error occurs
- **WHEN** upload/download authentication fails
- **THEN** Platform reports a stable unavailable reason and safe driver identity without returning the key path, token or raw response

### Requirement: Legacy backup commands SHALL remain compatibility surfaces

Existing backup CLI, function and HTTP contracts SHALL remain compatible until
the interface compatibility child selects Platform Backup. This includes
`trade backup create|push|restore|list`, existing function signatures and current
HTTP backup status. The implementation SHALL not mark the current
`scripts.backup.restore_backup_snapshot` extraction result as a verified
`RestoreOperation`; the legacy path remains explicitly uncertified.

The first Platform Backup implementation SHALL run side-by-side on temporary
roots, generate independently verifiable manifests and require an explicit
feature/capability selection. It SHALL not activate real data by default.
Legacy retirement requires command/API snapshots, corrupt/unsafe archive
fixtures, crash-at-every-state reconciliation, a successful prior-generation
rollback and the minimum compatibility window.

#### Scenario: The legacy restore command is still selected
- **WHEN** the Platform restore capability is disabled or not ready
- **THEN** the current command behavior remains compatible and status identifies the result as legacy/uncertified rather than fabricating staged verification

#### Scenario: Platform restore is piloted
- **WHEN** an authorized test selects the new path
- **THEN** it operates only on a temporary fixture root and does not overwrite or activate the repository's real data root

#### Scenario: Compatibility comparison fails
- **WHEN** legacy command/API snapshots or restored fixture checks differ unexpectedly
- **THEN** cutover is cancelled, the legacy selection stays active and all verified staging evidence is retained for diagnosis
