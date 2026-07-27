## ADDED Requirements

### Requirement: Kernel SHALL contain only admitted framework-free primitives

Kernel SHALL contain only identity, UTC time/deadline, digest, structural
contract error, result and envelope metadata primitives whose semantics are
identical for at least two Contexts, have no business owner and require no
third-party framework. Its module allowlist is exactly `ids`, `time`, `digest`,
`errors`, `result` and `envelope`. Kernel SHALL NOT contain generic reference,
policy, actor, provider, Dataset, Study, recommendation, portfolio, process,
HTTP, database, filesystem, DataFrame, scheduler, EventBus or native-runtime
semantics.

#### Scenario: A business value appears reusable
- **WHEN** two modules use similarly shaped values but the values have different business owners or invariants
- **THEN** the values remain in their owner contracts and are not promoted to Kernel

#### Scenario: A Kernel module is imported from an installed wheel
- **WHEN** a clean Python environment imports every public Kernel symbol
- **THEN** no `trade_py`, `trade_web`, FastAPI, Pydantic, pandas, ORM, database, native extension or application bootstrap module is imported

### Requirement: Kernel identifiers SHALL be opaque, bounded and namespace-qualified

An opaque identifier SHALL contain a namespace of 1-64 ASCII lower-case
letters/digits plus `._-` and a value of 1-128 printable ASCII characters
excluding whitespace/control. It SHALL have no ordering semantics and SHALL
serialize both fields explicitly. Generated identifiers SHALL use UUID4 without
exposing generation as business meaning. A legacy integer identifier SHALL be
represented in a named legacy namespace and SHALL NOT be silently treated as a
globally generated target identifier.

#### Scenario: A legacy Web run ID is mapped
- **WHEN** compatibility code maps `job_runs.id=17`
- **THEN** it produces an opaque identity such as namespace `legacy.job_run` and value `17` without implying UUID generation or cross-table uniqueness

#### Scenario: An invalid identifier is decoded
- **WHEN** an identifier has an empty field, unsupported character, unbounded value or unknown object shape
- **THEN** decoding fails with a structural contract error before a domain or adapter is called

### Requirement: Kernel time and deadline values SHALL preserve UTC and elapsed-time semantics

Wire instants SHALL be timezone-aware UTC values serialized in the sole
canonical form `YYYY-MM-DDTHH:MM:SS.ffffffZ`: a four-digit year from `0001`
through `9999`, seconds from `00` through `59`, exactly six fractional-second
digits including `.000000`, and the literal `Z` suffix. Version 1 wire decoding
SHALL accept only that form; it SHALL reject an offset spelling such as
`+00:00`, omitted or non-six-digit fractional seconds, precision beyond
microseconds and leap seconds. Direct Python construction SHALL accept only an
aware datetime whose UTC offset is exactly zero and SHALL preserve its
microsecond value without an implicit timezone conversion. Durations SHALL be
integer milliseconds in 1-86,400,000. A durable deadline SHALL record a UTC
wall-clock instant, while a local wait SHALL derive and consume a monotonic
remaining duration; wall-clock movement SHALL NOT extend a local wait.

`Deadline` SHALL be an in-memory execution-budget value containing
`wall_clock_expires_at: UtcInstant` and a finite process-local
`monotonic_expires_at`. The composite `Deadline` and its monotonic component
SHALL NOT be public wire values, durable fields or cross-process evidence.
Public receipt/view fields named `deadline` SHALL contain only the canonical
`UtcInstant` wall-clock evidence. Decoding that evidence SHALL NOT reconstruct,
rebind, start or extend a monotonic budget; each executable attempt requires a
new local `Deadline` admitted by its trusted boundary.

#### Scenario: A naive datetime is supplied
- **WHEN** a caller constructs or decodes a time without timezone evidence
- **THEN** the Kernel rejects it rather than assuming local time or UTC

#### Scenario: Equivalent or over-precise UTC spellings are supplied
- **WHEN** a wire value uses `+00:00`, omits the six fractional digits, carries more than six fractional digits or uses a leap second
- **THEN** version 1 rejects it rather than creating a second spelling or rounding a different instant into the canonical identity

#### Scenario: Canonical fractional-second boundaries are serialized
- **WHEN** accepted UTC values carry zero, one or 999,999 microseconds
- **THEN** they serialize respectively with `.000000Z`, `.000001Z` and `.999999Z`

#### Scenario: Wall clock changes during a wait
- **WHEN** the system wall clock moves after a local deadline has been admitted
- **THEN** the wait remains bounded by the original monotonic duration and the wire receipt retains the declared UTC deadline

#### Scenario: A durable deadline is decoded
- **WHEN** a receipt or view containing canonical UTC deadline evidence is decoded in another process
- **THEN** the decoded value remains a `UtcInstant` and no local monotonic deadline or remaining-duration claim is created

#### Scenario: A producer encodes a monotonic deadline
- **WHEN** a public DTO contains `monotonic_expires_at`, a floating-point deadline or the composite Kernel `Deadline`
- **THEN** exact version 1 decoding rejects the field/value rather than treating process-local clock state as durable evidence

### Requirement: Content digests SHALL identify bytes with an explicit algorithm

A content digest SHALL include its algorithm and normalized value. Version 1
SHALL admit SHA-256 lower-case hexadecimal digests only. A digest SHALL
identify exact bytes and SHALL NOT by itself claim schema, owner, version,
quality, provenance or publication state.

#### Scenario: Equal bytes have equal content identity
- **WHEN** two owners hash the same canonical byte sequence with SHA-256
- **THEN** their content digests compare equal while their owner-specific immutable references may remain distinct

#### Scenario: A malformed digest is supplied
- **WHEN** a digest uses an unsupported algorithm, upper-case or non-hex value, or a value of the wrong length
- **THEN** construction fails without normalizing it into a different identity

### Requirement: Results and structural errors SHALL be explicit and immutable

A Kernel result SHALL contain exactly one of a value or a structural contract
error and SHALL not rely on implicit truthiness. Structural errors SHALL use a
closed safe code and detail of at most 1,024 UTF-8 bytes; they SHALL NOT
serialize a live exception, cause, traceback, SQL, path, credential or raw
payload.

#### Scenario: Both result branches are supplied
- **WHEN** construction receives both a value and an error or receives neither
- **THEN** the result is rejected as an invalid contract state

#### Scenario: An adapter exception reaches a contract boundary
- **WHEN** compatibility code catches a legacy exception
- **THEN** it maps only reviewed safe facts into an owner error contract and does not place the exception object in a Kernel result

### Requirement: Envelope metadata SHALL be transport-neutral and owner-payload typed

Kernel envelope metadata SHALL contain versioned message identity,
correlation/causation identity and `envelope_created_at` only. Creation time is
the transport-envelope construction clock and SHALL NOT be interpreted as
provider event/publication/observed/received/available time, PIT evidence or
finality. The envelope payload SHALL be supplied by a typed owner codec. Kernel
SHALL NOT dispatch, persist, retry, acknowledge or inspect business payload
dictionaries.

The v1 causal relation SHALL be exact. A root command/query receives a new
`message_id`, sets `correlation_id=message_id` and has no `causation_id`. A
child command/event receives a new `message_id`, inherits its verified parent's
`correlation_id` and sets `causation_id` to that parent's `message_id`.
Transport retry or redelivery of the same logical envelope preserves all three
identities and exact `DurableEnvelopeProjectionV1` bytes; attempt metadata
remains outside that projection. A newly submitted idempotent duplicate is a
new root or child envelope under these rules, while any returned existing
receipt preserves the original admitted envelope identities. Durable replay
preserves the historical envelope identities and exact projection bytes; a
message newly derived during replay follows the child rule and points to the
replayed historical `message_id`. No adapter may accept caller payload values
as trusted correlation or causation evidence.

Platform message contracts SHALL describe an owner codec with an immutable
descriptor binding owner namespace, schema name/version, payload purpose,
maximum canonical bytes, content policy and deterministic codec identity. This
child SHALL supply only the Platform-owned descriptor value/validator and pure
registry collision/freeze invariants; the later Platform foundation child SHALL
implement the Bootstrap registry builder. Bootstrap SHALL assemble and freeze
that registry before ingress, with unique keys and no more than 4,096
descriptors. A 4,097th descriptor SHALL fail before ingress. Exact registry-key
resolution SHALL use binary search over the immutable registry-key-sorted
descriptor tuple, with at most 13 key comparisons, and SHALL NOT linearly scan
all descriptors. Increasing the v1 capacity SHALL require a reviewed contract
change rather than runtime configuration.
Each retained descriptor identity and its executable owner codec SHALL remain
resolvable for at least the full retention horizon of every durable projection,
replay key and receipt that depends on it. Retained historical entries SHALL
count toward 4,096. Retirement SHALL require owner proof that no retained
durable identity references the codec; Bootstrap readiness SHALL fail when
retention and the capacity bound cannot both be satisfied.
The immutable local registry entry SHALL be a non-wire
`RegisteredOwnerCodecV1` pairing exactly one descriptor with one typed
`OwnerPayloadCodec[T]` encode/decode capability validated against that
descriptor. Binary lookup SHALL resolve that binding rather than a descriptor
without executable validation. The capability object, callback and
implementation identity SHALL NOT enter a public DTO, projection, digest or
diagnostic.
`OwnerPayloadCodec[T]` SHALL expose exactly immutable `descriptor`,
`manifest`, `encode(payload: T) -> bytes` and
`decode(canonical_payload: bytes) -> T`. Its `OwnerCodecManifestV1` SHALL contain
exactly manifest schema version 1, the descriptor registry-key fields,
`max_canonical_bytes`, `content_policy`, a reviewed canonical-schema
`ContentDigest` and a 1-96 character ASCII lower-case codec revision token.
Manifest bytes SHALL use ASCII domain `trade.owner-payload-codec-manifest.v1`,
one NUL and the same unsigned four-byte big-endian component framing as the
durable projection. The exact ordered components SHALL be: manifest schema
version; owner namespace; schema name; schema version; payload purpose;
maximum canonical bytes; content policy; canonical-schema digest algorithm;
canonical-schema digest value; codec revision. Integers and tokens SHALL use
the same canonical encoding rules as the projection. The descriptor fields
SHALL equal the manifest fields and
`descriptor.codec_identity` SHALL equal
`ContentDigest.from_bytes(exact_manifest_bytes)`.

Registry freeze SHALL reject a descriptor/capability/manifest mismatch before
ingress. For every accepted canonical byte sequence, `decode` SHALL return the
typed owner value and exactly one subsequent `encode` SHALL reproduce identical
bytes; otherwise the product is `OWNER_PAYLOAD_INVALID`. Fresh typed admission
SHALL create a non-wire immutable `ValidatedOwnerPayloadV1[T]` by one `encode`;
durable bytes SHALL create it by one `decode` plus one canonical re-encode.
Envelope construction SHALL consume that validated product without invoking
the codec again. A codec exception, non-`bytes` result, over-limit output,
noncanonical round trip or descriptor mismatch SHALL fail closed and SHALL NOT
produce a partial validated payload.
Kernel supplies only envelope composition and SHALL NOT own the descriptor or
registry. A codec validates wire shape only and SHALL NOT confer authority,
rights, publication, quality or PIT proof. The immutable-ref-only rule applies
to cross-Context/canonical Platform envelopes. A Capture inbound adapter MAY
boundedly receive/stage raw push, stream, import or provider input inside
Capture, but external/raw provider, news, L2 or stream content SHALL cross the
canonical boundary only as a committed owner-controlled Capture artifact
reference.

Platform `CommandEnvelope` and `QueryEnvelope` SHALL be admission-local,
non-wire and non-durable composites containing current verified authority and a
process-local `Deadline`. They SHALL have no whole-object encoder or decoder.
The owner codec SHALL be the sole source of `canonical_payload`; one envelope
construction SHALL call `owner_codec.encode(typed_payload)` exactly once and
SHALL prove the returned immutable bytes are byte-for-byte equal to any supplied
canonical bytes before fingerprinting, projection or execution. The same
returned bytes object SHALL be retained as the envelope and projection
`canonical_payload`; projection construction SHALL NOT decode, re-encode or
stage a second payload-sized copy. The projection encoder SHALL use one bounded
destination buffer, and digest derivation SHALL consume that exact immutable
projection encoding rather than encoding the owner payload or projection again.
Their only durable/transport identity SHALL be the inert
`DurableEnvelopeProjectionV1`, which contains exactly projection version 1,
every `EnvelopeMeta` field, the complete `OwnerCodecDescriptor` identity and
policy fields, and the exact owner canonical payload bytes. It SHALL exclude
`ActorContext`, `Deadline`, remaining time, raw idempotency keys, keyed
fingerprints, attempt counters, transport headers and framework state.
`trade.platform.contracts.messages` SHALL own that projection value, its sole
exact binary encoder/decoder and
`derive_durable_envelope_digest_v1(encoded_projection: bytes)`. Kernel SHALL
remain limited to envelope metadata composition. One projection operation
SHALL call `encode_durable_envelope_projection_v1` once; the digest function
SHALL consume that exact returned byte sequence without encoding again.

Projection bytes SHALL start with ASCII
`trade.durable-envelope-projection.v1`, one NUL byte, then these exact ordered
components: projection version; metadata schema name/version; message namespace
and value; correlation namespace and value; a one-byte causation presence
marker and, when present, causation namespace and value; canonical envelope
creation instant; descriptor owner namespace, schema name/version, payload
purpose, maximum canonical bytes, content policy, digest algorithm and digest
value; and exact canonical payload bytes. Each component SHALL be framed by an
unsigned four-byte big-endian length. Integers SHALL use canonical positive
base-10 ASCII, the presence marker SHALL be exactly byte `0` or `1`, and the
whole projection SHALL be at most 65,536 bytes. Lengths and component count
SHALL be validated before allocation. A descriptor's
`max_canonical_bytes` SHALL remain a standalone payload ceiling; projection
construction SHALL still reject framing plus metadata plus payload above
65,536 bytes.
The projection SHALL NOT have a canonical JSON representation or per-object
`schema_name`; its fixed domain plus projection version identify its schema,
and the framed bytes above SHALL be its only public/durable codec. Its digest
SHALL equal
`ContentDigest.from_bytes(encoded_projection)`, where `encoded_projection` is
the exact single result of `encode_durable_envelope_projection_v1(projection)`.
Payload-only, re-encoded, JSON, legacy-envelope and adapter-specific digests
SHALL fail identity verification.

Projection decoding SHALL yield only an authority-free
`DurableEnvelopeProjectionV1`. It SHALL NOT construct a command/query envelope,
verified actor or local `Deadline`. Execution after decode SHALL require exact
descriptor resolution against the current frozen registry, owner-codec payload
revalidation and separately verified current authority. A new owner identity
SHALL receive a newly admitted local deadline. An existing operation/process
identity SHALL instead require its authoritative immutable deadline fact plus
matching live monotonic-clock-domain binding and SHALL be capped by that
binding's remaining time without extending immutable UTC evidence. An expired
owner SHALL be rejected; restart, transfer or loss of the binding SHALL fail
closed rather than deriving time from UTC. A new owner deadline SHALL require a
new causally linked owner identity. Changing only current authority, a matching
live binding or the newly admitted deadline for that new owner identity SHALL
NOT change projection bytes; changing projected metadata, descriptor identity
or canonical payload SHALL change the projection or fail validation.

Projection and registry failure identity SHALL be the closed
`MessageContractFailureCodeV1` set:
`DURABLE_PROJECTION_MALFORMED`, `DURABLE_PROJECTION_TOO_LARGE`,
`OWNER_CODEC_NOT_FOUND`, `OWNER_CODEC_IDENTITY_MISMATCH`,
`OWNER_PAYLOAD_INVALID`, `CODEC_REGISTRY_DUPLICATE_KEY`,
`CODEC_REGISTRY_CAPACITY_EXCEEDED`,
`CODEC_REGISTRY_BINDING_MISMATCH` and
`CODEC_REGISTRY_REQUIRED_CODEC_UNAVAILABLE`. The first five SHALL be
request-scoped admission failures and produce no authority, deadline, operation
or dispatch. Only code plus bounded owner/schema/version/purpose identity MAY
cross an interface; payload, actor evidence, paths and codec internals SHALL be
absent. The final four SHALL be Bootstrap-readiness failures that block
ingress. Duplicate and capacity products SHALL contain only code and descriptor
count. Binding-mismatch SHALL contain only its code and descriptor count.
Required-codec-unavailable SHALL contain only its code, descriptor count and
required-manifest entry count; neither product SHALL expose a descriptor,
dependent identity, payload, path or codec implementation.

The future Platform persistence owner SHALL maintain a complete bounded
`RequiredOwnerCodecManifestV1` instead of proving retention by scanning durable
projections, replay keys or receipts at startup. It SHALL contain a generation
and at most 4,096 immutable entries sorted by exact registry key. Each
`RequiredOwnerCodecManifestEntryV1` SHALL contain only owner namespace, schema
name/version, payload purpose, exact codec identity, retained-reference count
from 1 through 9,223,372,036,854,775,807 and latest required retention
`UtcInstant`. A durable dependent identity SHALL increment or create its entry
in the same owner-local transaction that makes the identity visible; final
retirement SHALL decrement or remove it only in the transaction that makes the
last dependent identity permanently unresolvable and only after its retention
horizon. Duplicate keys, zero counts, incomplete generations or more than 4,096
entries SHALL be corruption.

Bootstrap SHALL read one complete manifest generation through a bounded
Platform persistence port under one finite local `Deadline`. It SHALL compare
at most 4,096 entries to the frozen registry, using at most 13 registry-key
comparisons per entry. It SHALL NOT enumerate or scan durable projection,
replay, receipt or audit rows. A missing manifest generation, timeout,
incomplete/corrupt manifest, missing required registry key, descriptor/capability
identity mismatch, or inability to satisfy both retention and the 4,096-entry
registry limit SHALL block ingress with
`CODEC_REGISTRY_REQUIRED_CODEC_UNAVAILABLE`. Request-time lookup of an
unregistered non-required identity SHALL remain `OWNER_CODEC_NOT_FOUND`.
Descriptor/capability/manifest mismatch while freezing a proposed static
registry entry SHALL instead block ingress with
`CODEC_REGISTRY_BINDING_MISMATCH`, whether or not a historical manifest
currently requires that key.

#### Scenario: An EventBus event is mapped
- **WHEN** compatibility code maps a durable legacy event
- **THEN** it copies durable metadata accepted by the owner codec and excludes the live EventBus back-reference and unreviewed arbitrary payload

#### Scenario: A caller supplies an arbitrary mapping payload
- **WHEN** no registered owner codec exists for that mapping
- **THEN** canonical envelope construction fails rather than accepting `dict[str, Any]` as a public contract

#### Scenario: Typed payload and supplied bytes disagree
- **WHEN** an envelope caller supplies canonical bytes that differ from the owner codec's exact encoding of the typed DTO
- **THEN** construction fails before fingerprinting, projection or execution

#### Scenario: A durable projection digest is derived
- **WHEN** replay binds a historical durable envelope
- **THEN** it hashes the sole exact framed projection bytes and every projected component affects the digest while actor and local deadline do not

#### Scenario: A retained descriptor is proposed for retirement
- **WHEN** any retained projection, replay key or resolvable receipt still depends on its executable codec
- **THEN** retirement is refused and the retained entry continues to count toward the registry capacity

#### Scenario: Registry assembly cannot become ready
- **WHEN** Bootstrap sees a duplicate key, 4,097 entries or cannot retain a required historical codec
- **THEN** ingress remains blocked with one closed readiness code and no descriptor or payload dump

#### Scenario: Bootstrap validates retained codec requirements
- **WHEN** a complete required-codec manifest generation contains no more than 4,096 entries
- **THEN** Bootstrap validates it under one finite deadline with at most 13 registry comparisons per entry and without scanning dependent durable rows

#### Scenario: A required historical codec is unavailable
- **WHEN** the bounded manifest is absent, incomplete or corrupt, or any required key cannot resolve the exact executable codec identity
- **THEN** Bootstrap returns `CODEC_REGISTRY_REQUIRED_CODEC_UNAVAILABLE`, blocks ingress and exposes only descriptor and manifest counts

#### Scenario: One typed payload becomes a durable projection
- **WHEN** an envelope is constructed, projected and digested
- **THEN** the owner codec encodes exactly once, the same immutable payload bytes are reused, one bounded projection encoding is digested, and no second payload-sized staging copy is created

#### Scenario: A child command is emitted from a verified parent event
- **WHEN** a handler creates a new command from an admitted parent envelope
- **THEN** the command gets a new message identity, inherits the parent's correlation identity and names the parent message as its direct causation

#### Scenario: An envelope is redelivered or replayed
- **WHEN** transport redelivers the same logical envelope or an operator replays a durable historical envelope
- **THEN** the exact durable projection bytes and causal identities remain unchanged and retry/replay attempt metadata stays outside the projection

#### Scenario: A whole admission envelope is serialized
- **WHEN** generic or owner code attempts to encode `CommandEnvelope` or `QueryEnvelope` including its verified actor or local deadline
- **THEN** serialization fails because only the authority-free durable projection has a public canonical codec

#### Scenario: A durable projection is decoded for execution
- **WHEN** a historical durable projection is decoded in another attempt or process
- **THEN** it produces neither authority nor a remaining-time budget and cannot execute until trusted ingress attaches separately verified current authority plus either a matching live owner binding or a new causally linked owner identity with a newly admitted deadline

#### Scenario: Two codecs claim one wire identity
- **WHEN** Bootstrap receives duplicate owner/schema/version/purpose descriptors or a codec identity is not deterministic
- **THEN** registry assembly fails before command or event ingress starts

#### Scenario: A capability does not match its descriptor
- **WHEN** manifest fields or manifest digest differ from the registered descriptor
- **THEN** registry freeze returns `CODEC_REGISTRY_BINDING_MISMATCH` before ingress and no descriptor, callback or implementation detail enters the diagnostic

#### Scenario: Durable payload bytes are not canonical
- **WHEN** one codec decode followed by one encode does not reproduce the exact input bytes
- **THEN** validation returns `OWNER_PAYLOAD_INVALID` without constructing an executable envelope

#### Scenario: The codec registry exceeds its capacity
- **WHEN** Bootstrap attempts to freeze 4,097 descriptors
- **THEN** assembly fails before ingress rather than creating an unbounded or linearly scanned registry

#### Scenario: A news body fits under the envelope limit
- **WHEN** an owner codec is asked to inline external news content rather than a committed Capture artifact reference
- **THEN** it rejects the payload because the byte budget does not grant external-content admission
