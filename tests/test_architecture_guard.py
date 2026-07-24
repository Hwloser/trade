from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from trade_py.devtools.architecture_guard import (
    BASELINE_FILENAME,
    DEFAULT_LIMITS,
    PRODUCER_NONLITERAL_TARGET,
    PRODUCER_PATH_BUDGET,
    PRODUCER_RESULT_BUDGET,
    PRODUCER_SOURCE_BUDGET,
    PRODUCER_TIMEOUT,
    PRODUCER_TOOL_FAILURE,
    PRODUCER_UNDECLARED_WRITER,
    PRODUCER_UNRESOLVED_IMPORT,
    PRODUCER_UNRESOLVED_LAYOUT,
    PRODUCER_UNSAFE_SOURCE,
    DiscoveryLimits,
    validate_architecture_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_WRITE_TABLE = "trade_py.data.warehouse.io.write_table"
CANONICAL_UPSERT_TABLE = "trade_py.data.warehouse.io.upsert_table"
DEFAULT_APP = (
    "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
    "layout = WarehouseLayout.from_data_root('data')\n"
    'write_table(layout, "ods", "events", frame=None)\n'
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_repo(tmp_path: Path, sources: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "architecture@example.test")
    _git(repo, "config", "user.name", "Architecture Guard")
    for relative, text in sources.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    _git(repo, "branch", "-M", "master")
    return repo


def _producer_identity(
    app: str,
    *,
    layer: str,
    table: str,
    writer: str,
) -> tuple[int, int, str, str]:
    tree = ast.parse(app)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and len(node.args) >= 3
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == layer
            and isinstance(node.args[2], ast.Constant)
            and node.args[2].value == table
        ):
            digest = hashlib.sha256(
                ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
            ).hexdigest()
            return node.lineno, node.col_offset, ast.unparse(node), digest
    raise AssertionError(f"fixture has no {writer} producer for {layer}.{table}")


def _toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _baseline(
    *,
    producer_source: str = "trade_py/app.py",
    producer_literal: str = 'write_table(layout, "ods", "events", frame=None)',
    producer_layer: str = "ods",
    producer_table: str = "events",
    producer_writer: str = CANONICAL_WRITE_TABLE,
    producer_app: str = DEFAULT_APP,
    classification: str = "candidate",
    target_context: str = "datasets",
    artifact_id: str = "warehouse-parquet",
    extra: str = "",
) -> str:
    producer_line, producer_column, normalized_literal, producer_digest = _producer_identity(
        producer_app,
        layer=producer_layer,
        table=producer_table,
        writer=producer_writer,
    )
    producer_literal = normalized_literal
    return f'''schema_version = 1
target_source_root = "src/trade"
target_import_root = "trade"
legacy_package_roots = ["trade_py", "trade_web"]
target_contexts = ["kernel", "capture", "datasets", "studies", "decision_support", "processes", "platform", "interfaces", "bootstrap"]

[[source_facts]]
id = "legacy-db"
source = "trade_py/db.py"
literal = "LEGACY_DB = 1"
current_owner = "legacy"
required_child = "dataset-product-boundary"

[[tables]]
logical_name = "legacy_records"
current_owner = "legacy"
semantic_kind = "legacy-record"
classification = "deferred"
target_context = "deferred"
reason = "Requires evidence before ownership transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db.py", literal = "CREATE TABLE legacy_records", role = "bootstrap" }},
  {{ source = "trade_py/migrations.py", literal = "ALTER TABLE legacy_records", role = "alter" }},
]

[[tables]]
logical_name = "causal_decision_snapshots"
current_owner = "legacy"
semantic_kind = "causal-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Study or Decision Support migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS causal_decision_snapshots", role = "bootstrap" }},
]

[[tables]]
logical_name = "causal_validation_outcomes"
current_owner = "legacy"
semantic_kind = "causal-validation-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Study migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS causal_validation_outcomes", role = "bootstrap" }},
]

[[tables]]
logical_name = "causal_reward_punishment"
current_owner = "legacy"
semantic_kind = "causal-feedback-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Study migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS causal_reward_punishment", role = "bootstrap" }},
]

[[tables]]
logical_name = "factors"
current_owner = "legacy"
semantic_kind = "factor-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Dataset or Study migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS factors", role = "bootstrap" }},
]

[[tables]]
logical_name = "factor_registry"
current_owner = "legacy"
semantic_kind = "factor-metadata-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Study migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS factor_registry", role = "bootstrap" }},
]

[[tables]]
logical_name = "kg_nodes"
current_owner = "legacy"
semantic_kind = "knowledge-graph-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Dataset or Study migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS kg_nodes", role = "bootstrap" }},
]

[[tables]]
logical_name = "kg_relations"
current_owner = "legacy"
semantic_kind = "knowledge-graph-relation-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Study migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS kg_relations", role = "bootstrap" }},
]

[[tables]]
logical_name = "kg_edge_candidates"
current_owner = "legacy"
semantic_kind = "knowledge-graph-candidate-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Study migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS kg_edge_candidates", role = "bootstrap" }},
]

[[tables]]
logical_name = "model_registry"
current_owner = "legacy"
semantic_kind = "model-registry-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Study migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS model_registry", role = "bootstrap" }},
]

[[tables]]
logical_name = "model_eval_runs"
current_owner = "legacy"
semantic_kind = "model-evaluation-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Study migration."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS model_eval_runs", role = "bootstrap" }},
]

[[tables]]
logical_name = "Recommendation"
current_owner = "legacy"
semantic_kind = "historical-recommendation-record"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Decision Support migration."
required_child = "decision-support-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS Recommendation", role = "migration" }},
]

[[tables]]
logical_name = "RecommendationTrace"
current_owner = "legacy"
semantic_kind = "historical-recommendation-trace"
classification = "deferred"
target_context = "deferred"
reason = "Target ownership requires a reviewed Decision Support migration."
required_child = "decision-support-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS RecommendationTrace", role = "migration" }},
]

[[artifacts]]
id = "warehouse-parquet"
source = "trade_py/warehouse.py"
literal = 'f"{{table}}.parquet"'
current_owner = "legacy"
role = "legacy-artifact"
classification = "candidate"
target_context = "datasets"
reason = "Legacy output requires DatasetVersion migration."
required_child = "dataset-product-boundary"

[[artifacts]]
id = "catalog-sqlite-projection"
source = "trade_py/observatory/catalog/store.py"
literal = 'return base / "catalog.sqlite", base / "generation.json"'
current_owner = "legacy"
role = "rebuildable-catalog-projection"
classification = "candidate"
target_context = "datasets"
reason = "The catalog SQLite file is a rebuildable projection."
required_child = "dataset-product-boundary"

[[artifacts]]
id = "catalog-generation-pointer"
source = "trade_py/observatory/catalog/store.py"
literal = 'return base / "catalog.sqlite", base / "generation.json"'
current_owner = "legacy"
role = "catalog-generation-pointer"
classification = "candidate"
target_context = "datasets"
reason = "The generation pointer is compatibility and recovery input."
required_child = "dataset-product-boundary"

[[artifacts]]
id = "kline-reconciliation-operation-pointer"
source = "trade_py/data/operations/checks.py"
literal = 'path = root / "market" / "kline" / "reconciliation" / "current.json"'
current_owner = "legacy"
role = "data-operation-reconciliation-pointer"
classification = "deferred"
target_context = "deferred"
reason = "The operation check reads a legacy reconciliation pointer."
required_child = "dataset-product-boundary"

[[capture_risks]]
id = "raw-record-single-publication-clock"
source = "trade_py/capture.py"
literal = "published_at"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "clock-collapse"
current_behavior = "One field represents multiple clocks."
required_migration_proof = "Independent clocks."

[[capture_risks]]
id = "cctv-date-only-publication-time"
source = "trade_py/cctv.py"
literal = "synthetic_noon"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "date-only-inferred-precision"
current_behavior = "Date-only values have synthetic time."
required_migration_proof = "Preserve source precision."

[[capture_risks]]
id = "warehouse-rss-fetched-time-substitution"
source = "trade_py/rss.py"
literal = "published_at or fetched_at"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "provider-timestamp-absence-substitution"
current_behavior = "Fetch time substitutes provider time."
required_migration_proof = "Separate provider and received clocks."

[[capture_risks]]
id = "rss-provider-time-fallback"
source = "trade_py/data/news/rss/base.py"
literal = "pub_time = datetime.now(timezone.utc)"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "provider-timestamp-absence-substitution"
current_behavior = "Missing RSS provider time is replaced with the local collection clock."
required_migration_proof = "Persist provider precision separately from received time."

[[capture_risks]]
id = "archive-date-only-publication-time"
source = "trade_py/archive.py"
literal = "archive_noon"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "date-only-inferred-precision"
current_behavior = "Archive dates gain synthetic time."
required_migration_proof = "Preserve date-only precision."

[[capture_risks]]
id = "rss-catalog-environment-override"
source = "trade_py/catalog.py"
literal = "RSS_OVERRIDE"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "catalog-environment-override-and-absent-rights-evidence"
current_behavior = "Environment can replace the feed catalog."
required_migration_proof = "Version SourceManifest rights."

[[capture_risks]]
id = "gdelt-catalog-db-config"
source = "trade_py/data/news/gdelt/source.py"
literal = 'load_catalog_payload("catalog.feeds.gdelt", "config/feeds/gdelt.json")'
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "db-first-provider-channel-config"
current_behavior = "GDELT channel query, language, enablement, and priority can change from DB-first catalog settings."
required_migration_proof = "Freeze SourceManifest channel configuration digest in CaptureRequest and use CaptureArtifactRef-only replay."

[[capture_risks]]
id = "gdelt-provider-time-fallback"
source = "trade_py/data/news/gdelt/source.py"
literal = "pub = datetime.now(timezone.utc)"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "provider-timestamp-absence-substitution"
current_behavior = "Invalid provider time uses collection time."
required_migration_proof = "Separate provider and received clocks."

[[capture_risks]]
id = "gdelt-streaming-local-state-and-refetch"
source = "trade_py/data/news/gdelt/source.py"
literal = "bronze_offsets = scan_bronze_channel_offsets(data_root)"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "provider-refetch-versus-local-artifact-replay-versus-stateful-stream-cursor"
current_behavior = "Streaming uses mutable state and provider fetches."
required_migration_proof = "Replay immutable CaptureArtifact segments."

[[capture_risks]]
id = "ingest-wal-replay"
source = "trade_py/wal.py"
literal = "replay_wal"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "provider-refetch-versus-local-artifact-replay-versus-wal-recovery"
current_behavior = "WAL replay writes legacy data."
required_migration_proof = "Replay immutable CaptureArtifact references."

[[capture_risks]]
id = "warehouse-semantic-quarantine"
source = "trade_py/quarantine.py"
literal = "quality_status = 'quarantined'"
current_owner = "legacy"
required_child = "dataset-product-boundary"
risk_kind = "transport-integrity-versus-downstream-semantic-quarantine"
current_behavior = "Semantic quality uses a legacy quarantine flag."
required_migration_proof = "Keep Capture transport and Dataset quality separate."

[[interfaces]]
id = "cli"
source = "trade"
literal = "legacy-cli"
current_owner = "legacy"
required_child = "cli-http-sdk-compatibility"
surface_kind = "cli-facade"
current_behavior = "Legacy entrypoint remains available."
compatibility_owner = "interfaces.cli.compat"

[[interfaces]]
id = "cli-domain"
source = "trade_py/cli/main.py"
literal = "CANONICAL_DOMAIN"
current_owner = "legacy"
required_child = "cli-http-sdk-compatibility"
surface_kind = "cli-domain"
current_behavior = "CLI routes a canonical domain."
compatibility_owner = "interfaces.cli.compat"

[[interfaces]]
id = "cli-compatibility"
source = "trade_py/cli/main.py"
literal = "LEGACY_DOMAIN"
current_owner = "legacy"
required_child = "cli-http-sdk-compatibility"
surface_kind = "cli-compatibility"
current_behavior = "CLI keeps a legacy domain."
compatibility_owner = "interfaces.cli.compat"

[[interfaces]]
id = "http-app"
source = "trade_web/backend/app.py"
literal = "def create_app"
current_owner = "legacy"
required_child = "cli-http-sdk-compatibility"
surface_kind = "http-app"
current_behavior = "HTTP application factory."
compatibility_owner = "interfaces.http.compat"

[[interfaces]]
id = "http-openapi"
source = "trade_web/backend/app.py"
literal = "FastAPI("
current_owner = "legacy"
required_child = "cli-http-sdk-compatibility"
surface_kind = "http-openapi"
current_behavior = "Generated OpenAPI surface."
compatibility_owner = "interfaces.http.compat"

[[interfaces]]
id = "http-router"
source = "trade_web/backend/router.py"
literal = "APIRouter"
current_owner = "legacy"
required_child = "cli-http-sdk-compatibility"
surface_kind = "http-router"
current_behavior = "HTTP route registration."
compatibility_owner = "interfaces.http.compat"

[[interfaces]]
id = "sse"
source = "trade_web/backend/sse.py"
literal = "text/event-stream"
current_owner = "legacy"
required_child = "cli-http-sdk-compatibility"
surface_kind = "sse"
current_behavior = "SSE response."
compatibility_owner = "interfaces.http.compat"

[[interfaces]]
id = "http-contract-test"
source = "tests/test_http_contract.py"
literal = "/api/v1/events"
current_owner = "tests"
required_child = "cli-http-sdk-compatibility"
surface_kind = "http-contract-test"
current_behavior = "HTTP contract assertion."
compatibility_owner = "interfaces.http.compat"

[[native_bindings]]
id = "native"
source = "engine/cmake/python_bindings.cmake"
literal = "nanobind_add_module(trade_py"
current_owner = "engine"
required_child = "python-package-and-web-layout"
current_binding = "trade_py"
reserved_binding = "_trade_native"

[[warehouse_producers]]
id = "writer"
source = "{producer_source}"
literal = "{_toml_string(producer_literal)}"
line = {producer_line}
column = {producer_column}
writer = "{producer_writer}"
call_digest = "{producer_digest}"
current_owner = "legacy"
required_child = "dataset-product-boundary"
layer = "{producer_layer}"
table = "{producer_table}"
path_role = "fixture"
artifact_id = "{artifact_id}"
classification = "{classification}"
target_context = "{target_context}"
reason = "Fixture declaration."
{extra}
'''


def _sources(app: str | None = None, *, baseline_app: str = DEFAULT_APP) -> dict[str, str]:
    application = app or DEFAULT_APP
    return {
        "architecture-baseline.toml": _baseline(producer_app=baseline_app),
        "trade": "#!/bin/sh\n# legacy-cli\n",
        "trade_py/__init__.py": "",
        "trade_py/db.py": (
            "LEGACY_DB = 1\n"
            'SQL = "CREATE TABLE legacy_records"\n'
            'CAUSAL_SQL = "CREATE TABLE causal_decision_snapshots"\n'
            'CAUSAL_VALIDATION_SQL = "CREATE TABLE causal_validation_outcomes"\n'
            'CAUSAL_FEEDBACK_SQL = "CREATE TABLE causal_reward_punishment"\n'
            'FACTORS_SQL = "CREATE TABLE factors"\n'
            'FACTOR_REGISTRY_SQL = "CREATE TABLE factor_registry"\n'
            'KG_SQL = "CREATE TABLE kg_nodes"\n'
            'KG_RELATIONS_SQL = "CREATE TABLE kg_relations"\n'
            'KG_CANDIDATES_SQL = "CREATE TABLE kg_edge_candidates"\n'
            'MODELS_SQL = "CREATE TABLE model_registry"\n'
            'MODEL_EVAL_SQL = "CREATE TABLE model_eval_runs"\n'
        ),
        "trade_py/migrations.py": ('SQL = "ALTER TABLE legacy_records ADD COLUMN value"\n'),
        "trade_py/db/__init__.py": "",
        "trade_py/db/trade_db.py": (
            'CAUSAL_SQL = "CREATE TABLE IF NOT EXISTS causal_decision_snapshots"\n'
            'CAUSAL_VALIDATION_SQL = "CREATE TABLE IF NOT EXISTS causal_validation_outcomes"\n'
            'CAUSAL_FEEDBACK_SQL = "CREATE TABLE IF NOT EXISTS causal_reward_punishment"\n'
            'FACTORS_SQL = "CREATE TABLE IF NOT EXISTS factors"\n'
            'FACTOR_REGISTRY_SQL = "CREATE TABLE IF NOT EXISTS factor_registry"\n'
            'KG_SQL = "CREATE TABLE IF NOT EXISTS kg_nodes"\n'
            'KG_RELATIONS_SQL = "CREATE TABLE IF NOT EXISTS kg_relations"\n'
            'KG_CANDIDATES_SQL = "CREATE TABLE IF NOT EXISTS kg_edge_candidates"\n'
            'MODELS_SQL = "CREATE TABLE IF NOT EXISTS model_registry"\n'
            'MODEL_EVAL_SQL = "CREATE TABLE IF NOT EXISTS model_eval_runs"\n'
        ),
        "trade_py/db/migrations.py": (
            'RECOMMENDATION_SQL = "CREATE TABLE IF NOT EXISTS Recommendation"\n'
            'RECOMMENDATION_TRACE_SQL = "CREATE TABLE IF NOT EXISTS RecommendationTrace"\n'
        ),
        "trade_py/warehouse.py": 'path = f"{table}.parquet"\n',
        "trade_py/capture.py": "published_at = None\n",
        "trade_py/cctv.py": "synthetic_noon = True\n",
        "trade_py/rss.py": "published_at or fetched_at\n",
        "trade_py/archive.py": "archive_noon = True\n",
        "trade_py/catalog.py": 'RSS_OVERRIDE = "RSS_OVERRIDE"\n',
        "trade_py/wal.py": "replay_wal = True\n",
        "trade_py/quarantine.py": "quality_status = 'quarantined'\n",
        "trade_py/data/__init__.py": "",
        "trade_py/data/news/__init__.py": "",
        "trade_py/data/news/rss/__init__.py": "",
        "trade_py/data/news/rss/base.py": "pub_time = datetime.now(timezone.utc)\n",
        "trade_py/data/news/gdelt/__init__.py": "",
        "trade_py/data/news/gdelt/source.py": (
            'payload = load_catalog_payload("catalog.feeds.gdelt", "config/feeds/gdelt.json")\n'
            "from datetime import datetime, timezone\n"
            "pub = datetime.now(timezone.utc)\n"
            "bronze_offsets = scan_bronze_channel_offsets(data_root)\n"
        ),
        "trade_py/data/warehouse/__init__.py": (
            "from trade_py.data.warehouse.io import WarehouseLayout, write_table, upsert_table\n"
        ),
        "trade_py/data/warehouse/io.py": (
            "class WarehouseLayout:\n"
            "    @classmethod\n"
            "    def from_data_root(cls, root):\n"
            "        return cls()\n"
            "def write_table(layout, layer, table, frame):\n"
            "    return None\n"
            "def upsert_table(layout, layer, table, frame, *, key_cols):\n"
            "    return None\n"
        ),
        "trade_py/data/operations/checks.py": (
            'path = root / "market" / "kline" / "reconciliation" / "current.json"\n'
        ),
        "trade_py/observatory/catalog/store.py": (
            'return base / "catalog.sqlite", base / "generation.json"\n'
        ),
        "trade_py/app.py": application,
        "trade_py/cli/main.py": "CANONICAL_DOMAIN = True\nLEGACY_DOMAIN = True\n",
        "trade_web/backend/app.py": "def create_app():\n    return FastAPI()\n",
        "trade_web/backend/router.py": "APIRouter = object()\n",
        "trade_web/backend/sse.py": 'MEDIA = "text/event-stream"\n',
        "tests/test_http_contract.py": 'EVENTS = "/api/v1/events"\n',
        "engine/cmake/python_bindings.cmake": "nanobind_add_module(trade_py bindings.cpp)\n",
    }


def _write_baseline(repo: Path, content: str) -> None:
    (repo / BASELINE_FILENAME).write_text(content, encoding="utf-8")


def _rule_ids(repo: Path, *, limits: DiscoveryLimits = DEFAULT_LIMITS) -> set[str]:
    return {
        finding.rule_id for finding in validate_architecture_baseline(repo, limits=limits).findings
    }


def test_repository_baseline_is_complete_and_source_only() -> None:
    report = validate_architecture_baseline(REPO_ROOT)

    assert report.ok, report.findings
    assert len(report.producers) == 19
    assert {(producer.source, producer.artifact_key) for producer in report.producers} >= {
        ("trade_py/cli/data.py", "dim.dim_data_source"),
        ("trade_py/cli/data.py", "ods.ods_fetch_attempt"),
        ("trade_py/data/warehouse/materialize.py", "ads.ads_warehouse_validation_report"),
    }


def test_repository_baseline_includes_review_required_provenance_and_interfaces() -> None:
    from trade_py.devtools.quality.toml_compat import tomllib

    baseline = tomllib.loads((REPO_ROOT / BASELINE_FILENAME).read_text(encoding="utf-8"))
    table_names = {table["logical_name"] for table in baseline["tables"]}
    capture_risk_ids = {risk["id"] for risk in baseline["capture_risks"]}
    interface_kinds = {item["surface_kind"] for item in baseline["interfaces"]}

    assert {
        "ingest_runs",
        "coverage",
        "enrichment_status",
        "causal_decision_snapshots",
        "causal_validation_outcomes",
        "causal_reward_punishment",
        "factors",
        "factor_registry",
        "kg_nodes",
        "kg_relations",
        "kg_edge_candidates",
        "model_registry",
        "model_eval_runs",
        "Recommendation",
        "RecommendationTrace",
    } <= table_names
    assert {
        "rss-provider-time-fallback",
        "gdelt-catalog-db-config",
        "gdelt-provider-time-fallback",
        "gdelt-streaming-local-state-and-refetch",
    } <= capture_risk_ids
    assert "http-openapi" in interface_kinds


def test_required_facts_and_target_context_vocabulary_fail_closed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")

    _write_baseline(
        repo,
        baseline.replace('id = "gdelt-catalog-db-config"', 'id = "removed-gdelt-config"', 1),
    )
    assert "architecture.baseline_malformed" in _rule_ids(repo)

    _write_baseline(
        repo,
        baseline.replace(
            'source = "trade_py/data/operations/checks.py"',
            'source = "trade_py/utils/data_inspector.py"',
            1,
        ),
    )
    assert "architecture.baseline_malformed" in _rule_ids(repo)

    _write_baseline(
        repo,
        baseline.replace(
            '"interfaces", "bootstrap"]',
            '"interfaces", "bootstrap", "evidence"]',
            1,
        ),
    )
    assert "architecture.baseline_malformed" in _rule_ids(repo)

    _write_baseline(
        repo,
        baseline.replace(
            'logical_name = "factor_registry"\ncurrent_owner = "legacy"\n'
            'semantic_kind = "factor-metadata-record"\nclassification = "deferred"',
            'logical_name = "factor_registry"\ncurrent_owner = "legacy"\n'
            'semantic_kind = "factor-metadata-record"\nclassification = "candidate"',
            1,
        ),
    )
    assert "architecture.baseline_malformed" in _rule_ids(repo)

    _write_baseline(
        repo,
        baseline.replace(
            'id = "rss-provider-time-fallback"\nsource = "trade_py/data/news/rss/base.py"',
            'id = "rss-provider-time-fallback"\nsource = "trade_py/data/news/gdelt/source.py"',
            1,
        ),
    )
    assert "architecture.baseline_malformed" in _rule_ids(repo)

    _write_baseline(repo, baseline)
    (repo / "trade_py/data/operations/checks.py").write_text(
        'path = root / "market" / "kline" / "reconciliation" / "retired.json"\n',
        encoding="utf-8",
    )
    assert "architecture.baseline_literal_mismatch" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("mutator", "expected_rule"),
    (
        (
            lambda text: text.replace("schema_version = 1", "schema_version = 2"),
            "architecture.baseline_malformed",
        ),
        (
            lambda text: text.replace(
                'target_context = "deferred"\nreason = "Requires evidence before ownership transfer."',
                'target_context = "datasets"\nreason = "Requires evidence before ownership transfer."',
                1,
            ),
            "architecture.baseline_invalid_classification",
        ),
        (
            lambda text: text.replace(
                'classification = "candidate"\ntarget_context = "datasets"\nreason = "Legacy output requires DatasetVersion migration."',
                'classification = "candidate"\ntarget_context = "datasets"\nadapter_scope = "forbidden"\nreason = "Legacy output requires DatasetVersion migration."',
                1,
            ),
            "architecture.baseline_non_authorizing_binding",
        ),
        (
            lambda text: text.replace(
                'artifact_id = "warehouse-parquet"', 'artifact_id = "missing"'
            ),
            "architecture.baseline_missing_producer_artifact",
        ),
        (
            lambda text: text.replace('role = "bootstrap"', 'role = "not-a-role"', 1),
            "architecture.baseline_incomplete_provenance",
        ),
    ),
)
def test_baseline_schema_and_non_authorizing_states_fail_closed(
    tmp_path: Path,
    mutator,
    expected_rule: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    _write_baseline(repo, mutator((repo / BASELINE_FILENAME).read_text(encoding="utf-8")))

    assert expected_rule in _rule_ids(repo)


def test_missing_or_changed_declared_source_fails_closed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    (repo / "trade_py/db.py").unlink()
    assert "architecture.baseline_missing_source" in _rule_ids(repo)

    repo = _init_repo(tmp_path / "changed", _sources())
    path = repo / "trade_py/capture.py"
    path.write_text("received_at = None\n", encoding="utf-8")
    assert "architecture.baseline_literal_mismatch" in _rule_ids(repo)


def test_canonical_direct_package_and_alias_writers_are_resolved(tmp_path: Path) -> None:
    app = (
        "from trade_py.data.warehouse.io import WarehouseLayout as Layout, write_table as write\n"
        "from trade_py.data.warehouse import upsert_table as merge\n"
        "layout = Layout.from_data_root('data')\n"
        'write(layout, "ods", "events", frame=None)\n'
        'merge(layout, "dwd", "articles", frame=None, key_cols=["id"])\n'
    )
    sources = _sources(app)
    alias_line, alias_column, alias_literal, alias_digest = _producer_identity(
        app,
        layer="dwd",
        table="articles",
        writer=CANONICAL_UPSERT_TABLE,
    )
    sources[BASELINE_FILENAME] = (
        _baseline(
            producer_literal='write(layout, "ods", "events", frame=None)',
            producer_app=app,
        )
        + f"""
[[warehouse_producers]]
id = "alias-upsert"
source = "trade_py/app.py"
literal = "{_toml_string(alias_literal)}"
line = {alias_line}
column = {alias_column}
writer = "trade_py.data.warehouse.io.upsert_table"
call_digest = "{alias_digest}"
current_owner = "legacy"
required_child = "dataset-product-boundary"
layer = "dwd"
table = "articles"
path_role = "fixture"
artifact_id = "warehouse-parquet"
classification = "candidate"
target_context = "datasets"
reason = "Fixture declaration."
"""
    )
    repo = _init_repo(tmp_path, sources)

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert {(producer.writer, producer.artifact_key) for producer in report.producers} == {
        ("trade_py.data.warehouse.io.write_table", "ods.events"),
        ("trade_py.data.warehouse.io.upsert_table", "dwd.articles"),
    }


@pytest.mark.parametrize(
    ("app", "expected_rule"),
    (
        (
            "from trade_py.data.warehouse import unknown_writer\n"
            "unknown_writer(layout, 'ods', 'events', frame=None)\n",
            PRODUCER_UNRESOLVED_IMPORT,
        ),
        (
            "from trade_py.data.warehouse import write_table\n"
            "write_table(unknown_layout, 'ods', 'events', frame=None)\n",
            PRODUCER_UNRESOLVED_LAYOUT,
        ),
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "write_table(layout, layer, 'events', frame=None)\n",
            PRODUCER_NONLITERAL_TARGET,
        ),
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "write_table(layout, 'ads', 'new_output', frame=None)\n",
            PRODUCER_UNDECLARED_WRITER,
        ),
    ),
)
def test_producer_discovery_fails_closed_for_unresolved_or_undeclared_writers(
    tmp_path: Path,
    app: str,
    expected_rule: str,
) -> None:
    repo = _init_repo(tmp_path, _sources(app))

    assert expected_rule in _rule_ids(repo)


def test_test_only_writer_is_not_scanned(tmp_path: Path) -> None:
    sources = _sources()
    sources["trade_py/tests/test_writer.py"] = (
        "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        'write_table(layout, "ads", "test_only", frame=None)\n'
    )
    repo = _init_repo(tmp_path, sources)

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert all(producer.table != "test_only" for producer in report.producers)


def test_path_and_source_budget_fail_before_partial_inventory(tmp_path: Path) -> None:
    sources = _sources()
    sources["trade_py/extra.py"] = "VALUE = 1\n"
    repo = _init_repo(tmp_path, sources)

    path_limits = DiscoveryLimits(
        max_raw_records=100,
        max_raw_path_bytes=100_000,
        max_included_paths=1,
        max_included_path_bytes=100_000,
        max_source_bytes=100_000,
        max_file_bytes=100_000,
    )
    source_limits = DiscoveryLimits(
        max_raw_records=100,
        max_raw_path_bytes=100_000,
        max_included_paths=100,
        max_included_path_bytes=100_000,
        max_source_bytes=10,
        max_file_bytes=100_000,
    )

    path_report = validate_architecture_baseline(repo, limits=path_limits)
    source_report = validate_architecture_baseline(repo, limits=source_limits)

    assert _rule_ids(repo, limits=path_limits) == {PRODUCER_PATH_BUDGET}
    assert _rule_ids(repo, limits=source_limits) == {PRODUCER_SOURCE_BUDGET}
    assert path_report.producers == ()
    assert source_report.producers == ()


def test_symlink_and_replaced_source_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support is unavailable")
    repo = _init_repo(tmp_path, _sources())
    target = repo / "trade_py/app.py"
    external = tmp_path / "external.py"
    external.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    target.unlink()
    target.symlink_to(external)
    assert PRODUCER_UNSAFE_SOURCE in _rule_ids(repo)

    repo = _init_repo(tmp_path / "replacement", _sources())
    import trade_py.devtools.architecture_guard as guard

    original = guard._read_descriptor

    def replace_then_read(descriptor: int, size: int, path: str) -> bytes:
        if path == "trade_py/app.py":
            app_path = repo / path
            backup = repo / "trade_py/app.backup.py"
            app_path.rename(backup)
            app_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        return original(descriptor, size, path)

    monkeypatch.setattr(guard, "_read_descriptor", replace_then_read)
    assert PRODUCER_UNSAFE_SOURCE in _rule_ids(repo)


def test_validator_does_not_use_data_or_runtime_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import sqlite3

    import duckdb
    import pandas as pd

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: pytest.fail("sqlite access"))
    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: pytest.fail("duckdb access"))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("parquet access"))

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings


def test_unsafe_baseline_source_path_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    content = (
        (repo / BASELINE_FILENAME)
        .read_text(encoding="utf-8")
        .replace(
            'source = "trade_py/db.py"',
            'source = "../outside.py"',
            1,
        )
    )
    _write_baseline(repo, content)

    assert "architecture.baseline_unsafe_source" in _rule_ids(repo)


def test_declared_producer_rename_or_deletion_fails(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    shutil.move(repo / "trade_py/app.py", repo / "trade_py/renamed.py")

    assert "architecture.baseline_missing_source" in _rule_ids(repo)


def test_source_only_evidence_policy_rejects_artifacts_without_opening_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    (repo / "data").mkdir()
    (repo / "data" / "sentinel.parquet").write_text("not parquet", encoding="utf-8")
    baseline = (
        (repo / BASELINE_FILENAME)
        .read_text(encoding="utf-8")
        .replace(
            'source = "trade_py/db.py"',
            'source = "data/sentinel.parquet"',
            1,
        )
    )
    _write_baseline(repo, baseline)

    import trade_py.devtools.architecture_guard as guard

    original = guard._safe_read_relative
    opened: list[str] = []

    def record_open(root: Path, relative: str, *, max_bytes: int) -> bytes:
        opened.append(relative)
        return original(root, relative, max_bytes=max_bytes)

    monkeypatch.setattr(guard, "_safe_read_relative", record_open)

    assert "architecture.baseline_unsafe_source" in _rule_ids(repo)
    assert "data/sentinel.parquet" not in opened


@pytest.mark.parametrize(
    "unsafe_source",
    (
        "trade_py/data/news/receipts/segment.py",
        "trade_py/data/news/manifest/segment.py",
        "trade_py/data/news/artifacts/segment.py",
    ),
)
def test_source_only_evidence_rejects_nested_artifact_paths_without_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_source: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    path = repo / unsafe_source
    path.parent.mkdir(parents=True)
    path.write_text("SENTINEL = 1\n", encoding="utf-8")
    baseline = (
        (repo / BASELINE_FILENAME)
        .read_text(encoding="utf-8")
        .replace(
            'source = "trade_py/db.py"',
            f'source = "{unsafe_source}"',
            1,
        )
    )
    _write_baseline(repo, baseline)

    import trade_py.devtools.architecture_guard as guard

    original = guard._safe_read_relative
    opened: list[str] = []

    def record_open(root: Path, relative: str, *, max_bytes: int) -> bytes:
        opened.append(relative)
        return original(root, relative, max_bytes=max_bytes)

    monkeypatch.setattr(guard, "_safe_read_relative", record_open)

    assert "architecture.baseline_unsafe_source" in _rule_ids(repo)
    assert unsafe_source not in opened


def test_source_evidence_is_memoized_and_aggregate_budgeted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    original = guard._safe_read_relative
    reads: list[str] = []

    def record_read(root: Path, relative: str, *, max_bytes: int) -> bytes:
        reads.append(relative)
        return original(root, relative, max_bytes=max_bytes)

    monkeypatch.setattr(guard, "_safe_read_relative", record_read)

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    # One verified evidence read and one producer-universe read, despite two facts.
    assert reads.count("trade_py/observatory/catalog/store.py") == 2

    budget_report = validate_architecture_baseline(
        repo,
        limits=DiscoveryLimits(max_aggregate_evidence_bytes=1),
    )
    assert {finding.rule_id for finding in budget_report.findings} == {
        "architecture.baseline_evidence_budget_exceeded"
    }
    assert budget_report.producers == ()


def test_git_discovery_ignores_inherited_index_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    alternate_index = tmp_path / "alternate.index"
    alternate_index.write_bytes(b"")
    monkeypatch.setenv("GIT_INDEX_FILE", str(alternate_index))

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert len(report.producers) == 1


def test_exact_producer_declarations_fail_for_changed_or_removed_calls(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    changed = (
        "from trade_py.data.warehouse import WarehouseLayout, upsert_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        'upsert_table(layout, "ods", "events", frame=None, key_cols=["id"])\n'
    )
    (repo / "trade_py/app.py").write_text(changed, encoding="utf-8")

    changed_rules = _rule_ids(repo)

    assert PRODUCER_UNDECLARED_WRITER in changed_rules
    assert "architecture.baseline_stale_producer_declaration" in changed_rules

    repo = _init_repo(tmp_path / "removed", _sources())
    (repo / "trade_py/app.py").write_text(
        "# write_table(layout, 'ods', 'events', frame=None)\n", encoding="utf-8"
    )
    assert "architecture.baseline_stale_producer_declaration" in _rule_ids(repo)


def test_rebound_or_shadowed_writer_is_not_misclassified(tmp_path: Path) -> None:
    app = (
        "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        "write_table = lambda *args, **kwargs: None\n"
        'write_table(layout, "ods", "events", frame=None)\n'
    )
    repo = _init_repo(tmp_path, _sources(app, baseline_app=DEFAULT_APP))

    rules = _rule_ids(repo)

    assert PRODUCER_UNRESOLVED_IMPORT in rules


def test_lexical_writer_shadowing_preserves_outer_producer_and_rejects_inner_call(
    tmp_path: Path,
) -> None:
    app = (
        "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        "def helper(write_table):\n"
        '    write_table(layout, "ods", "shadowed", frame=None)\n'
        'write_table(layout, "ods", "events", frame=None)\n'
    )
    repo = _init_repo(tmp_path, _sources(app))

    report = validate_architecture_baseline(repo)

    assert {finding.rule_id for finding in report.findings} == {PRODUCER_UNRESOLVED_IMPORT}
    assert report.producers == ()

    outer_only = (
        "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        "def helper(write_table):\n"
        "    return write_table\n"
        'write_table(layout, "ods", "events", frame=None)\n'
    )
    repo = _init_repo(
        tmp_path / "outer-only",
        _sources(outer_only, baseline_app=outer_only),
    )

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert [(producer.layer, producer.table) for producer in report.producers] == [
        ("ods", "events")
    ]


def test_relative_canonical_warehouse_import_is_resolved(tmp_path: Path) -> None:
    app = (
        "from .io import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        'write_table(layout, "ods", "events", frame=None)\n'
    )
    sources = _sources(app="VALUE = 1\n")
    sources["trade_py/data/warehouse/consumer.py"] = app
    sources[BASELINE_FILENAME] = _baseline(
        producer_source="trade_py/data/warehouse/consumer.py",
        producer_app=app,
    )
    repo = _init_repo(tmp_path, sources)

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert report.producers[0].source == "trade_py/data/warehouse/consumer.py"


def test_result_and_diagnostic_bounds_fail_closed(tmp_path: Path) -> None:
    app = (
        "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        'write_table(layout, "ods", "events", frame=None)\n'
        'write_table(layout, "ads", "other", frame=None)\n'
    )
    repo = _init_repo(tmp_path, _sources(app, baseline_app=DEFAULT_APP))
    producer_limits = DiscoveryLimits(max_discovered_producers=1)
    producer_report = validate_architecture_baseline(repo, limits=producer_limits)

    assert {finding.rule_id for finding in producer_report.findings} == {PRODUCER_RESULT_BUDGET}
    assert producer_report.producers == ()

    duplicate_facts = "\n".join(
        """
[[source_facts]]
id = "legacy-db"
source = "trade_py/db.py"
literal = "LEGACY_DB = 1"
current_owner = "legacy"
required_child = "dataset-product-boundary"
""".strip()
        for _ in range(5)
    )
    _write_baseline(
        repo,
        (repo / BASELINE_FILENAME).read_text(encoding="utf-8") + "\n" + duplicate_facts,
    )
    report_limits = DiscoveryLimits(max_findings=3, max_diagnostic_field_bytes=32)
    bounded_report = validate_architecture_baseline(repo, limits=report_limits)

    assert len(bounded_report.findings) == 3
    assert bounded_report.omitted_findings_count == 2
    assert bounded_report.findings[-1].rule_id == "architecture.guard_result_truncated"
    assert all(
        len(finding.message.encode("utf-8")) <= report_limits.max_diagnostic_field_bytes
        for finding in bounded_report.findings
    )


def test_ast_and_producer_literal_budgets_fail_before_partial_inventory(tmp_path: Path) -> None:
    nested_expression = "[" * 80 + "0" + "]" * 80
    app = (
        "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        f'write_table(layout, "ods", "events", {nested_expression})\n'
    )
    repo = _init_repo(tmp_path, _sources(app, baseline_app=DEFAULT_APP))

    ast_report = validate_architecture_baseline(
        repo,
        limits=DiscoveryLimits(max_ast_nodes_per_file=10),
    )
    literal_report = validate_architecture_baseline(
        repo,
        limits=DiscoveryLimits(max_producer_literal_bytes=32),
    )

    assert {finding.rule_id for finding in ast_report.findings} == {PRODUCER_RESULT_BUDGET}
    assert {finding.rule_id for finding in literal_report.findings} == {PRODUCER_RESULT_BUDGET}
    assert ast_report.producers == ()
    assert literal_report.producers == ()


def test_git_discovery_timeout_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nexec sleep 1\n", encoding="utf-8")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    report = validate_architecture_baseline(repo, limits=DiscoveryLimits(git_timeout_seconds=0.01))

    assert {finding.rule_id for finding in report.findings} == {PRODUCER_TIMEOUT}
    assert report.producers == ()


def test_git_discovery_timeout_cleans_background_stdout_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    child_pid_path = tmp_path / "background-child.pid"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        '#!/bin/sh\nsleep 30 &\nprintf "%s\\n" "$!" > "$ARCH_GUARD_CHILD_PID_FILE"\nexit 0\n',
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ARCH_GUARD_CHILD_PID_FILE", str(child_pid_path))

    report = validate_architecture_baseline(repo, limits=DiscoveryLimits(git_timeout_seconds=0.01))

    assert {finding.rule_id for finding in report.findings} == {PRODUCER_TIMEOUT}
    assert report.producers == ()
    child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("timed-out Git descendant still exists after process-group cleanup")


def test_git_discovery_bounds_unterminated_stdout_and_nonzero_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        '#!/bin/sh\nprintf "x%.0s" $(seq 1 4096)\n',
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    truncated_report = validate_architecture_baseline(
        repo,
        limits=DiscoveryLimits(max_git_record_bytes=64, git_timeout_seconds=1),
    )

    assert {finding.rule_id for finding in truncated_report.findings} == {PRODUCER_PATH_BUDGET}
    assert truncated_report.producers == ()

    fake_git.write_text(
        '#!/bin/sh\nprintf "diagnostic%.0s" $(seq 1 4096) >&2\nexit 9\n',
        encoding="utf-8",
    )
    failed_report = validate_architecture_baseline(
        repo,
        limits=DiscoveryLimits(max_diagnostic_field_bytes=64, git_timeout_seconds=1),
    )

    assert {finding.rule_id for finding in failed_report.findings} == {PRODUCER_TOOL_FAILURE}
    assert failed_report.producers == ()
    assert len(failed_report.findings[0].message.encode("utf-8")) <= 64
