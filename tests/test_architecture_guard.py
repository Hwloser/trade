from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    _call_digest,
    validate_architecture_baseline,
)
from trade_py.devtools.toml_compat import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_WRITE_TABLE = "trade_py.data.warehouse.io.write_table"
CANONICAL_UPSERT_TABLE = "trade_py.data.warehouse.io.upsert_table"
DEFAULT_APP = (
    "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
    "layout = WarehouseLayout.from_data_root('data')\n"
    'write_table(layout, "ods", "events", frame=None)\n'
)
_AUDITED_SCHEMA_EVOLUTION_PROVENANCE = {
    "event_log": (
        ("trade_py/db/trade_db.py", "CREATE TABLE IF NOT EXISTS event_log", "bootstrap"),
        ("trade_py/db/migrations.py", "INSERT OR IGNORE INTO event_log", "data_transform"),
    ),
    "event_handler_runs": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS event_handler_runs",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "CREATE TABLE IF NOT EXISTS event_handler_runs",
            "migration",
        ),
    ),
    "pipeline_dag": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS pipeline_dag",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE pipeline_dag ADD COLUMN config_json TEXT DEFAULT '{}'",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE pipeline_dag ADD COLUMN sync_source TEXT",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE pipeline_dag ADD COLUMN sync_dataset TEXT",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE pipeline_dag ADD COLUMN mode TEXT DEFAULT 'batch'",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "UPDATE pipeline_dag SET sync_source=?, sync_dataset=?",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "UPDATE pipeline_dag SET mode='streaming' WHERE job_name=? AND mode='batch'",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "UPDATE pipeline_dag SET mode='both' WHERE job_name=? AND mode='batch'",
            "data_transform",
        ),
        ("trade_py/db/migrations.py", "DELETE FROM pipeline_dag", "data_transform"),
        (
            "trade_py/db/migrations.py",
            "UPDATE pipeline_dag SET enabled=0 WHERE job_name='sentiment_pipeline'",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "UPDATE pipeline_dag SET enabled=0 WHERE job_name='event_pipeline'",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "UPDATE pipeline_dag SET source=?, emits='', description=?",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "UPDATE pipeline_dag SET description='BTC assurance-gated UTC 日线同步' WHERE job_name='crypto_btc_fetch'",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "UPDATE pipeline_dag SET enabled=0 WHERE job_name='cross_asset_fetch'",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "UPDATE pipeline_dag SET config_json=?, description=? WHERE id=?",
            "data_transform",
        ),
    ),
    "asset_registry": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS asset_registry",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "CREATE INDEX IF NOT EXISTS idx_asset_class ON asset_registry(asset_class, enabled, priority)",
            "migration",
        ),
        ("trade_py/db/migrations.py", "INSERT INTO asset_registry", "data_transform"),
        (
            "trade_py/db/migrations.py",
            "UPDATE asset_registry SET config_json=?, updated_at=CURRENT_TIMESTAMP WHERE asset_id=?",
            "data_transform",
        ),
    ),
    "event_propagations": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS event_propagations",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE event_propagations ADD COLUMN rel_path TEXT",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE event_propagations ADD COLUMN validated_at TIMESTAMP",
            "alter",
        ),
    ),
    "job_runs": (
        ("trade_py/db/trade_db.py", "CREATE TABLE IF NOT EXISTS job_runs", "bootstrap"),
        ("trade_py/db/migrations.py", "ALTER TABLE job_runs ADD COLUMN stage TEXT", "alter"),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE job_runs ADD COLUMN trigger_event_id INTEGER",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE job_runs ADD COLUMN result_summary TEXT",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE job_runs ADD COLUMN symbols_processed INTEGER",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE job_runs ADD COLUMN elapsed_ms INTEGER",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE job_runs ADD COLUMN completed_at TIMESTAMP",
            "alter",
        ),
    ),
    "instruments": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS instruments",
            "bootstrap",
        ),
        (
            "trade_py/db/trade_db.py",
            "ALTER TABLE instruments ADD COLUMN total_shares INTEGER DEFAULT 0",
            "alter",
        ),
        (
            "trade_py/db/trade_db.py",
            "ALTER TABLE instruments ADD COLUMN float_shares INTEGER DEFAULT 0",
            "alter",
        ),
        (
            "trade_py/db/trade_db.py",
            "ALTER TABLE instruments ADD COLUMN market_name TEXT NOT NULL DEFAULT ''",
            "alter",
        ),
    ),
    "kg_relations": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS kg_relations",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE kg_relations ADD COLUMN direction INTEGER NOT NULL DEFAULT 1",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE kg_relations ADD COLUMN typical_days INTEGER NOT NULL DEFAULT 0",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE kg_relations ADD COLUMN confidence REAL NOT NULL DEFAULT 0.0",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE kg_relations ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 0",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE kg_relations ADD COLUMN evidence_json TEXT",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE kg_relations ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "alter",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE kg_relations ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "alter",
        ),
        ("trade_py/db/migrations.py", "UPDATE kg_relations", "data_transform"),
        (
            "trade_py/db/migrations.py",
            "UPDATE kg_relations SET weight = ABS(weight) WHERE weight < 0",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "UPDATE kg_relations SET status = 'active' WHERE status IS NULL OR status = ''",
            "data_transform",
        ),
    ),
    "kg_edge_candidates": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS kg_edge_candidates",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "CREATE TABLE IF NOT EXISTS kg_edge_candidates",
            "migration",
        ),
    ),
    "model_registry": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS model_registry",
            "bootstrap",
        ),
        (
            "trade_py/db/trade_db.py",
            "ALTER TABLE model_registry ADD COLUMN target_name TEXT",
            "alter",
        ),
        (
            "trade_py/db/trade_db.py",
            "ALTER TABLE model_registry ADD COLUMN backend TEXT DEFAULT 'lgbm'",
            "alter",
        ),
        (
            "trade_py/db/trade_db.py",
            "ALTER TABLE model_registry ADD COLUMN artifact_format TEXT DEFAULT 'joblib'",
            "alter",
        ),
        (
            "trade_py/db/trade_db.py",
            "ALTER TABLE model_registry ADD COLUMN feature_set TEXT",
            "alter",
        ),
        (
            "trade_py/db/trade_db.py",
            "ALTER TABLE model_registry ADD COLUMN dataset_snapshot_id INTEGER",
            "alter",
        ),
        (
            "trade_py/db/trade_db.py",
            "ALTER TABLE model_registry ADD COLUMN promotion_state TEXT NOT NULL DEFAULT 'active'",
            "alter",
        ),
        (
            "trade_py/db/trade_db.py",
            "UPDATE model_registry SET target_name=COALESCE(target_name, model_name)",
            "data_transform",
        ),
        ("trade_py/db/trade_db.py", "UPDATE model_registry SET backend=", "data_transform"),
        (
            "trade_py/db/trade_db.py",
            "UPDATE model_registry SET artifact_format=",
            "data_transform",
        ),
        (
            "trade_py/db/trade_db.py",
            "UPDATE model_registry SET promotion_state=",
            "data_transform",
        ),
    ),
    "signals": (
        ("trade_py/db/trade_db.py", "CREATE TABLE IF NOT EXISTS signals", "bootstrap"),
        ("trade_py/db/migrations.py", "INSERT OR IGNORE INTO signals", "data_transform"),
    ),
    "sector_members": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS sector_members",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "INSERT OR IGNORE INTO sector_members",
            "data_transform",
        ),
    ),
    "market_events": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS market_events",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "INSERT OR IGNORE INTO market_events",
            "data_transform",
        ),
    ),
    "sync_state": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS sync_state",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "INSERT OR IGNORE INTO sync_state",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "INSERT OR REPLACE INTO sync_state",
            "data_transform",
        ),
    ),
    "ui_snapshots": (
        (
            "trade_py/db/trade_db.py",
            "CREATE TABLE IF NOT EXISTS ui_snapshots",
            "bootstrap",
        ),
        (
            "trade_py/db/migrations.py",
            "CREATE TABLE IF NOT EXISTS ui_snapshots",
            "migration",
        ),
    ),
    "signal_cache_v2": (
        (
            "trade_py/db/migrations.py",
            "CREATE TABLE IF NOT EXISTS signal_cache_v2",
            "migration",
        ),
        (
            "trade_py/db/migrations.py",
            "INSERT OR IGNORE INTO signal_cache_v2",
            "data_transform",
        ),
        (
            "trade_py/db/migrations.py",
            "ALTER TABLE signal_cache_v2 RENAME TO signal_cache",
            "alter",
        ),
    ),
    "bus_events": (
        (
            "trade_py/db/migrations.py",
            "CREATE TABLE IF NOT EXISTS bus_events",
            "migration",
        ),
        ("trade_py/db/migrations.py", "DROP TABLE IF EXISTS bus_events", "alter"),
    ),
}
_REQUIRED_TABLE_FIXTURES = (
    (
        "event_log",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS event_log",
        "candidate",
        "platform",
        "process-manager-and-platform-boundary",
        "bootstrap",
    ),
    (
        "pipeline_dag",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS pipeline_dag",
        "deferred",
        "deferred",
        "process-manager-and-platform-boundary",
        "bootstrap",
    ),
    (
        "asset_registry",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS asset_registry",
        "candidate",
        "capture",
        "capture-boundary",
        "bootstrap",
    ),
    (
        "settings",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS settings",
        "candidate",
        "platform",
        "process-manager-and-platform-boundary",
        "bootstrap",
    ),
    (
        "watchlist",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS watchlist",
        "candidate",
        "decision_support",
        "decision-support-boundary",
        "bootstrap",
    ),
    (
        "signals",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS signals",
        "candidate",
        "decision_support",
        "decision-support-boundary",
        "bootstrap",
    ),
    (
        "job_runs",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS job_runs",
        "candidate",
        "platform",
        "process-manager-and-platform-boundary",
        "bootstrap",
    ),
    (
        "instruments",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS instruments",
        "candidate",
        "datasets",
        "dataset-product-boundary",
        "bootstrap",
    ),
    (
        "sector_members",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS sector_members",
        "candidate",
        "datasets",
        "dataset-product-boundary",
        "bootstrap",
    ),
    (
        "sync_state",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS sync_state",
        "candidate",
        "capture",
        "capture-boundary",
        "bootstrap",
    ),
    (
        "trading_calendar",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS trading_calendar",
        "candidate",
        "datasets",
        "dataset-product-boundary",
        "bootstrap",
    ),
    (
        "planned_events",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS planned_events",
        "candidate",
        "datasets",
        "dataset-product-boundary",
        "bootstrap",
    ),
    (
        "agenda_queue",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS agenda_queue",
        "candidate",
        "processes",
        "process-manager-and-platform-boundary",
        "bootstrap",
    ),
    (
        "backup_snapshots",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS backup_snapshots",
        "candidate",
        "platform",
        "process-manager-and-platform-boundary",
        "bootstrap",
    ),
    (
        "ui_snapshots",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS ui_snapshots",
        "candidate",
        "interfaces",
        "cli-http-sdk-compatibility",
        "bootstrap",
    ),
    (
        "readiness_recovery_actions",
        "trade_py/db/trade_db.py",
        "CREATE TABLE IF NOT EXISTS readiness_recovery_actions",
        "candidate",
        "processes",
        "process-manager-and-platform-boundary",
        "bootstrap",
    ),
    (
        "schema_migrations",
        "trade_py/db/migrations.py",
        "CREATE TABLE IF NOT EXISTS schema_migrations",
        "candidate",
        "platform",
        "process-manager-and-platform-boundary",
        "migration",
    ),
    (
        "signal_cache_v2",
        "trade_py/db/migrations.py",
        "CREATE TABLE IF NOT EXISTS signal_cache_v2",
        "deferred",
        "deferred",
        "decision-support-boundary",
        "migration",
    ),
    (
        "bus_events",
        "trade_py/db/migrations.py",
        "CREATE TABLE IF NOT EXISTS bus_events",
        "deferred",
        "deferred",
        "process-manager-and-platform-boundary",
        "migration",
    ),
)
_REQUIRED_TABLE_EXTRA_PROVENANCE = {
    table: requirements[1:] for table, requirements in _AUDITED_SCHEMA_EVOLUTION_PROVENANCE.items()
}


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
            digest = _call_digest(node)
            return node.lineno, node.col_offset, ast.unparse(node), digest
    raise AssertionError(f"fixture has no {writer} producer for {layer}.{table}")


def _toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _required_table_declarations() -> str:
    declarations = []
    for (
        name,
        source,
        literal,
        classification,
        target_context,
        required_child,
        role,
    ) in _REQUIRED_TABLE_FIXTURES:
        provenance = [f'  {{ source = "{source}", literal = "{literal}", role = "{role}" }},']
        provenance.extend(
            f'  {{ source = "{extra_source}", literal = "{extra_literal}", '
            f'role = "{extra_role}" }},'
            for extra_source, extra_literal, extra_role in _REQUIRED_TABLE_EXTRA_PROVENANCE.get(
                name, ()
            )
        )
        provenance_text = "\n".join(provenance)
        declarations.append(
            f'''
[[tables]]
logical_name = "{name}"
current_owner = "legacy"
semantic_kind = "reviewed-legacy-schema"
classification = "{classification}"
target_context = "{target_context}"
reason = "Fixture declaration."
required_child = "{required_child}"
provenance = [
{provenance_text}
]
'''.strip()
        )
    return "\n\n".join(declarations)


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
logical_name = "event_handler_runs"
current_owner = "legacy"
semantic_kind = "event-delivery-state"
classification = "candidate"
target_context = "platform"
reason = "Target ownership requires a reviewed Platform migration."
required_child = "process-manager-and-platform-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS event_handler_runs", role = "bootstrap" }},
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS event_handler_runs", role = "migration" }},
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
  {{ source = "trade_py/db/migrations.py", literal = "ALTER TABLE kg_relations ADD COLUMN direction INTEGER NOT NULL DEFAULT 1", role = "alter" }},
  {{ source = "trade_py/db/migrations.py", literal = "ALTER TABLE kg_relations ADD COLUMN typical_days INTEGER NOT NULL DEFAULT 0", role = "alter" }},
  {{ source = "trade_py/db/migrations.py", literal = "ALTER TABLE kg_relations ADD COLUMN confidence REAL NOT NULL DEFAULT 0.0", role = "alter" }},
  {{ source = "trade_py/db/migrations.py", literal = "ALTER TABLE kg_relations ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 0", role = "alter" }},
  {{ source = "trade_py/db/migrations.py", literal = "ALTER TABLE kg_relations ADD COLUMN evidence_json TEXT", role = "alter" }},
  {{ source = "trade_py/db/migrations.py", literal = "ALTER TABLE kg_relations ADD COLUMN status TEXT NOT NULL DEFAULT 'active'", role = "alter" }},
  {{ source = "trade_py/db/migrations.py", literal = "ALTER TABLE kg_relations ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP", role = "alter" }},
  {{ source = "trade_py/db/migrations.py", literal = "UPDATE kg_relations", role = "data_transform" }},
  {{ source = "trade_py/db/migrations.py", literal = "UPDATE kg_relations SET weight = ABS(weight) WHERE weight < 0", role = "data_transform" }},
  {{ source = "trade_py/db/migrations.py", literal = "UPDATE kg_relations SET status = 'active' WHERE status IS NULL OR status = ''", role = "data_transform" }},
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
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS kg_edge_candidates", role = "migration" }},
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
  {{ source = "trade_py/db/trade_db.py", literal = "ALTER TABLE model_registry ADD COLUMN target_name TEXT", role = "alter" }},
  {{ source = "trade_py/db/trade_db.py", literal = "ALTER TABLE model_registry ADD COLUMN backend TEXT DEFAULT 'lgbm'", role = "alter" }},
  {{ source = "trade_py/db/trade_db.py", literal = "ALTER TABLE model_registry ADD COLUMN artifact_format TEXT DEFAULT 'joblib'", role = "alter" }},
  {{ source = "trade_py/db/trade_db.py", literal = "ALTER TABLE model_registry ADD COLUMN feature_set TEXT", role = "alter" }},
  {{ source = "trade_py/db/trade_db.py", literal = "ALTER TABLE model_registry ADD COLUMN dataset_snapshot_id INTEGER", role = "alter" }},
  {{ source = "trade_py/db/trade_db.py", literal = "ALTER TABLE model_registry ADD COLUMN promotion_state TEXT NOT NULL DEFAULT 'active'", role = "alter" }},
  {{ source = "trade_py/db/trade_db.py", literal = "UPDATE model_registry SET target_name=COALESCE(target_name, model_name)", role = "data_transform" }},
  {{ source = "trade_py/db/trade_db.py", literal = "UPDATE model_registry SET backend=", role = "data_transform" }},
  {{ source = "trade_py/db/trade_db.py", literal = "UPDATE model_registry SET artifact_format=", role = "data_transform" }},
  {{ source = "trade_py/db/trade_db.py", literal = "UPDATE model_registry SET promotion_state=", role = "data_transform" }},
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

[[tables]]
logical_name = "source_health_daily"
current_owner = "legacy"
semantic_kind = "source-health-quality-projection"
classification = "candidate"
target_context = "datasets"
reason = "Requires immutable Dataset quality lineage before ownership transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS source_health_daily", role = "bootstrap" }},
]

[[tables]]
logical_name = "source_eval_daily"
current_owner = "legacy"
semantic_kind = "source-evaluation-quality-projection"
classification = "candidate"
target_context = "datasets"
reason = "Requires immutable Dataset quality lineage before ownership transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS source_eval_daily", role = "bootstrap" }},
]

[[tables]]
logical_name = "event_eval_runs"
current_owner = "legacy"
semantic_kind = "event-study-evaluation-record"
classification = "candidate"
target_context = "studies"
reason = "Requires registered Study and immutable Dataset inputs before ownership transfer."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS event_eval_runs", role = "bootstrap" }},
]

[[tables]]
logical_name = "dataset_snapshots"
current_owner = "legacy"
semantic_kind = "dataset-snapshot-projection"
classification = "candidate"
target_context = "datasets"
reason = "Requires immutable DatasetSnapshot references before ownership transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS dataset_snapshots", role = "bootstrap" }},
]

[[tables]]
logical_name = "daily_quality_gate"
current_owner = "legacy"
semantic_kind = "dataset-quality-gate"
classification = "candidate"
target_context = "datasets"
reason = "Requires Dataset release policy evidence before ownership transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS daily_quality_gate", role = "bootstrap" }},
]

[[tables]]
logical_name = "event_templates"
current_owner = "legacy"
semantic_kind = "event-template-record"
classification = "deferred"
target_context = "deferred"
reason = "Requires feature classification evidence before ownership transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS event_templates", role = "bootstrap" }},
]

[[tables]]
logical_name = "market_events"
current_owner = "legacy"
semantic_kind = "market-event-record"
classification = "deferred"
target_context = "deferred"
reason = "Requires immutable Dataset lineage before ownership transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS market_events", role = "bootstrap" }},
  {{ source = "trade_py/db/migrations.py", literal = "INSERT OR IGNORE INTO market_events", role = "data_transform" }},
]

[[tables]]
logical_name = "event_propagations"
current_owner = "legacy"
semantic_kind = "event-propagation-record"
classification = "deferred"
target_context = "deferred"
reason = "Requires feature and validation ownership evidence before transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/trade_db.py", literal = "CREATE TABLE IF NOT EXISTS event_propagations", role = "bootstrap" }},
  {{ source = "trade_py/db/migrations.py", literal = "ALTER TABLE event_propagations ADD COLUMN rel_path TEXT", role = "alter" }},
  {{ source = "trade_py/db/migrations.py", literal = "ALTER TABLE event_propagations ADD COLUMN validated_at TIMESTAMP", role = "alter" }},
]

[[tables]]
logical_name = "ArticleEvent"
current_owner = "legacy"
semantic_kind = "canonical-article-event-record"
classification = "candidate"
target_context = "datasets"
reason = "Requires immutable Dataset lineage before ownership transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS ArticleEvent", role = "migration" }},
]

[[tables]]
logical_name = "InfluenceSignal"
current_owner = "legacy"
semantic_kind = "source-influence-record"
classification = "deferred"
target_context = "deferred"
reason = "Requires source and Study ownership evidence before transfer."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS InfluenceSignal", role = "migration" }},
]

[[tables]]
logical_name = "Evidence"
current_owner = "legacy"
semantic_kind = "derived-evidence-record"
classification = "deferred"
target_context = "deferred"
reason = "Requires Dataset and Study boundary evidence before transfer."
required_child = "study-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS Evidence", role = "migration" }},
]

[[tables]]
logical_name = "BeliefState"
current_owner = "legacy"
semantic_kind = "decision-belief-state"
classification = "deferred"
target_context = "deferred"
reason = "Requires Decision Support audit and Study provenance before transfer."
required_child = "decision-support-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS BeliefState", role = "migration" }},
]

[[tables]]
logical_name = "AttentionScore"
current_owner = "legacy"
semantic_kind = "decision-attention-record"
classification = "deferred"
target_context = "deferred"
reason = "Requires Decision Support audit and Study provenance before transfer."
required_child = "decision-support-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS AttentionScore", role = "migration" }},
]

[[tables]]
logical_name = "BeliefTransition"
current_owner = "legacy"
semantic_kind = "decision-belief-transition"
classification = "deferred"
target_context = "deferred"
reason = "Requires Decision Support audit and Study provenance before transfer."
required_child = "decision-support-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS BeliefTransition", role = "migration" }},
]

[[tables]]
logical_name = "QualityReport"
current_owner = "legacy"
semantic_kind = "dataset-quality-report"
classification = "candidate"
target_context = "datasets"
reason = "Requires Dataset quality lifecycle evidence before ownership transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS QualityReport", role = "migration" }},
]

[[tables]]
logical_name = "FreshnessStatus"
current_owner = "legacy"
semantic_kind = "dataset-freshness-projection"
classification = "candidate"
target_context = "datasets"
reason = "Requires immutable Dataset availability evidence before transfer."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/db/migrations.py", literal = "CREATE TABLE IF NOT EXISTS FreshnessStatus", role = "migration" }},
]

[[tables]]
logical_name = "catalog_meta"
current_owner = "legacy"
semantic_kind = "rebuildable-catalog-projection"
classification = "candidate"
target_context = "datasets"
reason = "The catalog is a rebuildable projection."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/observatory/catalog/store.py", literal = "CREATE TABLE catalog_meta", role = "bootstrap" }},
]

[[tables]]
logical_name = "catalog_runs"
current_owner = "legacy"
semantic_kind = "rebuildable-catalog-projection"
classification = "candidate"
target_context = "datasets"
reason = "Catalog runs are projection facts."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/observatory/catalog/store.py", literal = "CREATE TABLE runs", role = "bootstrap" }},
]

[[tables]]
logical_name = "catalog_releases"
current_owner = "legacy"
semantic_kind = "rebuildable-catalog-projection"
classification = "candidate"
target_context = "datasets"
reason = "Catalog releases are projection facts."
required_child = "dataset-product-boundary"
provenance = [
  {{ source = "trade_py/observatory/catalog/store.py", literal = "CREATE TABLE releases", role = "bootstrap" }},
]

{_required_table_declarations()}

[[artifacts]]
id = "warehouse-parquet"
source = "trade_py/data/warehouse/io.py"
literal = 'f"{{table}}.parquet"'
current_owner = "legacy"
role = "legacy-warehouse-artifact-family"
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
id = "crypto-ads-current-pointer"
source = "trade_py/data/warehouse/crypto_store.py"
literal = 'CRYPTO_VALIDATION_CURRENT = "_crypto_validation_current.json"'
current_owner = "legacy"
role = "legacy-current-pointer"
classification = "candidate"
target_context = "datasets"
reason = "The current pointer is a compatibility and rollback input."
required_child = "dataset-product-boundary"

[[artifacts]]
id = "crypto-ads-validation-receipt"
source = "trade_py/data/warehouse/crypto_store.py"
literal = 'receipt_root = ads_root / "_validation_receipts"'
current_owner = "legacy"
role = "completion-receipt"
classification = "candidate"
target_context = "datasets"
reason = "The receipt is a compatibility and rollback input."
required_child = "dataset-product-boundary"

[[artifacts]]
id = "btc-compatibility-pointer"
source = "trade_py/data/market/crypto/store.py"
literal = 'self.current_path = self.crypto_root / "btc_current.json"'
current_owner = "legacy"
role = "legacy-current-pointer"
classification = "candidate"
target_context = "datasets"
reason = "The BTC pointer is a compatibility and rollback input."
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

[[artifacts]]
id = "kline-reconciliation-pointer"
source = "trade_py/utils/data_inspector.py"
literal = 'return KLINE_DIR(data_root) / "reconciliation" / "current.json"'
current_owner = "legacy"
role = "legacy-reconciliation-pointer"
classification = "deferred"
target_context = "deferred"
reason = "The inspection path reads a legacy reconciliation pointer."
required_child = "dataset-product-boundary"

[[capture_risks]]
id = "raw-record-single-publication-clock"
source = "trade_py/intelligence/raw_record.py"
literal = "published_at: datetime"
current_owner = "trade_py.intelligence"
required_child = "capture-boundary"
risk_kind = "provider-observed-received-available-revision-clocks-collapsed"
current_behavior = "RawRecord exposes one published_at field for all temporal semantics."
required_migration_proof = "Independent provider, observed, received, available, revision, and finality clocks."

[[capture_risks]]
id = "cctv-date-only-publication-time"
source = "trade_py/data/news/akshare_news.py"
literal = "pub = datetime(cur.year, cur.month, cur.day, 12, 0, 0, tzinfo=CST)"
current_owner = "trade_py.data.news"
required_child = "capture-boundary"
risk_kind = "date-only-inferred-precision"
current_behavior = "A date-only provider value is converted to a synthetic noon timestamp."
required_migration_proof = "Preserve source precision and prohibit unproven point-in-time publication claims."

[[capture_risks]]
id = "eastmoney-stock-timezone-overwrite"
source = "trade_py/data/news/akshare_news.py"
literal = "pub = pub_raw.to_pydatetime().replace(tzinfo=CST)"
current_owner = "trade_py.data.news"
required_child = "capture-boundary"
risk_kind = "provider-timezone-and-precision-overwrite"
current_behavior = "Parsed provider timestamps are relabeled CST without preserving source timezone or precision."
required_migration_proof = "Preserve provider timezone and precision, and record observed, received, and available clocks before point-in-time use."

[[capture_risks]]
id = "warehouse-rss-fetched-time-substitution"
source = "trade_py/data/warehouse/fetch.py"
literal = '"published_at": published_at or fetched_at'
current_owner = "trade_py.data.warehouse"
required_child = "capture-boundary"
risk_kind = "provider-timestamp-absence-substitution"
current_behavior = "Missing provider publication time falls back to fetch time."
required_migration_proof = "Record provider time and received time separately in CaptureArtifact metadata."

[[capture_risks]]
id = "rss-provider-time-fallback"
source = "trade_py/data/news/rss/base.py"
literal = "pub_time = datetime.now(timezone.utc)"
current_owner = "trade_py.data.news.rss"
required_child = "capture-boundary"
risk_kind = "provider-timestamp-absence-substitution"
current_behavior = "RSS entries without a provider timestamp substitute the local collection clock."
required_migration_proof = "Persist provider precision separately from observed and received time, and prohibit synthetic event-time PIT claims."

[[capture_risks]]
id = "archive-date-only-publication-time"
source = "trade_py/data/news/rss/archive.py"
literal = "return datetime.combine(day, time(12, 0), tzinfo=timezone.utc)"
current_owner = "trade_py.data.news.rss"
required_child = "capture-boundary"
risk_kind = "date-only-inferred-precision"
current_behavior = "Archive day values become synthetic UTC noon timestamps."
required_migration_proof = "Retain date-only precision and use an explicit availability policy."

[[capture_risks]]
id = "rss-catalog-environment-override"
source = "trade_py/data/news/rss/catalog.py"
literal = 'override = os.environ.get("TRADE_RSS_FEED_INDEX_PATH")'
current_owner = "trade_py.data.news.rss"
required_child = "capture-boundary"
risk_kind = "catalog-environment-override-and-absent-rights-evidence"
current_behavior = "An environment variable can replace the feed index without an immutable SourceManifest."
required_migration_proof = "Versioned SourceManifest with source rights, credentials, and override audit evidence."

[[capture_risks]]
id = "gdelt-catalog-db-config"
source = "trade_py/data/news/gdelt/source.py"
literal = 'load_catalog_payload("catalog.feeds.gdelt", "config/feeds/gdelt.json")'
current_owner = "trade_py.data.news.gdelt"
required_child = "capture-boundary"
risk_kind = "db-first-provider-channel-config"
current_behavior = "GDELT channel query, language, enablement, and priority are selected from mutable DB-first catalog settings."
required_migration_proof = "Freeze a SourceManifest channel configuration digest in CaptureRequest and support CaptureArtifactRef-only replay without provider access."

[[capture_risks]]
id = "gdelt-provider-time-fallback"
source = "trade_py/data/news/gdelt/source.py"
literal = "pub = datetime.now(timezone.utc)"
current_owner = "trade_py.data.news.gdelt"
required_child = "capture-boundary"
risk_kind = "provider-timestamp-absence-substitution"
current_behavior = "Invalid or absent GDELT seendate is replaced with the local collection clock."
required_migration_proof = "Persist provider precision separately from received time and prohibit synthetic event-time PIT claims."

[[capture_risks]]
id = "gdelt-streaming-local-state-and-refetch"
source = "trade_py/data/news/gdelt/source.py"
literal = "bronze_offsets = scan_bronze_channel_offsets(data_root)"
current_owner = "trade_py.data.news.gdelt"
required_child = "capture-boundary"
risk_kind = "provider-refetch-versus-local-artifact-replay-versus-stateful-stream-cursor"
current_behavior = "Streaming scans mutable Bronze Parquet and database cursor state while re-fetching the provider and writing Parquet."
required_migration_proof = "Capture checkpoints and immutable segments must support provider-free replay, revision identity, and bounded retry receipts."

[[capture_risks]]
id = "ingest-wal-replay"
source = "trade_py/data/ingest/batch.py"
literal = "self._recover_wal()"
current_owner = "trade_py.data.ingest"
required_child = "capture-boundary"
risk_kind = "provider-refetch-versus-local-artifact-replay-versus-wal-recovery"
current_behavior = "WAL recovery writes legacy parquet before a formal Capture receipt exists."
required_migration_proof = "Provider-free replay from immutable CaptureArtifact references and explicit replay receipts."

[[capture_risks]]
id = "warehouse-semantic-quarantine"
source = "trade_py/data/warehouse/articles.py"
literal = 'quality_status = "quarantined"'
current_owner = "trade_py.data.warehouse"
required_child = "dataset-product-boundary"
risk_kind = "transport-integrity-versus-downstream-semantic-quarantine"
current_behavior = "Article semantic quality marks rows quarantined in the warehouse transform."
required_migration_proof = "Capture transport failures remain distinct from Datasets semantic quality quarantine."

[[capture_risks]]
id = "influence-signal-runtime-publication-time"
source = "trade_py/intelligence/feed_scorer.py"
literal = "published_at = datetime.now(timezone.utc).isoformat()"
current_owner = "trade_py.intelligence.feed_scorer"
required_child = "study-boundary"
risk_kind = "runtime-evaluation-time-substituted-for-publication-time"
current_behavior = "Feed scorer uses the local evaluation clock as InfluenceSignal published_at, which is then used to select the most recent reliability record."
required_migration_proof = "Separate source publication, observed, received, evaluation, available, and revision clocks before a Dataset or Study publishes an InfluenceSignal-derived result."

[[dynamic_sql_limitations]]
id = "recommendation-dynamic-columns"
logical_name = "Recommendation"
source = "trade_py/db/migrations.py"
literal = 'conn.execute(f"ALTER TABLE Recommendation ADD COLUMN {{col_def}}")'
limitation_kind = "dynamic_ddl"
owning_child = "decision-support-boundary"
non_authorizing = true
limitation = "The f-string column definition is dynamic DDL and is non-authorizing until the Decision Support migration adds reviewed SQL-normalization or runtime migration evidence."

[[dynamic_sql_limitations]]
id = "recommendation-trace-dynamic-columns"
logical_name = "RecommendationTrace"
source = "trade_py/db/migrations.py"
literal = 'conn.execute(f"ALTER TABLE RecommendationTrace ADD COLUMN {{col_def}}")'
limitation_kind = "dynamic_ddl"
owning_child = "decision-support-boundary"
non_authorizing = true
limitation = "The f-string column definition is dynamic DDL and is non-authorizing until the Decision Support migration adds reviewed SQL-normalization or runtime migration evidence."

[[dynamic_sql_limitations]]
id = "factor-registry-dynamic-columns"
logical_name = "factor_registry"
source = "trade_py/db/migrations.py"
literal = 'f"ALTER TABLE factor_registry ADD COLUMN {{col}} REAL NOT NULL DEFAULT {{default}}"'
limitation_kind = "dynamic_ddl"
owning_child = "study-boundary"
non_authorizing = true
limitation = "The f-string column and default are dynamic DDL and are non-authorizing until the Study migration adds reviewed SQL-normalization or runtime migration evidence."

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
        "trade": "#!/bin/sh\nlegacy-cli\n",
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
            'SETTINGS_SQL = "CREATE TABLE IF NOT EXISTS settings"\n'
            'WATCHLIST_SQL = "CREATE TABLE IF NOT EXISTS watchlist"\n'
            'SIGNALS_SQL = "CREATE TABLE IF NOT EXISTS signals"\n'
            'EVENT_LOG_SQL = "CREATE TABLE IF NOT EXISTS event_log"\n'
            'PIPELINE_DAG_SQL = "CREATE TABLE IF NOT EXISTS pipeline_dag"\n'
            'ASSET_REGISTRY_SQL = "CREATE TABLE IF NOT EXISTS asset_registry"\n'
            'JOB_RUNS_SQL = "CREATE TABLE IF NOT EXISTS job_runs"\n'
            'EVENT_HANDLER_RUNS_SQL = "CREATE TABLE IF NOT EXISTS event_handler_runs"\n'
            'INSTRUMENTS_SQL = "CREATE TABLE IF NOT EXISTS instruments"\n'
            'SECTOR_MEMBERS_SQL = "CREATE TABLE IF NOT EXISTS sector_members"\n'
            'SYNC_STATE_SQL = "CREATE TABLE IF NOT EXISTS sync_state"\n'
            'TRADING_CALENDAR_SQL = "CREATE TABLE IF NOT EXISTS trading_calendar"\n'
            'PLANNED_EVENTS_SQL = "CREATE TABLE IF NOT EXISTS planned_events"\n'
            'AGENDA_QUEUE_SQL = "CREATE TABLE IF NOT EXISTS agenda_queue"\n'
            'BACKUP_SNAPSHOTS_SQL = "CREATE TABLE IF NOT EXISTS backup_snapshots"\n'
            'UI_SNAPSHOTS_SQL = "CREATE TABLE IF NOT EXISTS ui_snapshots"\n'
            'RECOVERY_ACTIONS_SQL = "CREATE TABLE IF NOT EXISTS readiness_recovery_actions"\n'
            'SOURCE_HEALTH_SQL = "CREATE TABLE IF NOT EXISTS source_health_daily"\n'
            'SOURCE_EVAL_SQL = "CREATE TABLE IF NOT EXISTS source_eval_daily"\n'
            'EVENT_EVAL_SQL = "CREATE TABLE IF NOT EXISTS event_eval_runs"\n'
            'DATASET_SNAPSHOT_SQL = "CREATE TABLE IF NOT EXISTS dataset_snapshots"\n'
            'QUALITY_GATE_SQL = "CREATE TABLE IF NOT EXISTS daily_quality_gate"\n'
            'EVENT_TEMPLATE_SQL = "CREATE TABLE IF NOT EXISTS event_templates"\n'
            'MARKET_EVENT_SQL = "CREATE TABLE IF NOT EXISTS market_events"\n'
            'EVENT_PROPAGATION_SQL = "CREATE TABLE IF NOT EXISTS event_propagations"\n'
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
            'INSTRUMENTS_ALTER_TOTAL_SHARES = "ALTER TABLE instruments ADD COLUMN total_shares INTEGER DEFAULT 0"\n'
            'INSTRUMENTS_ALTER_FLOAT_SHARES = "ALTER TABLE instruments ADD COLUMN float_shares INTEGER DEFAULT 0"\n'
            "INSTRUMENTS_ALTER_MARKET_NAME = \"ALTER TABLE instruments ADD COLUMN market_name TEXT NOT NULL DEFAULT ''\"\n"
            'MODEL_REGISTRY_ALTER_TARGET = "ALTER TABLE model_registry ADD COLUMN target_name TEXT"\n'
            "MODEL_REGISTRY_ALTER_BACKEND = \"ALTER TABLE model_registry ADD COLUMN backend TEXT DEFAULT 'lgbm'\"\n"
            "MODEL_REGISTRY_ALTER_FORMAT = \"ALTER TABLE model_registry ADD COLUMN artifact_format TEXT DEFAULT 'joblib'\"\n"
            'MODEL_REGISTRY_ALTER_FEATURE_SET = "ALTER TABLE model_registry ADD COLUMN feature_set TEXT"\n'
            'MODEL_REGISTRY_ALTER_SNAPSHOT = "ALTER TABLE model_registry ADD COLUMN dataset_snapshot_id INTEGER"\n'
            "MODEL_REGISTRY_ALTER_PROMOTION = \"ALTER TABLE model_registry ADD COLUMN promotion_state TEXT NOT NULL DEFAULT 'active'\"\n"
            'MODEL_REGISTRY_TARGET_BACKFILL = "UPDATE model_registry SET target_name=COALESCE(target_name, model_name)"\n'
            'MODEL_REGISTRY_BACKEND_BACKFILL = "UPDATE model_registry SET backend="\n'
            'MODEL_REGISTRY_FORMAT_BACKFILL = "UPDATE model_registry SET artifact_format="\n'
            'MODEL_REGISTRY_PROMOTION_BACKFILL = "UPDATE model_registry SET promotion_state="\n'
        ),
        "trade_py/db/migrations.py": (
            'MIGRATIONS_SQL = "CREATE TABLE IF NOT EXISTS schema_migrations"\n'
            'SIGNAL_CACHE_SQL = "CREATE TABLE IF NOT EXISTS signal_cache_v2"\n'
            'BUS_EVENTS_SQL = "CREATE TABLE IF NOT EXISTS bus_events"\n'
            'BUS_EVENTS_DROP_SQL = "DROP TABLE IF EXISTS bus_events"\n'
            'SIGNAL_CACHE_INSERT_SQL = "INSERT OR IGNORE INTO signal_cache_v2"\n'
            'SIGNAL_CACHE_RENAME_SQL = "ALTER TABLE signal_cache_v2 RENAME TO signal_cache"\n'
            'EVENT_LOG_INSERT_SQL = "INSERT OR IGNORE INTO event_log"\n'
            'SIGNALS_INSERT_SQL = "INSERT OR IGNORE INTO signals"\n'
            'SECTOR_MEMBERS_INSERT_SQL = "INSERT OR IGNORE INTO sector_members"\n'
            'MARKET_EVENTS_INSERT_SQL = "INSERT OR IGNORE INTO market_events"\n'
            'SYNC_STATE_IGNORE_SQL = "INSERT OR IGNORE INTO sync_state"\n'
            'SYNC_STATE_REPLACE_SQL = "INSERT OR REPLACE INTO sync_state"\n'
            'EVENT_PROPAGATIONS_ALTER_PATH = "ALTER TABLE event_propagations ADD COLUMN rel_path TEXT"\n'
            'EVENT_PROPAGATIONS_ALTER_VALIDATED = "ALTER TABLE event_propagations ADD COLUMN validated_at TIMESTAMP"\n'
            'JOB_RUNS_ALTER_STAGE = "ALTER TABLE job_runs ADD COLUMN stage TEXT"\n'
            'JOB_RUNS_ALTER_EVENT = "ALTER TABLE job_runs ADD COLUMN trigger_event_id INTEGER"\n'
            'JOB_RUNS_ALTER_SUMMARY = "ALTER TABLE job_runs ADD COLUMN result_summary TEXT"\n'
            'JOB_RUNS_ALTER_SYMBOLS = "ALTER TABLE job_runs ADD COLUMN symbols_processed INTEGER"\n'
            'JOB_RUNS_ALTER_ELAPSED = "ALTER TABLE job_runs ADD COLUMN elapsed_ms INTEGER"\n'
            'JOB_RUNS_ALTER_COMPLETED = "ALTER TABLE job_runs ADD COLUMN completed_at TIMESTAMP"\n'
            'KG_RELATIONS_ALTER_DIRECTION = "ALTER TABLE kg_relations ADD COLUMN direction INTEGER NOT NULL DEFAULT 1"\n'
            'KG_RELATIONS_ALTER_DAYS = "ALTER TABLE kg_relations ADD COLUMN typical_days INTEGER NOT NULL DEFAULT 0"\n'
            'KG_RELATIONS_ALTER_CONFIDENCE = "ALTER TABLE kg_relations ADD COLUMN confidence REAL NOT NULL DEFAULT 0.0"\n'
            'KG_RELATIONS_ALTER_SAMPLES = "ALTER TABLE kg_relations ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 0"\n'
            'KG_RELATIONS_ALTER_EVIDENCE = "ALTER TABLE kg_relations ADD COLUMN evidence_json TEXT"\n'
            "KG_RELATIONS_ALTER_STATUS = \"ALTER TABLE kg_relations ADD COLUMN status TEXT NOT NULL DEFAULT 'active'\"\n"
            'KG_RELATIONS_ALTER_UPDATED = "ALTER TABLE kg_relations ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"\n'
            'KG_RELATIONS_DIRECTION_BACKFILL = "UPDATE kg_relations"\n'
            'KG_RELATIONS_WEIGHT_BACKFILL = "UPDATE kg_relations SET weight = ABS(weight) WHERE weight < 0"\n'
            "KG_RELATIONS_STATUS_BACKFILL = \"UPDATE kg_relations SET status = 'active' WHERE status IS NULL OR status = ''\"\n"
            "PIPELINE_DAG_ALTER_CONFIG = \"ALTER TABLE pipeline_dag ADD COLUMN config_json TEXT DEFAULT '{}'\"\n"
            'PIPELINE_DAG_ALTER_SOURCE = "ALTER TABLE pipeline_dag ADD COLUMN sync_source TEXT"\n'
            'PIPELINE_DAG_ALTER_DATASET = "ALTER TABLE pipeline_dag ADD COLUMN sync_dataset TEXT"\n'
            "PIPELINE_DAG_ALTER_MODE = \"ALTER TABLE pipeline_dag ADD COLUMN mode TEXT DEFAULT 'batch'\"\n"
            'PIPELINE_DAG_SYNC_BACKFILL = "UPDATE pipeline_dag SET sync_source=?, sync_dataset=?"\n'
            "PIPELINE_DAG_STREAM_MODE = \"UPDATE pipeline_dag SET mode='streaming' WHERE job_name=? AND mode='batch'\"\n"
            "PIPELINE_DAG_BOTH_MODE = \"UPDATE pipeline_dag SET mode='both' WHERE job_name=? AND mode='batch'\"\n"
            'PIPELINE_DAG_DELETE = "DELETE FROM pipeline_dag"\n'
            "PIPELINE_DAG_DISABLE_SENTIMENT = \"UPDATE pipeline_dag SET enabled=0 WHERE job_name='sentiment_pipeline'\"\n"
            "PIPELINE_DAG_DISABLE_EVENT = \"UPDATE pipeline_dag SET enabled=0 WHERE job_name='event_pipeline'\"\n"
            "PIPELINE_DAG_MOVE_CROSS_ASSET = \"UPDATE pipeline_dag SET source=?, emits='', description=?\"\n"
            "PIPELINE_DAG_BTC_DESCRIPTION = \"UPDATE pipeline_dag SET description='BTC assurance-gated UTC 日线同步' WHERE job_name='crypto_btc_fetch'\"\n"
            "PIPELINE_DAG_DISABLE_CROSS_ASSET = \"UPDATE pipeline_dag SET enabled=0 WHERE job_name='cross_asset_fetch'\"\n"
            'PIPELINE_DAG_CONFIG_BACKFILL = "UPDATE pipeline_dag SET config_json=?, description=? WHERE id=?"\n'
            'ASSET_REGISTRY_INDEX = "CREATE INDEX IF NOT EXISTS idx_asset_class ON asset_registry(asset_class, enabled, priority)"\n'
            'ASSET_REGISTRY_SEED = "INSERT INTO asset_registry"\n'
            'ASSET_REGISTRY_CONFIG = "UPDATE asset_registry SET config_json=?, updated_at=CURRENT_TIMESTAMP WHERE asset_id=?"\n'
            'EVENT_HANDLER_RUNS_SQL = "CREATE TABLE IF NOT EXISTS event_handler_runs"\n'
            'KG_CANDIDATES_SQL = "CREATE TABLE IF NOT EXISTS kg_edge_candidates"\n'
            'UI_SNAPSHOTS_SQL = "CREATE TABLE IF NOT EXISTS ui_snapshots"\n'
            'ARTICLE_EVENT_SQL = "CREATE TABLE IF NOT EXISTS ArticleEvent"\n'
            'INFLUENCE_SIGNAL_SQL = "CREATE TABLE IF NOT EXISTS InfluenceSignal"\n'
            'EVIDENCE_SQL = "CREATE TABLE IF NOT EXISTS Evidence"\n'
            'BELIEF_STATE_SQL = "CREATE TABLE IF NOT EXISTS BeliefState"\n'
            'ATTENTION_SCORE_SQL = "CREATE TABLE IF NOT EXISTS AttentionScore"\n'
            'BELIEF_TRANSITION_SQL = "CREATE TABLE IF NOT EXISTS BeliefTransition"\n'
            'QUALITY_REPORT_SQL = "CREATE TABLE IF NOT EXISTS QualityReport"\n'
            'FRESHNESS_STATUS_SQL = "CREATE TABLE IF NOT EXISTS FreshnessStatus"\n'
            'RECOMMENDATION_SQL = "CREATE TABLE IF NOT EXISTS Recommendation"\n'
            'RECOMMENDATION_TRACE_SQL = "CREATE TABLE IF NOT EXISTS RecommendationTrace"\n'
            "RECOMMENDATION_DYNAMIC = 'conn.execute(f\"ALTER TABLE Recommendation ADD COLUMN {col_def}\")'\n"
            "RECOMMENDATION_TRACE_DYNAMIC = 'conn.execute(f\"ALTER TABLE RecommendationTrace ADD COLUMN {col_def}\")'\n"
            "FACTOR_REGISTRY_DYNAMIC = 'f\"ALTER TABLE factor_registry ADD COLUMN {col} REAL NOT NULL DEFAULT {default}\"'\n"
        ),
        "trade_py/intelligence/raw_record.py": "published_at: datetime\n",
        "trade_py/intelligence/feed_scorer.py": (
            "published_at = datetime.now(timezone.utc).isoformat()\n"
        ),
        "trade_py/data/__init__.py": "",
        "trade_py/data/ingest/__init__.py": "",
        "trade_py/data/ingest/batch.py": "self._recover_wal()\n",
        "trade_py/data/news/__init__.py": "",
        "trade_py/data/news/akshare_news.py": (
            "pub = datetime(cur.year, cur.month, cur.day, 12, 0, 0, tzinfo=CST)\n"
            "pub = pub_raw.to_pydatetime().replace(tzinfo=CST)\n"
        ),
        "trade_py/data/news/rss/__init__.py": "",
        "trade_py/data/news/rss/base.py": "pub_time = datetime.now(timezone.utc)\n",
        "trade_py/data/news/rss/archive.py": (
            "return datetime.combine(day, time(12, 0), tzinfo=timezone.utc)\n"
        ),
        "trade_py/data/news/rss/catalog.py": (
            'override = os.environ.get("TRADE_RSS_FEED_INDEX_PATH")\n'
        ),
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
        "trade_py/data/warehouse/fetch.py": (
            'row = {"published_at": published_at or fetched_at}\n'
        ),
        "trade_py/data/warehouse/articles.py": 'quality_status = "quarantined"\n',
        "trade_py/data/warehouse/io.py": (
            "class WarehouseLayout:\n"
            "    @classmethod\n"
            "    def from_data_root(cls, root):\n"
            "        return cls()\n"
            "def write_table(layout, layer, table, frame):\n"
            "    return None\n"
            "def upsert_table(layout, layer, table, frame, *, key_cols):\n"
            "    return None\n"
            'path = f"{table}.parquet"\n'
        ),
        "trade_py/data/warehouse/crypto_store.py": (
            'CRYPTO_VALIDATION_CURRENT = "_crypto_validation_current.json"\n'
            'receipt_root = ads_root / "_validation_receipts"\n'
        ),
        "trade_py/data/market/crypto/store.py": (
            'self.current_path = self.crypto_root / "btc_current.json"\n'
        ),
        "trade_py/data/operations/checks.py": (
            'path = root / "market" / "kline" / "reconciliation" / "current.json"\n'
        ),
        "trade_py/utils/data_inspector.py": (
            'return KLINE_DIR(data_root) / "reconciliation" / "current.json"\n'
        ),
        "trade_py/observatory/catalog/store.py": (
            'CATALOG_META_SQL = "CREATE TABLE catalog_meta"\n'
            'CATALOG_RUNS_SQL = "CREATE TABLE runs"\n'
            'CATALOG_RELEASES_SQL = "CREATE TABLE releases"\n'
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


def test_producer_call_digest_ignores_ast_context_metadata() -> None:
    tree = ast.parse(
        'result = write_table(layout, "ods", "events", frame=None)\n',
    )
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    expected_digest = _call_digest(call)
    layout_argument = call.args[0]
    assert isinstance(layout_argument, ast.Name)

    layout_argument.ctx = ast.Store()

    assert _call_digest(call) == expected_digest


def test_repository_baseline_includes_review_required_provenance_and_interfaces() -> None:
    baseline = tomllib.loads((REPO_ROOT / BASELINE_FILENAME).read_text(encoding="utf-8"))
    tables_by_name = {table["logical_name"]: table for table in baseline["tables"]}
    table_names = {table["logical_name"] for table in baseline["tables"]}
    capture_risk_ids = {risk["id"] for risk in baseline["capture_risks"]}
    interface_kinds = {item["surface_kind"] for item in baseline["interfaces"]}

    assert {
        "ingest_runs",
        "coverage",
        "enrichment_status",
        "settings",
        "watchlist",
        "signals",
        "job_runs",
        "instruments",
        "sector_members",
        "sync_state",
        "trading_calendar",
        "planned_events",
        "agenda_queue",
        "backup_snapshots",
        "ui_snapshots",
        "readiness_recovery_actions",
        "schema_migrations",
        "signal_cache_v2",
        "bus_events",
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
        "source_health_daily",
        "source_eval_daily",
        "event_eval_runs",
        "dataset_snapshots",
        "daily_quality_gate",
        "event_templates",
        "market_events",
        "event_propagations",
        "ArticleEvent",
        "InfluenceSignal",
        "Evidence",
        "BeliefState",
        "AttentionScore",
        "BeliefTransition",
        "QualityReport",
        "FreshnessStatus",
        "Recommendation",
        "RecommendationTrace",
    } <= table_names
    assert {
        "influence-signal-runtime-publication-time",
        "eastmoney-stock-timezone-overwrite",
        "rss-provider-time-fallback",
        "gdelt-catalog-db-config",
        "gdelt-provider-time-fallback",
        "gdelt-streaming-local-state-and-refetch",
    } <= capture_risk_ids
    assert {
        "recommendation-dynamic-columns",
        "recommendation-trace-dynamic-columns",
        "factor-registry-dynamic-columns",
    } == {item["id"] for item in baseline["dynamic_sql_limitations"]}
    for table_name, literal in (
        ("catalog_meta", "CREATE TABLE catalog_meta"),
        ("catalog_runs", "CREATE TABLE runs"),
        ("catalog_releases", "CREATE TABLE releases"),
    ):
        assert {
            "source": "trade_py/observatory/catalog/store.py",
            "literal": literal,
            "role": "bootstrap",
        } in tables_by_name[table_name]["provenance"]
    assert "http-openapi" in interface_kinds
    for table_name, source, literal, role in (
        (
            "event_handler_runs",
            "trade_py/db/migrations.py",
            "CREATE TABLE IF NOT EXISTS event_handler_runs",
            "migration",
        ),
        (
            "kg_edge_candidates",
            "trade_py/db/migrations.py",
            "CREATE TABLE IF NOT EXISTS kg_edge_candidates",
            "migration",
        ),
        (
            "ui_snapshots",
            "trade_py/db/migrations.py",
            "CREATE TABLE IF NOT EXISTS ui_snapshots",
            "migration",
        ),
    ):
        assert {
            "source": source,
            "literal": literal,
            "role": role,
        } in tables_by_name[table_name]["provenance"]


def test_required_facts_and_target_context_vocabulary_fail_closed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")

    for artifact_id in (
        "warehouse-parquet",
        "catalog-sqlite-projection",
        "catalog-generation-pointer",
        "crypto-ads-current-pointer",
        "crypto-ads-validation-receipt",
        "btc-compatibility-pointer",
        "kline-reconciliation-operation-pointer",
        "kline-reconciliation-pointer",
    ):
        _write_baseline(repo, baseline.replace(f'id = "{artifact_id}"', 'id = "removed"', 1))
        assert "architecture.baseline_malformed" in _rule_ids(repo)

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
            'logical_name = "event_eval_runs"\ncurrent_owner = "legacy"\nsemantic_kind = "event-study-evaluation-record"\nclassification = "candidate"\ntarget_context = "studies"',
            'logical_name = "event_eval_runs"\ncurrent_owner = "legacy"\nsemantic_kind = "event-study-evaluation-record"\nclassification = "candidate"\ntarget_context = "datasets"',
            1,
        ),
    )
    assert "architecture.baseline_malformed" in _rule_ids(repo)

    _write_baseline(
        repo,
        baseline.replace(
            'logical_name = "InfluenceSignal"\ncurrent_owner = "legacy"\nsemantic_kind = "source-influence-record"\nclassification = "deferred"\ntarget_context = "deferred"\nreason = "Requires source and Study ownership evidence before transfer."\nrequired_child = "study-boundary"',
            'logical_name = "InfluenceSignal"\ncurrent_owner = "legacy"\nsemantic_kind = "source-influence-record"\nclassification = "candidate"\ntarget_context = "datasets"\nreason = "Requires source and Study ownership evidence before transfer."\nrequired_child = "study-boundary"',
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
    "risk_id",
    (
        "raw-record-single-publication-clock",
        "cctv-date-only-publication-time",
        "eastmoney-stock-timezone-overwrite",
        "warehouse-rss-fetched-time-substitution",
        "rss-provider-time-fallback",
        "archive-date-only-publication-time",
        "rss-catalog-environment-override",
        "gdelt-catalog-db-config",
        "gdelt-provider-time-fallback",
        "gdelt-streaming-local-state-and-refetch",
        "ingest-wal-replay",
        "warehouse-semantic-quarantine",
        "influence-signal-runtime-publication-time",
    ),
)
def test_required_capture_risk_bindings_fail_closed_on_removal(
    tmp_path: Path,
    risk_id: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline.replace(f'id = "{risk_id}"', 'id = "removed"', 1))

    assert "architecture.baseline_malformed" in _rule_ids(repo)


def test_capture_risk_inventory_rejects_unreviewed_record(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline
        + """

[[capture_risks]]
id = "unreviewed-temporal-risk"
source = "trade_py/intelligence/raw_record.py"
literal = "published_at: datetime"
current_owner = "trade_py.intelligence"
required_child = "capture-boundary"
risk_kind = "unreviewed-temporal-claim"
current_behavior = "Unreviewed."
required_migration_proof = "Unreviewed."
""",
    )

    assert "architecture.baseline_malformed" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("path", "source"),
    (
        (
            "trade",
            "# legacy-cli\n",
        ),
        (
            "engine/cmake/python_bindings.cmake",
            "# nanobind_add_module(trade_py bindings.cpp)\n",
        ),
    ),
)
def test_non_python_comments_do_not_satisfy_source_evidence(
    tmp_path: Path,
    path: str,
    source: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    (repo / path).write_text(source, encoding="utf-8")

    assert "architecture.baseline_literal_mismatch" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("risk_id", "field"),
    tuple(
        (risk_id, field)
        for risk_id in (
            "raw-record-single-publication-clock",
            "cctv-date-only-publication-time",
            "eastmoney-stock-timezone-overwrite",
            "warehouse-rss-fetched-time-substitution",
            "rss-provider-time-fallback",
            "archive-date-only-publication-time",
            "rss-catalog-environment-override",
            "gdelt-catalog-db-config",
            "gdelt-provider-time-fallback",
            "gdelt-streaming-local-state-and-refetch",
            "ingest-wal-replay",
            "warehouse-semantic-quarantine",
            "influence-signal-runtime-publication-time",
        )
        for field in (
            "source",
            "literal",
            "current_owner",
            "required_child",
            "risk_kind",
            "current_behavior",
            "required_migration_proof",
        )
    ),
)
def test_required_capture_risk_bindings_reject_every_field_mutation(
    tmp_path: Path,
    risk_id: str,
    field: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    start = baseline.index(f'id = "{risk_id}"')
    end = baseline.find("\n[[", start + 1)
    risk_record = baseline[start:] if end == -1 else baseline[start:end]
    record_field = re.search(
        rf"^{re.escape(field)} = (['\"])(?P<value>.*?)\1$",
        risk_record,
        flags=re.MULTILINE,
    )
    assert record_field is not None
    replacement = f'{field} = "mutated-{risk_id}-{field}"'
    mutated_record = (
        risk_record[: record_field.start()] + replacement + risk_record[record_field.end() :]
    )
    _write_baseline(repo, baseline[:start] + mutated_record + baseline[start + len(risk_record) :])

    assert "architecture.baseline_malformed" in _rule_ids(repo)


def _approved_adapter_source(*, include_unrelated: bool = False) -> str:
    source = """\
APPROVED_RECORD_DDL = "CREATE TABLE approved_records"


def persist_approved(session):
    with session.transaction():
        session.execute("INSERT INTO approved_records (id) VALUES (?)")


def load_approved(session):
    return session.execute("SELECT id FROM approved_records").fetchall()


def load_approved_compat(session):
    return session.query("SELECT id FROM approved_records")
"""
    if include_unrelated:
        source += """\

def persist_unrelated(session):
    with session.transaction():
        session.execute("INSERT INTO unrelated_records (id) VALUES (?)")


def load_unrelated(session):
    return session.execute("SELECT id FROM unrelated_records").fetchall()


def load_unrelated_compat(session):
    return session.query("SELECT id FROM unrelated_records")
"""
    return source


def _nested_approved_adapter_source(scope_kind: str) -> str:
    nested_statements = {
        "function": (
            "        def unused():\n"
            '            session.execute("INSERT INTO approved_records (id) VALUES (?)")\n',
            "    def unused():\n"
            '        return session.execute("SELECT id FROM approved_records").fetchall()\n',
            '    def unused():\n        return session.query("SELECT id FROM approved_records")\n',
        ),
        "async_function": (
            "        async def unused():\n"
            '            session.execute("INSERT INTO approved_records (id) VALUES (?)")\n',
            "    async def unused():\n"
            '        return session.execute("SELECT id FROM approved_records").fetchall()\n',
            "    async def unused():\n"
            '        return session.query("SELECT id FROM approved_records")\n',
        ),
        "lambda": (
            '        unused = lambda: session.execute("INSERT INTO approved_records (id) VALUES (?)")\n',
            '    unused = lambda: session.execute("SELECT id FROM approved_records").fetchall()\n',
            '    unused = lambda: session.query("SELECT id FROM approved_records")\n',
        ),
        "class": (
            "        class Unused:\n"
            "            def run(self):\n"
            '                session.execute("INSERT INTO approved_records (id) VALUES (?)")\n',
            "    class Unused:\n"
            "        def run(self):\n"
            '            return session.execute("SELECT id FROM approved_records").fetchall()\n',
            "    class Unused:\n"
            "        def run(self):\n"
            '            return session.query("SELECT id FROM approved_records")\n',
        ),
    }
    writer, reader, compatibility = nested_statements[scope_kind]
    return (
        'APPROVED_RECORD_DDL = "CREATE TABLE approved_records"\n\n\n'
        "def persist_approved(session):\n"
        "    with session.transaction():\n"
        f"{writer}\n"
        "\n"
        "def load_approved(session):\n"
        f"{reader}\n"
        "\n"
        "def load_approved_compat(session):\n"
        f"{compatibility}"
    )


def _unreachable_approved_adapter_callable(field: str, unreachable_kind: str) -> str:
    callable_name = f"dead_{field}"
    if field in {"writer_evidence", "transaction_evidence"}:
        operation = (
            "    with session.transaction():\n"
            '        session.execute("INSERT INTO approved_records (id) VALUES (?)")\n'
        )
    else:
        operation = '    return session.execute("SELECT id FROM approved_records").fetchall()\n'

    if unreachable_kind == "if_false":
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return f"\n\ndef {callable_name}(session):\n    if False:\n{body}"
    if unreachable_kind in {"if_zero", "if_none", "if_empty_string", "if_empty_tuple"}:
        test = {
            "if_zero": "0",
            "if_none": "None",
            "if_empty_string": '""',
            "if_empty_tuple": "()",
        }[unreachable_kind]
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return f"\n\ndef {callable_name}(session):\n    if {test}:\n{body}"
    if unreachable_kind in {"while_zero", "while_none"}:
        test = {"while_zero": "0", "while_none": "None"}[unreachable_kind]
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return f"\n\ndef {callable_name}(session):\n    while {test}:\n{body}"
    if unreachable_kind == "try_else_after_terminal":
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return (
            f"\n\ndef {callable_name}(session):\n"
            "    try:\n"
            "        return None\n"
            "    except RuntimeError:\n"
            "        return None\n"
            "    else:\n"
            f"{body}"
        )
    if unreachable_kind == "try_handler_after_return":
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return (
            f"\n\ndef {callable_name}(session):\n"
            "    try:\n"
            "        return None\n"
            "    except RuntimeError:\n"
            f"{body}"
        )
    if unreachable_kind == "if_truthy_else":
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return (
            f"\n\ndef {callable_name}(session):\n"
            "    if (1,):\n"
            "        return None\n"
            "    else:\n"
            f"{body}"
        )
    if unreachable_kind == "if_not_one":
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return f"\n\ndef {callable_name}(session):\n    if not 1:\n{body}"
    if unreachable_kind == "if_literal_comparison":
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return f"\n\ndef {callable_name}(session):\n    if 1 == 0:\n{body}"
    if unreachable_kind == "while_break":
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return f"\n\ndef {callable_name}(session):\n    while True:\n        break\n{body}"
    if unreachable_kind == "for_empty":
        body = "".join(f"    {line}\n" for line in operation.rstrip().splitlines())
        return f"\n\ndef {callable_name}(session):\n    for _ in ():\n{body}"
    if unreachable_kind == "generator_empty":
        sql = (
            "INSERT INTO approved_records (id) VALUES (?)"
            if field in {"writer_evidence", "transaction_evidence"}
            else "SELECT id FROM approved_records"
        )
        return (
            f'\n\ndef {callable_name}(session):\n    return (session.execute("{sql}") for _ in ())'
        )
    if unreachable_kind == "after_return":
        return f"\n\ndef {callable_name}(session):\n    return None\n{operation}"
    if unreachable_kind == "after_raise":
        return (
            f'\n\ndef {callable_name}(session):\n    raise RuntimeError("unreachable")\n{operation}'
        )
    raise AssertionError(f"unsupported unreachable proof kind: {unreachable_kind}")


def _approved_binding_declaration() -> str:
    return (
        "[[tables]]\n"
        'logical_name = "approved_records"\n'
        'current_owner = "legacy"\n'
        'semantic_kind = "approved-fixture-record"\n'
        'classification = "approved_binding"\n'
        'target_context = "datasets"\n'
        'reason = "Fixture approved binding."\n'
        'required_child = "dataset-product-boundary"\n'
        'adapter_scope = "datasets.adapters.persistence.warehouse"\n'
        'writer_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "INSERT INTO approved_records (id) VALUES (?)", callable = "persist_approved" }\n'
        'reader_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "SELECT id FROM approved_records", callable = "load_approved" }\n'
        'transaction_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "INSERT INTO approved_records (id) VALUES (?)", callable = "persist_approved" }\n'
        'compatibility_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "SELECT id FROM approved_records", callable = "load_approved_compat" }\n'
        "provenance = [\n"
        '  { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "CREATE TABLE approved_records", role = "bootstrap" },\n'
        "]"
    )


def test_approved_binding_requires_structural_adapter_proofs(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source(), encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    approved = _approved_binding_declaration()
    approved_baseline = baseline + "\n" + approved

    _write_baseline(repo, approved_baseline)
    assert validate_architecture_baseline(repo).ok

    for invalid_scope in (
        "datasets.adapters.",
        "datasets.adapters.persistence.",
        "datasets.adapters.persistence..warehouse",
        "datasets.adapters.persistence.warehouse-invalid",
        "capture.adapters.persistence.warehouse",
    ):
        _write_baseline(
            repo,
            approved_baseline.replace(
                'adapter_scope = "datasets.adapters.persistence.warehouse"',
                f'adapter_scope = "{invalid_scope}"',
                1,
            ),
        )
        assert "architecture.baseline_invalid_classification" in _rule_ids(repo)

    _write_baseline(
        repo,
        approved_baseline.replace(
            'writer_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "INSERT INTO approved_records (id) VALUES (?)", callable = "persist_approved" }',
            'writer_evidence = "arbitrary prose"',
            1,
        ),
    )
    assert "architecture.baseline_malformed" in _rule_ids(repo)

    _write_baseline(
        repo,
        approved_baseline.replace(
            'reader_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "SELECT id FROM approved_records", callable = "load_approved" }',
            'reader_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "SELECT id FROM approved_records", callable = "unknown_callable" }',
            1,
        ),
    )
    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)

    source.write_text(
        'WRITER_PROOF = "INSERT INTO approved_records (id) VALUES (?)"\n'
        'READER_PROOF = "SELECT id FROM approved_records"\n'
        'TRANSACTION_PROOF = "INSERT INTO approved_records (id) VALUES (?)"\n'
        'COMPATIBILITY_PROOF = "SELECT id FROM approved_records"\n',
        encoding="utf-8",
    )
    _write_baseline(repo, approved_baseline)
    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (
            "writer_evidence",
            'writer_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "INSERT INTO unrelated_records (id) VALUES (?)", callable = "persist_unrelated" }',
        ),
        (
            "reader_evidence",
            'reader_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "SELECT id FROM unrelated_records", callable = "load_unrelated" }',
        ),
        (
            "transaction_evidence",
            'transaction_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "INSERT INTO unrelated_records (id) VALUES (?)", callable = "persist_unrelated" }',
        ),
        (
            "compatibility_evidence",
            'compatibility_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "SELECT id FROM unrelated_records", callable = "load_unrelated_compat" }',
        ),
    ),
)
def test_approved_binding_rejects_same_adapter_proof_for_different_table(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source(include_unrelated=True), encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    approved = "\n" + _approved_binding_declaration() + "\n"
    approved_baseline = baseline + approved
    _write_baseline(repo, approved_baseline)
    assert validate_architecture_baseline(repo).ok

    original = {
        "writer_evidence": 'writer_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "INSERT INTO approved_records (id) VALUES (?)", callable = "persist_approved" }',
        "reader_evidence": 'reader_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "SELECT id FROM approved_records", callable = "load_approved" }',
        "transaction_evidence": 'transaction_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "INSERT INTO approved_records (id) VALUES (?)", callable = "persist_approved" }',
        "compatibility_evidence": 'compatibility_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "SELECT id FROM approved_records", callable = "load_approved_compat" }',
    }[field]
    _write_baseline(repo, approved_baseline.replace(original, replacement, 1))

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


@pytest.mark.parametrize("scope_kind", ("function", "async_function", "lambda", "class"))
def test_approved_binding_rejects_nonexecuted_nested_callable_proofs(
    tmp_path: Path,
    scope_kind: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_nested_approved_adapter_source(scope_kind), encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


def test_approved_binding_requires_transaction_and_persistence_receivers_to_match(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source().replace(
            "with session.transaction():",
            "with unrelated.transaction():",
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


def test_approved_binding_accepts_explicit_transaction_alias(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source().replace(
            'with session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'with session.transaction() as tx:\n        tx.execute("INSERT INTO approved_records (id) VALUES (?)")',
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert validate_architecture_baseline(repo).ok


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            'with session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'with session.transaction():\n        globals()["session"] = unrelated\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
        ),
        (
            'with session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'with session.transaction():\n        globals().update({"session": unrelated})\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
        ),
        (
            'with session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'with session.transaction() as tx:\n        globals()["tx"] = unrelated\n        tx.execute("INSERT INTO approved_records (id) VALUES (?)")',
        ),
    ),
)
def test_approved_binding_rejects_dynamic_transaction_receiver_rebinding(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source().replace(old, new, 1), encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


def test_approved_binding_rejects_nested_with_dead_transaction_proof(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source().replace(
            'with session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'with session.transaction():\n        with lock:\n            return None\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


def test_approved_binding_rejects_multi_item_transaction_with_proof(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source().replace(
            'with session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'with lock, session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


def test_approved_binding_rejects_read_only_transaction_proof(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source().replace(
            'def persist_approved(session):\n    with session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'def persist_approved(session):\n    session.execute("INSERT INTO approved_records (id) VALUES (?)")\n    with session.transaction():\n        session.execute("SELECT id FROM approved_records")',
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline
        + "\n"
        + _approved_binding_declaration().replace(
            'transaction_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "INSERT INTO approved_records (id) VALUES (?)", callable = "persist_approved" }',
            'transaction_evidence = { source = "src/trade/datasets/adapters/persistence/warehouse.py", literal = "SELECT id FROM approved_records", callable = "persist_approved" }',
            1,
        ),
    )

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("field", "old", "new"),
    (
        (
            "writer_evidence",
            'session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'session.execute("SELECT ?", "INSERT INTO approved_records (id) VALUES (?)")',
        ),
        (
            "reader_evidence",
            'session.execute("SELECT id FROM approved_records")',
            'session.execute("SELECT ?", "SELECT id FROM approved_records")',
        ),
        (
            "transaction_evidence",
            'session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'session.execute("SELECT ?", "INSERT INTO approved_records (id) VALUES (?)")',
        ),
        (
            "compatibility_evidence",
            'session.query("SELECT id FROM approved_records")',
            'session.query("SELECT ?", "SELECT id FROM approved_records")',
        ),
    ),
)
def test_approved_binding_rejects_sql_only_in_persistence_parameters(
    tmp_path: Path,
    field: str,
    old: str,
    new: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source().replace(old, new, 1), encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo), field


def test_approved_binding_requires_exact_callable_sql_literal(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source()
        + '\n\nUNRELATED_SQL = "INSERT INTO approved_records (id) VALUES (?)"\n',
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline
        + "\n"
        + _approved_binding_declaration().replace(
            'literal = "INSERT INTO approved_records (id) VALUES (?)", callable = "persist_approved"',
            'literal = "INSERT INTO approved_records", callable = "persist_approved"',
            1,
        ),
    )

    report = validate_architecture_baseline(repo)

    finding = next(
        item
        for item in report.findings
        if item.rule_id == "architecture.baseline_invalid_classification"
    )
    assert finding.path == "src/trade/datasets/adapters/persistence/warehouse.py"
    assert finding.line == 4
    assert "writer_evidence callable persist_approved" in finding.message
    assert "first static SQL argument" in finding.remediation


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            'with session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'with session.transaction():\n        session = unrelated\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
        ),
        (
            'with session.transaction():\n        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            'with session.transaction() as tx:\n        tx = unrelated\n        tx.execute("INSERT INTO approved_records (id) VALUES (?)")',
        ),
    ),
)
def test_approved_binding_rejects_rebound_transaction_receivers(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source().replace(old, new, 1), encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("field", "old", "new"),
    (
        (
            "reader_evidence",
            'return session.execute("SELECT id FROM approved_records").fetchall()',
            'return session.execute("DELETE FROM approved_records WHERE id IN (SELECT id FROM approved_records)").fetchall()',
        ),
        (
            "compatibility_evidence",
            'return session.query("SELECT id FROM approved_records")',
            'return session.query("DELETE FROM approved_records; SELECT id FROM approved_records")',
        ),
    ),
)
def test_approved_binding_rejects_mutating_reader_or_compatibility_proofs(
    tmp_path: Path,
    field: str,
    old: str,
    new: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source().replace(old, new, 1), encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo), field


@pytest.mark.parametrize(
    "field",
    ("writer_evidence", "reader_evidence", "transaction_evidence", "compatibility_evidence"),
)
@pytest.mark.parametrize(
    "unreachable_kind",
    (
        "if_false",
        "if_zero",
        "if_none",
        "if_empty_string",
        "if_empty_tuple",
        "while_zero",
        "while_none",
        "try_else_after_terminal",
        "try_handler_after_return",
        "if_truthy_else",
        "if_not_one",
        "if_literal_comparison",
        "while_break",
        "for_empty",
        "generator_empty",
        "after_return",
        "after_raise",
    ),
)
def test_approved_binding_rejects_unreachable_direct_scope_proofs(
    tmp_path: Path,
    field: str,
    unreachable_kind: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source()
        + _unreachable_approved_adapter_callable(field, unreachable_kind),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    original = {
        "writer_evidence": (
            'writer_evidence = { source = "src/trade/datasets/adapters/persistence/'
            'warehouse.py", literal = "INSERT INTO approved_records (id) VALUES (?)", callable = '
            '"persist_approved" }'
        ),
        "reader_evidence": (
            'reader_evidence = { source = "src/trade/datasets/adapters/persistence/'
            'warehouse.py", literal = "SELECT id FROM approved_records", callable = '
            '"load_approved" }'
        ),
        "transaction_evidence": (
            'transaction_evidence = { source = "src/trade/datasets/adapters/persistence/'
            'warehouse.py", literal = "INSERT INTO approved_records (id) VALUES (?)", callable = '
            '"persist_approved" }'
        ),
        "compatibility_evidence": (
            'compatibility_evidence = { source = "src/trade/datasets/adapters/persistence/'
            'warehouse.py", literal = "SELECT id FROM approved_records", callable = '
            '"load_approved_compat" }'
        ),
    }[field]
    replacement = original.replace(
        {
            "writer_evidence": "persist_approved",
            "reader_evidence": "load_approved",
            "transaction_evidence": "persist_approved",
            "compatibility_evidence": "load_approved_compat",
        }[field],
        f"dead_{field}",
    )
    _write_baseline(
        repo, baseline + "\n" + _approved_binding_declaration().replace(original, replacement)
    )

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("redefinition", "description"),
    (
        (
            "\n\ndef persist_approved(session):\n    return None\n",
            "duplicate definition",
        ),
        (
            "\n\npersist_approved = lambda session: None\n",
            "subsequent assignment",
        ),
        (
            "\n\ndel persist_approved\n",
            "subsequent deletion",
        ),
        (
            "\n\nif enabled:\n    persist_approved = lambda session: None\n",
            "nested control-flow assignment",
        ),
        (
            "\n\nfrom rebound import *\n",
            "wildcard import",
        ),
        (
            '\n\nglobals()["persist_approved"] = replacement\n',
            "module namespace assignment",
        ),
        (
            '\n\nexec("persist_approved = replacement")\n',
            "dynamic execution",
        ),
        (
            '\n\nsetattr(module, "persist_approved", replacement)\n',
            "dynamic attribute assignment",
        ),
    ),
)
def test_approved_binding_rejects_rebound_proof_callable(
    tmp_path: Path,
    redefinition: str,
    description: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source() + redefinition, encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo), description


@pytest.mark.parametrize(
    ("redefinition", "description"),
    (
        (
            '\n\nglobals().pop("persist_approved")\n',
            "module namespace deletion",
        ),
        (
            "\n\nglobals().clear()\n",
            "module namespace clear",
        ),
        (
            '\n\nglobals().pop("persist_approved")\nvars().setdefault("persist_approved", replacement)\n',
            "module namespace delete and rebind",
        ),
        (
            '\n\ndelattr(module, "persist_approved")\n',
            "dynamic attribute deletion",
        ),
        (
            '\n\nnamespace = globals()\nnamespace.pop("persist_approved")\n',
            "module namespace alias mutation",
        ),
        (
            '\n\neval("persist_approved = replacement")\n',
            "dynamic evaluation",
        ),
    ),
)
def test_approved_binding_rejects_callable_namespace_mutation(
    tmp_path: Path,
    redefinition: str,
    description: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source() + redefinition, encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo), description


@pytest.mark.parametrize(
    ("definition", "description"),
    (
        (
            '\n\nclass SideEffect:\n    globals().pop("persist_approved")\n',
            "class body mutation",
        ),
        (
            '\n\n@globals().pop("persist_approved")\nclass SideEffect:\n    pass\n',
            "class decorator mutation",
        ),
        (
            '\n\nclass SideEffect(globals().pop("persist_approved")):\n    pass\n',
            "class base mutation",
        ),
        (
            '\n\nclass SideEffect(metaclass=globals().pop("persist_approved")):\n    pass\n',
            "class metaclass mutation",
        ),
        (
            '\n\ndef unrelated(value=globals().pop("persist_approved")):\n    return value\n',
            "function default mutation",
        ),
        (
            '\n\n@globals().pop("persist_approved")\ndef unrelated():\n    return None\n',
            "function decorator mutation",
        ),
    ),
)
def test_approved_binding_rejects_definition_time_namespace_mutation(
    tmp_path: Path,
    definition: str,
    description: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source() + definition, encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo), description


def test_approved_binding_rejects_decorated_proof_callable(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source().replace(
            "def persist_approved(session):",
            "@decorator\ndef persist_approved(session):",
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


@pytest.mark.parametrize(
    "limits",
    (
        DiscoveryLimits(max_callable_proof_operations=1),
        DiscoveryLimits(max_callable_proof_sql_bytes=1),
    ),
)
def test_approved_binding_proof_collection_is_budgeted(
    tmp_path: Path,
    limits: DiscoveryLimits,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source().replace(
            '        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            '        session.execute("INSERT INTO approved_records (id) VALUES (?)")\n'
            '        session.execute("INSERT INTO approved_records (id) VALUES (?)")',
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    report = validate_architecture_baseline(repo, limits=limits)

    assert "architecture.baseline_evidence_budget_exceeded" in {
        finding.rule_id for finding in report.findings
    }


def test_approved_binding_proof_ast_budget_fails_without_crashing(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    expression = " + ".join("1" for _ in range(64))
    source.write_text(
        _approved_adapter_source().replace(
            "def persist_approved(session):\n",
            f"def persist_approved(session):\n    value = {expression}\n",
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    report = validate_architecture_baseline(repo, limits=DiscoveryLimits(max_ast_depth=24))

    assert {finding.rule_id for finding in report.findings} == {
        "architecture.baseline_evidence_budget_exceeded"
    }
    assert report.producers == ()


def test_approved_binding_proof_within_ast_depth_budget_validates(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    expression = "+" * 300 + "1"
    source.write_text(
        _approved_adapter_source().replace(
            "def persist_approved(session):\n",
            f"def persist_approved(session):\n    value = {expression}\n",
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    report = validate_architecture_baseline(repo, limits=DiscoveryLimits(max_ast_depth=512))

    assert report.ok
    assert report.producers


def test_callable_proof_recursion_failure_is_terminally_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source(), encoding="utf-8")
    import trade_py.devtools.architecture_guard as guard

    calls = 0

    def raise_recursion(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal calls
        calls += 1
        raise RecursionError("fixture recursion limit")

    monkeypatch.setattr(guard, "_summarize_callable_proof", raise_recursion)
    evidence = guard._EvidenceReader(repo, DEFAULT_LIMITS)

    for _ in range(2):
        with pytest.raises(guard._GuardError) as exc_info:
            evidence.callable_proof_summary(
                "src/trade/datasets/adapters/persistence/warehouse.py",
                "persist_approved",
            )
        assert exc_info.value.finding.rule_id == "architecture.baseline_evidence_budget_exceeded"

    assert calls == 1


def test_approved_binding_rejects_table_suffix_and_comment_only_proofs(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        _approved_adapter_source().replace(
            "INSERT INTO approved_records",
            "INSERT INTO approved_records_backup",
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    approved = _approved_binding_declaration().replace(
        "INSERT INTO approved_records",
        "INSERT INTO approved_records_backup",
        1,
    )
    _write_baseline(repo, baseline + "\n" + approved)

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)

    source.write_text(
        _approved_adapter_source().replace(
            "SELECT id FROM approved_records",
            "SELECT id FROM unrelated_records /* SELECT id FROM approved_records */",
            1,
        ),
        encoding="utf-8",
    )
    _write_baseline(repo, baseline + "\n" + _approved_binding_declaration())

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


@pytest.mark.parametrize(
    "quoted_identifier",
    (
        '"FROM approved_records"',
        "`FROM approved_records`",
        "[FROM approved_records]",
    ),
)
def test_approved_binding_rejects_quoted_identifier_pseudo_read_proof(
    tmp_path: Path,
    quoted_identifier: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    literal = f"SELECT 1 AS {quoted_identifier}"
    source.write_text(
        _approved_adapter_source().replace(
            'session.execute("SELECT id FROM approved_records")',
            f"session.execute({literal!r})",
            1,
        ),
        encoding="utf-8",
    )
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline
        + "\n"
        + _approved_binding_declaration().replace(
            'literal = "SELECT id FROM approved_records", callable = "load_approved"',
            f"literal = '{literal}', callable = \"load_approved\"",
            1,
        ),
    )

    assert "architecture.baseline_invalid_classification" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("section", "classification"),
    (
        ("[[artifacts]]", "approved_binding"),
        ("[[warehouse_producers]]", "approved_binding"),
    ),
)
def test_non_table_records_cannot_become_approved_bindings(
    tmp_path: Path,
    section: str,
    classification: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    start = baseline.index(section)
    before = baseline[:start]
    record = baseline[start:]
    _write_baseline(
        repo,
        before
        + record.replace('classification = "candidate"', f'classification = "{classification}"', 1),
    )

    assert "architecture.baseline_non_authorizing_binding" in _rule_ids(repo)


@pytest.mark.parametrize(
    "unknown_resource",
    ("providers", "streams", "object_stores", "vectors", "unstructured_resources"),
)
def test_baseline_rejects_unknown_non_sql_resource_declarations(
    tmp_path: Path,
    unknown_resource: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline + f'\n\n[[{unknown_resource}]]\nid = "unapproved-{unknown_resource}"\n'
        'classification = "approved_binding"\n',
    )

    assert "architecture.baseline_malformed" in _rule_ids(repo)


@pytest.mark.parametrize(
    "section",
    (
        "[[source_facts]]",
        "[[capture_risks]]",
        "[[dynamic_sql_limitations]]",
        "[[interfaces]]",
        "[[native_bindings]]",
    ),
)
def test_unclassifiable_baseline_facts_reject_authorization_fields(
    tmp_path: Path,
    section: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline.replace(section, section + '\nclassification = "approved_binding"', 1),
    )

    assert "architecture.baseline_non_authorizing_binding" in _rule_ids(repo)


def test_required_table_bindings_reject_prefix_only_ddl_evidence(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    migration_source = repo / "trade_py/db/migrations.py"
    migration_source.write_text(
        migration_source.read_text(encoding="utf-8").replace(
            'RECOMMENDATION_SQL = "CREATE TABLE IF NOT EXISTS Recommendation"\n',
            "",
            1,
        ),
        encoding="utf-8",
    )

    assert "architecture.baseline_literal_mismatch" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("literal", "replacement"),
    (
        ("DELETE FROM pipeline_dag", "DELETE FROM pipeline_dag_archive"),
        (
            "UPDATE pipeline_dag SET sync_source=?, sync_dataset=?",
            "UPDATE pipeline_dag_archive SET sync_source=?, sync_dataset=?",
        ),
        ("INSERT OR IGNORE INTO event_log", "INSERT OR IGNORE INTO event_log_archive"),
    ),
)
def test_required_table_bindings_reject_suffix_only_dml_evidence(
    tmp_path: Path,
    literal: str,
    replacement: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "trade_py/db/migrations.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace(literal, replacement, 1),
        encoding="utf-8",
    )

    assert "architecture.baseline_literal_mismatch" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("table_name", "source", "literal", "role"),
    tuple(
        (table_name, source, literal, role)
        for table_name, requirements in _AUDITED_SCHEMA_EVOLUTION_PROVENANCE.items()
        for source, literal, role in requirements
    ),
)
def test_audited_schema_evolution_provenance_fails_closed(
    tmp_path: Path,
    table_name: str,
    source: str,
    literal: str,
    role: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    provenance_record = f'  {{ source = "{source}", literal = "{literal}", role = "{role}" }},\n'
    assert provenance_record in baseline, f"{table_name}: {provenance_record}"
    _write_baseline(repo, baseline.replace(provenance_record, "", 1))

    assert "architecture.baseline_malformed" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("table_name", "literal"),
    (
        ("catalog_meta", "CREATE TABLE catalog_meta"),
        ("catalog_runs", "CREATE TABLE runs"),
        ("catalog_releases", "CREATE TABLE releases"),
    ),
)
def test_catalog_projection_bootstrap_provenance_fails_closed(
    tmp_path: Path,
    table_name: str,
    literal: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    record = (
        '  { source = "trade_py/observatory/catalog/store.py", '
        f'literal = "{literal}", role = "bootstrap" }},\n'
    )
    assert record in baseline, table_name
    _write_baseline(
        repo, baseline.replace(record, record.replace('"bootstrap"', '"data_transform"'), 1)
    )

    assert "architecture.baseline_malformed" in _rule_ids(repo)


@pytest.mark.parametrize("table_name", ("catalog_meta", "catalog_runs", "catalog_releases"))
def test_catalog_projection_table_declarations_are_required(
    tmp_path: Path, table_name: str
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    start = baseline.index(f'[[tables]]\nlogical_name = "{table_name}"')
    end = baseline.find("\n[[tables]]", start + 1)
    record = baseline[start:] if end == -1 else baseline[start:end]
    _write_baseline(repo, baseline.replace(record, "", 1))

    assert "architecture.baseline_malformed" in _rule_ids(repo)


@pytest.mark.parametrize(
    "limitation_id",
    (
        "recommendation-dynamic-columns",
        "recommendation-trace-dynamic-columns",
        "factor-registry-dynamic-columns",
    ),
)
def test_required_dynamic_sql_limitations_fail_closed(
    tmp_path: Path,
    limitation_id: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline.replace(f'id = "{limitation_id}"', 'id = "removed"', 1))

    assert "architecture.baseline_malformed" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("original", "replacement"),
    (
        (
            'limitation_kind = "dynamic_ddl"',
            'limitation_kind = "dynamic_data_transform"',
        ),
        (
            'owning_child = "decision-support-boundary"',
            'owning_child = "study-boundary"',
        ),
        (
            "non-authorizing until the Decision Support migration adds reviewed SQL-normalization",
            "authorizing immediately",
        ),
    ),
)
def test_required_dynamic_sql_limitations_reject_binding_mutation(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline.replace(original, replacement, 1))

    assert "architecture.baseline_malformed" in _rule_ids(repo)


def test_dynamic_sql_limitations_reject_unbounded_data_transform_kind(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline.replace(
            'limitation_kind = "dynamic_ddl"',
            'limitation_kind = "dynamic_data_transform"',
            1,
        ),
    )

    assert "architecture.baseline_incomplete_provenance" in _rule_ids(repo)


def test_dynamic_sql_limitations_reject_unreviewed_site(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline
        + """

[[dynamic_sql_limitations]]
id = "unreviewed-dynamic-sql"
logical_name = "Recommendation"
source = "trade_py/db/migrations.py"
literal = 'conn.execute(f"UPDATE Recommendation SET state = {state}")'
limitation_kind = "dynamic_ddl"
owning_child = "decision-support-boundary"
non_authorizing = true
limitation = "Unreviewed."
""",
    )

    assert "architecture.baseline_malformed" in _rule_ids(repo)


def test_dynamic_sql_limitations_require_explicit_non_authorizing_marker(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(repo, baseline.replace("non_authorizing = true\n", "", 1))

    assert "architecture.baseline_non_authorizing_binding" in _rule_ids(repo)


def test_dynamic_sql_limitation_cannot_authorize_its_table(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "src/trade/datasets/adapters/persistence/warehouse.py"
    source.parent.mkdir(parents=True)
    source.write_text(_approved_adapter_source(), encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    replacement = _approved_binding_declaration().replace(
        'logical_name = "approved_records"',
        'logical_name = "Recommendation"',
        1,
    )
    original_start = baseline.index('[[tables]]\nlogical_name = "Recommendation"')
    original_end = baseline.find("\n[[tables]]", original_start + 1)
    original_table = (
        baseline[original_start:] if original_end == -1 else baseline[original_start:original_end]
    )
    mutated = (
        baseline[:original_start] + replacement + baseline[original_start + len(original_table) :]
    )
    _write_baseline(repo, mutated)

    assert "architecture.baseline_non_authorizing_binding" in _rule_ids(repo)


@pytest.mark.parametrize(
    ("mutator", "expected_rule"),
    (
        (
            lambda text: text.replace("schema_version = 1", "schema_version = 2"),
            "architecture.baseline_malformed",
        ),
        (
            lambda text: text.replace("schema_version = 1", "schema_version = true"),
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
                'classification = "deferred"\ntarget_context = "deferred"\nreason = "Requires evidence before ownership transfer."',
                'classification = "deferred"\ntarget_context = "deferred"\nadapter_scope = "forbidden"\nreason = "Requires evidence before ownership transfer."',
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
            lambda text: text.replace('artifact_id = "warehouse-parquet"', "artifact_id = []"),
            "architecture.baseline_malformed",
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


@pytest.mark.parametrize(
    "replacement",
    (
        "# LEGACY_DB = 1\n",
        '"""LEGACY_DB = 1"""\n',
        'f"LEGACY_DB = {1}"\n',
        'b"LEGACY_DB = 1"\n',
    ),
)
def test_comments_and_inert_strings_do_not_satisfy_source_evidence(
    tmp_path: Path,
    replacement: str,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    (repo / "trade_py/db.py").write_text(replacement, encoding="utf-8")

    assert "architecture.baseline_literal_mismatch" in _rule_ids(repo)


def test_unicode_before_inert_string_does_not_satisfy_source_evidence(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    replacement = "prefix = '" + "\u00e9" * 20 + "'; 'LEGACY_DB = 1'\n"
    (repo / "trade_py/db.py").write_text(replacement, encoding="utf-8")

    assert "architecture.baseline_literal_mismatch" in _rule_ids(repo)


def test_toml_recursion_failure_is_reported_without_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    def raise_recursion(_: str) -> None:
        raise RecursionError("fixture recursion limit")

    monkeypatch.setattr(guard.tomllib, "loads", raise_recursion)

    report = validate_architecture_baseline(repo)

    assert {finding.rule_id for finding in report.findings} == {"architecture.baseline_malformed"}
    assert report.producers == ()


def test_missing_or_changed_declared_source_fails_closed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    (repo / "trade_py/db.py").unlink()
    assert "architecture.baseline_missing_source" in _rule_ids(repo)

    repo = _init_repo(tmp_path / "changed", _sources())
    path = repo / "trade_py/intelligence/raw_record.py"
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


def test_fifo_source_is_rejected_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO support is unavailable")
    repo = _init_repo(tmp_path, _sources())
    source = repo / "trade_py/app.py"
    source.unlink()
    os.mkfifo(source)

    started = time.monotonic()
    report = validate_architecture_baseline(repo)
    elapsed_seconds = time.monotonic() - started

    assert {finding.rule_id for finding in report.findings} == {PRODUCER_UNSAFE_SOURCE}
    assert report.producers == ()
    assert elapsed_seconds < 1.0


def test_nonregular_production_python_index_entry_fails_closed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, _sources())
    source = repo / "trade_py/index_symlink.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    blob = subprocess.run(
        ["git", "hash-object", "-w", "--", "trade_py/index_symlink.py"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(repo, "update-index", "--add", "--cacheinfo", f"120000,{blob},trade_py/index_symlink.py")

    report = validate_architecture_baseline(repo)

    assert {finding.rule_id for finding in report.findings} == {PRODUCER_UNSAFE_SOURCE}
    assert report.producers == ()


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


def test_validator_uses_only_admitted_source_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    original = guard._safe_read_relative
    opened: list[str] = []

    def record_read(root: Path, relative: str, *, max_bytes: int) -> bytes:
        opened.append(relative)
        return original(root, relative, max_bytes=max_bytes)

    monkeypatch.setattr(guard, "_safe_read_relative", record_read)

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert BASELINE_FILENAME in opened
    assert all(
        relative == BASELINE_FILENAME
        or relative.startswith("trade_py/")
        or relative.startswith("trade_web/")
        or relative.startswith("tests/")
        or relative.startswith("engine/")
        or relative == "trade"
        for relative in opened
    )
    assert not any(
        relative.startswith(("data/", "warehouse/", "market/"))
        or relative.endswith((".db", ".sqlite", ".parquet"))
        or any(
            part
            in {"artifacts", "manifest", "manifests", "pointer", "pointers", "receipt", "receipts"}
            for part in relative.split("/")
        )
        for relative in opened
    )


def test_architecture_guard_cold_import_does_not_load_quality_runner(
    tmp_path: Path,
) -> None:
    source_root = REPO_ROOT
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(source_root)!r})\n"
        "import trade_py.devtools.architecture_guard\n"
        "assert 'trade_py.devtools.quality.runner' not in sys.modules\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


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
        limits=DiscoveryLimits(max_aggregate_evidence_bytes=1, max_findings=4),
    )
    assert "architecture.baseline_evidence_budget_exceeded" in {
        finding.rule_id for finding in budget_report.findings
    }
    assert budget_report.producers == ()


def test_source_evidence_executable_text_is_transformed_once_per_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    repeated_source = "REPEATED_EVIDENCE = True\n"
    repeated_path = repo / "tests/repeated_evidence.py"
    repeated_path.write_text(repeated_source, encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline
        + """
[[source_facts]]
id = "repeated-evidence-one"
source = "tests/repeated_evidence.py"
literal = "REPEATED_EVIDENCE"
current_owner = "legacy"
required_child = "dataset-product-boundary"

[[source_facts]]
id = "repeated-evidence-two"
source = "tests/repeated_evidence.py"
literal = "REPEATED_EVIDENCE"
current_owner = "legacy"
required_child = "dataset-product-boundary"
""",
    )
    original_parse = guard.ast.parse
    original_tokens = guard.tokenize.generate_tokens
    original_live_source = guard._live_python_source_text
    parses = 0
    tokenizations = 0
    transformations = 0
    transforming_repeated_source = False

    def record_parse(source: str, *args: object, **kwargs: object) -> ast.AST:
        nonlocal parses
        if source == repeated_source:
            parses += 1
        return original_parse(source, *args, **kwargs)

    def record_tokens(readline):
        nonlocal tokenizations
        if transforming_repeated_source:
            tokenizations += 1
        return original_tokens(readline)

    def record_live_source(text: str, *, source: str, max_tokens: int) -> str:
        nonlocal transformations, transforming_repeated_source
        if text != repeated_source:
            return original_live_source(text, source=source, max_tokens=max_tokens)
        transformations += 1
        transforming_repeated_source = True
        try:
            return original_live_source(text, source=source, max_tokens=max_tokens)
        finally:
            transforming_repeated_source = False

    monkeypatch.setattr(guard.ast, "parse", record_parse)
    monkeypatch.setattr(guard.tokenize, "generate_tokens", record_tokens)
    monkeypatch.setattr(guard, "_live_python_source_text", record_live_source)

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert transformations == 1
    assert parses == 1
    # One pass admits the token budget, and one pass masks executable text.
    assert tokenizations == 2


def test_source_evidence_literals_are_batched_once_per_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    repeated_source = "\n".join(f'FACT_{index} = "evidence-{index}"' for index in range(128)) + "\n"
    (repo / "tests/repeated_literals.py").write_text(repeated_source, encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    declarations = "\n".join(
        (
            "[[source_facts]]\n"
            f'id = "repeated-literal-{index}"\n'
            'source = "tests/repeated_literals.py"\n'
            f'literal = "evidence-{index}"\n'
            'current_owner = "legacy"\n'
            'required_child = "dataset-product-boundary"\n'
        )
        for index in range(128)
    )
    _write_baseline(repo, baseline + "\n" + declarations)
    original = guard._literal_matches_for_source
    source_scans = 0

    def record_matches(text: str, literals: set[str]) -> Mapping[str, bool]:
        nonlocal source_scans
        if text == repeated_source:
            source_scans += 1
        return original(text, literals)

    monkeypatch.setattr(guard, "_literal_matches_for_source", record_matches)

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert source_scans == 1


def test_batched_literal_matching_matches_single_literal_semantics() -> None:
    import trade_py.devtools.architecture_guard as guard

    text = "CREATE TABLE records_archive\nCREATE TABLE IF NOT EXISTS records\naaaaaa\n"
    literals = {
        "CREATE TABLE records",
        "CREATE TABLE records_archive",
        "CREATE TABLE IF NOT EXISTS records",
        "aaaaab",
        "aaaaaa",
    }

    assert guard._literal_matches_for_source(text, literals) == {
        literal: guard._literal_is_present(text, literal) for literal in literals
    }


def test_batched_literal_matching_handles_overlapping_absent_prefixes() -> None:
    import trade_py.devtools.architecture_guard as guard

    text = "a" * 32_768
    literals = {("a" * width) + "b" for width in range(1, 128)}

    started = time.monotonic()
    matches = guard._literal_matches_for_source(text, literals)
    elapsed_seconds = time.monotonic() - started

    assert matches == {literal: False for literal in literals}
    assert elapsed_seconds < 3.0


@pytest.mark.parametrize(
    ("literal_count", "literal_byte_budget"),
    (
        (1, 1024),
        (256, 8),
    ),
)
def test_batched_literal_matching_fails_closed_on_per_source_budget(
    tmp_path: Path,
    literal_count: int,
    literal_byte_budget: int,
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    repeated_source = (
        "\n".join(f'EVIDENCE_{index} = "evidence-{index}"' for index in range(2)) + "\n"
    )
    (repo / "tests/repeated_literals.py").write_text(repeated_source, encoding="utf-8")
    evidence = guard._EvidenceReader(
        repo,
        DiscoveryLimits(
            max_evidence_literals_per_source=literal_count,
            max_evidence_literal_bytes_per_source=literal_byte_budget,
        ),
    )
    source = "tests/repeated_literals.py"
    evidence.prime_literal_matches(
        ((source, "evidence-0"), (source, "evidence-1")),
    )

    with pytest.raises(guard._GuardError) as error:
        evidence.literal_is_present(source, "evidence-0")

    assert error.value.finding.rule_id == "architecture.baseline_evidence_budget_exceeded"
    assert error.value.finding.remediation == (
        "Reduce duplicate source literals, split the governed evidence source, "
        "or make a reviewed per-source literal-budget increase."
    )


def test_many_inert_strings_are_masked_with_linear_span_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    source = repo / "trade_py/db.py"
    source.write_text(
        "LEGACY_DB = 1\n"
        'SQL = "CREATE TABLE legacy_records"\n' + ("'inert evidence text'\n" * 4_000),
        encoding="utf-8",
    )
    original_spans = guard._inert_python_string_spans
    span_iteration_calls = 0

    def record_spans(
        tree: ast.AST,
        line_offsets: tuple[int, ...],
        utf8_column_maps: dict[int, Any],
    ) -> tuple[tuple[int, int], ...]:
        spans = original_spans(tree, line_offsets, utf8_column_maps)

        class CountingSpans(tuple):
            def __iter__(self):
                nonlocal span_iteration_calls
                span_iteration_calls += 1
                return super().__iter__()

        return CountingSpans(spans)

    monkeypatch.setattr(guard, "_inert_python_string_spans", record_spans)

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    # The monotonic index uses len/index access; it must not rescan every span
    # for every STRING token as the former overlap predicate did.
    assert span_iteration_calls == 0


def test_same_line_inert_strings_reuse_utf8_column_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    source = repo / "trade_py/db.py"
    inert_expression_count = 16_000
    source.write_text(
        "LEGACY_DB = 1\n"
        'SQL = "CREATE TABLE legacy_records"\n' + ("'é' " * inert_expression_count) + "\n",
        encoding="utf-8",
    )
    original_column_map = guard._utf8_column_map
    mapped_lines: list[str] = []

    def record_column_map(line: str):
        mapped_lines.append(line)
        return original_column_map(line)

    monkeypatch.setattr(guard, "_utf8_column_map", record_column_map)

    started = time.monotonic()
    report = validate_architecture_baseline(repo)
    elapsed_seconds = time.monotonic() - started

    assert report.ok, report.findings
    assert mapped_lines.count("'é' " * inert_expression_count + "\n") == 1
    assert elapsed_seconds < 3.0


def test_high_cardinality_python_evidence_fails_before_ast_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    source = repo / "trade_py/db.py"
    source.write_text(
        'LEGACY_DB = 1\nSQL = "CREATE TABLE legacy_records"\n' + ("'é';" * 200_000) + "\n",
        encoding="utf-8",
    )
    original_parse = guard.ast.parse
    parsed_sources: list[str] = []

    def record_parse(source: str, *args: object, **kwargs: object) -> ast.AST:
        parsed_sources.append(source)
        return original_parse(source, *args, **kwargs)

    monkeypatch.setattr(guard.ast, "parse", record_parse)

    report = validate_architecture_baseline(
        repo,
        limits=DiscoveryLimits(max_evidence_python_tokens=1_000),
    )

    assert {finding.rule_id for finding in report.findings} == {
        "architecture.baseline_evidence_budget_exceeded"
    }
    assert report.producers == ()
    assert not any("'é';" in source_text for source_text in parsed_sources)


def test_repeated_over_budget_python_evidence_is_terminally_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    import trade_py.devtools.architecture_guard as guard

    repeated_source = "REPEATED = " + ("'é';" * 2_000) + "\n"
    repeated_path = repo / "tests/repeated_over_budget.py"
    repeated_path.write_text(repeated_source, encoding="utf-8")
    baseline = (repo / BASELINE_FILENAME).read_text(encoding="utf-8")
    _write_baseline(
        repo,
        baseline
        + """
[[source_facts]]
id = "repeated-over-budget-one"
source = "tests/repeated_over_budget.py"
literal = "REPEATED"
current_owner = "legacy"
required_child = "dataset-product-boundary"

[[source_facts]]
id = "repeated-over-budget-two"
source = "tests/repeated_over_budget.py"
literal = "REPEATED"
current_owner = "legacy"
required_child = "dataset-product-boundary"
""",
    )
    original_admission = guard._validate_python_evidence_token_budget
    original_parse = guard.ast.parse
    admissions = 0
    parsed_sources: list[str] = []

    def record_admission(text: str, *, source: str, max_tokens: int) -> None:
        nonlocal admissions
        if text == repeated_source:
            admissions += 1
        original_admission(text, source=source, max_tokens=max_tokens)

    def record_parse(source: str, *args: object, **kwargs: object) -> ast.AST:
        parsed_sources.append(source)
        return original_parse(source, *args, **kwargs)

    monkeypatch.setattr(guard, "_validate_python_evidence_token_budget", record_admission)
    monkeypatch.setattr(guard.ast, "parse", record_parse)

    report = validate_architecture_baseline(
        repo,
        limits=DiscoveryLimits(max_evidence_python_tokens=1_000),
    )

    assert {finding.rule_id for finding in report.findings} == {
        "architecture.baseline_evidence_budget_exceeded"
    }
    assert report.producers == ()
    assert admissions == 1
    assert not any(source_text == repeated_source for source_text in parsed_sources)
    assert {finding.remediation for finding in report.findings} == {
        "Reduce or split the declared Python source evidence, or make a reviewed "
        "governed Python-token-budget increase."
    }


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


@pytest.mark.parametrize(
    "app",
    (
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "def helper():\n"
            '    write_table(layout, "ods", "before_assignment", frame=None)\n'
            "    write_table = lambda *args, **kwargs: None\n"
        ),
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "from foreign_module import write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            'write_table(layout, "ods", "foreign_rebound", frame=None)\n'
        ),
    ),
)
def test_noncanonical_python_bindings_do_not_register_producers(tmp_path: Path, app: str) -> None:
    repo = _init_repo(tmp_path, _sources(app, baseline_app=DEFAULT_APP))

    report = validate_architecture_baseline(repo)

    assert {finding.rule_id for finding in report.findings} == {PRODUCER_UNRESOLVED_IMPORT}
    assert report.producers == ()


def test_function_local_non_writer_import_does_not_trigger_writer_diagnostic(
    tmp_path: Path,
) -> None:
    app = (
        "from helpers import render\n"
        "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        "def helper():\n"
        "    from helpers import render\n"
        "    return render()\n"
        'write_table(layout, "ods", "events", frame=None)\n'
    )
    repo = _init_repo(tmp_path, _sources(app, baseline_app=app))

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert [(producer.layer, producer.table) for producer in report.producers] == [
        ("ods", "events")
    ]


@pytest.mark.parametrize(
    "app",
    (
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "from foreign_module import *\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            'write_table(layout, "ods", "foreign_star_import", frame=None)\n'
        ),
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "[\n"
            '    write_table(layout, "ods", "comprehension_shadow", frame=None)\n'
            "    for write_table in [lambda *args, **kwargs: None]\n"
            "]\n"
        ),
    ),
)
def test_uncertain_lexical_bindings_do_not_register_producers(tmp_path: Path, app: str) -> None:
    repo = _init_repo(tmp_path, _sources(app, baseline_app=DEFAULT_APP))

    report = validate_architecture_baseline(repo)

    assert PRODUCER_UNRESOLVED_IMPORT in {finding.rule_id for finding in report.findings}
    assert report.producers == ()


@pytest.mark.parametrize(
    "app",
    (
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "def helper():\n"
            "    del write_table\n"
            '    write_table(layout, "ods", "deleted_writer", frame=None)\n'
        ),
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "def helper(value):\n"
            "    match value:\n"
            "        case write_table:\n"
            "            pass\n"
            '    write_table(layout, "ods", "match_capture", frame=None)\n'
        ),
    ),
)
def test_delete_and_match_capture_do_not_register_producers(tmp_path: Path, app: str) -> None:
    repo = _init_repo(tmp_path, _sources(app, baseline_app=DEFAULT_APP))

    report = validate_architecture_baseline(repo)

    assert PRODUCER_UNRESOLVED_IMPORT in {finding.rule_id for finding in report.findings}
    assert report.producers == ()


@pytest.mark.parametrize(
    "app",
    (
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "def helper():\n"
            "    global write_table\n"
            '    write_table(layout, "ods", "events", frame=None)\n'
            "    write_table = lambda *args, **kwargs: None\n"
        ),
        (
            "def outer():\n"
            "    from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "    layout = WarehouseLayout.from_data_root('data')\n"
            "    def helper():\n"
            "        nonlocal write_table\n"
            '        write_table(layout, "ods", "events", frame=None)\n'
            "        write_table = lambda *args, **kwargs: None\n"
            "    return helper\n"
        ),
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "class WriterContainer:\n"
            "    write_table = staticmethod(lambda *args, **kwargs: None)\n"
            "    def emit(self):\n"
            '        return write_table(layout, "ods", "events", frame=None)\n'
        ),
    ),
)
def test_lexical_global_nonlocal_and_class_method_writers_are_resolved(
    tmp_path: Path,
    app: str,
) -> None:
    repo = _init_repo(tmp_path, _sources(app, baseline_app=app))

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    assert len(report.producers) == 1


def test_producer_discovery_scales_with_irrelevant_imports_and_sibling_scopes(
    tmp_path: Path,
) -> None:
    irrelevant_source = "".join(
        f"import ignored_{index:05d}\ndef sibling_{index:05d}():\n    return None\n"
        for index in range(15_000)
    )
    app = (
        "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        f"{irrelevant_source}"
        'write_table(layout, "ods", "events", frame=None)\n'
    )
    assert len(app.encode("utf-8")) <= DEFAULT_LIMITS.max_file_bytes
    repo = _init_repo(tmp_path, _sources(app, baseline_app=app))

    started = time.monotonic()
    report = validate_architecture_baseline(repo)
    elapsed_seconds = time.monotonic() - started

    assert report.ok, report.findings
    assert elapsed_seconds < 3.0


@pytest.mark.parametrize(
    "app",
    (
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "def annotation(value: write_table(layout, 'ods', 'annotation', frame=None)) -> None:\n"
            "    return None\n"
        ),
        (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            "class Container(metaclass=write_table(layout, 'ods', 'class_keyword', frame=None)):\n"
            "    pass\n"
        ),
    ),
)
def test_writer_calls_in_annotations_and_class_keywords_are_discovered(
    tmp_path: Path,
    app: str,
) -> None:
    repo = _init_repo(tmp_path, _sources(app, baseline_app=DEFAULT_APP))

    report = validate_architecture_baseline(repo)

    assert PRODUCER_UNDECLARED_WRITER in {finding.rule_id for finding in report.findings}
    assert report.producers == ()


def test_deep_production_ast_fails_closed_without_recursion_error(tmp_path: Path) -> None:
    nested_expression = "[" * 128 + "0" + "]" * 128
    app = (
        "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
        "layout = WarehouseLayout.from_data_root('data')\n"
        f'write_table(layout, "ods", "events", {nested_expression})\n'
    )
    repo = _init_repo(tmp_path, _sources(app, baseline_app=DEFAULT_APP))

    report = validate_architecture_baseline(repo, limits=DiscoveryLimits(max_ast_depth=32))

    assert {finding.rule_id for finding in report.findings} == {PRODUCER_RESULT_BUDGET}
    assert report.producers == ()


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
    for max_findings, expected_emitted, expected_omitted in (
        (0, 0, 5),
        (1, 1, 5),
        (3, 3, 3),
        (4, 4, 2),
        (5, 5, 0),
    ):
        report_limits = DiscoveryLimits(
            max_findings=max_findings,
            max_diagnostic_field_bytes=32,
        )
        bounded_report = validate_architecture_baseline(repo, limits=report_limits)

        assert len(bounded_report.findings) == expected_emitted
        assert bounded_report.omitted_findings_count == expected_omitted
        if expected_emitted and expected_omitted:
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


def test_git_discovery_success_cleans_detached_pipe_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, _sources())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    child_pid_path = tmp_path / "successful-child.pid"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        "sleep 30 </dev/null >/dev/null 2>&1 &\n"
        'printf "%s\n" "$!" > "$ARCH_GUARD_CHILD_PID_FILE"\n'
        "printf '100644 deadbeef 0\ttrade_py/app.py\\000'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ARCH_GUARD_CHILD_PID_FILE", str(child_pid_path))

    report = validate_architecture_baseline(repo)

    assert report.ok, report.findings
    child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("successful Git descendant still exists after process-group cleanup")


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
