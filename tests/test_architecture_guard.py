from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from trade_py.devtools.architecture_guard import (
    BASELINE_FILENAME,
    DEFAULT_LIMITS,
    PRODUCER_NONLITERAL_TARGET,
    PRODUCER_PATH_BUDGET,
    PRODUCER_SOURCE_BUDGET,
    PRODUCER_UNDECLARED_WRITER,
    PRODUCER_UNRESOLVED_IMPORT,
    PRODUCER_UNRESOLVED_LAYOUT,
    PRODUCER_UNSAFE_SOURCE,
    DiscoveryLimits,
    validate_architecture_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _baseline(
    *,
    producer_source: str = "trade_py/app.py",
    producer_literal: str = 'write_table(layout, "ods", "events", frame=None)',
    producer_layer: str = "ods",
    producer_table: str = "events",
    classification: str = "candidate",
    target_context: str = "datasets",
    artifact_id: str = "warehouse-parquet",
    extra: str = "",
) -> str:
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

[[capture_risks]]
id = "raw-clock"
source = "trade_py/capture.py"
literal = "published_at"
current_owner = "legacy"
required_child = "capture-boundary"
risk_kind = "clock-collapse"
current_behavior = "One field represents multiple clocks."
required_migration_proof = "Independent clocks."

[[interfaces]]
id = "cli"
source = "trade"
literal = "legacy-cli"
current_owner = "legacy"
required_child = "cli-http-sdk-compatibility"
surface_kind = "cli-facade"
current_behavior = "Legacy entrypoint remains available."
compatibility_owner = "interfaces.cli.compat"

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
literal = '{producer_literal}'
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


def _sources(app: str | None = None) -> dict[str, str]:
    return {
        "architecture-baseline.toml": _baseline(),
        "trade": "#!/bin/sh\n# legacy-cli\n",
        "trade_py/__init__.py": "",
        "trade_py/db.py": 'LEGACY_DB = 1\nSQL = "CREATE TABLE legacy_records"\n',
        "trade_py/migrations.py": 'SQL = "ALTER TABLE legacy_records ADD COLUMN value"\n',
        "trade_py/warehouse.py": 'path = f"{table}.parquet"\n',
        "trade_py/capture.py": "published_at = None\n",
        "trade_py/data/__init__.py": "",
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
        "trade_py/app.py": app
        or (
            "from trade_py.data.warehouse import WarehouseLayout, write_table\n"
            "layout = WarehouseLayout.from_data_root('data')\n"
            'write_table(layout, "ods", "events", frame=None)\n'
        ),
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
    sources[BASELINE_FILENAME] = (
        _baseline(
            producer_literal='write(layout, "ods", "events", frame=None)',
        )
        + """
[[warehouse_producers]]
id = "alias-upsert"
source = "trade_py/app.py"
literal = 'merge(layout, "dwd", "articles", frame=None, key_cols=["id"])'
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
