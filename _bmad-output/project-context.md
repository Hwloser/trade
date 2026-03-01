---
project_name: 'trade'
user_name: 'huanwei'
date: '2026-03-01'
sections_completed: ['technology_stack', 'critical_rules', 'architecture_boundary', 'data_layout', 'code_organization']
existing_patterns_found: 46
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## Technology Stack & Versions

### C++ Core (Primary computation engine)
- **Standard:** C++20 (`CMAKE_CXX_STANDARD 20`)
- **Compiler:** Apple Clang (macOS ARM / Apple Silicon)
- **Build system:** CMake 3.25+ with Ninja, preset `default` (`cmake --preset default`)
- **Libraries (Homebrew only — NOT vcpkg):**
  - Arrow/Parquet: `Arrow::arrow_shared`, `Parquet::parquet_shared`
  - SQLite3, Eigen3, spdlog, fmt, nlohmann_json, yaml-cpp, pugixml
  - httplib, libcurl, re2, OpenSSL
  - DuckDB: git submodule `vendor/duckdb`, **static** link
  - LightGBM: optional (`#ifdef HAVE_LIGHTGBM`)
  - ONNX Runtime: optional (`#ifdef HAVE_ONNXRUNTIME`)
- **Testing:** GoogleTest via `add_trade_test(name src)` macro, `gtest_discover_tests`

### Python (Glue layer — offline / data ingestion only)
- **Version:** >=3.10 (venv managed by `uv`)
- **Install:** `uv sync --extra ml --extra dev`
- **Key deps:**
  - `akshare>=1.12` — data source for kline, fundamentals, fund flow
  - `pandas>=2.0`, `pyarrow>=14.0`, `duckdb>=1.0`
  - `anthropic>=0.40.0` — Claude Haiku for sentiment LLM
  - `lightgbm>=4.0`, `scikit-learn>=1.4`, `shap>=0.45` (ml extra)
  - `networkx>=3.0` — sector knowledge graph

---

## Critical Implementation Rules

### C++ Rules

1. **`#pragma once`** — All headers use `#pragma once`. Never use `#ifndef` include guards.

2. **Namespace:** All C++ code lives in `namespace trade { ... }`. No exceptions.

3. **Optional features:** Guard with `#ifdef HAVE_LIGHTGBM` / `#ifdef HAVE_ONNXRUNTIME`. Never assume these are always available.

4. **Apple Clang default-initializer bug:** Never write `ClassName(Config cfg = {})` when `Config` has default member initializers inside an enclosing class. Always split into two constructors:
   ```cpp
   ClassName();                    // uses default Config{}
   explicit ClassName(Config cfg); // takes explicit config
   ```

5. **`CMAKE_NO_SYSTEM_FROM_IMPORTED ON`:** Must be set in `CMakeLists.txt`. Prevents `/usr/local/include` header contamination (old fmt/gtest headers override Homebrew). Already set — never remove it.

6. **GTest include dirs fix:** `cmake/tests.cmake` strips SYSTEM include dirs from GTest targets. Do not undo this.

7. **`std::hash<Date>`:** `Date = std::chrono::sys_days`. An explicit `std::hash<trade::Date>` specialization is in `include/trade/common/types.h`. Always include `types.h` before using `Date` as `unordered_map` key.

8. **`PriceCents`:** Prices are `int64_t` cents (`kPriceScale = 10000.0`). Use `to_double()` / `to_cents()` helpers. Never store prices as raw `double` in model structs.

9. **`Eigen::MatrixXd` column-major:** LightGBM C API expects column-major data. `Eigen::MatrixXd` is column-major by default — always use this type. Never use `Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>` for feature matrices passed to LightGBM.

10. **LightGBM C API signatures:**
    - `LGBM_BoosterSaveModelToString` has **7 parameters** (not 6)
    - `feature_importance_named` takes `int` (0=split, 1=gain), not a string

11. **DuckDB static linking:** Always link all four targets together:
    ```cmake
    duckdb_static
    duckdb_generated_extension_loader
    parquet_extension
    core_functions_extension
    ```

12. **`market_name_from_enum`:** Defined `inline` in `instrument.h`. Do not redefine locally anywhere.

13. **Module object pattern:** Each module is an OBJECT library registered via `trade_add_object_module`. All OBJECT libs aggregate into `trade_core` static library. New source files must be added to `cmake/trade_core_sources.cmake` under the appropriate module list — never link a `.cpp` file directly to a target.

14. **New test files:** Register via `add_trade_test(test_name tests/unit/test_name.cpp)` in `cmake/tests.cmake`. Wrap LightGBM-dependent tests in `if(HAVE_LIGHTGBM)`.

15. **spdlog/fmt ABI:** `spdlog` must be compiled against the same `fmt` major version. After `brew upgrade fmt`, always run `brew reinstall spdlog` and `brew reinstall ccache`.

### Python Rules

1. **`from __future__ import annotations`:** Every Python file starts with this import (first non-comment line).

2. **akshare lazy import:** `import akshare as ak` goes inside functions, never at module top level. This avoids mandatory load-time dependency.

3. **Symbol format:** Canonical symbol is `NNNNNN.SH` / `.SZ` / `.BJ`. Always normalize via `_ensure_symbol()`. Exchange suffix inference rule: first digit 6 or 9 → `.SH`, first digit 4 or 8 → `.BJ`, otherwise → `.SZ`.

4. **Parquet column order:** Must match C++ `ParquetReader` exactly:
   ```
   symbol, date, open, high, low, close, volume, amount, turnover_rate, prev_close, vwap
   ```
   Never reorder or omit columns.

5. **Volume units:** Volume in 手 (lots = 100 shares), amount in 元 (CNY). Do NOT convert to shares or thousands.

6. **SQLite schema compatibility:** `InstrumentsDB` must use the exact DDL as C++ `MetadataStore`. Market enum integers must match `types.h`. Add new columns with `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` to avoid breaking existing DB files.

7. **Watermark keys:** Use `source="akshare"`, `dataset="kline"` (or appropriate dataset name). Must be consistent across all fetch calls to the same dataset.

8. **Monthly Parquet partitioning:** `data/kline/YYYY-MM/{symbol}.parquet` where symbol dots are replaced by underscores: `symbol.replace(".", "_")`.

9. **Rate limiting:** Always add `time.sleep(delay_ms / 1000.0)` between akshare API calls. Default delay: 200 ms. Never issue back-to-back requests without a sleep.

10. **Logger pattern:** `logger = logging.getLogger(__name__)` at module level. Use `logger.info/warning/error/debug`. Never use `print()` for operational messages.

---

## Architecture Boundary Rules

1. **Hot path → C++ only:** DuckDB reads, `TechnicalSignal`, `EventPropagator`, `FeatureExtractor`, ONNX inference, `RiskManager`. Any code executing at decision time must be C++.

2. **Cold path → Python only:** LLM API calls (Claude Haiku via `anthropic`), LightGBM training → ONNX export (C++ loads the ONNX model), data ingestion via akshare (T+1 only), Jupyter notebooks, visualization.

3. **Data ingestion is Python-only:** The C++ provider/collector files (`src/provider/`, `src/collector/`, `src/app/pipelines/download_pipeline.cpp`) are excluded from CMake compilation. Do not re-add them to `cmake/trade_core_sources.cmake`. The `cmd_collect` and `cmd_silver` CLI commands deliberately print a deprecation message and return 1.

4. **Feature count contract:** `N_FEATURES = 55` (Groups A–F: 7+5+25+6+8+4). Any change to the feature set must update **both** C++ `FeatureExtractor` and Python `feature_builder.py` simultaneously, and update `N_FEATURES`.

5. **Config system:** YAML files in `config/`. C++ loads via `Config::load(path)`. Python reads the same YAML directly with `yaml.safe_load()`. Config structs are in `include/trade/common/config.h`.

---

## Data Layout Rules

| Path | Contents |
|------|----------|
| `data/.metadata/trade.db` | DuckDB/SQLite metadata DB — shared between C++ `MetadataStore` and Python `InstrumentsDB` |
| `data/kline/YYYY-MM/{symbol}.parquet` | Monthly OHLCV Parquet (monthly partitioned) |
| `data/models/` | Trained model files (LightGBM `.txt`, ONNX `.onnx`) |
| `data/sentiment/` | Bronze/Silver/Gold sentiment pipeline outputs |
| `data/journal/decisions.parquet` | Decision journal (logged by `decision_journal.py`) |

---

## Code Organization

### Directory Layout
```
include/trade/{module}/          # C++ headers (#pragma once + namespace trade)
src/{module}/                    # C++ implementations
tests/unit/test_{module}.cpp     # GoogleTest unit tests
tests/integration/               # Integration tests
cmake/trade_core_sources.cmake   # Source file registry for all modules
cmake/tests.cmake                # Test registration
python/trade_py/                 # Python glue layer modules
python/scripts/                  # Python CLI entry points
config/                          # YAML configuration files
vendor/duckdb/                   # DuckDB git submodule
_bmad-output/                    # BMAD workflow outputs
```

### Naming Conventions

| Context | Convention | Example |
|---------|-----------|---------|
| C++ files | `snake_case.h` / `snake_case.cpp` | `lgbm_model.cpp` |
| C++ types (class/struct/enum) | `PascalCase` | `LgbmModel`, `Market` |
| C++ enum members | `kPascalCase` | `Market::kSH` |
| C++ private members | trailing underscore | `booster_`, `loaded_` |
| C++ constants | `kCamelCase` | `kPriceScale`, `kSWIndustryCount` |
| Python files | `snake_case.py` | `kline_fetcher.py` |
| Python classes | `PascalCase` | `KlineFetcher` |
| Python private functions | leading underscore | `_to_akshare_code()`, `_ensure_symbol()` |

### C++ Module Registration Pattern
To add a new C++ module:
1. Create `include/trade/{module}/` headers and `src/{module}/` sources
2. Add source files to `cmake/trade_core_sources.cmake` under a new `trade_add_object_module({module} ...)` block
3. Add tests via `add_trade_test(test_{module} tests/unit/test_{module}.cpp)` in `cmake/tests.cmake`
4. Do **not** create a separate CMake target — everything aggregates into `trade_core`
