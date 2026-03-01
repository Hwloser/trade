# 项目精简与重构规划

**版本：** 1.0
**日期：** 2026-03-01
**目标：** 整洁、清晰、单一入口

---

## 一、现状问题清单

### 可以直接删除的历史遗留

| 路径 | 原因 |
|------|------|
| `data/raw/` | 旧 C++ 采集管道遗留，已被 akshare 完全替代 |
| `data/metadata.db` | 与 `data/.metadata/trade.db` 重复，旧版本 |
| `config/modules/storage_baidu.yaml` | 百度网盘已替换为 Google Drive，且文件内含硬编码密钥 ⚠️ |
| `config/modules/provider_eastmoney.yaml` | EastMoney C++ provider 已废弃，Python/akshare 替代 |
| `vcpkg.json` | 项目改用 Homebrew，vcpkg 已放弃 |
| `include/trade/provider/` | 5 个头文件，全部从 CMake 排除，已废弃 |
| `src/provider/` | 对应实现，全部废弃 |
| `include/trade/collector/` | 已废弃 |
| `src/collector/` | 已废弃 |
| `include/trade/storage/baidu_netdisk_client.h` | 百度已删除 |
| `src/storage/baidu_netdisk_client.cpp` | 同上 |
| `python/bindings/` | nanobind 未启用（BUILD_PYTHON_BINDINGS=OFF），空目录 |

### 配置文件问题

- `config/modules/data_ingestion.yaml` 里的 `raw_dir`、`silver_dir`、`write_raw_layer`、`write_silver_layer` 对应已删除的 raw/silver 层，全部无效
- C++ `Config` 结构体里的 `EastMoneyConfig`、`IngestionConfig` 的大量字段、`StorageConfig` 的百度字段全部无用
- `RiskConfig`、`BacktestConfig`、`TradingCostConfig` 应该让用户在 UI 里调，放 YAML 文件很不方便

### Python 代码散乱

- `python/scripts/` 只有 `run_collector.py`（其余脚本在 `trade_py/` 外），其他 `run_sentinel.py`、`build_graph.py` 等散落在不明目录
- `python/trade_py/` 里的 `viz.py`、`notebook_helpers.py` 是 Jupyter 专用工具，和核心逻辑混在一起
- 没有 `python/app/`，Streamlit 无处安家

---

## 二、配置分层方案

### 原则：YAML 只放"基础设施"，"用户参数"进数据库

```
YAML（很少改，改了要重启）      SQLite settings 表（随时在 UI 改）
─────────────────────────      ─────────────────────────────────
数据根目录路径                   风险参数（目标波动率、仓位上限）
Google Drive 凭证               回测参数（初始资金、最大持仓数）
Admin token                     信号阈值（窗口质量分 cutoff）
RSS feed URL 列表               调度时间（晨报生成时间）
Jin10/Xueqiu API keys          自选池（watchlist）
                                UI 偏好
```

### 精简后的 YAML（3 个文件，去掉 3 个）

**删除：**
- `config/modules/storage_baidu.yaml`
- `config/modules/provider_eastmoney.yaml`
- `config/modules/data_ingestion.yaml`（内容重写，见下）

**保留并重写：**

`config/config.yaml`（主入口）：
```yaml
# 基础设施配置（不常改的）。用户参数请在 Web UI 的"设置"页面中修改。
includes:
  - modules/storage.yaml    # Google Drive（可选）
  - modules/security.yaml   # Admin token
  - modules/sentiment.yaml  # RSS/Jin10/Xueqiu API keys

data:
  data_root: "data"
```

`config/modules/storage.yaml`（合并 google_drive，去掉百度）：
```yaml
storage:
  enabled: false
  backend: "local"          # local | google_drive
  write_mode: "local_only"
  hot_days: 30
  google_drive_key_file: ""
  google_drive_folder_id: ""
  google_drive_timeout_ms: 30000
  google_drive_retry_count: 2
```

`config/modules/security.yaml`：保持不变
`config/modules/sentiment.yaml`：保持不变（RSS feeds + API keys）

### SQLite settings 表（新增）

```sql
CREATE TABLE IF NOT EXISTS settings (
    key          TEXT PRIMARY KEY,
    value        TEXT NOT NULL,
    value_type   TEXT NOT NULL DEFAULT 'string', -- string|int|float|bool|json
    category     TEXT NOT NULL DEFAULT 'general',-- risk|backtest|signal|ui|scheduler
    label        TEXT,          -- 中文显示名，给 UI 用
    description  TEXT,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 默认值（INSERT OR IGNORE，保证升级不覆盖用户设置）
INSERT OR IGNORE INTO settings VALUES
  -- 风险参数
  ('risk.target_annual_vol',   '0.11',  'float',  'risk', '目标年化波动率', NULL, CURRENT_TIMESTAMP),
  ('risk.max_single_weight',   '0.10',  'float',  'risk', '单股最大仓位',   NULL, CURRENT_TIMESTAMP),
  ('risk.max_industry_weight', '0.35',  'float',  'risk', '行业最大仓位',   NULL, CURRENT_TIMESTAMP),
  ('risk.base_cash_pct',       '0.10',  'float',  'risk', '基础现金比例',   NULL, CURRENT_TIMESTAMP),
  -- 回测参数
  ('backtest.initial_capital', '1000000', 'float', 'backtest', '初始资金（元）', NULL, CURRENT_TIMESTAMP),
  ('backtest.max_positions',   '25',    'int',    'backtest', '最大持仓数',   NULL, CURRENT_TIMESTAMP),
  -- 信号参数
  ('signal.window_act_threshold',   '80', 'int', 'signal', '出手窗口质量分 cutoff', NULL, CURRENT_TIMESTAMP),
  ('signal.window_watch_threshold', '60', 'int', 'signal', '观察窗口质量分 cutoff', NULL, CURRENT_TIMESTAMP),
  -- 调度参数
  ('scheduler.brief_time',     '09:10', 'string', 'scheduler', '晨报生成时间', NULL, CURRENT_TIMESTAMP),
  ('scheduler.scan_interval',  '5',     'int',    'scheduler', '盘中扫描间隔（分钟）', NULL, CURRENT_TIMESTAMP),
  -- 交易成本
  ('cost.stamp_tax_rate',      '0.0005',  'float', 'risk', '印花税率',   NULL, CURRENT_TIMESTAMP),
  ('cost.commission_rate',     '0.00025', 'float', 'risk', '佣金率',     NULL, CURRENT_TIMESTAMP),
  ('cost.commission_min_yuan', '5.0',     'float', 'risk', '最低佣金（元）', NULL, CURRENT_TIMESTAMP);
```

---

## 三、精简后的 C++ Config 结构体

删除废弃字段后，`config.h` 只剩真正需要的：

```cpp
struct DataConfig {
    std::string data_root = "data";
    std::string models_dir = "models";      // 只保留 models_dir，其他路径由 StoragePath 推导
};

// IngestionConfig 完全删除（Python 端管理数据采集，C++ 不需要）

// EastMoneyConfig 完全删除（EastMoney provider 已废弃）

struct StorageConfig {
    bool enabled = false;
    std::string backend = "local";          // local | google_drive
    std::string write_mode = "local_only";
    int hot_days = 30;
    std::string google_drive_key_file = "";
    std::string google_drive_folder_id = "";
    int google_drive_timeout_ms = 30000;
    int google_drive_retry_count = 2;
    // 删除：百度字段、compaction 字段（移入内部默认值）
};

struct SecurityConfig {                     // 不变
    std::string default_role = "user";
    std::string admin_token = "";
};

struct SentimentConfig {                    // 不变（保留 RSS/API keys）
    ...
};

// RiskConfig、BacktestConfig、TradingCostConfig 从 Config 中移除
// → 改由 Python settings_db.py 从 SQLite 读取，C++ 不再需要

struct Config {
    DataConfig data;
    StorageConfig storage;
    SecurityConfig security;
    SentimentConfig sentiment;

    static Config load(const std::string& path);
    static Config defaults();
};
```

---

## 四、目标目录结构

```
trade/
│
├── CMakeLists.txt
├── CMakePresets.json
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .gitmodules
│
├── cmake/
│   ├── trade_core_sources.cmake
│   └── tests.cmake
│
├── config/                         # 基础设施配置（不常改）
│   ├── config.yaml                 # 主入口，includes 下面3个
│   └── modules/
│       ├── storage.yaml            # Google Drive（合并，去掉百度）
│       ├── security.yaml           # Admin token
│       └── sentiment.yaml          # RSS feeds / API keys
│
├── include/trade/                  # C++ 计算库头文件
│   ├── common/                     # types.h, config.h, time_utils.h
│   ├── model/                      # Bar, Instrument, Account, ...
│   ├── storage/                    # DuckDB, Parquet, MetadataStore, GoogleDrive
│   │   └── (删除 baidu_netdisk_client.h)
│   ├── features/                   # 特征计算器
│   ├── signal/                     # 信号传播
│   ├── sentiment/                  # 情感分析
│   ├── ml/                         # LightGBM, ONNX
│   ├── regime/                     # 市场状态检测
│   ├── risk/                       # VaR, Kelly, 仓位管理
│   ├── backtest/                   # 回测引擎
│   ├── decision/                   # 决策报告、信号合成
│   ├── stats/                      # 统计工具
│   ├── app/pipelines/              # 训练流水线
│   └── cli/                        # CLI 辅助（保留，支持 trade_cli 二进制）
│   └── (删除 provider/, collector/)
│
├── src/                            # C++ 实现（镜像 include/trade/）
│   └── (同上，删除对应废弃目录)
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── python/
│   │
│   ├── app/                        # Streamlit Web 应用（唯一用户入口）
│   │   ├── ui.py                   # 主入口: uv run streamlit run python/app/ui.py
│   │   ├── tabs/
│   │   │   ├── briefing.py         # Tab 1: 今日晨报
│   │   │   ├── analysis.py         # Tab 2: 标的分析
│   │   │   ├── monitor.py          # Tab 3: 信号监控
│   │   │   ├── journal.py          # Tab 4: 决策日志
│   │   │   └── settings.py         # Tab 5: 设置（读写 settings 表）
│   │   └── services/
│   │       ├── cpp_bridge.py       # subprocess 封装 trade_cli
│   │       └── briefing_gen.py     # 晨报生成逻辑
│   │
│   ├── trade_py/                   # 核心 Python 库（按功能分子目录）
│   │   ├── __init__.py
│   │   │
│   │   ├── db/                     # 数据库层
│   │   │   ├── instruments_db.py   # 股票列表、水位（现有）
│   │   │   └── settings_db.py      # 用户设置读写（新增）
│   │   │
│   │   ├── data/                   # 数据采集
│   │   │   ├── kline_fetcher.py
│   │   │   ├── fundamental_fetcher.py
│   │   │   ├── fund_flow_fetcher.py
│   │   │   └── cross_asset_fetcher.py   # 新增（黄金/BTC/汇率）
│   │   │
│   │   ├── signals/                # 信号计算
│   │   │   ├── window_scorer.py         # 新增（窗口质量分）
│   │   │   └── smart_money.py           # 新增（机构建仓识别）
│   │   │
│   │   ├── analysis/               # 分析与模型
│   │   │   ├── feature_builder.py
│   │   │   ├── model_trainer.py
│   │   │   ├── label_builder.py
│   │   │   └── knowledge_graph.py
│   │   │
│   │   ├── intelligence/           # LLM / NLP 情报
│   │   │   ├── claude_client.py
│   │   │   ├── rss_fetcher.py
│   │   │   ├── sentiment_pipeline.py
│   │   │   └── nlp_train.py
│   │   │
│   │   └── journal/                # 决策追踪
│   │       └── decision_journal.py
│   │
│   └── scripts/                    # 批处理脚本（调度器调用）
│       ├── scheduler.py            # 新增（统一调度入口）
│       ├── run_collector.py        # 数据采集
│       ├── run_sentiment.py        # 情感分析
│       └── build_graph.py          # 知识图谱
│
├── notebooks/                      # Jupyter 探索（不影响生产）
│   └── (notebook_helpers.py、viz.py 移至这里)
│
├── docs/
│
└── vendor/duckdb/                  # git submodule
```

---

## 五、Python 文件迁移对照表

| 原位置 | 新位置 | 处置 |
|--------|--------|------|
| `trade_py/instruments_db.py` | `trade_py/db/instruments_db.py` | 移动 |
| `trade_py/kline_fetcher.py` | `trade_py/data/kline_fetcher.py` | 移动 |
| `trade_py/fundamental_fetcher.py` | `trade_py/data/fundamental_fetcher.py` | 移动 |
| `trade_py/fund_flow_fetcher.py` | `trade_py/data/fund_flow_fetcher.py` | 移动 |
| `trade_py/feature_builder.py` | `trade_py/analysis/feature_builder.py` | 移动 |
| `trade_py/model_trainer.py` | `trade_py/analysis/model_trainer.py` | 移动 |
| `trade_py/label_builder.py` | `trade_py/analysis/label_builder.py` | 移动 |
| `trade_py/knowledge_graph.py` | `trade_py/analysis/knowledge_graph.py` | 移动 |
| `trade_py/claude_client.py` | `trade_py/intelligence/claude_client.py` | 移动 |
| `trade_py/rss_fetcher.py` | `trade_py/intelligence/rss_fetcher.py` | 移动 |
| `trade_py/sentiment_pipeline.py` | `trade_py/intelligence/sentiment_pipeline.py` | 移动 |
| `trade_py/nlp_train.py` | `trade_py/intelligence/nlp_train.py` | 移动 |
| `trade_py/decision_journal.py` | `trade_py/journal/decision_journal.py` | 移动 |
| `trade_py/viz.py` | `notebooks/viz.py` | 移动（Jupyter 专用） |
| `trade_py/notebook_helpers.py` | `notebooks/notebook_helpers.py` | 移动 |
| `trade_py/report_generator.py` | `trade_py/journal/report_generator.py` | 移动 |
| `trade_py/event_db.py` | `trade_py/db/event_db.py` | 移动 |
| — | `trade_py/db/settings_db.py` | **新增** |
| — | `trade_py/data/cross_asset_fetcher.py` | **新增** |
| — | `trade_py/signals/window_scorer.py` | **新增** |
| — | `trade_py/signals/smart_money.py` | **新增** |
| — | `python/app/ui.py` | **新增** |
| — | `python/scripts/scheduler.py` | **新增** |

---

## 六、新增第5个 Tab：设置

在 Streamlit Web UI 中新增设置 Tab，让用户直接在页面中修改参数：

```
┌─ ⚙️ 设置 ──────────────────────────────────────────┐
│                                                    │
│ 风险参数                                           │
│  目标年化波动率        [  11  ] %                  │
│  单股最大仓位          [  10  ] %                  │
│  行业最大仓位          [  35  ] %                  │
│  基础现金比例          [  10  ] %                  │
│                                                    │
│ 信号参数                                           │
│  出手窗口质量分 cutoff [  80  ] / 100              │
│  观察窗口质量分 cutoff [  60  ] / 100              │
│                                                    │
│ 调度参数                                           │
│  晨报生成时间          [ 09:10 ]                   │
│  盘中扫描间隔          [   5  ] 分钟               │
│                                                    │
│ 交易成本                                           │
│  印花税率     [ 0.0005 ]   佣金率    [ 0.00025 ]  │
│  最低佣金     [   5    ] 元                        │
│                                                    │
│                              [保存设置]            │
└────────────────────────────────────────────────────┘
```

---

## 七、删除操作清单（执行顺序）

> 执行前确认：以下均为安全删除，不影响任何现有功能

```bash
# 1. 删除旧数据
rm -rf data/raw/
rm -f  data/metadata.db

# 2. 删除废弃配置（含敏感信息）
rm -f config/modules/storage_baidu.yaml
rm -f config/modules/provider_eastmoney.yaml

# 3. 删除废弃 C++ 代码
rm -rf include/trade/provider/
rm -rf src/provider/
rm -rf include/trade/collector/
rm -rf src/collector/
rm -f  include/trade/storage/baidu_netdisk_client.h
rm -f  src/storage/baidu_netdisk_client.cpp

# 4. 删除空目录
rm -rf python/bindings/

# 5. 删除 legacy 依赖文件
rm -f vcpkg.json
```

---

## 八、实施顺序

```
Step 1（清理，无风险）
  → 执行删除操作清单
  → 简化 config/modules/data_ingestion.yaml → 重命名不再使用的内容
  → 重写 config/config.yaml（3个模块）

Step 2（重构 Python 目录）
  → 在 trade_py/ 下建 db/ data/ signals/ analysis/ intelligence/ journal/
  → 移动文件（更新 import 路径）
  → 建 python/app/ 骨架

Step 3（简化 C++ Config）
  → 删除 EastMoneyConfig、IngestionConfig（或大幅瘦身）
  → 删除 RiskConfig、BacktestConfig、TradingCostConfig（移到 Python/SQLite）
  → 确认 340 个测试仍然通过

Step 4（新增 settings 表 + settings_db.py）
  → 建表 + 写入默认值
  → 建 settings_db.py（CRUD）
  → Streamlit 设置 Tab 接通

Step 5（Streamlit UI 骨架上线）
  → python/app/ui.py（5个 Tab 空框架）
  → uv run streamlit run python/app/ui.py 能打开
```
