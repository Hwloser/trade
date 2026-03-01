# UI 页面规划 + 数据架构规划

**版本：** 1.0
**日期：** 2026-03-01
**关联：** project-evolution-plan.md（Phase 7-11）

---

## 一、UI 四个页面（Tab）详细规划

### Tab 1 📋 今日晨报（Morning Brief）
> **每天用这里开始你的交易日，5分钟决策完毕**

**内容结构：**
```
┌─ 今日作战简报 2026-03-01 ────────────────────────┐
│ 宏观环境  ██████░░░░  6.2/10                     │
│  └─ 避险情绪升温 | 北向持续流出 | 美股夜盘+0.3%  │
│                                                  │
│ 今日建议：⚠️ 观望                                │
│  根因：散热板块融资余额增速创3月新高（分发信号）  │
│                                                  │
│ 重点关注（最多3只）：                            │
│  600000.SH  窗口质量 73/100  [等换手率降低]      │
│  000858.SZ  窗口质量 81/100  [大单净流入第3天]   │
│                                                  │
│ 昨日预测回溯：                                   │
│  000858.SZ 预测↑62% → 实际+1.8% ✅              │
│  600000.SH 预测↑55% → 实际-0.3% ❌              │
│                                                  │
│ 今日情报（3句话）：                              │
│  1. 证监会公告措辞从"引导"→"规范"（监管收紧早期）│
│  2. OpenAI 宣布扩建数据中心 → 关注电力板块       │
│  3. 美联储会议纪要今晚发布 → 注意黄金/汇率波动   │
└──────────────────────────────────────────────────┘
```

**依赖数据：**
- 宏观环境分：`data/signals/macro/YYYY-MM-DD.json`
- 窗口质量分：`data/signals/window/YYYY-MM-DD.parquet`
- 昨日回溯：`data/journal/decisions.parquet`
- 三句话情报：`data/briefs/intelligence/YYYY-MM-DD.md`（LLM生成）

**触发时间：** 每个交易日 9:10 自动生成，9:20 可读

---

### Tab 2 📊 标的分析（Stock Analysis）
> **输入一只股票，获得完整的判决书**

**内容结构：**
```
┌─ 标的分析 ──────────────────────────────────────┐
│ 输入：[600000.SH    ] [分析] [加入自选]          │
├─────────────────────────────────────────────────┤
│ K线图（带量能、MACD）     [1M] [3M] [6M] [1Y]  │
│ ══════════════════════════════════════════        │
│                                                  │
│ 判决书                                           │
│  概率：看涨 62%（置信区间 55%-69%）              │
│  窗口质量：81/100                                │
│  行动条件："若大盘跌幅<0.5%且价格>23.5，可建仓"  │
│                                                  │
│ 支持证据 ✅                    反驳证据 ❌        │
│  大单净流入第3天                RSI超买(72)       │
│  换手率骤降至2.1%              北向小幅流出       │
│  价格在区间震荡5日             市盈率偏高         │
│                                                  │
│ 魔鬼代理人 👹                                    │
│  "反方认为：该股过去3次类似形态中2次在5日内      │
│   继续下跌。当前融资余额仍在高位，浮筹未出清。"  │
│                                                  │
│ 风险指标                                         │
│  日波动率 1.8% | 年化波动率 28% | VaR99% 4.2%   │
│  Quarter Kelly 仓位建议：12%                     │
│                                                  │
│ [记录决策] [加入等待池]                          │
└──────────────────────────────────────────────────┘
```

**依赖数据：**
- K线：`data/kline/YYYY-MM/{symbol}.parquet`
- 特征/信号：调用 `trade_cli features --symbol XXX`
- 风险指标：调用 `trade_cli risk --symbol XXX`
- 魔鬼代理人：Python LLM 调用（Claude Haiku）
- 基本面：`data/fundamentals/{symbol}.parquet`
- 资金流：`data/fund_flow/{symbol}.parquet`

---

### Tab 3 ⚡ 信号监控（Signal Monitor）
> **盘中使用，自选股的实时信号面板**

**内容结构：**
```
┌─ 信号监控（每5分钟刷新）────────────────────────┐
│ 上次更新：10:35:02   [手动刷新]                 │
│                                                  │
│ 自选池                                           │
│  股票        窗口分  信号      变化              │
│  600000.SH   73     大单转正  +0.8% ↑           │
│  000858.SZ   81     建仓中    +1.2% ↑ ⭐        │
│  300750.SZ   45     观望      -0.3% ↓           │
│  002594.SZ   38     分发区间  +2.1% ↑ ⚠️        │
│                                                  │
│ 宏观快讯                                         │
│  10:20 美元指数突破104 → 避险情绪+              │
│  09:45 煤炭板块大单净流入排名第1                 │
│                                                  │
│ 跨资产                                           │
│  黄金  +0.4%   BTC  +1.2%   原油  -0.8%        │
│  避险联动强度：中等 (42/100)                     │
│                                                  │
│ 今日不操作的理由：                               │
│  大盘成交量不足，整体窗口质量偏低                │
└──────────────────────────────────────────────────┘
```

**依赖数据：**
- 盘中实时：akshare 实时行情 API（每5分钟轮询）
- 跨资产：akshare 期货/黄金/外汇数据
- 宏观快讯：RSS 聚合 + NLP 实时筛选
- 自选池：`data/.metadata/trade.db` watchlist 表

**注意：** 非交易时段自动停止轮询，显示"收盘状态"

---

### Tab 4 📓 决策日志（Decision Journal）
> **复盘、学习、发现自己的偏差**

**内容结构：**
```
┌─ 决策日志 ──────────────────────────────────────┐
│ [新增决策记录]                                   │
│                                                  │
│ 近期决策                                         │
│  日期       股票      方向  概率  实际   结果     │
│  2026-02-28 000858.SZ 买入  62%  +1.8%  ✅      │
│  2026-02-27 600000.SH 买入  55%  -0.3%  ❌      │
│  2026-02-25 300750.SZ 观望  --   -2.1%  ✅避开  │
│                                                  │
│ 我的偏差分析                                     │
│  ⚠ 你在科技板块的胜率 34%（系统性过度乐观）      │
│  ⚠ 你的止损设得太窄：平均持仓3天就止损           │
│  ✅ 你的"观望"判断准确率 71%                     │
│                                                  │
│ 月度绩效                                         │
│  总信号 24 | 操作 8 | 胜率 62.5% | 避开损失 4次  │
│  "不做"信号准确率：71%                           │
│                                                  │
│ [导出报告]  [查看历史]                           │
└──────────────────────────────────────────────────┘
```

**依赖数据：**
- 决策记录：`data/journal/decisions.parquet`（已有）
- 回溯结果：T+5/T+20/T+60 自动写入
- 偏差分析：Python 统计计算

---

## 二、数据现状分析（需先清理）

### 当前发现的两套并行结构 ⚠️

```
data/
  raw/cn_a/daily/2026/bucket=XX/part-000.parquet   ← 旧格式（32个bucket分区）
  kline/YYYY-MM/{symbol}.parquet                    ← 新格式（akshare月度分区）
  metadata.db                                       ← 旧metadata DB
  .metadata/trade.db                                ← 新metadata DB（C++/Python共用）
  knowledge_graph/sector_graph.json                 ← 知识图谱
```

**问题：** 两个 DB，两套 kline 格式，C++ 读新格式，旧格式是遗留物

### 数据清理行动

| 路径 | 处置 |
|------|------|
| `data/raw/` | **归档或删除** — 旧 C++ 采集管道遗留，akshare 已完全替代 |
| `data/metadata.db` | **删除** — 与 `.metadata/trade.db` 重复，旧的 |
| `data/kline/` | **保留** — 主要 kline 存储，继续使用 |
| `data/.metadata/trade.db` | **保留** — C++ MetadataStore + Python InstrumentsDB 共用 |
| `data/knowledge_graph/` | **保留并扩展** — Phase 9 会升级为三层图 |

---

## 三、目标数据架构

```
data/
  ├── .metadata/
  │   └── trade.db              # 主数据库（SQLite）
  │       表: instruments         股票列表
  │       表: downloads           下载记录/水位
  │       表: watermarks          增量更新水位
  │       表: watchlist           自选池（新增）
  │       表: signal_cache        信号缓存（新增）
  │
  ├── kline/                    # K线数据（按月分区）
  │   └── YYYY-MM/
  │       └── {SYMBOL}.parquet  # 如 600000_SH.parquet
  │
  ├── fundamentals/             # 基本面数据（Phase 5已有）
  │   └── {SYMBOL}.parquet      # PE/PB/ROE/现金流等
  │
  ├── fund_flow/                # 资金流数据（Phase 6已有）
  │   └── {SYMBOL}.parquet      # 大单/超大单净流入
  │
  ├── sentiment/                # 情感分析（Phase 2已有）
  │   ├── bronze/               # 原始RSS/新闻
  │   ├── silver/               # 清洗后
  │   └── gold/                 # 最终情感分数
  │
  ├── signals/                  # 计算后的信号（新增）
  │   ├── window/
  │   │   └── YYYY-MM-DD.parquet  # 每日窗口质量分（全市场扫描）
  │   ├── smart_money/
  │   │   └── YYYY-MM-DD.parquet  # 机构建仓信号
  │   └── macro/
  │       └── YYYY-MM-DD.json     # 宏观环境评分
  │
  ├── cross_asset/              # 跨资产数据（新增，Phase 9）
  │   ├── gold.parquet          # 黄金日线
  │   ├── btc.parquet           # BTC日线
  │   └── fx_cnh.parquet        # 美元/人民币汇率
  │
  ├── macro_events/             # 宏观事件记录（新增，Phase 9）
  │   └── events.parquet        # 事件类型/强度/发生时间/资产影响
  │
  ├── briefs/                   # 晨报输出（新增，Phase 7）
  │   ├── YYYY-MM-DD.md         # 晨报正文（Markdown）
  │   └── intelligence/
  │       └── YYYY-MM-DD.md     # 三句话情报（LLM生成）
  │
  ├── journal/                  # 决策日志（Phase 6已有）
  │   └── decisions.parquet     # 所有决策记录+回溯结果
  │
  ├── models/                   # 模型文件
  │   ├── lgbm_factor_v1.model
  │   └── *.onnx
  │
  └── knowledge_graph/          # 知识图谱（Phase 3已有，Phase 9升级）
      └── sector_graph.json     # 三层图：事件→资产→边权重
```

---

## 四、数据来源与更新调度

### 数据源清单

| 数据 | 来源 | 更新频率 | 方法 |
|------|------|---------|------|
| K线 OHLCV | akshare `stock_zh_a_hist` | 交易日收盘后 | `kline_fetcher.py` |
| 基本面 | akshare `stock_financial_abstract_ths` | 季报后 | `fundamental_fetcher.py` |
| 资金流 | akshare `stock_individual_fund_flow` | 交易日收盘后 | `fund_flow_fetcher.py` |
| 情感/舆情 | RSS + Claude Haiku | 每日 | `run_sentiment.py` |
| 黄金日线 | akshare `spot_hist_sge` | 交易日 | `cross_asset_fetcher.py`（新建）|
| BTC日线 | akshare `crypto_hist_okex` | 每日 | 同上 |
| 美元/人民币 | akshare `currency_hist` | 交易日 | 同上 |
| 宏观新闻 | RSS（财新/Reuters/Bloomberg中文）| 每小时 | `vip_monitor.py`（新建）|
| 监管公告 | 证监会/上交所/深交所官网 | 每日 | `regulatory_monitor.py`（新建）|

### 每日调度时间表

```
02:00  cross_asset_fetcher.py     → 更新黄金/BTC/汇率（夜盘数据）
07:00  regulatory_monitor.py     → 抓取昨日晚间监管公告
08:00  intelligence_digest.py    → LLM生成三句话情报
08:30  run_sentiment.py          → 情感分析
09:05  kline_fetcher.py          → 更新昨日K线（T+1）
09:05  fund_flow_fetcher.py      → 更新昨日资金流
09:10  window_scorer（C++桥接）  → 全自选池窗口质量评分
09:15  morning_brief.py          → 生成晨报 Markdown
09:20  [用户打开 Streamlit]      ← 读取完成的晨报

10:00-15:00  盘中每5分钟：
       signal_scanner（C++桥接） → 扫描异动信号（仅自选池）

15:15  fund_flow_fetcher.py      → 更新当日资金流（盘后数据）
15:20  fundamental_fetcher.py    → 检查是否有新财报发布
15:30  prediction_retrospective.py → T+5/T+20/T+60 回溯写入 journal
```

---

## 五、.metadata/trade.db 新增表

在现有 `instruments`、`downloads`、`watermarks` 三表基础上新增：

```sql
-- 自选池
CREATE TABLE IF NOT EXISTS watchlist (
    symbol      TEXT PRIMARY KEY,
    added_at    DATE NOT NULL,
    note        TEXT,
    active      INTEGER NOT NULL DEFAULT 1
);

-- 信号缓存（避免重复计算，每日覆盖）
CREATE TABLE IF NOT EXISTS signal_cache (
    date        DATE NOT NULL,
    symbol      TEXT NOT NULL,
    window_score INTEGER,
    smart_money_signal INTEGER,
    large_order_trend TEXT,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);

-- 宏观事件记录
CREATE TABLE IF NOT EXISTS macro_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_date  DATE NOT NULL,
    event_type  TEXT NOT NULL,   -- 'geopolitical'/'policy'/'tech_shock'等
    source      TEXT,            -- 事件来源描述
    intensity   REAL,            -- 0.0-1.0
    affected_assets TEXT,        -- JSON数组 ["gold","btc","energy"]
    notes       TEXT
);
```

---

## 六、实施顺序（对应 Phase 7 优先）

```
Week 1：数据清理 + UI骨架
  1. 删除 data/raw/ 和 data/metadata.db（旧遗留）
  2. 建 data/signals/、data/cross_asset/、data/briefs/ 目录结构
  3. 新增 watchlist + signal_cache + macro_events 表（ALTER/CREATE）
  4. 建 Streamlit 骨架（4个空 tab，能跑起来）

Week 2：Tab 1 晨报 + 数据采集扩展
  5. cross_asset_fetcher.py（黄金/BTC/汇率）
  6. morning_brief.py（晨报生成器）
  7. Tab 1 接通数据显示

Week 3：Tab 2 标的分析
  8. cpp_bridge.py（封装 trade_cli subprocess）
  9. Tab 2 K线图 + 判决书展示

Week 4：Tab 3 信号监控 + Tab 4 决策日志
  10. Tab 3 自选池面板（盘中轮询）
  11. Tab 4 决策记录 + 偏差分析
```
