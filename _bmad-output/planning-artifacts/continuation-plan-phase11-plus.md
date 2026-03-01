# Trade 项目后续计划 — Phase 11+

**版本：** 1.0
**日期：** 2026-03-02
**前置状态：** Phase 1–10 全部完成，351/351 C++ 测试通过，N_FEATURES = 57

---

## 已完成回顾

| 阶段 | 关键交付物 | 状态 |
|------|-----------|------|
| Phase 1 | DuckDB 存储、MetadataStore、月度 Parquet | ✅ |
| Phase 2 | 情感分析流水线（Claude Haiku + RSS） | ✅ |
| Phase 3 | 行业知识图谱 + C++ EventPropagator | ✅ |
| Phase 4 | Python ML 流水线（LightGBM + SHAP） | ✅ |
| Phase 5–6 | 基本面获取、N_FEATURES=55、Group F 情感质量 | ✅ |
| Phase 7 | Streamlit Web UI、晨报生成器、日度调度器 | ✅ |
| Phase 8 | SmartMoney CMF、散户热度、分发区间探测；N_FEATURES→57 | ✅ |
| Phase 9 | 跨资产信号（CrossAssetSignal）、监管语气探测 | ✅ |
| Phase 10 | Verdict JSON、魔鬼代理人、多角色辩论 | ✅ |

---

## Phase 11 — ONNX 推理热路径 ⭐ 最高优先级

> **目标：** 将 LightGBM 模型导出为 ONNX，接入 C++ `#ifdef HAVE_ONNXRUNTIME` 推理路径，彻底消除 Python 运行时依赖。

### 11.1 Python 端：模型导出
- 在 `model_trainer.py` 中增加 `export_to_onnx(model, output_path)` 方法
- 使用 `lightgbm.to_onnx(input_shape=(1, N_FEATURES))` 导出
- 输出到 `data/models/lgbm_factor_v1.onnx`
- 验证脚本：用 `onnxruntime` 对比 LightGBM 原始预测结果（误差 < 1e-5）

### 11.2 C++ 端：ONNXRuntime 推理
- 创建 `include/trade/ml/onnx_model.h` + `src/ml/onnx_model.cpp`
- 接口：`OnnxModel::load(path)`, `predict_one(Eigen::VectorXd)`, `predict_batch(MatrixXd)`
- 在 `cmd_predict` 中：有 `.onnx` 文件时优先用 ONNX，fallback 到 LightGBM
- 编译条件：`#ifdef HAVE_ONNXRUNTIME`（Homebrew `onnxruntime` 包）

### 11.3 CMake
- 在 `CMakeLists.txt` 中查找 `find_package(onnxruntime)` 并设置 `HAVE_ONNXRUNTIME`
- 新增 `add_trade_test(test_onnx tests/unit/test_onnx.cpp)`（仅在 `HAVE_ONNXRUNTIME` 时）

**验收标准：** `trade_cli predict --symbol 600036.SH` 输出 JSON 且 `method: "onnx"` 字段存在。

---

## Phase 12 — Streamlit UI 增强

> **目标：** 将 Phase 8–10 新模块（魔鬼代理人、多角色辩论、等待成本、宏观信号）接入 Web UI。

### 12.1 Tab 2 (标的分析) 增强
- 在判决书下方新增 **"👹 对抗性分析"** 折叠区域
  - 调用 `DevilAdvocate.challenge(symbol, thesis=action_condition)` 生成反驳
  - 调用 `MultiPersonaDebate.run(symbol, context)` 渲染多角色辩论
  - 有 API Key 时实时调用 Claude Haiku；无 Key 时显示规则模板

### 12.2 Tab 3 (信号监控) 增强
- 新增 **"🌍 宏观环境"** 指标卡：
  - 显示 `CrossAssetSignal.latest()` 的 `risk_on/off_score` 和 `macro_env_score`
  - 黄金/BTC/USD-CNH 近5日涨跌
  - 监管语气仪表盘（`RegulatoryToneMonitor.latest().tone_score`）

### 12.3 新增 Tab 6 (等待追踪器)
- 表格展示 `PatienceTracker.cost_all()` 结果
- 支持新增/删除自选观察记录
- 颜色编码：ABORT=红、ENTER=绿、WAIT=蓝

---

## Phase 13 — 模型漂移检测 + 自动再训练

> **目标：** 当特征分布偏移或预测准确率下降时，自动触发 Python 再训练流程。

### 13.1 漂移检测器 `python/trade_py/ml/drift_detector.py`
- Population Stability Index (PSI) 监控每个特征列
- PSI > 0.2 → 触发警告；PSI > 0.3 → 触发再训练
- 每日由调度器在 09:05 后运行

### 13.2 再训练触发器
- 在 `scheduler.py` 新增 `job_check_drift()` 在 09:20 运行
- 漂移超阈值时调用 `model_trainer.train()`
- 训练完成后自动导出 ONNX（依赖 Phase 11）

### 13.3 性能追踪
- 扩展 `decision_journal.py`：记录模型预测值 + 实际5日收益
- 新增 `model_performance_report()` 方法，计算 IC / Rank IC / Hit Rate
- 在晨报中插入昨日模型回验结果

---

## Phase 14 — 券商 API 接入（可选，需配置）

> **目标：** 在 `pre_trade_check.cpp` 验证通过后，可路由真实委托。

### 14.1 Python 端：券商适配层
- 抽象基类 `BrokerClient`（接口：`get_position`, `place_order`, `cancel_order`）
- 实现：`EastMoneyBroker`（东方财富通 EDP API）
- 配置：`config/broker.yaml`（account_id, token, paper_trading=true）

### 14.2 C++ 端：订单路由
- 在 `OrderManager` 中新增 `route_to_broker(order)` 方法
- 调用 Python 券商适配层（通过 subprocess 或 Python C API）
- 默认 `paper_trading=true`，需在配置中显式关闭

### 14.3 风险护栏
- 单笔委托金额上限 = `risk.max_single_position × NAV`
- 禁止在非交易时段发单
- 所有实盘委托写入 `data/journal/orders.parquet` 审计日志

**验收标准：** `paper_trading=true` 模式下完成一次完整买卖循环，日志记录完整。

---

## Phase 15 — 强化学习仓位调节（长期）

> **目标：** 在 ML 预测基础上，用 RL Agent 自适应调整仓位大小，替代静态 Kelly 公式。

### 15.1 环境设计
- State: `[bull_prob, annual_vol, macro_env_score, days_in_position, current_drawdown]`
- Action: `[-1, -0.5, 0, 0.5, 1]`（缩仓/半仓/不动/加仓/满仓）
- Reward: `risk_adjusted_return - transaction_cost`

### 15.2 实现选型
- 离线 RL：在历史数据上训练（Offline PPO 或 Conservative Q-Learning）
- 库：`stable_baselines3`（纯 Python，不引入 C++ 依赖）
- 导出为规则查表（避免在 C++ 热路径中嵌入神经网络）

---

## Phase 16 — L2 行情数据（长期）

> **目标：** 接入逐笔委托数据，填充当前占位符 `auction_vol_ratio`，提升 Group C 精度。

### 16.1 数据源
- akshare `stock_bid_ask_em` 获取实时五档行情
- 历史逐笔：需对接数据商（Wind/聚宽/通联）或交易所历史数据包

### 16.2 新特征
- `auction_vol_ratio`：集合竞价量 / 昨日均量（填充当前占位符 0.0）
- `order_imbalance`：(买量 - 卖量) / (买量 + 卖量)，五档加权
- `depth_asymmetry`：买一~三档累积量 vs 卖一~三档累积量比值

---

## 优先级排序

```
Phase 11 (ONNX) ──→ Phase 12 (UI增强) ──→ Phase 13 (漂移检测)
       └──────────────────────────────────────→ Phase 14 (券商接入)
                                               Phase 15 (RL) [长期]
                                               Phase 16 (L2) [长期]
```

**建议下一步：Phase 11 — 先走通 ONNX 导出验证，再接 C++ 推理。**

---

## 技术约束提醒（AI 实现时必读）

- `N_FEATURES = 57`（Groups A–G，不可随意增减，需同步 C++ 和 Python）
- 所有新 C++ 模块使用 `#pragma once` + `namespace trade`
- 新 Python 文件首行必须 `from __future__ import annotations`
- akshare 调用必须在函数内 `import akshare as ak`（非模块级）
- ONNX 特征向量列顺序必须与 Python `ALL_FEATURE_COLS` 完全一致
- 参见 `_bmad-output/project-context.md` 获取完整规则列表
