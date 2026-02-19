set(TRADE_CORE_SOURCES
    # Common
    src/common/config.cpp
    src/common/time_utils.cpp
    # Model
    src/model/bar.cpp
    src/model/instrument.cpp
    # Storage
    src/storage/parquet_writer.cpp
    src/storage/parquet_reader.cpp
    src/storage/metadata_store.cpp
    src/storage/storage_path.cpp
    src/storage/baidu_netdisk_client.cpp
    # Provider
    src/provider/provider_factory.cpp
    src/provider/http_client.cpp
    src/provider/eastmoney_provider.cpp
    # Normalizer
    src/normalizer/bar_normalizer.cpp
    # Validator
    src/validator/data_validator.cpp
    # Collector
    src/collector/collector.cpp
    # Features
    src/features/feature_engine.cpp
    src/features/preprocessor.cpp
    src/features/momentum.cpp
    src/features/volatility.cpp
    src/features/liquidity.cpp
    src/features/fund_flow.cpp
    src/features/price_limit.cpp
    src/features/industry.cpp
    src/features/interaction.cpp
    src/features/calendar.cpp
    src/features/fundamental.cpp
    src/features/feature_monitor.cpp
    # Stats
    src/stats/descriptive.cpp
    src/stats/correlation.cpp
    src/stats/attribution.cpp
    # ML
    src/ml/lgbm_model.cpp
    src/ml/model_trainer.cpp
    src/ml/model_evaluator.cpp
    # Risk
    src/risk/covariance.cpp
    src/risk/var.cpp
    src/risk/kelly.cpp
    src/risk/position_sizer.cpp
    src/risk/drawdown.cpp
    src/risk/stress_test.cpp
    src/risk/risk_monitor.cpp
    src/risk/risk_attribution.cpp
    # Regime
    src/regime/regime_detector.cpp
    # Backtest
    src/backtest/backtest_engine.cpp
    src/backtest/broker_sim.cpp
    src/backtest/portfolio_state.cpp
    src/backtest/strategy.cpp
    src/backtest/slippage.cpp
    src/backtest/performance.cpp
    src/backtest/validation.cpp
    src/backtest/reporting.cpp
    # Sentiment
    src/sentiment/rss_source.cpp
    src/sentiment/xueqiu_source.cpp
    src/sentiment/jin10_source.cpp
    src/sentiment/text_cleaner.cpp
    src/sentiment/symbol_linker.cpp
    src/sentiment/rule_sentiment.cpp
    src/sentiment/sentiment_factor.cpp
    # Decision
    src/decision/signal_combiner.cpp
    src/decision/universe_filter.cpp
    src/decision/portfolio_opt.cpp
    src/decision/order_manager.cpp
    src/decision/pre_trade_check.cpp
    src/decision/decision_report.cpp
)
