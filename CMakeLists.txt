cmake_minimum_required(VERSION 3.25)
project(trade VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Apple Clang searches /usr/local/include (built-in) before -isystem dirs.
# Use -I (not -isystem) for imported targets so Homebrew headers win.
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)

# Options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build nanobind Python bindings" OFF)

# Find required dependencies (Homebrew on macOS)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(pugixml CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(re2 CONFIG REQUIRED)

# LightGBM - optional (not in vcpkg for macOS ARM, install via brew or source)
find_package(LightGBM CONFIG QUIET)
if(NOT LightGBM_FOUND)
    # Try pkg-config or manual path
    find_library(LIGHTGBM_LIB NAMES lightgbm _lightgbm lib_lightgbm
                 PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(LIGHTGBM_INCLUDE NAMES LightGBM/c_api.h
              PATHS /opt/homebrew/include /usr/local/include)
    if(LIGHTGBM_LIB AND LIGHTGBM_INCLUDE)
        set(HAVE_LIGHTGBM ON)
        add_library(LightGBM::lightgbm UNKNOWN IMPORTED)
        set_target_properties(LightGBM::lightgbm PROPERTIES
            IMPORTED_LOCATION "${LIGHTGBM_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${LIGHTGBM_INCLUDE}")
        message(STATUS "LightGBM found at ${LIGHTGBM_LIB}")
    else()
        set(HAVE_LIGHTGBM OFF)
        message(STATUS "LightGBM not found - ML module disabled. Install: brew install lightgbm")
    endif()
else()
    set(HAVE_LIGHTGBM ON)
    message(STATUS "LightGBM found via CMake config")
endif()

if(HAVE_LIGHTGBM)
    add_compile_definitions(HAVE_LIGHTGBM)
endif()

# ONNX Runtime - optional
find_package(onnxruntime CONFIG QUIET)
if(onnxruntime_FOUND)
    set(HAVE_ONNXRUNTIME ON)
    add_compile_definitions(HAVE_ONNXRUNTIME)
    message(STATUS "ONNX Runtime found - sentiment ONNX inference enabled")
else()
    set(HAVE_ONNXRUNTIME OFF)
    message(STATUS "ONNX Runtime not found - using rule-based sentiment only")
endif()

# Core library
add_library(trade_core
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
    # Provider
    src/provider/provider_factory.cpp
    src/provider/akshare_provider.cpp
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

target_include_directories(trade_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(trade_core PUBLIC
    Arrow::arrow_shared
    Parquet::parquet_shared
    SQLite::SQLite3
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    httplib::httplib
    fmt::fmt
    yaml-cpp::yaml-cpp
    pugixml::pugixml
    CURL::libcurl
    re2::re2
)

if(HAVE_LIGHTGBM)
    target_link_libraries(trade_core PUBLIC LightGBM::lightgbm)
endif()

if(HAVE_ONNXRUNTIME)
    target_sources(trade_core PRIVATE src/sentiment/onnx_sentiment.cpp)
    target_link_libraries(trade_core PUBLIC onnxruntime::onnxruntime)
endif()

# CLI tool
add_executable(trade_cli tools/trade_cli.cpp)
target_link_libraries(trade_cli PRIVATE trade_core)

# Tests
if(BUILD_TESTING)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
    include(GoogleTest)

    # Fix GTest include: convert SYSTEM include to normal include
    # to ensure Homebrew headers are found before /usr/local/include
    get_target_property(_gtest_inc GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
    if(_gtest_inc)
        set_target_properties(GTest::gtest PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ""
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
        target_include_directories(GTest::gtest INTERFACE ${_gtest_inc})
    endif()
    get_target_property(_gmain_inc GTest::gtest_main INTERFACE_INCLUDE_DIRECTORIES)
    if(_gmain_inc)
        set_target_properties(GTest::gtest_main PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ""
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
        target_include_directories(GTest::gtest_main INTERFACE ${_gmain_inc})
    endif()

    function(add_trade_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE trade_core GTest::gtest_main)
        gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 30)
    endfunction()

    add_trade_test(test_types tests/unit/test_types.cpp)
    add_trade_test(test_bar tests/unit/test_bar.cpp)
    add_trade_test(test_parquet tests/unit/test_parquet.cpp)
    add_trade_test(test_metadata tests/unit/test_metadata.cpp)
    add_trade_test(test_normalizer tests/unit/test_normalizer.cpp)
    add_trade_test(test_validator tests/unit/test_validator.cpp)
    add_trade_test(test_features tests/unit/test_features.cpp)
    add_trade_test(test_stats tests/unit/test_stats.cpp)
    if(LightGBM_FOUND)
        add_trade_test(test_ml tests/unit/test_ml.cpp)
    endif()
    add_trade_test(test_risk tests/unit/test_risk.cpp)
    add_trade_test(test_backtest tests/unit/test_backtest.cpp)
    add_trade_test(test_sentiment tests/unit/test_sentiment.cpp)
    add_trade_test(test_decision tests/unit/test_decision.cpp)
    add_trade_test(test_regime tests/unit/test_regime.cpp)

    add_trade_test(test_pipeline tests/integration/test_pipeline.cpp)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.4.0
    )
    FetchContent_MakeAvailable(nanobind)

    nanobind_add_module(trade_py
        python/bindings/module.cpp
        python/bindings/bind_model.cpp
        python/bindings/bind_features.cpp
        python/bindings/bind_ml.cpp
        python/bindings/bind_risk.cpp
        python/bindings/bind_regime.cpp
        python/bindings/bind_backtest.cpp
        python/bindings/bind_sentiment.cpp
        python/bindings/bind_decision.cpp
    )
    target_link_libraries(trade_py PRIVATE trade_core)
endif()
