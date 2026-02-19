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

# Dependencies
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

# LightGBM - optional
find_package(LightGBM CONFIG QUIET)
if(NOT LightGBM_FOUND)
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

# Core library sources
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/trade_core_sources.cmake)
add_library(trade_core ${TRADE_CORE_SOURCES})

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

# Output layout
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CLI tool (moved to repo root)
add_executable(trade_cli trade_cli.cpp)
target_link_libraries(trade_cli PRIVATE trade_core)
target_compile_definitions(trade_cli PRIVATE
    TRADE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Symlinks in build dir
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/config
     ${CMAKE_BINARY_DIR}/config SYMBOLIC)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/data
     ${CMAKE_BINARY_DIR}/data SYMBOLIC)

# Tests
if(BUILD_TESTING)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests.cmake)
    trade_configure_tests()
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/python_bindings.cmake)
    trade_configure_python_bindings()
endif()
