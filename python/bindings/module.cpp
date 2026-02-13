#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Forward declarations of binding functions
void bind_model(nb::module_& m);
void bind_features(nb::module_& m);
void bind_ml(nb::module_& m);
void bind_risk(nb::module_& m);
void bind_regime(nb::module_& m);
void bind_backtest(nb::module_& m);
void bind_sentiment(nb::module_& m);
void bind_decision(nb::module_& m);

NB_MODULE(trade_py, m) {
    m.doc() = "C++ quantitative trading system - Python bindings";

    bind_model(m);
    bind_features(m);
    bind_ml(m);
    bind_risk(m);
    bind_regime(m);
    bind_backtest(m);
    bind_sentiment(m);
    bind_decision(m);
}
