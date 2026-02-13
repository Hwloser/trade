#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/eigen/dense.h>
#include "trade/sentiment/sentiment_model.h"
#include "trade/sentiment/rule_sentiment.h"
#include "trade/sentiment/text_cleaner.h"
#include "trade/sentiment/symbol_linker.h"
#include "trade/sentiment/sentiment_factor.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_sentiment(nb::module_& m) {
    auto sentiment = m.def_submodule("sentiment", "Sentiment analysis module");

    // -----------------------------------------------------------------------
    // SentimentResult
    // -----------------------------------------------------------------------

    nb::class_<trade::SentimentResult>(sentiment, "SentimentResult")
        .def(nb::init<>())
        .def_rw("positive", &trade::SentimentResult::positive)
        .def_rw("neutral", &trade::SentimentResult::neutral)
        .def_rw("negative", &trade::SentimentResult::negative)
        .def_rw("confidence", &trade::SentimentResult::confidence)
        .def("direction", &trade::SentimentResult::direction)
        .def("net_score", &trade::SentimentResult::net_score);

    // -----------------------------------------------------------------------
    // RuleSentiment
    // -----------------------------------------------------------------------

    nb::class_<trade::RuleSentiment>(sentiment, "RuleSentiment")
        .def(nb::init<>())
        .def("name", &trade::RuleSentiment::name)
        .def("predict", &trade::RuleSentiment::predict, "text"_a)
        .def("predict_batch", &trade::RuleSentiment::predict_batch, "texts"_a)
        .def("is_ready", &trade::RuleSentiment::is_ready)
        .def("load_dict", &trade::RuleSentiment::load_dict, "path"_a)
        .def("load_positive_dict", &trade::RuleSentiment::load_positive_dict, "path"_a)
        .def("load_negative_dict", &trade::RuleSentiment::load_negative_dict, "path"_a)
        .def("add_positive_word", &trade::RuleSentiment::add_positive_word,
             "word"_a, "weight"_a = 1.0)
        .def("add_negative_word", &trade::RuleSentiment::add_negative_word,
             "word"_a, "weight"_a = -1.0)
        .def("add_negation_word", &trade::RuleSentiment::add_negation_word, "word"_a)
        .def("set_negation_window", &trade::RuleSentiment::set_negation_window, "chars"_a)
        .def("negation_window", &trade::RuleSentiment::negation_window)
        .def("set_neutral_threshold", &trade::RuleSentiment::set_neutral_threshold, "t"_a)
        .def("neutral_threshold", &trade::RuleSentiment::neutral_threshold)
        .def("positive_dict_size", &trade::RuleSentiment::positive_dict_size)
        .def("negative_dict_size", &trade::RuleSentiment::negative_dict_size);

    // -----------------------------------------------------------------------
    // TextCleaner (static methods only)
    // -----------------------------------------------------------------------

    nb::class_<trade::TextCleaner>(sentiment, "TextCleaner")
        .def_static("clean", &trade::TextCleaner::clean, "raw_text"_a)
        .def_static("remove_html_tags", &trade::TextCleaner::remove_html_tags, "text"_a)
        .def_static("remove_punctuation", &trade::TextCleaner::remove_punctuation, "text"_a)
        .def_static("normalize_whitespace", &trade::TextCleaner::normalize_whitespace, "text"_a)
        .def_static("fullwidth_to_halfwidth", &trade::TextCleaner::fullwidth_to_halfwidth, "text"_a)
        .def_static("remove_stopwords",
            static_cast<std::string(*)(const std::string&)>(&trade::TextCleaner::remove_stopwords),
            "text"_a)
        .def_static("remove_stopwords_custom",
            static_cast<std::string(*)(const std::string&, const std::unordered_set<std::string>&)>(
                &trade::TextCleaner::remove_stopwords),
            "text"_a, "stopwords"_a)
        .def_static("load_stopwords", &trade::TextCleaner::load_stopwords, "path"_a)
        .def_static("normalize_financial_abbrev",
            &trade::TextCleaner::normalize_financial_abbrev, "text"_a)
        .def_static("content_hash", &trade::TextCleaner::content_hash, "text"_a);

    // -----------------------------------------------------------------------
    // SymbolMatch
    // -----------------------------------------------------------------------

    nb::enum_<trade::SymbolMatch::MatchType>(sentiment, "MatchType")
        .value("kFullName", trade::SymbolMatch::MatchType::kFullName)
        .value("kStockCode", trade::SymbolMatch::MatchType::kStockCode)
        .value("kAlias", trade::SymbolMatch::MatchType::kAlias);

    nb::class_<trade::SymbolMatch>(sentiment, "SymbolMatch")
        .def(nb::init<>())
        .def_rw("symbol", &trade::SymbolMatch::symbol)
        .def_rw("matched_text", &trade::SymbolMatch::matched_text)
        .def_rw("match_score", &trade::SymbolMatch::match_score)
        .def_rw("type", &trade::SymbolMatch::type);

    // -----------------------------------------------------------------------
    // SymbolLinker
    // -----------------------------------------------------------------------

    nb::class_<trade::SymbolLinker>(sentiment, "SymbolLinker")
        .def(nb::init<>())
        .def("build_index", &trade::SymbolLinker::build_index, "instruments"_a)
        .def("add_alias", &trade::SymbolLinker::add_alias, "symbol"_a, "alias"_a)
        .def("load_aliases", &trade::SymbolLinker::load_aliases, "path"_a)
        .def("link",
            static_cast<std::vector<trade::SymbolMatch>(trade::SymbolLinker::*)(
                const std::string&) const>(&trade::SymbolLinker::link),
            "text"_a)
        .def("link_with_universe",
            static_cast<std::vector<trade::SymbolMatch>(trade::SymbolLinker::*)(
                const std::string&, const std::vector<trade::Symbol>&) const>(
                &trade::SymbolLinker::link),
            "text"_a, "universe"_a)
        .def("link_symbols", &trade::SymbolLinker::link_symbols, "text"_a)
        .def("set_min_score", &trade::SymbolLinker::set_min_score, "score"_a)
        .def("min_score", &trade::SymbolLinker::min_score)
        .def("set_match_bare_code", &trade::SymbolLinker::set_match_bare_code, "enable"_a)
        .def("match_bare_code", &trade::SymbolLinker::match_bare_code)
        .def("index_size", &trade::SymbolLinker::index_size);

    // -----------------------------------------------------------------------
    // SentimentFactors
    // -----------------------------------------------------------------------

    nb::class_<trade::SentimentFactors>(sentiment, "SentimentFactors")
        .def(nb::init<>())
        .def_rw("symbol", &trade::SentimentFactors::symbol)
        .def_prop_rw("date",
            [](const trade::SentimentFactors& f) {
                return f.date.time_since_epoch().count();
            },
            [](trade::SentimentFactors& f, int d) {
                f.date = trade::Date(std::chrono::days(d));
            })
        .def_rw("net_sentiment", &trade::SentimentFactors::net_sentiment)
        .def_rw("neg_shock", &trade::SentimentFactors::neg_shock)
        .def_rw("sent_velocity", &trade::SentimentFactors::sent_velocity)
        .def_rw("sent_volatility", &trade::SentimentFactors::sent_volatility)
        .def_rw("source_dispersion", &trade::SentimentFactors::source_dispersion)
        .def_rw("sent_volume_cross", &trade::SentimentFactors::sent_volume_cross)
        .def_rw("sent_turnover_cross", &trade::SentimentFactors::sent_turnover_cross);

    // -----------------------------------------------------------------------
    // NlpResult
    // -----------------------------------------------------------------------

    nb::class_<trade::NlpResult>(sentiment, "NlpResult")
        .def(nb::init<>())
        .def_rw("symbol", &trade::NlpResult::symbol)
        .def_prop_rw("date",
            [](const trade::NlpResult& r) {
                return r.date.time_since_epoch().count();
            },
            [](trade::NlpResult& r, int d) {
                r.date = trade::Date(std::chrono::days(d));
            })
        .def_rw("source", &trade::NlpResult::source)
        .def_rw("sentiment", &trade::NlpResult::sentiment)
        .def_rw("article_count", &trade::NlpResult::article_count);

    // -----------------------------------------------------------------------
    // SentimentFactorCalculator
    // -----------------------------------------------------------------------

    nb::class_<trade::SentimentFactorCalculator::Config>(sentiment, "SentimentFactorConfig")
        .def(nb::init<>())
        .def_rw("ema_halflife", &trade::SentimentFactorCalculator::Config::ema_halflife)
        .def_rw("volatility_window", &trade::SentimentFactorCalculator::Config::volatility_window)
        .def_rw("abnormal_volume_lookback",
            &trade::SentimentFactorCalculator::Config::abnormal_volume_lookback);

    nb::class_<trade::SentimentFactorCalculator::MatrixOutput>(sentiment, "MatrixOutput")
        .def(nb::init<>())
        .def_rw("matrix", &trade::SentimentFactorCalculator::MatrixOutput::matrix)
        .def_rw("factor_names", &trade::SentimentFactorCalculator::MatrixOutput::factor_names)
        .def_rw("symbols", &trade::SentimentFactorCalculator::MatrixOutput::symbols)
        .def_prop_rw("dates",
            [](const trade::SentimentFactorCalculator::MatrixOutput& o) {
                std::vector<int> out;
                out.reserve(o.dates.size());
                for (const auto& d : o.dates) {
                    out.push_back(d.time_since_epoch().count());
                }
                return out;
            },
            [](trade::SentimentFactorCalculator::MatrixOutput& o, const std::vector<int>& v) {
                o.dates.clear();
                o.dates.reserve(v.size());
                for (int d : v) {
                    o.dates.push_back(trade::Date(std::chrono::days(d)));
                }
            });

    nb::class_<trade::SentimentFactorCalculator>(sentiment, "SentimentFactorCalculator")
        .def(nb::init<>())
        .def(nb::init<trade::SentimentFactorCalculator::Config>(), "config"_a)
        .def("compute", &trade::SentimentFactorCalculator::compute,
             "nlp_results"_a, "bar_map"_a)
        .def("to_matrix", [](const trade::SentimentFactorCalculator& calc,
                             const std::vector<trade::SentimentFactors>& factors,
                             const std::vector<trade::Symbol>& symbols,
                             const std::vector<int>& date_ints) {
            std::vector<trade::Date> dates;
            dates.reserve(date_ints.size());
            for (int d : date_ints) {
                dates.push_back(trade::Date(std::chrono::days(d)));
            }
            return calc.to_matrix(factors, symbols, dates);
        }, "factors"_a, "symbols"_a, "dates"_a)
        .def_static("factor_names", &trade::SentimentFactorCalculator::factor_names)
        .def("config", &trade::SentimentFactorCalculator::config,
             nb::rv_policy::reference_internal);
}
