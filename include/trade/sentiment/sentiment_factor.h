#pragma once

#include "trade/common/types.h"
#include "trade/model/bar.h"
#include "trade/sentiment/sentiment_model.h"
#include "trade/sentiment/text_source.h"

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

namespace trade {

// ============================================================================
// NlpResult: aggregated NLP output for one symbol on one day
// ============================================================================
struct NlpResult {
    Symbol symbol;
    Date date;
    std::string source;                       // originating source name
    SentimentResult sentiment;                // model output
    int article_count = 0;                    // number of texts for this symbol
};

// ============================================================================
// SentimentFactors: computed sentiment-based factors for one symbol on one day
// ============================================================================
struct SentimentFactors {
    Symbol symbol;
    Date date;

    // Core sentiment factors
    double net_sentiment = 0.0;       // (pos - neg) / (pos + neg + neu)
    double neg_shock = 0.0;           // neg_t - EMA(neg, 5d) -- most predictive
    double sent_velocity = 0.0;       // delta(sentiment) / delta(t)
    double sent_volatility = 0.0;     // std(sentiment, 5d)

    // Cross-source factors
    double source_dispersion = 0.0;   // std(sentiment across sources)

    // Cross-feature factors (sentiment x market data)
    double sent_volume_cross = 0.0;   // sentiment x abnormal_volume
    double sent_turnover_cross = 0.0; // sentiment x turnover_rate
};

// ============================================================================
// SentimentFactorCalculator: compute per-symbol, per-day sentiment factors
// ============================================================================
// Takes NLP results (from any ISentimentModel) and market bar data, then
// computes a set of predictive sentiment factors.
//
// Usage:
//   SentimentFactorCalculator calc;
//   auto factors = calc.compute(nlp_results, bar_data);
//   auto eigen_matrix = calc.to_matrix(factors, symbols, dates);
//
class SentimentFactorCalculator {
public:
    struct Config {
        int ema_halflife = 5;                 // days for EMA in neg_shock
        int volatility_window = 5;            // days for rolling std
        double abnormal_volume_lookback = 20; // days for volume z-score
    };

    SentimentFactorCalculator();
    explicit SentimentFactorCalculator(Config cfg);

    // -- Main computation ----------------------------------------------------

    // Compute sentiment factors for all symbols across all dates.
    // nlp_results: per-symbol, per-day NLP outputs (possibly multiple sources).
    // bar_map:     symbol -> BarSeries for market data (volume, turnover).
    // Returns a flat vector of SentimentFactors (one per symbol per day).
    std::vector<SentimentFactors> compute(
        const std::vector<NlpResult>& nlp_results,
        const std::unordered_map<Symbol, BarSeries>& bar_map) const;

    // -- Output as Eigen matrix for integration with FeatureEngine -----------

    // Convert factor vector to an Eigen matrix.
    // Rows = (symbol x date) observations in the same order as symbols/dates.
    // Columns = factor dimensions (net_sentiment, neg_shock, ...).
    // Returns the matrix and corresponding column names.
    struct MatrixOutput {
        Eigen::MatrixXd matrix;
        std::vector<std::string> factor_names;
        std::vector<Symbol> symbols;
        std::vector<Date> dates;
    };

    MatrixOutput to_matrix(
        const std::vector<SentimentFactors>& factors,
        const std::vector<Symbol>& symbols,
        const std::vector<Date>& dates) const;

    // -- Factor names --------------------------------------------------------

    // Names of all computed factors (column labels for the matrix).
    static std::vector<std::string> factor_names();

    // Number of sentiment factors produced.
    static constexpr int kNumFactors = 7;

    // -- Configuration -------------------------------------------------------
    const Config& config() const { return config_; }

private:
    // -- Individual factor computations --------------------------------------

    // net_sentiment: (pos - neg) / (pos + neg + neu) for one day.
    static double compute_net_sentiment(
        const std::vector<NlpResult>& day_results);

    // neg_shock: neg_t - EMA(neg, halflife) -- requires history.
    static Eigen::VectorXd compute_neg_shock(
        const Eigen::VectorXd& neg_series, int halflife);

    // sent_velocity: first difference of net_sentiment.
    static Eigen::VectorXd compute_sent_velocity(
        const Eigen::VectorXd& sentiment_series);

    // sent_volatility: rolling std of net_sentiment.
    static Eigen::VectorXd compute_sent_volatility(
        const Eigen::VectorXd& sentiment_series, int window);

    // source_dispersion: cross-source std for one day.
    static double compute_source_dispersion(
        const std::vector<NlpResult>& day_results);

    // abnormal_volume: (vol_t - mean_vol) / std_vol over lookback.
    static Eigen::VectorXd compute_abnormal_volume(
        const Eigen::VectorXd& volume_series, int lookback);

    Config config_;
};

} // namespace trade
