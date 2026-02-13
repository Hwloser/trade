#include "trade/sentiment/sentiment_factor.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>

namespace trade {

SentimentFactorCalculator::SentimentFactorCalculator() : config_{} {}

SentimentFactorCalculator::SentimentFactorCalculator(Config cfg)
    : config_(std::move(cfg)) {}

std::vector<SentimentFactors> SentimentFactorCalculator::compute(
    const std::vector<NlpResult>& nlp_results,
    const std::unordered_map<Symbol, BarSeries>& bar_map) const {

    if (nlp_results.empty()) return {};

    // Step 1: Group NLP results by (symbol, date)
    // Use ordered map so dates are sorted chronologically per symbol.
    std::unordered_map<Symbol,
        std::map<Date, std::vector<NlpResult>>> grouped;

    std::set<Symbol> all_symbols;
    for (const auto& r : nlp_results) {
        grouped[r.symbol][r.date].push_back(r);
        all_symbols.insert(r.symbol);
    }

    std::vector<SentimentFactors> output;

    // Step 2: For each symbol, compute time-series factors
    for (const auto& sym : all_symbols) {
        auto& date_map = grouped[sym];
        if (date_map.empty()) continue;

        // Collect sorted dates for this symbol
        std::vector<Date> dates;
        dates.reserve(date_map.size());
        for (const auto& [d, _] : date_map) {
            dates.push_back(d);
        }

        int n = static_cast<int>(dates.size());

        // Build daily time series
        Eigen::VectorXd net_sent(n);
        Eigen::VectorXd neg_series(n);
        std::vector<double> dispersion_vec(n);

        for (int i = 0; i < n; ++i) {
            const auto& day_results = date_map[dates[i]];
            net_sent(i) = compute_net_sentiment(day_results);

            // Compute average negative probability for neg_shock
            double neg_sum = 0.0;
            for (const auto& r : day_results) {
                neg_sum += r.sentiment.negative;
            }
            neg_series(i) = neg_sum / static_cast<double>(day_results.size());

            dispersion_vec[i] = compute_source_dispersion(day_results);
        }

        // Compute factor time series
        Eigen::VectorXd neg_shock = compute_neg_shock(neg_series, config_.ema_halflife);
        Eigen::VectorXd velocity = compute_sent_velocity(net_sent);
        Eigen::VectorXd volatility = compute_sent_volatility(net_sent, config_.volatility_window);

        // Get volume series from bar_map for cross-feature factors
        Eigen::VectorXd abnormal_vol = Eigen::VectorXd::Zero(n);
        auto bar_it = bar_map.find(sym);

        if (bar_it != bar_map.end() && !bar_it->second.bars.empty()) {
            const auto& bars = bar_it->second.bars;
            // Build a date->bar lookup for this symbol
            std::unordered_map<Date, const Bar*> bar_by_date;
            for (const auto& bar : bars) {
                bar_by_date[bar.date] = &bar;
            }

            // Build volume series aligned to our dates
            Eigen::VectorXd vol_series(n);
            for (int i = 0; i < n; ++i) {
                auto vit = bar_by_date.find(dates[i]);
                vol_series(i) = (vit != bar_by_date.end())
                    ? static_cast<double>(vit->second->volume)
                    : 0.0;
            }
            abnormal_vol = compute_abnormal_volume(
                vol_series, static_cast<int>(config_.abnormal_volume_lookback));

            // Assemble SentimentFactors per day
            for (int i = 0; i < n; ++i) {
                SentimentFactors sf;
                sf.symbol = sym;
                sf.date = dates[i];
                sf.net_sentiment = net_sent(i);
                sf.neg_shock = neg_shock(i);
                sf.sent_velocity = velocity(i);
                sf.sent_volatility = volatility(i);
                sf.source_dispersion = dispersion_vec[i];

                // Cross features
                auto vit = bar_by_date.find(dates[i]);
                if (vit != bar_by_date.end()) {
                    sf.sent_volume_cross = net_sent(i) * abnormal_vol(i);
                    sf.sent_turnover_cross = net_sent(i) * vit->second->turnover_rate;
                }

                output.push_back(sf);
            }
        } else {
            // No bar data -- fill without cross features
            for (int i = 0; i < n; ++i) {
                SentimentFactors sf;
                sf.symbol = sym;
                sf.date = dates[i];
                sf.net_sentiment = net_sent(i);
                sf.neg_shock = neg_shock(i);
                sf.sent_velocity = velocity(i);
                sf.sent_volatility = volatility(i);
                sf.source_dispersion = dispersion_vec[i];
                sf.sent_volume_cross = 0.0;
                sf.sent_turnover_cross = 0.0;
                output.push_back(sf);
            }
        }
    }

    return output;
}

SentimentFactorCalculator::MatrixOutput SentimentFactorCalculator::to_matrix(
    const std::vector<SentimentFactors>& factors,
    const std::vector<Symbol>& symbols,
    const std::vector<Date>& dates) const {
    MatrixOutput output;
    output.factor_names = factor_names();
    output.symbols = symbols;
    output.dates = dates;

    int num_rows = static_cast<int>(symbols.size() * dates.size());
    output.matrix = Eigen::MatrixXd::Zero(num_rows, kNumFactors);

    // Build lookup: (symbol, date) -> index into factors vector
    // Row layout: for symbol i, date j, row = i * dates.size() + j
    std::unordered_map<Symbol, size_t> sym_idx;
    for (size_t i = 0; i < symbols.size(); ++i) {
        sym_idx[symbols[i]] = i;
    }
    std::unordered_map<Date, size_t> date_idx;
    for (size_t j = 0; j < dates.size(); ++j) {
        date_idx[dates[j]] = j;
    }

    for (const auto& f : factors) {
        auto si = sym_idx.find(f.symbol);
        auto di = date_idx.find(f.date);
        if (si == sym_idx.end() || di == date_idx.end()) continue;

        int row = static_cast<int>(si->second * dates.size() + di->second);
        if (row >= num_rows) continue;

        output.matrix(row, 0) = f.net_sentiment;
        output.matrix(row, 1) = f.neg_shock;
        output.matrix(row, 2) = f.sent_velocity;
        output.matrix(row, 3) = f.sent_volatility;
        output.matrix(row, 4) = f.source_dispersion;
        output.matrix(row, 5) = f.sent_volume_cross;
        output.matrix(row, 6) = f.sent_turnover_cross;
    }

    return output;
}

std::vector<std::string> SentimentFactorCalculator::factor_names() {
    return {
        "net_sentiment",
        "neg_shock",
        "sent_velocity",
        "sent_volatility",
        "source_dispersion",
        "sent_volume_cross",
        "sent_turnover_cross"
    };
}

double SentimentFactorCalculator::compute_net_sentiment(
    const std::vector<NlpResult>& day_results) {
    if (day_results.empty()) return 0.0;

    double pos_sum = 0.0, neg_sum = 0.0, neu_sum = 0.0;
    for (const auto& r : day_results) {
        pos_sum += r.sentiment.positive;
        neg_sum += r.sentiment.negative;
        neu_sum += r.sentiment.neutral;
    }
    double total = pos_sum + neg_sum + neu_sum;
    if (total <= 0.0) return 0.0;
    return (pos_sum - neg_sum) / total;
}

Eigen::VectorXd SentimentFactorCalculator::compute_neg_shock(
    const Eigen::VectorXd& neg_series, int halflife) {
    int n = static_cast<int>(neg_series.size());
    Eigen::VectorXd result(n);
    if (n == 0) return result;

    // EMA decay factor: alpha = 1 - exp(-ln(2) / halflife)
    double alpha = 1.0 - std::exp(-std::log(2.0) / static_cast<double>(halflife));

    // Compute EMA
    Eigen::VectorXd ema(n);
    ema(0) = neg_series(0);
    for (int i = 1; i < n; ++i) {
        ema(i) = alpha * neg_series(i) + (1.0 - alpha) * ema(i - 1);
    }

    // neg_shock = neg_t - EMA(neg, halflife)
    for (int i = 0; i < n; ++i) {
        result(i) = neg_series(i) - ema(i);
    }
    return result;
}

Eigen::VectorXd SentimentFactorCalculator::compute_sent_velocity(
    const Eigen::VectorXd& sentiment_series) {
    // First difference of net_sentiment
    int n = static_cast<int>(sentiment_series.size());
    if (n <= 1) return Eigen::VectorXd::Zero(n);

    Eigen::VectorXd velocity(n);
    velocity(0) = 0.0;
    for (int i = 1; i < n; ++i) {
        velocity(i) = sentiment_series(i) - sentiment_series(i - 1);
    }
    return velocity;
}

Eigen::VectorXd SentimentFactorCalculator::compute_sent_volatility(
    const Eigen::VectorXd& sentiment_series, int window) {
    int n = static_cast<int>(sentiment_series.size());
    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
    if (n == 0 || window <= 1) return result;

    for (int i = 0; i < n; ++i) {
        // Determine the window range: [start, i] inclusive
        int start = std::max(0, i - window + 1);
        int count = i - start + 1;

        if (count < 2) {
            result(i) = 0.0;
            continue;
        }

        // Compute mean
        double sum = 0.0;
        for (int j = start; j <= i; ++j) {
            sum += sentiment_series(j);
        }
        double mean = sum / static_cast<double>(count);

        // Compute variance
        double var = 0.0;
        for (int j = start; j <= i; ++j) {
            double diff = sentiment_series(j) - mean;
            var += diff * diff;
        }
        // Use sample std dev (count - 1)
        var /= static_cast<double>(count - 1);
        result(i) = std::sqrt(var);
    }
    return result;
}

double SentimentFactorCalculator::compute_source_dispersion(
    const std::vector<NlpResult>& day_results) {
    if (day_results.size() <= 1) return 0.0;

    // Compute std of net_score across sources
    std::vector<double> scores;
    scores.reserve(day_results.size());
    for (const auto& r : day_results) {
        scores.push_back(r.sentiment.net_score());
    }

    double mean = std::accumulate(scores.begin(), scores.end(), 0.0)
                  / static_cast<double>(scores.size());
    double var = 0.0;
    for (double s : scores) {
        double diff = s - mean;
        var += diff * diff;
    }
    var /= static_cast<double>(scores.size());
    return std::sqrt(var);
}

Eigen::VectorXd SentimentFactorCalculator::compute_abnormal_volume(
    const Eigen::VectorXd& volume_series, int lookback) {
    int n = static_cast<int>(volume_series.size());
    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
    if (n == 0 || lookback <= 1) return result;

    for (int i = 0; i < n; ++i) {
        // Window: [start, i-1] -- use past data only (exclude current)
        int start = std::max(0, i - lookback);
        int count = i - start; // number of historical observations

        if (count < 2) {
            result(i) = 0.0;
            continue;
        }

        // Compute mean and std of volume over the lookback window
        double sum = 0.0;
        for (int j = start; j < i; ++j) {
            sum += volume_series(j);
        }
        double mean = sum / static_cast<double>(count);

        double var = 0.0;
        for (int j = start; j < i; ++j) {
            double diff = volume_series(j) - mean;
            var += diff * diff;
        }
        double std_dev = std::sqrt(var / static_cast<double>(count));

        if (std_dev > 0.0) {
            result(i) = (volume_series(i) - mean) / std_dev;
        } else {
            result(i) = 0.0;
        }
    }
    return result;
}

} // namespace trade
